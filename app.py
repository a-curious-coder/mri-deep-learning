from flask import Flask, render_template, request, jsonify, send_from_directory
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DEMO_MODE disables the training-trigger routes and skips importing the
# heavy training pipeline entirely - those routes call real TensorFlow
# training against a private clinical dataset this repo doesn't ship, and
# are not safe to expose on a public, resource-constrained deployment.
DEMO_MODE = os.environ.get('DEMO_MODE', 'false').lower() == 'true'

if not DEMO_MODE:
    from image_data.prepare_data import prepare_data
    from image_data.train_test_models import main as image_data_classification
    from tabular_data.tabular_data import main as tabular_data_main
    import image_data.constants as constants
else:
    class constants:
        IMAGE_SIZE = 128
        SLICE_MODE = "axial"
        TEST_SIZE = 0.2
        VAL_SIZE = 0.2

app = Flask(__name__)

# All sourced from OpenNeuro ds003592 (CC0) or this repo's own bundled
# scan - none of these were in the model's training set (Falah/Alzheimer_MRI
# on Hugging Face), so classifying them is a genuine unseen-data test rather
# than replaying something the model has already memorized.
EXAMPLES = {
    'example': {
        'label': 'Bundled example scan',
        'full': 'data/raw/example.nii',
        'preview': 'data/raw/example_preview.nii',
    },
    'sub-01': {
        'label': 'OpenNeuro sub-01 (F, 21)',
        'full': 'data/raw/examples/sub-01.nii',
        'preview': 'data/raw/examples/sub-01_preview.nii',
    },
    'sub-03': {
        'label': 'OpenNeuro sub-03 (F, 77)',
        'full': 'data/raw/examples/sub-03.nii',
        'preview': 'data/raw/examples/sub-03_preview.nii',
    },
    'sub-04': {
        'label': 'OpenNeuro sub-04 (M, 68)',
        'full': 'data/raw/examples/sub-04.nii',
        'preview': 'data/raw/examples/sub-04_preview.nii',
    },
}


@app.route('/')
def index():
    return render_template('index.html',
                           image_size=constants.IMAGE_SIZE,
                           slice_mode=constants.SLICE_MODE,
                           test_size=constants.TEST_SIZE,
                           val_size=constants.VAL_SIZE,
                           demo_mode=DEMO_MODE,
                           examples=EXAMPLES)


@app.route('/run', methods=['POST'])
def run():
    if DEMO_MODE:
        return jsonify({"error": "Training is disabled on this public demo"}), 403

    action = request.form['action']

    if action == 'image':
        prepare_data()
        image_data_classification()
        return "Image data processing complete"
    elif action == 'tabular':
        tabular_data_main()
        return "Tabular data processing complete"
    elif action == 'prepare':
        prepare_data()
        return "Data preparation complete"
    else:
        return "Invalid action", 400


@app.route('/update_settings', methods=['POST'])
def update_settings():
    data = request.json
    constants.IMAGE_SIZE = int(data['image_size'])
    constants.SLICE_MODE = data['slice_mode']
    constants.TEST_SIZE = float(data['test_size'])
    constants.VAL_SIZE = float(data['val_size'])
    return jsonify({"status": "success"})


@app.route('/classify', methods=['POST'])
def classify():
    from classifier import classify_nifti

    try:
        example_id = request.form.get('example_id')
        if example_id:
            example = EXAMPLES.get(example_id)
            if not example:
                return jsonify({"error": "Unknown example scan"}), 400
            # Classify the full-resolution original, not the compressed
            # preview the browser loaded for the 3D viewer.
            with open(example['full'], 'rb') as f:
                result = classify_nifti(f.read())
        elif 'file' in request.files:
            result = classify_nifti(request.files['file'].read())
        else:
            return jsonify({"error": "No file uploaded"}), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/examples/<example_id>/preview')
def serve_example_preview(example_id):
    example = EXAMPLES.get(example_id)
    if not example:
        return jsonify({"error": "Unknown example scan"}), 404
    directory, filename = os.path.split(example['preview'])
    return send_from_directory(directory, filename)


@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory('data', filename)


if __name__ == '__main__':
    app.run(debug=True)
