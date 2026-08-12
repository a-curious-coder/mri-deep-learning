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


@app.route('/')
def index():
    return render_template('index.html',
                           image_size=constants.IMAGE_SIZE,
                           slice_mode=constants.SLICE_MODE,
                           test_size=constants.TEST_SIZE,
                           val_size=constants.VAL_SIZE,
                           demo_mode=DEMO_MODE)


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

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        result = classify_nifti(request.files['file'].read())
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/static/data/raw/<path:filename>')
def serve_nifti(filename):
    return send_from_directory('data/raw', filename)


@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory('data', filename)


if __name__ == '__main__':
    app.run(debug=True)
