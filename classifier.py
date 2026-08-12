"""Inference-only classification using the pretrained demo model.

Separate from image_data/ (the private-dataset training pipeline). Loads
models/alzheimer_classifier.keras once and classifies a single 2D slice
extracted from an uploaded .nii scan. A single forward pass on a 128x128
grayscale image is lightweight - safe for a public, resource-constrained
deploy, unlike the training pipeline.

Known simplification: takes the middle slice along the volume's third
axis with no orientation/qform-based reformatting, consistent with how
the rest of this app already handles NIfTI axes. Good enough for a demo,
not a clinically rigorous slice-selection step.
"""
import tempfile

import nibabel as nib
import numpy as np
import tensorflow as tf
from PIL import Image

# A single 128x128 forward pass is sub-second on CPU either way, and pinning
# to CPU avoids GPU/cuDNN environment differences (driver, compute
# capability) between wherever this runs and whatever machine trained the
# model - the deploy target has no GPU at all, so this keeps local dev
# behaviour consistent with production instead of depending on local
# hardware happening to work.
tf.config.set_visible_devices([], "GPU")

IMAGE_SIZE = 128
CLASS_NAMES = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]
MODEL_PATH = "models/alzheimer_classifier.keras"

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def _extract_middle_slice(nifti_bytes):
    with tempfile.NamedTemporaryFile(suffix=".nii") as tmp:
        tmp.write(nifti_bytes)
        tmp.flush()
        data = nib.load(tmp.name).get_fdata().squeeze()

    if data.ndim < 3:
        raise ValueError("Scan must be a 3D volume")

    # The array's geometric middle index is NOT mid-brain - for a full
    # head scan (skull to neck) the true anatomical middle sits well past
    # 50% of the depth axis. Checked against this repo's own example.nii:
    # 50% (index 128/256) lands at eye-socket/orbit level, not brain tissue
    # at all. ~75% consistently lands in a clean mid-brain cross-section
    # instead. Still an approximation, not a real frame-selection model -
    # see README's own note that slice selection is a genuinely hard,
    # important problem, not solved here.
    depth = data.shape[2]
    index = int(depth * 0.75)
    return data[:, :, index]


def _preprocess(slice_data):
    normalized = slice_data - slice_data.min()
    max_val = normalized.max()
    if max_val > 0:
        normalized = normalized / max_val
    normalized = (normalized * 255).astype(np.uint8)

    image = Image.fromarray(normalized).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    array = np.array(image).astype("float32") / 255.0
    return array.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)


def classify_nifti(nifti_bytes):
    slice_data = _extract_middle_slice(nifti_bytes)
    input_array = _preprocess(slice_data)

    model = _get_model()
    probabilities = model.predict(input_array, verbose=0)[0]

    predictions = sorted(
        [{"label": name, "probability": float(p)} for name, p in zip(CLASS_NAMES, probabilities)],
        key=lambda x: x["probability"],
        reverse=True,
    )
    return {"predictions": predictions, "predicted_class": predictions[0]["label"]}
