"""Inference-only classification using the pretrained demo model.

Separate from image_data/ (the private-dataset training pipeline). Loads
models/alzheimer_classifier.keras once and classifies a single 2D slice
extracted from an uploaded .nii scan. A single forward pass on a 128x128
grayscale image is lightweight - safe for a public, resource-constrained
deploy, unlike the training pipeline.

Known simplification: takes a fixed-depth axial slice with no real
frame-selection model - see README's own note on this.
"""
import os
import tempfile

import nibabel as nib
import numpy as np
import tensorflow as tf
from deepbet import run_bet
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


def _skull_strip(nifti_path, brain_path, mask_path, tiv_path):
    # Training data (Falah/Alzheimer_MRI) is already skull-stripped;
    # raw uploads aren't, which was a real domain-shift gap between what
    # the model saw in training and what it sees here. Naive intensity/
    # morphology thresholding can't separate skull from brain (no gap
    # between them in the raw data) - this needs an actual trained
    # segmentation model. deepbet is small (~10MB weights), CPU-capable,
    # and runs in ~2s per scan.
    run_bet(
        [nifti_path], [brain_path], [mask_path], [tiv_path],
        threshold=0.5, n_dilate=0, no_gpu=True,
    )


def _extract_middle_slice(nifti_bytes):
    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
        tmp.write(nifti_bytes)
        tmp.flush()
        in_path = tmp.name

    brain_path = in_path + "_brain.nii"
    mask_path = in_path + "_mask.nii"
    tiv_path = in_path + "_tiv.csv"
    try:
        _skull_strip(in_path, brain_path, mask_path, tiv_path)
        img = nib.load(brain_path)
        # Different scanners/datasets store axes in different orders (e.g.
        # this repo's example.nii is RAS, but real-world uploads have come
        # in as PIR - axis 2 there is left-right, not top-bottom). Without
        # this, a fixed "axis 2 at 75% depth" heuristic silently picks the
        # wrong anatomical plane on differently-oriented scans. Canonicalize
        # to RAS+ first so axis 2 always means superior-inferior (axial).
        img = nib.as_closest_canonical(img)
        data = img.get_fdata().squeeze()
    finally:
        for path in (in_path, brain_path, mask_path, tiv_path):
            if os.path.exists(path):
                os.remove(path)

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
