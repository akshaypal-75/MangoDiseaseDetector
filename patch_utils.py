"""Patch extraction and weighted patch-based inference."""
import numpy as np
from PIL import Image
PATCH_SIZE = 128
MODEL_INPUT_SIZE = (224, 224)

def extract_patches(image: Image.Image, patch_size=PATCH_SIZE):
    arr = np.asarray(image.convert("RGB"))
    h, w, c = arr.shape
    patches = []
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = arr[y:y+patch_size, x:x+patch_size]
            ph, pw = patch.shape[:2]
            if ph < patch_size or pw < patch_size:
                padded = np.zeros((patch_size, patch_size, c), dtype=np.uint8)
                padded[:ph, :pw] = patch
                patch = padded
            patches.append(patch)
    return patches

def preprocess_patch(patch_array, target_size=MODEL_INPUT_SIZE):
    img = Image.fromarray(patch_array).resize((target_size[1], target_size[0]), Image.BILINEAR)
    return np.expand_dims(np.asarray(img, dtype=np.float32) / 255., axis=0)

def patch_based_predict(model, image: Image.Image, class_names: list):
    patches = extract_patches(image)
    predictions = np.array([model.predict(preprocess_patch(p), verbose=0)[0] for p in patches])
    healthy_idx = class_names.index("Healthy")
    weights = 1. - predictions[:, healthy_idx] + 0.05
    final_preds = predictions.mean(axis=0) if np.all(predictions[:, healthy_idx] > 0.7) else np.average(predictions, axis=0, weights=weights)
    idx = int(np.argmax(final_preds))
    return {"predicted_class": class_names[idx],
            "probabilities": {name: round(float(final_preds[i])*100, 2) for i, name in enumerate(class_names)},
            "num_patches": len(patches)}
