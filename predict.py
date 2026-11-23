"""Load the trained model and predict a mango leaf image."""
import argparse
import os
import tensorflow as tf
from PIL import Image
from patch_utils import patch_based_predict

MODEL_PATH = os.path.join("saved_model", "mango_model.h5")
CLASS_NAMES_PATH = os.path.join("saved_model", "class_names.txt")

def load_model_and_classes(model_path=MODEL_PATH, class_names_path=CLASS_NAMES_PATH):
    model = tf.keras.models.load_model(model_path)
    with open(class_names_path, encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]
    return model, class_names

def predict_image(image_path, model=None, class_names=None):
    if model is None or class_names is None:
        model, class_names = load_model_and_classes()
    image = Image.open(image_path).convert("RGB")
    return patch_based_predict(model, image, class_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict mango disease")
    parser.add_argument("--image", required=True)
    result = predict_image(parser.parse_args().image)
    print("Predicted class:", result["predicted_class"])
    print("Patches used:", result["num_patches"])
    for cls, prob in result["probabilities"].items():
        print(f"{cls:20s}: {prob:.2f}%")
