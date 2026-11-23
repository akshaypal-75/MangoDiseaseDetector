"""Grad-CAM heatmap generation for model explainability."""
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

def make_gradcam_heatmap(model, img_array, last_conv_layer_name=None):
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            try:
                if len(layer.output.shape) == 4:
                    last_conv_layer_name = layer.name
                    break
            except (AttributeError, TypeError):
                continue
    if last_conv_layer_name is None:
        raise ValueError("Could not identify a convolutional layer for Grad-CAM.")
    grad_model = tf.keras.Model(model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        score = predictions[:, class_idx]
    grads = tape.gradient(score, conv_outputs)
    if grads is None:
        raise RuntimeError("Unable to calculate Grad-CAM gradients.")
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_outputs[0] * weights, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    return heatmap.numpy()

def overlay_heatmap(original_image, heatmap, alpha=0.4):
    arr = np.asarray(original_image.convert("RGB"))
    h, w = arr.shape[:2]
    resized = cv2.resize(heatmap, (w, h))
    colored = cv2.applyColorMap(np.uint8(255 * resized), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return Image.fromarray(np.uint8(alpha * colored + (1-alpha) * arr))

def generate_gradcam(model, image):
    resized = image.convert("RGB").resize((224, 224))
    batch = np.expand_dims(np.asarray(resized, dtype=np.float32) / 255., axis=0)
    heatmap = make_gradcam_heatmap(model, batch)
    return overlay_heatmap(image, heatmap)
