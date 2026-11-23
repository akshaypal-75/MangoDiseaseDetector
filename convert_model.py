"""Convert a trained Keras model to HDF5 format."""
import tensorflow as tf
model = tf.keras.models.load_model("saved_model/mango_model.keras")
model.save("saved_model/mango_model.h5")
print("Conversion complete.")
