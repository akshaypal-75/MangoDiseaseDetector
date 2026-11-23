"""Train the mango disease classifier."""
import argparse
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from dataset_loader import get_data_generators
from model import build_model

def train(dataset_dir, epochs=10, save_dir="saved_model"):
    train_gen, val_gen, class_names = get_data_generators(dataset_dir)
    model = build_model(num_classes=len(class_names))
    os.makedirs(save_dir, exist_ok=True)
    model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=[
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint(os.path.join(save_dir, "mango_model.h5"),
                        monitor="val_accuracy", save_best_only=True, verbose=1)])
    with open(os.path.join(save_dir, "class_names.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(class_names))
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Mango Disease Detector")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    train(args.dataset, args.epochs)
