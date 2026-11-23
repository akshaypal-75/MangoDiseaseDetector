"""Load and augment the mango leaf image dataset."""
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

def get_data_generators(dataset_dir, val_split=0.2, max_per_class=200):
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=25, zoom_range=0.2,
        horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1,
        fill_mode="nearest", validation_split=val_split)
    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=val_split)
    train_gen = train_datagen.flow_from_directory(dataset_dir, target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical", subset="training",
        shuffle=True, seed=42)
    val_gen = val_datagen.flow_from_directory(dataset_dir, target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical", subset="validation",
        shuffle=False, seed=42)
    class_names = sorted(train_gen.class_indices, key=train_gen.class_indices.get)
    print(f"Classes: {class_names}; train: {train_gen.samples}; validation: {val_gen.samples}")
    return train_gen, val_gen, class_names
