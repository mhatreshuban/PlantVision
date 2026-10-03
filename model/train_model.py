import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
import json
import os

# -------------------------
# SETTINGS
# -------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

TRAIN_DIR = "dataset/train"
VALID_DIR = "dataset/valid"

# -------------------------
# LOAD DATASET
# -------------------------
print("Loading training dataset...")

train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

print("Loading validation dataset...")

valid_data = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

# Get class names
class_names = train_data.class_names
num_classes = len(class_names)

print("\nNumber of classes:", num_classes)
print("\nClasses:")

for i, name in enumerate(class_names):
    print(i, name)

# Save class names
with open("model/class_names.json", "w") as file:
    json.dump(class_names, file)

# Improve loading performance
AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(buffer_size=AUTOTUNE)
valid_data = valid_data.prefetch(buffer_size=AUTOTUNE)

# -------------------------
# DATA AUGMENTATION
# -------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# -------------------------
# MOBILENETV2
# -------------------------
print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# -------------------------
# CREATE MODEL
# -------------------------
inputs = tf.keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)
x = preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

# -------------------------
# COMPILE MODEL
# -------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------
# TRAIN MODEL
# -------------------------
print("\nStarting training...")

history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=EPOCHS
)

# -------------------------
# SAVE MODEL
# -------------------------
model.save("model/plant_disease_model.keras")

print("\nModel saved successfully!")

# -------------------------
# ACCURACY GRAPH
# -------------------------
plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("PlantVision Model Accuracy")
plt.legend()
plt.tight_layout()

plt.savefig("model/accuracy_graph.png")
plt.close()

# -------------------------
# LOSS GRAPH
# -------------------------
plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("PlantVision Model Loss")
plt.legend()
plt.tight_layout()

plt.savefig("model/loss_graph.png")
plt.close()

print("Graphs saved successfully!")
print("\nPlantVision training completed.")