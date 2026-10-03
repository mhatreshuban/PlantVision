import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# Settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load class names
with open("model/class_names.json", "r") as file:
    class_names = json.load(file)

# Load validation dataset
print("Loading validation dataset...")

valid_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/valid",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Load trained model
print("Loading PlantVision model...")

model = tf.keras.models.load_model(
    "model/plant_disease_model.keras"
)

# Evaluate
print("\nEvaluating model...")

loss, accuracy = model.evaluate(valid_data)

print("\nValidation Accuracy:", round(accuracy * 100, 2), "%")
print("Validation Loss:", round(loss, 4))

# Predictions
print("\nGenerating predictions...")

predictions = model.predict(valid_data)

predicted_labels = np.argmax(predictions, axis=1)

true_labels = np.concatenate([
    labels.numpy() for images, labels in valid_data
])

# Classification report
report = classification_report(
    true_labels,
    predicted_labels,
    target_names=class_names
)

print("\nCLASSIFICATION REPORT")
print(report)

# Save report
with open("model/classification_report.txt", "w") as file:
    file.write(report)

# Confusion matrix
cm = confusion_matrix(
    true_labels,
    predicted_labels
)

fig, ax = plt.subplots(figsize=(20, 20))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    ax=ax,
    xticks_rotation=90,
    cmap="Blues",
    colorbar=False
)

plt.title("PlantVision Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "model/confusion_matrix.png",
    dpi=200
)

plt.close()

print("\nEvaluation completed!")
print("Classification report saved.")
print("Confusion matrix saved.")