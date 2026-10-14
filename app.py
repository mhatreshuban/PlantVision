"""PlantVision Flask application, with lower-overhead TensorFlow inference."""
import os

# Set these before importing TensorFlow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")

import json
import logging
import sqlite3
import uuid
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from database import create_database
from disease_info import get_disease_info

try:
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
except RuntimeError:
    pass  # Threads may already be initialized.

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
DB_PATH = BASE_DIR / "plantvision.db"
MODEL_PATH = BASE_DIR / "model" / "plant_disease_model.keras"
LABELS_PATH = BASE_DIR / "model" / "class_names.json"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

create_database()

print("Loading PlantVision model...", flush=True)
model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
with open(LABELS_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)
print(f"PlantVision model loaded: {len(class_names)} classes", flush=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def split_prediction(prediction):
    if "___" in prediction:
        plant, disease = prediction.split("___", 1)
    else:
        plant, disease = "Unknown Plant", prediction
    plant = plant.replace("_", " ").strip()
    disease = disease.replace("_", " ").strip()
    if disease.lower().startswith(plant.lower() + " "):
        disease = disease[len(plant):].strip()
    if disease.lower() == "healthy":
        disease = "Healthy"
    return plant, disease


def predict_image(image_path):
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((224, 224))
        # Preserve the original preprocessing: no division by 255.
        image_array = np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)

    # Direct model call avoids the additional predict() data-adapter overhead.
    predictions = np.asarray(model(image_array, training=False).numpy()[0])
    if len(predictions) != len(class_names):
        raise ValueError("Model output and class labels have different lengths")

    sorted_indices = np.argsort(predictions)[::-1]
    highest_index = int(sorted_indices[0])
    confidence = float(predictions[highest_index] * 100)
    second_highest = float(predictions[sorted_indices[1]] * 100) if len(sorted_indices) > 1 else 0.0
    difference = confidence - second_highest

    print(
        f"Prediction: {class_names[highest_index]} ({confidence:.2f}%), "
        f"second: {second_highest:.2f}%, gap: {difference:.2f}%",
        flush=True,
    )

    # Preserve the original uncertainty thresholds.
    if confidence < 85 or difference < 25:
        return "Unknown / Unsupported Image", confidence, False
    return class_names[highest_index], confidence, True


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    prediction = None
    confidence = None
    image_filename = None
    valid_prediction = None
    result_info = None
    error = None

    if request.method == "POST":
        image = request.files.get("plant_image")
        if image is None or not image.filename:
            error = "Please select an image."
        elif not allowed_file(image.filename):
            error = "Only JPG, JPEG, PNG and WEBP images are allowed."
        else:
            original_filename = secure_filename(image.filename)
            extension = original_filename.rsplit(".", 1)[1].lower()
            unique_name = f"{uuid.uuid4()}.{extension}"
            image_path = UPLOAD_FOLDER / unique_name
            try:
                image.save(image_path)
                prediction, confidence, valid_prediction = predict_image(image_path)
                image_filename = unique_name
                if valid_prediction:
                    plant, disease = split_prediction(prediction)
                    is_healthy = disease.lower() == "healthy"
                    result_info = get_disease_info(plant, disease, healthy=is_healthy)
                    with sqlite3.connect(DB_PATH) as connection:
                        connection.execute(
                            """INSERT INTO analyses
                               (image_name, plant, disease, status, confidence)
                               VALUES (?, ?, ?, ?, ?)""",
                            (
                                unique_name,
                                result_info["plant"],
                                result_info["disease"],
                                result_info["status"],
                                confidence,
                            ),
                        )
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                app.logger.exception("Invalid image or prediction failure: %s", exc)
                error = "Unable to analyze this image. Please upload another image."
            except Exception:
                app.logger.exception("PlantVision prediction failed")
                error = "Unable to analyze this image. Please try again."

    return render_template(
        "analyze.html",
        prediction=prediction,
        confidence=confidence,
        image_filename=image_filename,
        valid_prediction=valid_prediction,
        result_info=result_info,
        error=error,
    )


@app.route("/dashboard")
def dashboard():
    with sqlite3.connect(DB_PATH) as connection:
        total = connection.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
        healthy = connection.execute(
            "SELECT COUNT(*) FROM analyses WHERE status = 'Healthy'"
        ).fetchone()[0]
        diseased = connection.execute(
            "SELECT COUNT(*) FROM analyses WHERE status = 'Disease Detected'"
        ).fetchone()[0]
        recent = connection.execute(
            """SELECT id, image_name, plant, disease, status, confidence, created_at
               FROM analyses ORDER BY id DESC LIMIT 10"""
        ).fetchall()
        common_diseases = connection.execute(
            """SELECT disease, COUNT(*) AS total FROM analyses
               WHERE status = 'Disease Detected'
               GROUP BY disease ORDER BY total DESC LIMIT 5"""
        ).fetchall()
    return render_template(
        "dashboard.html",
        total=total,
        healthy=healthy,
        diseased=diseased,
        recent=recent,
        common_diseases=common_diseases,
    )


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
