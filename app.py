from flask import Flask, render_template, request

import tensorflow as tf
import numpy as np

from PIL import Image

import sqlite3
import json
import os
import uuid

from werkzeug.utils import secure_filename

from disease_info import get_disease_info
from database import create_database


# ==========================================
# FLASK APPLICATION
# ==========================================

app = Flask(__name__)


# ==========================================
# SETTINGS
# ==========================================

UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp"
}


# ==========================================
# CREATE DATABASE
# ==========================================

create_database()


# ==========================================
# LOAD AI MODEL
# ==========================================

print("Loading PlantVision model...")

model = tf.keras.models.load_model(
    "model/plant_disease_model.keras"
)

with open(
    "model/class_names.json",
    "r"
) as file:

    class_names = json.load(file)


print("PlantVision model loaded successfully!")
print("Number of classes:", len(class_names))


# ==========================================
# CHECK FILE TYPE
# ==========================================

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ==========================================
# CONVERT MODEL CLASS TO READABLE RESULT
# ==========================================

def split_prediction(prediction):

    # Example:
    # Apple___Apple_scab
    #
    # becomes:
    # plant = Apple
    # disease = Apple scab

    if "___" in prediction:

        plant, disease = prediction.split(
            "___",
            1
        )

    else:

        plant = "Unknown Plant"
        disease = prediction


    # Replace underscores with spaces

    plant = plant.replace(
        "_",
        " "
    ).strip()

    disease = disease.replace(
        "_",
        " "
    ).strip()


    # Some classes contain the plant name
    # again inside the disease.

    # Example:
    # Apple___Apple_scab
    #
    # Apple scab -> scab

    plant_lower = plant.lower()

    if disease.lower().startswith(
        plant_lower + " "
    ):

        disease = disease[
            len(plant):
        ].strip()


    # Healthy class

    if disease.lower() == "healthy":

        disease = "Healthy"


    return plant, disease


# ==========================================
# IMAGE PREDICTION
# ==========================================

def predict_image(image_path):

    # Open image

    image = Image.open(
        image_path
    ).convert("RGB")


    # Resize

    image = image.resize(
        (224, 224)
    )


    # Convert to NumPy array

    image_array = np.array(
        image
    )


    # Add batch dimension

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # ======================================
    # MODEL PREDICTION
    # ======================================

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]


    # Highest prediction

    predicted_index = np.argmax(
        predictions
    )


    confidence = float(
        predictions[predicted_index]
        * 100
    )


    # ======================================
    # SECOND HIGHEST PREDICTION
    # ======================================

    sorted_indices = np.argsort(
        predictions
    )[::-1]


    highest_index = sorted_indices[0]

    second_index = sorted_indices[1]


    highest = float(
        predictions[highest_index]
        * 100
    )

    second_highest = float(
        predictions[second_index]
        * 100
    )

    difference = (
        highest
        -
        second_highest
    )


    # Get top predictions for terminal

    print("\n----------------------------")
    print("PlantVision Prediction")
    print("----------------------------")

    print(
        "Top prediction:",
        class_names[highest_index]
    )

    print(
        "Confidence:",
        round(highest, 2),
        "%"
    )

    print(
        "Second prediction:",
        class_names[second_index]
    )

    print(
        "Second confidence:",
        round(second_highest, 2),
        "%"
    )

    print(
        "Difference:",
        round(difference, 2),
        "%"
    )


    # ======================================
    # UNCERTAINTY CHECK
    # ======================================

    # Important:
    # This is only a heuristic.
    # It cannot guarantee detection of
    # every non-plant image.

    if (
        confidence < 85
        or
        difference < 25
    ):

        return (
            "Unknown / Unsupported Image",
            confidence,
            False
        )


    predicted_class = class_names[
        predicted_index
    ]


    return (
        predicted_class,
        confidence,
        True
    )


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================
# ANALYZE PAGE
# ==========================================

@app.route(
    "/analyze",
    methods=["GET", "POST"]
)
def analyze():

    prediction = None

    confidence = None

    image_filename = None

    valid_prediction = None

    result_info = None

    error = None


    # ======================================
    # POST REQUEST
    # ======================================

    if request.method == "POST":


        # ----------------------------------
        # CHECK FILE EXISTS
        # ----------------------------------

        if "plant_image" not in request.files:

            error = "Please select an image."

            return render_template(
                "analyze.html",
                error=error
            )


        image = request.files[
            "plant_image"
        ]


        # ----------------------------------
        # CHECK EMPTY FILE
        # ----------------------------------

        if image.filename == "":

            error = "Please select an image."

            return render_template(
                "analyze.html",
                error=error
            )


        # ----------------------------------
        # CHECK FILE TYPE
        # ----------------------------------

        if not allowed_file(
            image.filename
        ):

            error = (
                "Only JPG, JPEG, PNG "
                "and WEBP images are allowed."
            )

            return render_template(
                "analyze.html",
                error=error
            )


        # ==================================
        # CREATE UNIQUE FILE NAME
        # ==================================

        original_filename = secure_filename(
            image.filename
        )


        extension = original_filename.rsplit(
            ".",
            1
        )[1].lower()


        unique_name = (
            str(uuid.uuid4())
            +
            "."
            +
            extension
        )


        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            unique_name
        )


        # Save uploaded image

        image.save(
            image_path
        )


        try:

            # ==================================
            # AI PREDICTION
            # ==================================

            (
                prediction,
                confidence,
                valid_prediction
            ) = predict_image(
                image_path
            )


            image_filename = unique_name


            # ==================================
            # VALID PREDICTION
            # ==================================

            if valid_prediction:


                # ------------------------------
                # SPLIT PLANT + DISEASE
                # ------------------------------

                plant, disease = split_prediction(
                    prediction
                )


                # ------------------------------
                # HEALTH STATUS
                # ------------------------------

                is_healthy = (
                    disease.lower()
                    ==
                    "healthy"
                )


                # ------------------------------
                # GET DETAILED INFORMATION
                # ------------------------------

                result_info = get_disease_info(

                    plant,

                    disease,

                    healthy=is_healthy

                )


                # ==================================
                # SAVE RESULT TO DATABASE
                # ==================================

                with sqlite3.connect(
                    "plantvision.db"
                ) as connection:


                    cursor = connection.cursor()


                    cursor.execute(
                        """
                        INSERT INTO analyses
                        (
                            image_name,
                            plant,
                            disease,
                            status,
                            confidence
                        )

                        VALUES (?, ?, ?, ?, ?)
                        """,

                        (
                            unique_name,

                            result_info[
                                "plant"
                            ],

                            result_info[
                                "disease"
                            ],

                            result_info[
                                "status"
                            ],

                            confidence
                        )
                    )


                    connection.commit()


        # ==================================
        # ERROR HANDLING
        # ==================================

        except Exception as e:

            print(
                "Prediction error:",
                e
            )

            error = (
                "Unable to analyze this image. "
                "Please upload another image."
            )


    # ======================================
    # LOAD ANALYZE PAGE
    # ======================================

    return render_template(

        "analyze.html",

        prediction=prediction,

        confidence=confidence,

        image_filename=image_filename,

        valid_prediction=valid_prediction,

        result_info=result_info,

        error=error

    )


# ==========================================
# DASHBOARD PAGE
# ==========================================

@app.route("/dashboard")
def dashboard():


    with sqlite3.connect(
        "plantvision.db"
    ) as connection:


        cursor = connection.cursor()


        # ----------------------------------
        # TOTAL ANALYSES
        # ----------------------------------

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM analyses
            """
        )

        total = cursor.fetchone()[0]


        # ----------------------------------
        # HEALTHY
        # ----------------------------------

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM analyses
            WHERE status = 'Healthy'
            """
        )

        healthy = cursor.fetchone()[0]


        # ----------------------------------
        # DISEASE DETECTED
        # ----------------------------------

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM analyses
            WHERE status = 'Disease Detected'
            """
        )

        diseased = cursor.fetchone()[0]


        # ----------------------------------
        # RECENT ANALYSES
        # ----------------------------------

        cursor.execute(
            """
            SELECT
                id,
                image_name,
                plant,
                disease,
                status,
                confidence,
                created_at

            FROM analyses

            ORDER BY id DESC

            LIMIT 10
            """
        )

        recent = cursor.fetchall()


        # ----------------------------------
        # COMMON DISEASES
        # ----------------------------------

        cursor.execute(
            """
            SELECT
                disease,
                COUNT(*) AS total

            FROM analyses

            WHERE status = 'Disease Detected'

            GROUP BY disease

            ORDER BY total DESC

            LIMIT 5
            """
        )

        common_diseases = cursor.fetchall()


    return render_template(

        "dashboard.html",

        total=total,

        healthy=healthy,

        diseased=diseased,

        recent=recent,

        common_diseases=common_diseases

    )


# ==========================================
# ABOUT PAGE
# ==========================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )