# 🌿 PlantVision – AI Plant Disease Analyzer

PlantVision is an AI/ML-based web application that analyzes plant leaf images and predicts possible plant diseases.

The system uses a deep learning model trained on plant leaf images and provides the predicted plant, disease condition, AI confidence, symptoms, treatment guidance, nutrition guidance, and prevention information.

---

## 🎯 Project Objective

The main objective of PlantVision is to use Artificial Intelligence and Deep Learning for early plant disease screening.

The system allows a user to upload a leaf image and receive an AI-based disease analysis through a simple web interface.

---

## ✨ Features

- Upload plant leaf images
- AI-based plant disease classification
- Detects healthy and diseased leaves
- Displays prediction confidence
- Detailed disease description
- Common symptoms
- Disease spread information
- Treatment and plant-care guidance
- Nutrition and fertilizer guidance
- Prevention recommendations
- Analysis history dashboard
- SQLite database integration
- User-friendly Flask web interface

---

## 🧠 AI Model

PlantVision uses **MobileNetV2 with Transfer Learning** for image classification.

The pretrained MobileNetV2 base model is used for feature extraction and a custom classification layer predicts the plant/disease class.

### Model Information

- Architecture: MobileNetV2
- Technique: Transfer Learning
- Input Image Size: 224 × 224 pixels
- Number of Classes: 38
- Training Epochs: 5
- Validation Accuracy: approximately 94.42%

---

## 📊 Model Performance

During training, the model achieved approximately:

**Training Accuracy:** 93.90%

**Validation Accuracy:** 94.42%

The project also generates training graphs such as:

- Accuracy Graph
- Loss Graph

These help visualize model performance during training.

---

## 🗂️ Dataset

The project uses the **New Plant Diseases Dataset (Augmented)**, which is based on PlantVillage-style plant disease images.

The dataset contains multiple healthy and diseased leaf categories across different crop plants.

The dataset itself is not included in this GitHub repository because of its large size.

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- Transfer Learning
- Flask
- HTML
- CSS
- JavaScript
- SQLite
- NumPy
- Pillow
- Matplotlib
- Scikit-learn

---

## 📁 Project Structure

```text
PlantVision/
│
├── app.py
├── database.py
├── disease_info.py
├── requirements.txt
├── README.md
│
├── model/
│   ├── plant_disease_model.keras
│   ├── class_names.json
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── accuracy_graph.png
│   └── loss_graph.png
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── uploads/
│
├── templates/
│   ├── index.html
│   ├── analyze.html
│   ├── dashboard.html
│   └── about.html
│
└── dataset/
```

---

## ⚙️ How PlantVision Works

```text
User Uploads Leaf Image
          ↓
Image Preprocessing
          ↓
MobileNetV2 AI Model
          ↓
Disease Classification
          ↓
Plant + Disease Prediction
          ↓
Confidence Score
          ↓
Disease Information
          ↓
Treatment + Nutrition + Prevention
          ↓
Result Stored in SQLite Database
          ↓
Dashboard
```

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Enter the project folder

```bash
cd PlantVision
```

### 3. Create a virtual environment

```bash
python3 -m venv venv
```

### 4. Activate the virtual environment

On macOS/Linux:

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
python3 app.py
```

Open the local address shown by Flask in your browser.

---

## 📸 Screenshots

Screenshots of the following pages can be added here:

1. Home Page
2. Analyze Plant Page
3. Disease Prediction Result
4. Detailed Disease Report
5. Dashboard

---

## ⚠️ Limitations

PlantVision currently supports the plant and disease classes included in its training dataset.

The model may not correctly recognize plants or diseases outside those trained classes.

Validation accuracy was measured on the project's validation dataset and does not guarantee the same accuracy for real-world photographs.

Visually similar symptoms can also be caused by diseases, pests, nutrient deficiencies, watering problems, or environmental stress.

Therefore, PlantVision should be treated as an **AI-based preliminary screening and educational tool**, not a definitive laboratory diagnosis.

---

## 🔮 Future Improvements

- Support more plant species
- Support additional diseases
- Improve real-world image recognition
- Add plant/non-plant image detection
- Add a dedicated plant species identification model
- Add weather-based disease risk analysis
- Improve disease management recommendations
- Add mobile application support

---

## 👨‍💻 Developed By

**Shuban Mhatre**

Information Technology Engineering

---

## 📌 Conclusion

PlantVision demonstrates how Artificial Intelligence, Deep Learning, and web technologies can be combined to build a practical plant disease analysis system.

The project uses image classification to analyze leaf images and provides useful information about detected plant conditions, care, nutrition, and prevention.