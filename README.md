🌾 CropSense — AI Crop Recommendation System
A machine learning web application that recommends the best crop to grow based on soil nutrients and climate conditions, powered by a Support Vector Machine (SVM) model trained on 2200 samples across 22 crop types.
🔗 Live Demo → https://crop-predictor-zvpo.onrender.com

✨ Features

🤖 SVM Model — 98.86% test accuracy with RBF kernel
🌱 22 Crop Types — rice, maize, wheat, mango, banana, coffee and more
📊 Top 5 Predictions — shows confidence % for each crop
🎛️ Interactive Sliders — adjust soil and climate parameters in real time
🔗 Flask REST API — /predict, /health, /crops endpoints
📋 Copy & Export — copy results or export history as CSV
🕓 Prediction History — stores last 20 predictions in browser
📱 Responsive Design — works on mobile and desktop


🧠 Model Details
PropertyValueAlgorithmSupport Vector Machine (SVM)KernelRBF (Radial Basis Function)C10GammascaleTest Accuracy98.86%CV Accuracy98.45% ± 0.30%Training Samples1760Test Samples440Total Classes22 crops

🌿 Supported Crops
🌾 Rice
🌽 Maize
🫘 Chickpea
🫘 Kidney Beans
🫛 Pigeon Peas
🌿 Moth Beans
🌱 Mung Bean
⚫ Black Gram
🟤 Lentil
🍎 Pomegranate
🍌 Banana
🥭 Mango
🍇 Grapes
🍉 Watermelon
🍈 Muskmelon
🍎 Apple
🍊 Orange
🍑 Papaya
🥥 Coconut
🌸 Cotton
🟫 Jute
☕ Coffee

📥 Input Features
Feature  Unit  Range:
Nitrogen (N)mg/kg0 – 140
Phosphorus(P)mg/kg5 – 145
Potassium (K)mg/kg5 – 205
Temperature°C8 – 44
Humidity%14 – 100
Soil pH—3.5 – 10
Rainfallmm20 – 300

🛠️ Tech Stack
Layer:-      Technology
MLModel      scikit-learn
SVMBackend   Python, Flask, Flask-CORS
Frontend     HTML, CSS, JavaScript
Deployment   Render.com
Version      ControlGitHub
