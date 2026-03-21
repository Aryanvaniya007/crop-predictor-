"""
Crop Recommendation — SVM Model
=================================
Dataset : Crop_recommendation.csv
Features: N, P, K, temperature, humidity, ph, rainfall
Target  : label (22 crop classes)
Model   : Support Vector Machine (RBF kernel)

Run:
    python crop_prediction_svm.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
import os

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1.  Load Data
# ─────────────────────────────────────────────
print("=" * 55)
print("  CROP PREDICTION — SVM MODEL")
print("=" * 55)

DATA_PATH = "Crop_recommendation.csv"   # update path if needed
df = pd.read_csv(DATA_PATH)

print(f"\n[1] Dataset  →  {df.shape[0]} rows × {df.shape[1]} cols")
print(f"    Crops    :  {sorted(df['label'].unique())}")

# ─────────────────────────────────────────────
# 2.  Preprocessing
# ─────────────────────────────────────────────
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET   = "label"

X = df[FEATURES].values
y = df[TARGET].values

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Scale features  (critical for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train / Test split  (80 / 20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.20, random_state=42, stratify=y_enc
)
print(f"\n[2] Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")

# ─────────────────────────────────────────────
# 3.  Train SVM
# ─────────────────────────────────────────────
print("\n[3] Training SVM (RBF kernel) ...")

svm_model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True,   # enables predict_proba
    random_state=42
)
svm_model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 4.  Evaluate
# ─────────────────────────────────────────────
y_pred = svm_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

cv_scores = cross_val_score(svm_model, X_scaled, y_enc, cv=5, scoring="accuracy")

print(f"\n[4] Results:")
print(f"    Test  Accuracy : {test_acc * 100:.2f}%")
print(f"    CV    Accuracy : {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")

print("\n[5] Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ─────────────────────────────────────────────
# 5.  Plots
# ─────────────────────────────────────────────
os.makedirs("plots", exist_ok=True)

# ── Confusion Matrix ──
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(14, 12))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
ax.set_title("Confusion Matrix — SVM (RBF)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("plots/svm_confusion_matrix.png", dpi=150)
print("\n[Plot saved]  plots/svm_confusion_matrix.png")
plt.close()

# ── CV Score Bar ──
fig, ax = plt.subplots(figsize=(7, 4))
folds = [f"Fold {i+1}" for i in range(5)]
bars = ax.bar(folds, cv_scores * 100, color="#42A5F5", edgecolor="white", width=0.5)
ax.axhline(cv_scores.mean() * 100, color="#E53935", linestyle="--",
           linewidth=1.5, label=f"Mean: {cv_scores.mean()*100:.2f}%")
for bar, val in zip(bars, cv_scores * 100):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f"{val:.2f}%", ha="center", va="bottom", fontsize=9)
ax.set_ylim(95, 101)
ax.set_ylabel("Accuracy (%)")
ax.set_title("5-Fold Cross-Validation — SVM", fontsize=13, fontweight="bold")
ax.legend()
ax.set_facecolor("#F5F5F5")
fig.tight_layout()
fig.savefig("plots/svm_cv_scores.png", dpi=150)
print("[Plot saved]  plots/svm_cv_scores.png")
plt.close()

# ── Per-Class Accuracy Bar ──
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
class_acc = {cls: report[cls]["precision"] for cls in le.classes_}
sorted_acc = dict(sorted(class_acc.items(), key=lambda x: x[1]))

fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#EF9A9A" if v < 0.98 else "#A5D6A7" for v in sorted_acc.values()]
ax.barh(list(sorted_acc.keys()), [v * 100 for v in sorted_acc.values()],
        color=colors, edgecolor="white")
ax.axvline(98, color="#E53935", linestyle="--", linewidth=1.2, label="98% threshold")
ax.set_xlabel("Precision (%)")
ax.set_title("Per-Class Precision — SVM", fontsize=13, fontweight="bold")
ax.set_xlim(80, 102)
ax.legend()
ax.set_facecolor("#F5F5F5")
fig.tight_layout()
fig.savefig("plots/svm_per_class_precision.png", dpi=150)
print("[Plot saved]  plots/svm_per_class_precision.png")
plt.close()

# ─────────────────────────────────────────────
# 6.  Save Model
# ─────────────────────────────────────────────
with open("svm_crop_model.pkl", "wb") as f:
    pickle.dump({"model": svm_model, "scaler": scaler, "encoder": le}, f)
print("\n[Model saved]  svm_crop_model.pkl")

# ─────────────────────────────────────────────
# 7.  Prediction Function
# ─────────────────────────────────────────────
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    """
    Predict the best crop for given soil & climate inputs.

    Parameters
    ----------
    N, P, K     : int   – Nitrogen, Phosphorous, Potassium (mg/kg)
    temperature : float – °C
    humidity    : float – relative humidity %
    ph          : float – soil pH (0–14)
    rainfall    : float – mm

    Returns
    -------
    crop  : str  – recommended crop name
    probs : dict – top-5 crop probabilities (%)
    """
    arr = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    arr_scaled = scaler.transform(arr)
    idx  = svm_model.predict(arr_scaled)[0]
    crop = le.inverse_transform([idx])[0]

    proba = svm_model.predict_proba(arr_scaled)[0]
    top5  = sorted(zip(le.classes_, proba * 100), key=lambda x: -x[1])[:5]
    probs = {c: round(p, 2) for c, p in top5}

    return crop, probs


# ─────────────────────────────────────────────
# 8.  Demo Predictions
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  DEMO PREDICTIONS")
print("=" * 55)

demos = [
    (90,  42, 43, 20.9, 82.0, 6.5, 202.9, "rice"),
    (40,  40, 40, 23.0, 82.0, 6.5, 230.0, "rice"),
    (20,  70, 80, 30.0, 50.0, 6.5,  85.0, "cotton"),
    (30,  60, 60, 27.0, 60.0, 7.0, 100.0, "maize"),
    (0,   20, 20, 38.0, 35.0, 7.5,  40.0, "mungbean"),
]

for N, P, K, temp, hum, ph, rain, expected in demos:
    crop, probs = predict_crop(N, P, K, temp, hum, ph, rain)
    status = "✓" if crop == expected else "~"
    print(f"\n  N={N}, P={P}, K={K}, Temp={temp}°C, Hum={hum}%, pH={ph}, Rain={rain}mm")
    print(f"  → Predicted : {crop:<14}  (Expected: {expected}) {status}")
    top3 = list(probs.items())[:3]
    print(f"     Top-3 : " + "  |  ".join(f"{c}: {p}%" for c, p in top3))

print("\n" + "=" * 55)
print("  REUSE IN YOUR OWN CODE")
print("=" * 55)
print("""
  from crop_prediction_svm import predict_crop

  crop, probs = predict_crop(
      N=90, P=42, K=43,
      temperature=20.9,
      humidity=82.0,
      ph=6.5,
      rainfall=202.9
  )
  print("Recommended crop:", crop)
  print("Probabilities:", probs)
""")
print("Done! Plots saved in 'plots/' folder.\n")
