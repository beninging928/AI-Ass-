import os
import cv2
import numpy as np
import joblib
import tensorflow as tf
from skimage.feature import hog
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------
# SETTINGS
# -------------------------
IMG_SIZE = 64
DATA_DIR = "dataset/test"

# -------------------------
# FEATURE: Logistic Regression (MATCH TRAIN)
# -------------------------
def extract_features_lr(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.GaussianBlur(img, (3, 3), 0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog_feat = hog(
        gray,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        feature_vector=True
    )

    color_feat = cv2.calcHist(
        [img], [0, 1, 2], None,
        [8, 8, 8],
        [0, 256, 0, 256, 0, 256]
    )
    color_feat = cv2.normalize(color_feat, color_feat).flatten()

    edges = cv2.Canny(gray, 100, 200)
    edge_feat = edges.flatten()

    return np.hstack([hog_feat, color_feat, edge_feat])


# -------------------------
# FEATURE: SVM (MATCH TRAIN)
# -------------------------
def extract_features_svm(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    hog_feat = hog(
        gray,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True
    )

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    color_feat = cv2.calcHist(
        [img], [0, 1, 2], None,
        [8, 8, 8],
        [0, 256, 0, 256, 0, 256]
    )
    color_feat = cv2.normalize(color_feat, color_feat).flatten()

    return np.hstack([hog_feat, color_feat])


# -------------------------
# LOAD TEST DATA
# -------------------------
def load_data():
    X_cnn = []
    X_lr = []
    X_svm = []
    y = []

    class_names = sorted(os.listdir(DATA_DIR))

    for idx, label in enumerate(class_names):
        folder = os.path.join(DATA_DIR, label)

        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            # CNN input
            img_cnn = cv2.resize(img, (128, 128)) / 255.0
            X_cnn.append(img_cnn)

            # LR features
            X_lr.append(extract_features_lr(img))

            # SVM features
            X_svm.append(extract_features_svm(img))

            y.append(idx)

    return (
        np.array(X_cnn),
        np.array(X_lr),
        np.array(X_svm),
        np.array(y),
        class_names
    )


print("📂 Loading test data...")
X_cnn, X_lr, X_svm, y_true, class_names = load_data()

# -------------------------
# LOAD MODELS
# -------------------------
print("📦 Loading models...")
cnn_model = tf.keras.models.load_model("fruit_model_v2.h5")
svm_model = joblib.load("svm_best_v2.pkl")
lr_model = joblib.load("lr_improved.pkl")

# -------------------------
# PREDICTION
# -------------------------
print("🚀 Predicting...")

cnn_pred = np.argmax(cnn_model.predict(X_cnn), axis=1)
svm_pred = svm_model.predict(X_svm)
lr_pred = lr_model.predict(X_lr)

# -------------------------
# EVALUATION FUNCTION
# -------------------------
def evaluate_model(name, y_true, y_pred):
    print(f"\n========== {name} ==========")

    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# -------------------------
# RUN EVALUATION
# -------------------------
evaluate_model("CNN", y_true, cnn_pred)
evaluate_model("SVM", y_true, svm_pred)
evaluate_model("Logistic Regression", y_true, lr_pred)
