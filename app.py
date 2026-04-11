import streamlit as st
import gdown
import joblib
import os
import cv2
import numpy as np
import tensorflow as tf
from skimage.feature import hog
from PIL import Image
import pandas as pd

# --- SETTINGS ---
IMG_SIZE = 64
fruit_labels = ["Apple", "Avocado", "Banana", "Broccoli", "Capsicum", "Cauliflower", "Cucumber", "Lemon", "Mango", "Watermelon"]

@st.cache_resource
def load_all_models():
    model_configs = {
        "fruit_model_v2.h5": "13stvBP7-Ta7R2BKnrbuRuTIQtFhikiVw",
        "svm_best_v2.pkl": "1DDBGQNAUZBu4VNX61NObYjso_6jDVBko",
        "lr_improved.pkl": "1j632tPQnIkFzWcpOQNdqiOVgI3Tpl4qA"
    }
    for filename, file_id in model_configs.items():
        if not os.path.exists(filename):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, filename, quiet=False, fuzzy=True)

    return (
        tf.keras.models.load_model("fruit_model_v2.h5"),
        joblib.load("svm_best_v2.pkl"),
        joblib.load("lr_improved.pkl")
    )

# --- FEATURE EXTRACTION ---
def extract_features_lr(img):
    img_res = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_blur = cv2.GaussianBlur(img_res, (3, 3), 0)
    gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(gray, pixels_per_cell=(4, 4), cells_per_block=(2, 2), feature_vector=True)
    color_feat = cv2.calcHist([img_res], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    color_feat = cv2.normalize(color_feat, color_feat).flatten()
    edges = cv2.Canny(gray, 100, 200).flatten()
    return np.hstack([hog_feat, color_feat, edges])

def extract_features_svm(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_res = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    hog_feat = hog(gray_res, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    img_res = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    color_feat = cv2.calcHist([img_res], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    color_feat = cv2.normalize(color_feat, color_feat).flatten()
    return np.hstack([hog_feat, color_feat])

# --- APP START ---
with st.spinner("Initializing AI..."):
    model_cnn, model_svm, model_lr = load_all_models()

st.title("🍓 Fruit Multi-Model Classifier")
choice = st.sidebar.selectbox("Model Architecture", ["CNN", "SVM", "Logistic Regression"])

picture = st.camera_input("Take a photo")

if picture:
    img_raw = Image.open(picture)
    st.image(img_raw, caption="Your Photo", use_container_width=True)
    img_cv = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)

    with st.spinner("Analyzing probabilities..."):
        try:
            if choice == "CNN":
                img_cnn = cv2.resize(img_cv, (128, 128)) / 255.0
                probs = model_cnn.predict(np.expand_dims(img_cnn, axis=0))[0]
            elif choice == "SVM":
                feat = extract_features_svm(img_cv)
                probs = model_svm.predict_proba([feat])[0]
            else:
                feat = extract_features_lr(img_cv)
                probs = model_lr.predict_proba([feat])[0]

            # --- TOP RESULTS LOGIC ---
            top_indices = probs.argsort()[-3:][::-1]  # Get indices of top 3
            
            st.subheader(f"🥇 Prediction: {fruit_labels[top_indices[0]]}")
            st.progress(float(probs[top_indices[0]]))

            # Create a nice chart for possible fruits
            chart_data = pd.DataFrame({
                'Fruit': [fruit_labels[i] for i in top_indices],
                'Confidence (%)': [probs[i] * 100 for i in top_indices]
            })
            
            st.write("### 📊 Top 3 Possibilities")
            st.bar_chart(chart_data, x="Fruit", y="Confidence (%)")

            # Simple Table View
            st.table(chart_data)

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.info("Note: Ensure your .pkl models were trained with 'probability=True'.")
