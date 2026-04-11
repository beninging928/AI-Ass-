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

# --- STEP 1: DOWNLOAD & LOAD MODELS ---
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
            try:
                gdown.download(url, filename, quiet=False, fuzzy=True)
            except Exception as e:
                st.error(f"Error downloading {filename}: {e}")
                st.stop()

    m1 = tf.keras.models.load_model("fruit_model_v2.h5")
    m2 = joblib.load("svm_best_v2.pkl")
    m3 = joblib.load("lr_improved.pkl")
    return m1, m2, m3

# --- STEP 2: YOUR FEATURE EXTRACTION LOGIC ---
def extract_features_lr(img):
    img_res = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_blur = cv2.GaussianBlur(img_res, (3, 3), 0)
    gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    
    # HOG
    hog_feat = hog(gray, pixels_per_cell=(4, 4), cells_per_block=(2, 2), feature_vector=True)
    # Color
    color_feat = cv2.calcHist([img_res], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    color_feat = cv2.normalize(color_feat, color_feat).flatten()
    # Edges
    edges = cv2.Canny(gray, 100, 200).flatten()
    
    return np.hstack([hog_feat, color_feat, edges])

def extract_features_svm(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_res = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    
    # HOG
    hog_feat = hog(gray_res, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    # Color
    img_res = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    color_feat = cv2.calcHist([img_res], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    color_feat = cv2.normalize(color_feat, color_feat).flatten()
    
    return np.hstack([hog_feat, color_feat])

# --- INITIALIZE ---
with st.spinner("Downloading and Loading AI Models..."):
    model_cnn, model_svm, model_lr = load_all_models()

# --- STEP 3: UI ---
st.set_page_config(page_title="Fruit Analysis Dashboard", layout="centered")
st.title("🍓 Fruit Multi-Model Classifier")

st.sidebar.header("Model Selection")
choice = st.sidebar.selectbox("Architecture", ["CNN (Deep Learning)", "SVM (Traditional ML)", "Logistic Regression"])

picture = st.camera_input("Take a photo of a fruit")

if picture:
    # Convert picture to OpenCV format
    img_raw = Image.open(picture)
    st.image(img_raw, caption="Original Captured Image", use_container_width=True)
    img_cv = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)

    with st.spinner(f"Analyzing with {choice}..."):
        try:
            if "CNN" in choice:
                img_cnn = cv2.resize(img_cv, (128, 128)) / 255.0
                probs = model_cnn.predict(np.expand_dims(img_cnn, axis=0))[0]
            
            elif "SVM" in choice:
                feat = extract_features_svm(img_cv)
                # FIX: Check if model has predict_proba, otherwise use decision function
                if hasattr(model_svm, "predict_proba"):
                    probs = model_svm.predict_proba([feat])[0]
                else:
                    scores = model_svm.decision_function([feat])[0]
                    # Softmax workaround to turn raw scores into probabilities
                    exp_scores = np.exp(scores - np.max(scores))
                    probs = exp_scores / exp_scores.sum()
            
            else: # Logistic Regression
                feat = extract_features_lr(img_cv)
                probs = model_lr.predict_proba([feat])[0]

            # --- TOP 3 RESULTS ---
            top_indices = probs.argsort()[-3:][::-1]
            predicted_fruit = fruit_labels[top_indices[0]]
            
            st.write("---")
            st.header(f"🥇 Predicted: **{predicted_fruit}**")
            
            # Data for chart
            chart_data = pd.DataFrame({
                'Fruit': [fruit_labels[i] for i in top_indices],
                'Confidence (%)': [float(probs[i] * 100) for i in top_indices]
            })

            # Visual Display
            st.write("### 📊 Top 3 Possibilities")
            st.bar_chart(chart_data, x="Fruit", y="Confidence (%)")
            
            # Detailed View
            st.table(chart_data.set_index('Fruit'))

        except Exception as e:
            st.error(f"Analysis Error: {e}")
            st.info("Check if your model training dimensions match the app settings.")
