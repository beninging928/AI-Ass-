import streamlit as st
import gdown
import joblib
import os
import cv2
import numpy as np
import tensorflow as tf
from skimage.feature import hog
from PIL import Image

# --- SETTINGS ---
IMG_SIZE = 64
fruit_labels = ["Apple", "Avocado", "Banana", "Broccoli", "Capsicum", "Cauliflower", "Cucumber", "Lemon", "Mango", "Watermelon"]

# --- STEP 1: DOWNLOAD MODELS FROM GOOGLE DRIVE ---
@st.cache_resource
def load_all_models():
    # Mapping filenames to your Google Drive IDs
    model_configs = {
        "fruit_model_v2.h5": "13stvBP7-Ta7R2BKnrbuRuTIQtFhikiVw",
        "svm_best_v2.pkl": "1DDBGQNAUZBu4VNX61NObYjso_6jDVBko",
        "lr_improved.pkl": "1j632tPQnIkFzWcpOQNdqiOVgI3Tpl4qA"
    }

    for filename, file_id in model_configs.items():
        if not os.path.exists(filename):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, filename, quiet=False, fuzzy=True)

    m1 = tf.keras.models.load_model("fruit_model_v2.h5")
    m2 = joblib.load("svm_best_v2.pkl")
    m3 = joblib.load("lr_improved.pkl")
    return m1, m2, m3

# --- STEP 2: YOUR FEATURE EXTRACTION LOGIC ---
def extract_features_lr(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # HOG Features
    hog_feat = hog(gray, pixels_per_cell=(4, 4), cells_per_block=(2, 2), feature_vector=True)
    
    # Color Histogram Features
    color_feat = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    color_feat = cv2.normalize(color_feat, color_feat).flatten()
    
    # Edge Features
    edges = cv2.Canny(gray, 100, 200)
    edge_feat = edges.flatten()
    
    return np.hstack([hog_feat, color_feat, edge_feat])

def extract_features_svm(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    
    # HOG Features
    hog_feat = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    color_feat = cv2.calcHist([img_resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    color_feat = cv2.normalize(color_feat, color_feat).flatten()
    
    return np.hstack([hog_feat, color_feat])

# --- INITIALIZE ---
with st.spinner("Downloading and loading models from Drive..."):
    model_cnn, model_svm, model_lr = load_all_models()

# --- STEP 3: STREAMLIT UI ---
st.title("🍎 Advanced Fruit Detector")
st.sidebar.header("Options")
choice = st.sidebar.selectbox("Choose AI Model", ["CNN (Deep Learning)", "SVM (HOG + Color)", "Logistic Regression (Complex)"])

picture = st.camera_input("Snapshot a fruit")

if picture:
    # Convert Streamlit upload to OpenCV BGR format
    img_raw = Image.open(picture)
    img_cv = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)

    with st.spinner(f"Running {choice} analysis..."):
        try:
            if "CNN" in choice:
                # CNN expects 128x128 normalized
                img_cnn = cv2.resize(img_cv, (128, 128)) / 255.0
                pred = model_cnn.predict(np.expand_dims(img_cnn, axis=0))
                idx = np.argmax(pred)
            
            elif "SVM" in choice:
                feat = extract_features_svm(img_cv)
                idx = int(model_svm.predict([feat])[0])
                
            else: # Logistic Regression
                feat = extract_features_lr(img_cv)
                idx = int(model_lr.predict([feat])[0])

            # --- DISPLAY RESULTS ---
            detected_name = fruit_labels[idx]
            
            st.success(f"### Result: {detected_name}")
            
            # Simple visual "detection" box
            h, w, _ = img_cv.shape
            cv2.rectangle(img_cv, (20, 20), (w-20, h-20), (0, 255, 0), 10)
            
            # Convert back to RGB for Streamlit display
            res_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            st.image(res_img, caption=f"Analyzed by {choice}")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.info("This is usually caused by a mismatch in feature sizes. Check your training dimensions!")
