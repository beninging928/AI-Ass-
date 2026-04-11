import streamlit as st
import os
import cv2
import numpy as np
import joblib
import tensorflow as tf
from skimage.feature import hog
from PIL import Image

# --- SETTINGS ---
IMG_SIZE = 64
fruit_labels = ["Apple", "Avocado", "Banana", "Broccoli", "Capsicum", "Cauliflower", "Cucumber", "Lemon", "Mango", "Watermelon"]

# --- FEATURE EXTRACTION (YOUR LOGIC) ---
def extract_features_lr(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    hog_feat = hog(gray, pixels_per_cell=(4, 4), cells_per_block=(2, 2), feature_vector=True)
    
    color_feat = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    color_feat = cv2.normalize(color_feat, color_feat).flatten()
    
    edges = cv2.Canny(gray, 100, 200)
    edge_feat = edges.flatten()
    
    return np.hstack([hog_feat, color_feat, edge_feat])

def extract_features_svm(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    
    hog_feat = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    
    img_color = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    color_feat = cv2.calcHist([img_color], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    color_feat = cv2.normalize(color_feat, color_feat).flatten()
    
    return np.hstack([hog_feat, color_feat])

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    # Note: Ensure these filenames match exactly what gdown downloads
    m1 = tf.keras.models.load_model("fruit_model_v2.h5")
    m2 = joblib.load("svm_best_v2.pkl")
    m3 = joblib.load("lr_improved.pkl")
    return m1, m2, m3

model_cnn, model_svm, model_lr = load_models()

# --- UI ---
st.title("🍎 Live Fruit Detector")
choice = st.selectbox("Select Model", ["CNN", "SVM", "Logistic Regression"])

picture = st.camera_input("Take a photo of a fruit")

if picture:
    # Convert Streamlit picture to OpenCV format
    img_raw = Image.open(picture)
    img_cv = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)

    with st.spinner("Processing..."):
        if choice == "CNN":
            img_cnn = cv2.resize(img_cv, (128, 128)) / 255.0
            pred = model_cnn.predict(np.expand_dims(img_cnn, axis=0))
            idx = np.argmax(pred)
        
        elif choice == "SVM":
            feat = extract_features_svm(img_cv)
            idx = model_svm.predict([feat])[0]
            
        else: # Logistic Regression
            feat = extract_features_lr(img_cv)
            idx = model_lr.predict([feat])[0]

    st.success(f"Prediction: {fruit_labels[idx]}")
    st.image(img_raw, caption=f"Analyzed by {choice}")
