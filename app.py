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

# --- SETTINGS & DATABASE ---
IMG_SIZE_RF = 128  # RF used 128
IMG_SIZE_SVM = 64  # SVM used 64
CONFIDENCE_THRESHOLD = 0.45 

fruit_labels = ["Apple", "Avocado", "Banana", "Broccoli", "Capsicum", "Cauliflower", "Cucumber", "Lemon", "Mango", "Watermelon"]

model_metrics = {
    "CNN": {"Accuracy": 0.9250, "F1": 0.92, "Note": "High Performance Wy-v2"},
    "SVM": {"Accuracy": 0.5403, "F1": 0.54, "Note": "Stable Baseline"},
    "Random Forest": {"Accuracy": 0.8850, "F1": 0.88, "Note": "Replaced Logistic Regression"} # Update with your RF result
}

fruit_info = {
    "Apple": {"emoji": "🍎", "fact": "Apples are 25% air, which is why they float!", "calories": "52 kcal/100g"},
    "Avocado": {"emoji": "🥑", "fact": "Avocados are actually large berries.", "calories": "160 kcal/100g"},
    "Banana": {"emoji": "🍌", "fact": "Bananas are slightly radioactive!", "calories": "89 kcal/100g"},
    "Broccoli": {"emoji": "🥦", "fact": "Broccoli has more protein per calorie than steak.", "calories": "34 kcal/100g"},
    "Capsicum": {"emoji": "🫑", "fact": "Red peppers are just ripened green peppers.", "calories": "20 kcal/100g"},
    "Cauliflower": {"emoji": "🥦", "fact": "The name means 'cabbage flower'.", "calories": "25 kcal/100g"},
    "Cucumber": {"emoji": "🥒", "fact": "Up to 96% water.", "calories": "15 kcal/100g"},
    "Lemon": {"emoji": "🍋", "fact": "Prevents other fruits from browning.", "calories": "29 kcal/100g"},
    "Mango": {"emoji": "🥭", "fact": "Most consumed fruit in the world.", "calories": "60 kcal/100g"},
    "Watermelon": {"emoji": "🍉", "fact": "Every part, including the rind, is edible.", "calories": "30 kcal/100g"}
}

@st.cache_resource
def load_all_models():
    # Make sure these files exist in your folder!
    model_configs = {
        "fruit_model_v2_wy.h5": "YOUR_CNN_ID", 
        "svm_best_v2.pkl": "1DDBGQNAUZBu4VNX61NObYjso_6jDVBko",
        "fruit_rf_model.pkl": "1EkN8HSRlWzcRXOu7EsKdcOkueZBOV8BL" 
    }
    for filename, file_id in model_configs.items():
        if not os.path.exists(filename) and file_id != "LOCAL_OR_DRIVE_ID":
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, filename, quiet=False, fuzzy=True)
    
    return (
        tf.keras.models.load_model("fruit_model_v2_wy.h5", compile=False),
        joblib.load("svm_best_v2.pkl"),
        joblib.load("fruit_rf_model.pkl")
    )

# --- NEW RF FEATURE EXTRACTION (Matches your new training code) ---
def extract_rf(img_bgr):
    img = cv2.resize(img_bgr, (IMG_SIZE_RF, IMG_SIZE_RF))
    
    # 1. COLOR: HSV Histogram
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    color_feat = hist.flatten()

    # 2. SHAPE: HOG
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(gray, orientations=9, pixels_per_cell=(16, 16), 
                   cells_per_block=(2, 2), feature_vector=True)

    return np.hstack([color_feat, hog_feat])

# --- OLD SVM FEATURE EXTRACTION ---
def extract_svm(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_res = cv2.resize(gray, (IMG_SIZE_SVM, IMG_SIZE_SVM))
    hog_feat = hog(gray_res, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
    img_res = cv2.resize(img_bgr, (IMG_SIZE_SVM, IMG_SIZE_SVM))
    color_feat = cv2.calcHist([img_res],[0,1,2],None,[8,8,8],[0,256,0,256,0,256])
    color_feat = cv2.normalize(color_feat, color_feat).flatten()
    return np.hstack([hog_feat, color_feat])

# --- UI SETUP ---
st.set_page_config(page_title="Pro Fruit AI Dashboard", layout="wide")
model_cnn, model_svm, model_rf = load_all_models()

st.title("🍎 Pro Fruit AI Analysis")

c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    tabs = st.tabs(["📸 Snapshot", "📁 Upload"])
    with tabs[0]: picture = st.camera_input("Scan Fruit")
    with tabs[1]: upload = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

input_img = picture if picture else upload

if input_img:
    img_raw = Image.open(input_img)
    img_cv = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    
    with st.spinner("Analyzing with Ensembles..."):
        # 1. CNN (RGB)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        cnn_in = cv2.resize(img_rgb, (128,128)) / 255.0
        cnn_probs = model_cnn.predict(np.expand_dims(cnn_in, axis=0), verbose=0)[0]
        
        # 2. SVM (BGR - 64px)
        svm_feat = extract_svm(img_cv)
        svm_probs = model_svm.predict_proba([svm_feat])[0]
            
        # 3. RANDOM FOREST (BGR - 128px + HSV)
        rf_feat = extract_rf(img_cv)
        rf_probs = model_rf.predict_proba([rf_feat])[0]

    # --- WEIGHTED CALCULATION ---
    # Since RF is usually stronger than LR, let's give it a good weight.
    weighted_probs = (cnn_probs * 0.5) + (rf_probs * 0.3) + (svm_probs * 0.2)
    best_conf = np.max(weighted_probs)
    final_idx = np.argmax(weighted_probs)
    final_fruit = fruit_labels[final_idx]

    st.divider()

    if best_conf < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ **No supported fruit detected.**")
    else:
        m_cols = st.columns(3)
        info_list = [("CNN", cnn_probs, m_cols[0]), ("Random Forest", rf_probs, m_cols[1]), ("SVM", svm_probs, m_cols[2])]
        
        for name, probs, col in info_list:
            with col:
                idx = np.argmax(probs)
                st.metric(name, fruit_labels[idx], f"{probs[idx]*100:.1f}%")
                with st.expander("📊 Metrics"):
                    st.write(f"Accuracy: {model_metrics[name]['Accuracy']:.1%}")
                top3 = probs.argsort()[-3:][::-1]
                df = pd.DataFrame({'Fruit': [fruit_labels[i] for i in top3], 'Prob': [probs[i]*100 for i in top3]})
                st.bar_chart(df, x="Fruit", y="Prob", height=180)

        st.divider()
        info = fruit_info.get(final_fruit, {"emoji": "❓", "fact": "N/A", "calories": "N/A"})
        v1, v2 = st.columns([1, 2])
        with v1: st.image(img_raw, use_container_width=True)
        with v2:
            st.header(f"{info['emoji']} Verdict: {final_fruit}")
            st.success(f"Confidence Level: {best_conf*100:.1f}%")
            st.info(f"**Did you know?** {info['fact']}")
