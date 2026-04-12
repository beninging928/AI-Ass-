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
IMG_SIZE = 64
CONFIDENCE_THRESHOLD = 0.45 

fruit_labels = ["Apple", "Avocado", "Banana", "Broccoli", "Capsicum", "Cauliflower", "Cucumber", "Lemon", "Mango", "Watermelon"]

# 1. UPDATED METRICS (Based on your high-accuracy results)
model_metrics = {
    "CNN": {"Accuracy": 0.9250, "F1": 0.92, "Note": "Optimized Wy-v2 Model"}, # Update with your exact high score
    "SVM": {"Accuracy": 0.5403, "F1": 0.54, "Note": "Best at Watermelon"},
    "Logistic Regression": {"Accuracy": 0.4417, "F1": 0.44, "Note": "Balanced performance"}
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
    # 2. UPDATE THE DRIVE ID FOR YOUR NEW MODEL HERE
    model_configs = {
        "fruit_model_v2_wy.h5": "15cCVNTiTD3bmarY4UivdOt8vSSS3KjVs", 
        "svm_best_v2.pkl": "1DDBGQNAUZBu4VNX61NObYjso_6jDVBko",
        "lr_improved.pkl": "1j632tPQnIkFzWcpOQNdqiOVgI3Tpl4qA"
    }
    for filename, file_id in model_configs.items():
        if not os.path.exists(filename):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, filename, quiet=False, fuzzy=True)
    
    return (
        tf.keras.models.load_model("fruit_model_v2_wy.h5", compile=False),
        joblib.load("svm_best_v2.pkl"),
        joblib.load("lr_improved.pkl")
    )

# --- FEATURE EXTRACTION (Traditional ML uses BGR) ---
def extract_lr(img_bgr):
    img_res = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img_blur = cv2.GaussianBlur(img_res, (3, 3), 0)
    gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(gray, pixels_per_cell=(4,4), cells_per_block=(2,2), feature_vector=True)
    color_feat = cv2.calcHist([img_res],[0,1,2],None,[8,8,8],[0,256,0,256,0,256])
    color_feat = cv2.normalize(color_feat, color_feat).flatten()
    edge_feat = cv2.Canny(gray, 100, 200).flatten()
    return np.hstack([hog_feat, color_feat, edge_feat])

def extract_svm(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_res = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    hog_feat = hog(gray_res, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
    img_res = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    color_feat = cv2.calcHist([img_res],[0,1,2],None,[8,8,8],[0,256,0,256,0,256])
    color_feat = cv2.normalize(color_feat, color_feat).flatten()
    return np.hstack([hog_feat, color_feat])

# --- UI SETUP ---
st.set_page_config(page_title="Pro Fruit AI Dashboard", layout="wide")
model_cnn, model_svm, model_lr = load_all_models()

st.title("🍎 Pro Fruit AI Analysis")
with st.expander("✅ See Detectable Fruits & Vegetables"):
    cols = st.columns(5)
    for i, label in enumerate(fruit_labels):
        emoji = fruit_info.get(label, {}).get("emoji", "▫️")
        cols[i % 5].write(f"{emoji} **{label}**")

c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    tabs = st.tabs(["📸 Snapshot", "📁 Upload"])
    with tabs[0]: picture = st.camera_input("Scan Fruit")
    with tabs[1]: upload = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

input_img = picture if picture else upload

if input_img:
    img_raw = Image.open(input_img)
    img_cv = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR) # OpenCV default is BGR
    
    with st.spinner("Analyzing..."):
        # 3. CNN PREPROCESSING (RGB FIX)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB) # Convert to RGB for CNN
        cnn_in = cv2.resize(img_rgb, (128,128)) / 255.0
        cnn_probs = model_cnn.predict(np.expand_dims(cnn_in, axis=0), verbose=0)[0]
        
        # SVM & LR (Use original BGR)
        svm_feat = extract_svm(img_cv)
        if hasattr(model_svm, "predict_proba"):
            svm_probs = model_svm.predict_proba([svm_feat])[0]
        else:
            scores = model_svm.decision_function([svm_feat])[0]
            e_s = np.exp(scores - np.max(scores))
            svm_probs = e_s / e_s.sum()
            
        lr_feat = extract_lr(img_cv)
        lr_probs = model_lr.predict_proba([lr_feat])[0]

    # --- UPDATED WEIGHTED CALCULATION ---
    # Now that CNN is the strongest, we give it the most weight (e.g., 70%)
    weighted_probs = (cnn_probs * 0.7) + (svm_probs * 0.2) + (lr_probs * 0.1)
    best_conf = np.max(weighted_probs)
    final_idx = np.argmax(weighted_probs)
    final_fruit = fruit_labels[final_idx]

    st.divider()

    if best_conf < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ **No supported fruit detected.**")
        st.image(img_raw, width=300)
    else:
        m_cols = st.columns(3)
        info_list = [("CNN (Strongest)", cnn_probs, m_cols[0]), ("SVM", svm_probs, m_cols[1]), ("Logistic Regression", lr_probs, m_cols[2])]
        
        for name, probs, col in info_list:
            with col:
                idx = np.argmax(probs)
                st.metric(name, fruit_labels[idx], f"{probs[idx]*100:.1f}% Match")
                with st.expander("📊 Test Metrics"):
                    st.write(f"Accuracy: {model_metrics[name]['Accuracy']:.2%}")
                top3 = probs.argsort()[-3:][::-1]
                df = pd.DataFrame({'Fruit': [fruit_labels[i] for i in top3], 'Prob': [probs[i]*100 for i in top3]})
                st.bar_chart(df, x="Fruit", y="Prob", height=180)

        st.divider()
        info = fruit_info.get(final_fruit, {"emoji": "❓", "fact": "N/A", "calories": "N/A"})
        v1, v2 = st.columns([1, 2])
        with v1: st.image(img_raw, use_container_width=True)
        with v2:
            st.header(f"{info['emoji']} Consensus Verdict: {final_fruit}")
            st.success(f"Confidence Level: {best_conf*100:.1f}%")
            st.info(f"**Did you know?** {info['fact']}")
            st.markdown(f"**Estimated Energy:** {info['calories']}")
