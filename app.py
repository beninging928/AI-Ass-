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
fruit_labels = ["Apple", "Avocado", "Banana", "Broccoli", "Capsicum", "Cauliflower", "Cucumber", "Lemon", "Mango", "Watermelon"]

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

def extract_lr(img):
    img_res = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(gray, pixels_per_cell=(4, 4), cells_per_block=(2, 2), feature_vector=True)
    color_feat = cv2.calcHist([img_res], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    color_feat = cv2.normalize(color_feat, color_feat).flatten()
    edges = cv2.Canny(gray, 100, 200).flatten()
    return np.hstack([hog_feat, color_feat, edges])

def extract_svm(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_res = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    hog_feat = hog(gray_res, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    img_res = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    color_feat = cv2.calcHist([img_res], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    color_feat = cv2.normalize(color_feat, color_feat).flatten()
    return np.hstack([hog_feat, color_feat])

# --- APP START ---
st.set_page_config(page_title="Fruit AI", layout="wide")
model_cnn, model_svm, model_lr = load_all_models()

st.title("🍎 Fruit Recognition Dashboard")

# --- SMALLER CAMERA LAYOUT ---
# We use columns to center and shrink the camera feed
col_left, col_mid, col_right = st.columns([1, 2, 1]) 
with col_mid:
    picture = st.camera_input("Scan your fruit")

if picture:
    img_raw = Image.open(picture)
    img_cv = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    
    with st.spinner("Analyzing across all models..."):
        # CNN
        cnn_in = cv2.resize(img_cv, (128, 128)) / 255.0
        cnn_probs = model_cnn.predict(np.expand_dims(cnn_in, axis=0))[0]
        
        # SVM
        svm_feat = extract_svm(img_cv)
        if hasattr(model_svm, "predict_proba"):
            svm_probs = model_svm.predict_proba([svm_feat])[0]
        else:
            scores = model_svm.decision_function([svm_feat])[0]
            exp_s = np.exp(scores - np.max(scores))
            svm_probs = exp_s / exp_s.sum()
        
        # Logistic Regression
        lr_feat = extract_lr(img_cv)
        lr_probs = model_lr.predict_proba([lr_feat])[0]

    # --- ENSEMBLE VERDICT LOGIC ---
    # We take the average probability across all 3 models for a 'Fair' verdict
    final_probs = (cnn_probs + svm_probs + lr_probs) / 3
    final_idx = np.argmax(final_probs)
    final_fruit = fruit_labels[final_idx]
    
    st.divider()

    # Display Verdict
    info = fruit_info.get(final_fruit, {"emoji": "❓", "fact": "N/A", "calories": "N/A"})
    st.header(f"Final Verdict: {info['emoji']} {final_fruit}")
    
    # Results Columns
    c1, c2, c3 = st.columns(3)
    
    models_list = [
        {"name": "CNN Model", "p": cnn_probs, "ui": c1},
        {"name": "SVM Model", "p": svm_probs, "ui": c2},
        {"name": "Logistic Reg", "p": lr_probs, "ui": c3}
    ]

    for m in models_list:
        with m["ui"]:
            idx = np.argmax(m["p"])
            conf = m["p"][idx] * 100
            st.metric(m["name"], fruit_labels[idx], f"{conf:.1f}%")
            
            # Mini Probability Chart
            top3 = m["p"].argsort()[-3:][::-1]
            df = pd.DataFrame({
                'Fruit': [fruit_labels[i] for i in top3],
                'Match': [m["p"][i]*100 for i in top3]
            })
            st.bar_chart(df, x="Fruit", y="Match", height=180)

    # Health Info Box
    st.info(f"💡 **Fun Fact:** {info['fact']} | **Energy:** {info['calories']}")
