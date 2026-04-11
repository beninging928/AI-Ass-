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
    "Avocado": {"emoji": "🥑", "fact": "Avocados are actually large berries with a single seed.", "calories": "160 kcal/100g"},
    "Banana": {"emoji": "🍌", "fact": "Bananas are slightly radioactive due to potassium content!", "calories": "89 kcal/100g"},
    "Broccoli": {"emoji": "🥦", "fact": "Broccoli contains more protein per calorie than steak.", "calories": "34 kcal/100g"},
    "Capsicum": {"emoji": "🫑", "fact": "Red bell peppers are just ripened green peppers.", "calories": "20 kcal/100g"},
    "Cauliflower": {"emoji": "🥦", "fact": "The name means 'cabbage flower' in Italian.", "calories": "25 kcal/100g"},
    "Cucumber": {"emoji": "🥒", "fact": "Cucumbers can be up to 96% water.", "calories": "15 kcal/100g"},
    "Lemon": {"emoji": "🍋", "fact": "Lemon juice can prevent other fruits from browning.", "calories": "29 kcal/100g"},
    "Mango": {"emoji": "🥭", "fact": "Mangos are the most consumed fruit in the world.", "calories": "60 kcal/100g"},
    "Watermelon": {"emoji": "🍉", "fact": "Every part of a watermelon, including the rind, is edible.", "calories": "30 kcal/100g"}
}

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
            gdown.download(url, filename, quiet=False, fuzzy=True)

    return (
        tf.keras.models.load_model("fruit_model_v2.h5"),
        joblib.load("svm_best_v2.pkl"),
        joblib.load("lr_improved.pkl")
    )

# --- STEP 2: FEATURE EXTRACTION ---
def extract_lr(img):
    img_res = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_blur = cv2.GaussianBlur(img_res, (3, 3), 0)
    gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
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
st.set_page_config(page_title="Pro Fruit AI", page_icon="🍎", layout="wide")
model_cnn, model_svm, model_lr = load_all_models()

st.title("🍎 Pro Fruit AI Dashboard")
st.markdown("Analyze your fruit using three different AI architectures simultaneously.")

# --- INPUT SECTION (SMALLER) ---
col_space1, col_input, col_space2 = st.columns([1, 2, 1])

with col_input:
    input_tab1, input_tab2 = st.tabs(["📸 Camera Snap", "📁 Upload Image"])
    with input_tab1:
        picture = st.camera_input("Take a photo")
    with input_tab2:
        uploaded_file = st.file_uploader("Choose a file from your computer", type=["jpg", "png", "jpeg"])

# Select the source
source = picture if picture else uploaded_file

if source:
    img_raw = Image.open(source)
    img_cv = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    
    with st.spinner("Analyzing with all models..."):
        # CNN
        cnn_in = cv2.resize(img_cv, (128, 128)) / 255.0
        cnn_probs = model_cnn.predict(np.expand_dims(cnn_in, axis=0))[0]
        cnn_idx = np.argmax(cnn_probs)
        
        # SVM
        svm_feat = extract_svm(img_cv)
        if hasattr(model_svm, "predict_proba"):
            svm_probs = model_svm.predict_proba([svm_feat])[0]
        else:
            scores = model_svm.decision_function([svm_feat])[0]
            exp_s = np.exp(scores - np.max(scores))
            svm_probs = exp_s / exp_s.sum()
        svm_idx = np.argmax(svm_probs)
        
        # Logistic Regression
        lr_feat = extract_lr(img_cv)
        lr_probs = model_lr.predict_proba([lr_feat])[0]
        lr_idx = np.argmax(lr_probs)

    # --- UI LAYOUT: SIDE-BY-SIDE RESULTS ---
    st.divider()
    col1, col2, col3 = st.columns(3)

    models_data = [
        {"name": "CNN (Deep Learning)", "idx": cnn_idx, "prob": cnn_probs, "col": col1},
        {"name": "SVM (Traditional)", "idx": svm_idx, "prob": svm_probs, "col": col2},
        {"name": "Logistic Regression", "idx": lr_idx, "prob": lr_probs, "col": col3}
    ]

    for m in models_data:
        with m["col"]:
            fruit = fruit_labels[m["idx"]]
            conf = m["prob"][m["idx"]] * 100
            st.metric(m["name"], f"{fruit}", f"{conf:.1f}% Match")
            
            # Mini Bar Chart
            top3_idx = m["prob"].argsort()[-3:][::-1]
            df = pd.DataFrame({
                'Fruit': [fruit_labels[i] for i in top3_idx],
                'Conf': [m["prob"][i]*100 for i in top3_idx]
            })
            st.bar_chart(df, x="Fruit", y="Conf", height=200)

    # --- FINAL VERDICT SECTION ---
    st.divider()
    # Using CNN as the primary verdict
    final_fruit = fruit_labels[cnn_idx]
    info = fruit_info.get(final_fruit, {"emoji": "❓", "fact": "No data", "calories": "N/A"})

    v1, v2 = st.columns([1, 2])
    with v1:
        st.image(img_raw, use_container_width=True, caption="Source Image")
    with v2:
        st.header(f"{info['emoji']} Final Verdict: {final_fruit}")
        st.info(f"**Did you know?** {info['fact']}")
        st.write(f"**Energy Content:** {info['calories']}")
        
        if st.button("Search for Recipes"):
            st.write(f"[Click here for {final_fruit} recipes](https://www.google.com/search?q={final_fruit}+recipes)")
