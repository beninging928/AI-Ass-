import streamlit as st
import gdown
import joblib
import os
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fruit Detection Hub", layout="centered")

# --- STEP 1: DOWNLOAD & LOAD MODELS ---
@st.cache_resource
def load_all_models():
    # File IDs for model1.h5, model2.pkl, model3.pkl
    model_configs = {
        "model1.h5": "13stvBP7-Ta7R2BKnrbuRuTIQtFhikiVw",
        "model2.pkl": "1DDBGQNAUZBu4VNX61NObYjso_6jDVBko",
        "model3.pkl": "1j632tPQnIkFzWcpOQNdqiOVgI3Tpl4qA"
    }

    for filename, file_id in model_configs.items():
        if not os.path.exists(filename):
            url = f'https://drive.google.com/uc?id={file_id}'
            try:
                gdown.download(url, filename, quiet=False, fuzzy=True)
            except Exception:
                st.error(f"Failed to download {filename}. Check Drive permissions!")
                st.stop()

    # Model 1: CNN (TensorFlow)
    m1 = tf.keras.models.load_model("model1.h5")
    # Model 2: SVM (Scikit-Learn)
    m2 = joblib.load("model2.pkl")
    # Model 3: Logistic Regression (Scikit-Learn)
    m3 = joblib.load("model3.pkl")

    return m1, m2, m3

# Initialize
with st.spinner("Loading AI Models..."):
    model1, model2, model3 = load_all_models()

# --- STEP 2: UI SETUP ---
st.sidebar.title("🛠 Settings")
page = st.sidebar.radio("Navigate", ["Home", "Model Playground"])

# Exact list based on your alphabetical training folders
fruit_labels = [
    "Apple", "Avocado", "Banana", "Broccoli", "Capsicum", 
    "Cauliflower", "Cucumber", "Lemon", "Mango", "Watermelon"
]

if page == "Home":
    st.title("🍎 Fruit Analysis Hub")
    st.markdown("""
    Compare how different AI architectures perform on fruit detection:
    * **Model 1:** CNN (Deep Learning)
    * **Model 2:** SVM (Traditional ML)
    * **Model 3:** Logistic Regression (Statistical ML)
    """)
else:
    st.title("📷 Model Playground")
    
    # Selection mapping to specific variables
    choice = st.selectbox("Select Model Architecture", ["CNN Model", "SVM Model", "Logistic Regression"])
    
    # CRITICAL: This must match your training pixel size (e.g., 64x64 or 128x128)
    target_size = (128, 128) 

    picture = st.camera_input("Snapshot a fruit")

    if picture:
        img = Image.open(picture)
        st.write("---")
        
        try:
            with st.spinner("Scaling and Analyzing..."):
                if choice == "CNN Model":
                    # --- CNN SCALE (e.g., 128x128) ---
                    img_resized = img.resize((128, 128))
                    img_array = np.array(img_resized) / 255.0
                    inp = np.expand_dims(img_array, axis=0)
                    
                    prediction = model1.predict(inp)
                    result_index = np.argmax(prediction)
                    confidence = np.max(prediction) * 100
                
                else:
                    # --- ML MODELS SCALE (e.g., 64x64) ---
                    # Change (64, 64) to whatever size you used in Colab for SVM/LogReg
                    ml_size = (64, 64) 
                    img_resized = img.resize(ml_size)
                    img_array = np.array(img_resized) / 255.0
                    
                    # Flatten the image into a 1D row of numbers
                    flat_inp = img_array.reshape(1, -1)
                    
                    if choice == "SVM Model":
                        result_index = int(model2.predict(flat_inp)[0])
                    else:
                        result_index = int(model3.predict(flat_inp)[0])
                    
                    confidence = 100 

            # Show Result
            detected_fruit = fruit_labels[result_index]
            st.success(f"Detected: {detected_fruit}")
