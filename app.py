import streamlit as st
import gdown
import joblib
import os
import numpy as np
from PIL import Image
import tensorflow as tf

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Multi-Model Hub", layout="centered")

# --- STEP 1: DOWNLOAD & LOAD MODELS ---
@st.cache_resource
def load_all_models():
    # Mapping of filenames to Google Drive File IDs
    model_configs = {
        "model1.h5": "13stvBP7-Ta7R2BKnrbuRuTIQtFhikiVw",
        "model2.pkl": "1DDBGQNAUZBu4VNX61NObYjso_6jDVBko",
        "model3.pkl": "1j632tPQnIkFzWcpOQNdqiOVgI3Tpl4qA"
    }

    for filename, file_id in model_configs.items():
        if not os.path.exists(filename):
            url = f'https://drive.google.com/uc?id={file_id}'
            try:
                # fuzzy=True helps bypass Google's "large file" warning page
                gdown.download(url, filename, quiet=False, fuzzy=True)
            except Exception as e:
                st.error(f"Failed to download {filename}. Check Drive permissions!")
                st.stop()

    # Load with correct libraries
    m1 = tf.keras.models.load_model("model1.h5")
    m2 = joblib.load("model2.pkl")
    m3 = joblib.load("model3.pkl")

    return m1, m2, m3

# Initialize
with st.spinner("Loading AI Models... Please wait."):
    model1, model2, model3 = load_all_models()

# --- STEP 2: SIDEBAR NAVIGATION ---
st.sidebar.title("🛠 Dashboard")
page = st.sidebar.radio("Navigation", ["Home", "Model Playground"])

if page == "Home":
    st.title("🤖 Welcome to the AI Model Hub")
    st.markdown("""
    This application allows you to toggle between three different trained models 
    and test them in real-time using your camera.
    
    1. **Model 1 (H5):** Deep Learning CNN.
    2. **Model 2 (PKL):** Scikit-Learn Classifier.
    3. **Model 3 (PKL):** Scikit-Learn Pipeline.
    """)
    st.info("👈 Select 'Model Playground' from the sidebar to begin.")

else:
    st.title("📷 Model Playground")
    
    model_choice = st.selectbox("Select Model", ["Detector (H5)", "Classifier (PKL)", "Analyzer (PKL)"])
    target_size = (128, 128) 

    picture = st.camera_input("Take a snapshot")

    if picture:
        # --- TEST 1: See if image even loads ---
        st.info("✅ Image captured! Processing...")
        img = Image.open(picture)
        st.image(img, caption="What the AI sees", use_container_width=True)

        # --- TEST 2: Processing ---
        img_resized = img.resize(target_size)
        img_array = np.array(img_resized) / 255.0
        
        st.write("---")
        st.subheader("Results")

        try:
            if "H5" in model_choice:
                st.write("🔄 Running H5 Model...")
                img_tensor = np.expand_dims(img_array, axis=0)
                prediction = model1.predict(img_tensor)
                result = np.argmax(prediction)
                st.success(f"Final Prediction: {result}")

            else:
                st.write(f"🔄 Running {model_choice}...")
                flat_img = img_array.flatten().reshape(1, -1)
                
                # Check which model to use
                if "Classifier" in model_choice:
                    prediction = model2.predict(flat_img)
                else:
                    prediction = model3.predict(flat_img)
                
                st.success(f"Final Prediction: {prediction[0]}")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
