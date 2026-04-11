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
        # Load and display original
        img = Image.open(picture)
        
        # Pre-process image
        img_resized = img.resize(target_size)
        img_array = np.array(img_resized) / 255.0  # Normalize to 0-1
        
        st.write("---")
        
        try:
            with st.spinner("Analyzing..."):
                if choice == "CNN Model":
                    # CNN expects 4D: (Batch, Width, Height, Channels)
                    inp = np.expand_dims(img_array, axis=0)
                    prediction = model1.predict(inp)
                    result_index = np.argmax(prediction)
                    confidence = np.max(prediction) * 100
                
                else:
                    # SVM and LogReg expect 2D Flat Vector: (1, Pixels*Channels)
                    # This converts (128, 128, 3) into (1, 49152)
                    flat_inp = img_array.reshape(1, -1)
                    
                    if choice == "SVM Model":
                        result_index = int(model2.predict(flat_inp)[0])
                    else:
                        result_index = int(model3.predict(flat_inp)[0])
                    
                    confidence = 100 # Default for non-probabilistic ML models

            # Get Fruit Name
            detected_fruit = fruit_labels[result_index]

            # --- DRAWING THE "DETECTION" BOX ---
            # Classifier models see the whole image, so we frame the whole image
            draw = ImageDraw.Draw(img)
            draw.rectangle([15, 15, img.size[0]-15, img.size[1]-15], outline="lime", width=12)
            
            # Show Results
            st.image(img, caption=f"Analyzed via {choice}", use_container_width=True)
            
            col1, col2 = st.columns(2)
            col1.success(f"**Result:** {detected_fruit}")
            col2.info(f"**Confidence:** {confidence:.1f}%")

        except ValueError as e:
            st.error("📐 Dimension Mismatch Error!")
            st.write(f"The model expects a different image size than {target_size}. Check your Colab training code!")
            st.expander("Technical details").write(e)
