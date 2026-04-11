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
    fruit_labels = ["Apple", "Avocado", "Banana", "Broccoli", "Capsicum", "Cauliflower", "Cucumber", "Lemon", "Mango", "Watermelon"]
    if picture:
    img = Image.open(picture)
    
    # Pre-process
    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    
    with st.spinner("Analyzing fruit..."):
        if "H5" in model_choice:
            prediction = model1.predict(np.expand_dims(img_array, axis=0))
            result_index = np.argmax(prediction)
            confidence = np.max(prediction) * 100
        else:
            flat_img = img_array.flatten().reshape(1, -1)
            result_index = model2.predict(flat_img)[0]
            confidence = 100  # Scikit-learn doesn't always provide probability

        # --- STEP 3: SHOW HUMAN NAMES ---
        detected_fruit = fruit_labels[result_index]

        # Display result with a nice UI
        st.subheader(f"Fruit Detected: {detected_fruit}")
        
        # This adds a "Metric" card for confidence
        st.metric(label="AI Confidence", value=f"{confidence:.1f}%")

        if confidence > 70:
            st.success(f"I am pretty sure this is a {detected_fruit}!")
        else:
            st.warning("I'm not very confident, but it looks like a " + detected_fruit)

        # To "draw" a label on the image:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        # Drawing a simple rectangle around the edge to simulate a "match"
        draw.rectangle([10, 10, img.size[0]-10, img.size[1]-10], outline="green", width=10)
        
        st.image(img, caption=f"Result: {detected_fruit}")
