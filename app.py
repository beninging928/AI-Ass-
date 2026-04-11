import streamlit as st
import gdown
import joblib
import os
import numpy as np
from PIL import Image
import tensorflow as tf

# --- STEP 1: DOWNLOAD MODELS FROM GOOGLE DRIVE ---
@st.cache_resource
def load_all_models():
    model_configs = {
        "model1.h5": "13stvBP7-Ta7R2BKnrbuRuTIQtFhikiVw",
        "model2.pkl": "1DDBGQNAUZBu4VNX61NObYjso_6jDVBko",
        "model3.pkl": "1j632tPQnIkFzWcpOQNdqiOVgI3Tpl4qA"
    }

    for filename, file_id in model_configs.items():
        if not os.path.exists(filename):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, filename, quiet=False, fuzzy=True)

    m1 = tf.keras.models.load_model("model1.h5")
    m2 = joblib.load("model2.pkl")
    m3 = joblib.load("model3.pkl")

    return m1, m2, m3

model1, model2, model3 = load_all_models()

# --- STEP 2: APP UI ---
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Home", "Model Playground"])

if selection == "Home":
    st.title("Welcome to the AI Model Hub")
    st.write("Select 'Model Playground' in the sidebar to start.")

else:
    st.title("📷 Model Playground")
    # FIX: This selectbox variable is used to decide which model to run below
    model_choice = st.selectbox("Select Model to Test", ["Detector", "Classifier", "Analyzer"])

    picture = st.camera_input("Take a photo")

    if picture:
        img = Image.open(picture)
        st.image(img, caption="Snapshot", use_container_width=True)

        # Pre-process
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_tensor = np.expand_dims(img_array, axis=0)

        st.write("### AI Result")

        # FIX: Matching the model_choice names exactly
        if model_choice == "Detector":
            prediction = model1.predict(img_tensor)
            result = np.argmax(prediction)
            st.success(f"Model 1 (H5) predicts: {result}")

        elif model_choice == "Classifier":
            flat_img = img_array.flatten().reshape(1, -1)
            prediction = model2.predict(flat_img)
            st.success(f"Model 2 (PKL) predicts: {prediction[0]}")

        else:
            flat_img = img_array.flatten().reshape(1, -1)
            prediction = model3.predict(flat_img)
            st.success(f"Model 3 (PKL) predicts: {prediction[0]}")
