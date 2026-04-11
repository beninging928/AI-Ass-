import streamlit as st
import gdown
import joblib
import os
import numpy as np
from PIL import Image
import tensorflow as tf  # Required for .h5 models

# --- STEP 1: DOWNLOAD MODELS FROM GOOGLE DRIVE ---
@st.cache_resource
def load_all_models():
    # Your specific File IDs from the folder you shared
    model_configs = {
        "model1.h5": "13stvBP7-Ta7R2BKnrbuRuTIQtFhikiVw",
        "model2.pkl": "1DDBGQNAUZBu4VNX61NObYjso_6jDVBko",
        "model3.pkl": "1j632tPQnIkFzWcpOQNdqiOVgI3Tpl4qA"
    }

    for filename, file_id in model_configs.items():
        if not os.path.exists(filename):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, filename, quiet=False)

    # Load with the correct library based on extension
    m1 = tf.keras.models.load_model("model1.h5")
    m2 = joblib.load("model2.pkl")
    m3 = joblib.load("model3.pkl")

    return m1, m2, m3

# Initialize models
model1, model2, model3 = load_all_models()

# --- STEP 2: APP UI ---
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Home", "Model Playground"])

if selection == "Home":
    st.title("Welcome to the AI Model Hub")
    st.write("Select 'Model Playground' in the sidebar to start.")

else:
    st.title("📷 Model Playground")
    model_choice = st.selectbox("Select Model to Test", ["CNN (H5)", "Model 2 (PKL)", "Model 3 (PKL)"])

    picture = st.camera_input("Take a photo")

    if picture:
      # 1. Open the image
      img = Image.open(picture)
      st.image(img, caption="Snapshot", use_container_width=True)

      # 2. Pre-process the image
      # Note: Most .h5 models need a specific size (e.g., 224x224)
      img_resized = img.resize((224, 224))
      img_array = np.array(img_resized) / 255.0  # Normalizing
      img_tensor = np.expand_dims(img_array, axis=0) # Add batch dimension (1, 224, 224, 3)

      st.write("### AI Result")

      if selection == "Detector": # Your .h5 model
          prediction = model1.predict(img_tensor)
          # Assuming classification, get the index of the highest score
          result = np.argmax(prediction)
          st.success(f"Model 1 (H5) predicts: {result}")

      elif selection == "Classifier": # Your .pkl model
          # .pkl models usually want flat data, not a 3D image
          flat_img = img_array.flatten().reshape(1, -1)
          prediction = model2.predict(flat_img)
          st.success(f"Model 2 (PKL) predicts: {prediction[0]}")

      else: # Your 3rd model
          flat_img = img_array.flatten().reshape(1, -1)
          prediction = model3.predict(flat_img)
          st.success(f"Model 3 (PKL) predicts: {prediction[0]}")

