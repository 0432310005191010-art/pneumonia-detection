import os
import requests
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image

st.set_page_config(page_title="Pneumonia Detection System", layout="centered")
st.title("Automated Pneumonia Detection System")
st.caption("ResNet50 + Grad-CAM | Team: Khan Abid Pias, Al Arafat Ovi & Saiful Islam Shova | Supervisor: Ratul Mandal")

MODEL_PATH = "resnet50_optimized.keras"
MODEL_URL = "https://huggingface.co/spaces/Saiful09/pneumonia-detection/resolve/main/resnet50_optimized.keras"

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI Model from Hugging Face (please wait ~20 seconds)..."):
            response = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_my_model()
last_conv_layer_name = [layer.name for layer in reversed(model.layers) if len(layer.output.shape) == 4][0]

uploaded_file = st.file_uploader("Upload Chest X-Ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img_resized = cv2.resize(img_np, (224, 224))
    img_array = preprocess_input(np.expand_dims(img_resized.astype('float32'), axis=0))

    with st.spinner("Analyzing X-Ray..."):
        prob = float(model.predict(img_array, verbose=0)[0][0])

        grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.squeeze(conv_outputs[0] @ pooled_grads[..., tf.newaxis])
        heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)

        heatmap_colored = cv2.applyColorMap(np.uint8(255 * cv2.resize(heatmap, (224, 224))), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        superimposed = cv2.addWeighted(img_resized, 0.6, heatmap_colored, 0.4, 0)

    if prob > 0.5:
        st.error(f"⚠️ PNEUMONIA DETECTED (Confidence: {prob*100:.2f}%)")
    else:
        st.success(f"✅ NORMAL CHEST X-RAY (Confidence: {(1-prob)*100:.2f}%)")

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_resized, caption="Original X-Ray", use_container_width=True)
    with col2:
        st.image(superimposed, caption="Grad-CAM Heatmap", use_container_width=True)
