import os
import time
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
        with st.spinner("Downloading AI Model for the first time (please wait ~20 seconds)..."):
            response = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_my_model()

# শেষ Conv লেয়ার খুঁজে বের করা (Grad-CAM এর জন্য)
last_conv_layer_name = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer_name = layer.name
        break

uploaded_file = st.file_uploader("Upload Chest X-Ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img_resized = cv2.resize(img_np, (224, 224))
    
    # আপলোড করা ছবিটি প্রিভিউ দেখানো
    st.image(img_resized, caption="Uploaded Chest X-Ray", width=250)

    # প্রেডিকশন করার জন্য বাটন
    if st.button("🔍 Analyze X-Ray (Detect Pneumonia)", type="primary"):
        # লোডিং প্রোগ্রেস বার
        progress_bar = st.progress(0, text="Initializing AI diagnostic engine...")
        time.sleep(0.3)
        progress_bar.progress(40, text="Extracting lung features with ResNet50...")
        
        img_array = preprocess_input(np.expand_dims(img_resized.astype('float32'), axis=0))
        prob = float(model.predict(img_array, verbose=0)[0][0])
        
        progress_bar.progress(75, text="Generating Grad-CAM explainability heatmap...")

        # সঠিক Grad-CAM হিটম্যাপ তৈরি (Softmax/Sigmoid লজিটসহ)
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            # Sigmoid আউটপুটের বদলে সরাসরি লজিট বা ক্লাস স্কোর নেওয়া
            class_channel = predictions[:, 0]
            
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()
        
        # হিটম্যাপ নরমালাইজেশন (যাতে লাল ও হলুদ রঙ স্পষ্ট আসে)
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        else:
            # ফলব্যাক: অ্যাক্টিভেশন ম্যাপের গড় নেওয়া
            heatmap = np.mean(np.maximum(conv_outputs.numpy(), 0), axis=-1)
            heatmap = heatmap / (np.max(heatmap) + 1e-10)

        heatmap_colored = cv2.applyColorMap(np.uint8(255 * cv2.resize(heatmap, (224, 224))), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        superimposed = cv2.addWeighted(img_resized, 0.55, heatmap_colored, 0.45, 0)

        progress_bar.progress(100, text="Analysis Complete!")
        time.sleep(0.2)
        progress_bar.empty()

        st.divider()
        st.subheader("Diagnostic Report:")

        # Positive বা Negative রেজাল্ট প্রদর্শন
        if prob > 0.5:
            st.error(f"### ⚠️ Result: PNEUMONIA POSITIVE\n**Confidence Score:** {prob*100:.2f}%")
        else:
            st.success(f"### ✅ Result: PNEUMONIA NEGATIVE (Normal Lung)\n**Confidence Score:** {(1-prob)*100:.2f}%")

        # পাশাপাশি ছবি দেখানো
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_resized, caption="Original Chest X-Ray", use_container_width=True)
        with col2:
            st.image(superimposed, caption="Grad-CAM Infected Region Heatmap", use_container_width=True)
