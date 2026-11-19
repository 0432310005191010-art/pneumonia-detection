import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import gradio as gr

# মডেল লোড করা
model = tf.keras.models.load_model("resnet50_optimized.keras")
last_conv_layer_name = [layer.name for layer in reversed(model.layers) if len(layer.output.shape) == 4][0]

def predict_pneumonia(input_img):
    if input_img is None:
        return "Please upload an X-Ray image.", None

    img_resized = cv2.resize(input_img, (224, 224))
    img_array = preprocess_input(np.expand_dims(img_resized.astype('float32'), axis=0))

    prob = float(model.predict(img_array, verbose=0)[0][0])
    if prob > 0.5:
        label = "⚠️ PNEUMONIA DETECTED"
        confidence = prob * 100
    else:
        label = "✅ NORMAL CHEST X-RAY"
        confidence = (1 - prob) * 100

    result_text = f"{label}\nConfidence Score: {confidence:.2f}%"

    # Grad-CAM হিটম্যাপ তৈরি
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

    return result_text, superimposed

demo = gr.Interface(
    fn=predict_pneumonia,
    inputs=gr.Image(label="Upload Chest X-Ray Image"),
    outputs=[
        gr.Textbox(label="Diagnostic Result", lines=2),
        gr.Image(label="Grad-CAM Explainability Heatmap")
    ],
    title="Automated Pneumonia Detection System (ResNet50 + Grad-CAM)",
    description="Team: Khan Abid Pias, Al Arafat Ovi & Saiful Islam Shova | Supervisor: Ratul Mandal"
)

demo.launch()