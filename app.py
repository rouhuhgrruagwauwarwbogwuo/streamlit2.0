import os
import cv2
import tempfile
import requests
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import h5py

# Hugging Face 模型網址
MODEL_URL = "https://huggingface.co/wuwuwu123123/deepfake/resolve/main/deepfake_cnn_model.h5"

@st.cache_resource
def download_model():
    model_path = os.path.join(tempfile.gettempdir(), "deepfake_cnn_model.h5")
    if not os.path.exists(model_path):
        response = requests.get(MODEL_URL)
        with open(model_path, "wb") as f:
            f.write(response.content)
    return load_model(model_path)

# 載入模型
custom_model = download_model()

# ResNet 模型
resnet_base = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(256, 256, 3))
resnet_model = Sequential([resnet_base, Dense(1, activation='sigmoid')])
resnet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 預處理函數
def preprocess_for_models(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256))

    resnet_input = preprocess_input(np.expand_dims(img_resized, axis=0))

    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    custom_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    custom_input = np.expand_dims(custom_rgb / 255.0, axis=0)

    return resnet_input, custom_input, img_rgb

# App UI
st.title("🕵️ Deepfake 偵測 App")

# --- 圖片上傳 ---
st.subheader("🖼️ 上傳圖片進行偵測")
image_file = st.file_uploader("📤 上傳一張圖片", type=["jpg", "jpeg", "png"], key="image_upload")
if image_file:
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    resnet_input, custom_input, display_img = preprocess_for_models(img)
    resnet_pred = resnet_model.predict(resnet_input)[0][0]
    custom_pred = custom_model.predict(custom_input)[0][0]

    st.image(display_img, caption="你上傳的圖片", use_container_width=True)
    st.markdown(f"**🔹 ResNet50 預測：** {'Deepfake' if resnet_pred > 0.5 else 'Real'} ({resnet_pred:.2%})")
    st.markdown(f"**🔸 自訂模型預測：** {'Deepfake' if custom_pred > 0.5 else 'Real'} ({custom_pred:.2%})")

# --- 影片上傳 ---
st.subheader("🎥 上傳影片進行偵測")
video_file = st.file_uploader("📤 上傳一段影片", type=["mp4", "mov", "avi"], key="video_upload")

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(video_file.read())
        temp_video_path = tmpfile.name

    cap = cv2.VideoCapture(temp_video_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    frame_interval = 10
    frame_count = 0
    predictions_resnet = []
    predictions_custom = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                resnet_input, custom_input, _ = preprocess_for_models(face)
                predictions_resnet.append(resnet_model.predict(resnet_input)[0][0])
                predictions_custom.append(custom_model.predict(custom_input)[0][0])

        frame_count += 1

    cap.release()

    if predictions_resnet and predictions_custom:
        avg_resnet = np.mean(predictions_resnet)
        avg_custom = np.mean(predictions_custom)
        st.markdown(f"**🔹 ResNet50 平均預測：** {'Deepfake' if avg_resnet > 0.5 else 'Real'} ({avg_resnet:.2%})")
        st.markdown(f"**🔸 自訂模型平均預測：** {'Deepfake' if avg_custom > 0.5 else 'Real'} ({avg_custom:.2%})")
    else:
        st.warning("⚠️ 未能從影片中偵測到人臉。")

