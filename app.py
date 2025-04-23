import os
import cv2
import h5py
import tempfile
import numpy as np
import streamlit as st
import requests
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing import image as keras_image

# ---------------------
# 模型下載與載入
# ---------------------
MODEL_URL = "https://huggingface.co/wuwuwu123123/deepfake/resolve/main/deepfake_cnn_model.h5"

@st.cache_resource
def download_model():
    model_path = os.path.join(tempfile.gettempdir(), "deepfake_cnn_model.h5")
    if not os.path.exists(model_path):
        response = requests.get(MODEL_URL)
        with open(model_path, "wb") as f:
            f.write(response.content)
    return load_model(model_path)

custom_model = download_model()

resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(256, 256, 3))
resnet_classifier = Sequential([
    resnet_model,
    Dense(1, activation='sigmoid')
])
resnet_classifier.compile(optimizer='adam', loss='binary_crossentropy')

# ---------------------
# 預處理函數
# ---------------------
def preprocess_for_models(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256))

    # ResNet50
    resnet_input = preprocess_input(np.expand_dims(img_resized, axis=0))

    # Custom CNN
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    clahe_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    custom_input = np.expand_dims(clahe_rgb / 255.0, axis=0)

    return resnet_input, custom_input, img_rgb

# ---------------------
# 影片處理函數
# ---------------------
def process_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    frame_count = 0
    frame_interval = 10
    resnet_preds, custom_preds = [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                resnet_input, custom_input, _ = preprocess_for_models(face)
                resnet_pred = resnet_classifier.predict(resnet_input, verbose=0)[0][0]
                custom_pred = custom_model.predict(custom_input, verbose=0)[0][0]
                resnet_preds.append(resnet_pred)
                custom_preds.append(custom_pred)
        frame_count += 1

    cap.release()
    os.unlink(tfile.name)

    return resnet_preds, custom_preds

# ---------------------
# Streamlit 介面
# ---------------------
st.title("🧠 Deepfake 偵測 App")
file = st.file_uploader("📤 上傳圖片或影片", type=["jpg", "jpeg", "png", "mp4"])

if file:
    file_type = file.type
    if "image" in file_type:
        # 處理圖片
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        resnet_input, custom_input, display_img = preprocess_for_models(img)

        resnet_pred = resnet_classifier.predict(resnet_input)[0][0]
        custom_pred = custom_model.predict(custom_input)[0][0]

        resnet_label = "Deepfake" if resnet_pred > 0.5 else "Real"
        custom_label = "Deepfake" if custom_pred > 0.5 else "Real"

        st.image(display_img, caption="你上傳的圖片", use_container_width=True)
        st.markdown(f"### 🔍 ResNet50 預測：**{resnet_label}** ({resnet_pred:.2%})")
        st.markdown(f"### 🔬 自訂模型預測：**{custom_label}** ({custom_pred:.2%})")

    elif "video" in file_type:
        st.video(file)
        st.info("🎞 正在分析影片，請稍候...")
        resnet_preds, custom_preds = process_video(file)

        if resnet_preds and custom_preds:
            avg_resnet = np.mean(resnet_preds)
            avg_custom = np.mean(custom_preds)

            resnet_label = "Deepfake" if avg_resnet > 0.5 else "Real"
            custom_label = "Deepfake" if avg_custom > 0.5 else "Real"

            st.markdown(f"### 📽 ResNet50 平均預測：**{resnet_label}** ({avg_resnet:.2%})")
            st.markdown(f"### 📽 自訂模型平均預測：**{custom_label}** ({avg_custom:.2%})")
        else:
            st.warning("⚠️ 未能從影片中偵測到人臉。")
