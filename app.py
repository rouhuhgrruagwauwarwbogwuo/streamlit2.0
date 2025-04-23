import os
import cv2
import time
import tempfile
import numpy as np
import requests
import h5py
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# 📦 模型下載（Custom CNN）
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

# 🔍 ResNet50 模型建立
resnet_base = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(256, 256, 3))
resnet_classifier = Sequential([
    resnet_base,
    Dense(1, activation='sigmoid')
])
resnet_classifier.compile(optimizer='adam', loss='binary_crossentropy')

# 📐 預處理
def preprocess_for_models(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256))

    # ResNet
    resnet_input = preprocess_input(np.expand_dims(img_resized, axis=0))

    # CNN
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    clahe_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    custom_input = np.expand_dims(clahe_rgb / 255.0, axis=0)

    return resnet_input, custom_input, img_rgb

# 🧠 預測
def predict_and_display(img):
    resnet_input, custom_input, display_img = preprocess_for_models(img)

    resnet_pred = resnet_classifier.predict(resnet_input)[0][0]
    custom_pred = custom_model.predict(custom_input)[0][0]

    st.image(display_img, caption="你上傳的圖片", use_container_width=True)

    st.markdown("### 🔍 預測結果")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("ResNet50", f"{'Deepfake' if resnet_pred > 0.5 else 'Real'}", f"{resnet_pred:.2%}")
    with col2:
        st.metric("Custom CNN", f"{'Deepfake' if custom_pred > 0.5 else 'Real'}", f"{custom_pred:.2%}")

# 🎥 處理影片
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ 無法打開影片")
        return

    resnet_scores = []
    custom_scores = []

    stframe = st.empty()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_count > 100:
            break

        resnet_input, custom_input, _ = preprocess_for_models(frame)
        resnet_pred = resnet_classifier.predict(resnet_input)[0][0]
        custom_pred = custom_model.predict(custom_input)[0][0]

        resnet_scores.append(resnet_pred)
        custom_scores.append(custom_pred)

        # 顯示每一幀
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, caption=f"Frame {frame_count}", channels="RGB", use_column_width=True)
        time.sleep(0.05)
        frame_count += 1

    cap.release()

    # 顯示結果
    st.markdown("### 📊 影片預測結果")
    col1, col2 = st.columns(2)
    with col1:
        avg_resnet = np.mean(resnet_scores)
        st.metric("ResNet50", f"{'Deepfake' if avg_resnet > 0.5 else 'Real'}", f"{avg_resnet:.2%}")
    with col2:
        avg_custom = np.mean(custom_scores)
        st.metric("Custom CNN", f"{'Deepfake' if avg_custom > 0.5 else 'Real'}", f"{avg_custom:.2%}")

# 🖼️ Streamlit 介面
st.title("🕵️ Deepfake 偵測 App")

st.sidebar.header("📂 選擇檔案類型")
file_type = st.sidebar.radio("請選擇你要上傳的類型", ["圖片", "影片"])

if file_type == "圖片":
    uploaded_file = st.file_uploader("📤 上傳圖片", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        predict_and_display(img)

elif file_type == "影片":
    uploaded_video = st.file_uploader("📤 上傳影片", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        st.video(tfile.name)
        process_video(tfile.name)
