import os
import numpy as np
import cv2
import tempfile
import requests
import h5py
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense
from PIL import Image

# 🔹 Hugging Face 模型下載網址
MODEL_URL = "https://huggingface.co/wuwuwu123123/deepfake/resolve/main/deepfake_cnn_model.h5"

@st.cache_resource
def download_model():
    model_path = os.path.join(tempfile.gettempdir(), "deepfake_cnn_model.h5")
    if not os.path.exists(model_path):
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                f.write(response.content)
        else:
            st.error("❌ 模型下載失敗，請確認 Hugging Face 模型網址是否正確。")
            raise Exception("模型下載失敗。")

    try:
        with h5py.File(model_path, 'r') as f:
            pass
    except OSError as e:
        st.error("❌ 模型檔案無法讀取，可能是損壞或格式錯誤。")
        raise

    return load_model(model_path)

# 🔹 載入模型
try:
    custom_model = download_model()
except Exception:
    st.stop()

# 🔹 ResNet50 模型建立（改為 224x224）
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
resnet_classifier = Sequential([
    resnet_model,
    Dense(1, activation='sigmoid')
])
resnet_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 🔹 預處理函數

def preprocess_for_models(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))

    resnet_input = preprocess_input(np.expand_dims(img_resized, axis=0))

    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    clahe_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    custom_input = np.expand_dims(clahe_rgb / 255.0, axis=0)

    return resnet_input, custom_input, img_rgb

# 🔹 處理影片函數

def process_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    frame_interval = 5
    frame_count = 0
    resnet_preds, custom_preds = [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face = frame[y:y + h, x:x + w]
                resnet_input, custom_input, _ = preprocess_for_models(face)

                resnet_pred = resnet_classifier.predict(resnet_input, verbose=0)[0][0]
                custom_pred = custom_model.predict(custom_input, verbose=0)[0][0]

                resnet_preds.append(resnet_pred)
                custom_preds.append(custom_pred)

        frame_count += 1

    cap.release()
    avg_resnet = np.mean(resnet_preds) if resnet_preds else 0
    avg_custom = np.mean(custom_preds) if custom_preds else 0

    return avg_resnet, avg_custom

# 🔹 Streamlit App

st.title("🕵️ Deepfake 偵測 App")
mode = st.radio("請選擇上傳模式：", ["圖片", "影片"])

if mode == "圖片":
    uploaded_file = st.file_uploader("📤 上傳一張圖片", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        resnet_input, custom_input, display_img = preprocess_for_models(img)
        resnet_pred = resnet_classifier.predict(resnet_input, verbose=0)[0][0]
        custom_pred = custom_model.predict(custom_input, verbose=0)[0][0]

        st.image(display_img, caption="你上傳的圖片", use_container_width=True)
        st.markdown(f"#### ResNet50 預測: {'Deepfake' if resnet_pred > 0.5 else 'Real'} ({resnet_pred:.2%})")
        st.markdown(f"#### Custom CNN 預測: {'Deepfake' if custom_pred > 0.5 else 'Real'} ({custom_pred:.2%})")

elif mode == "影片":
    uploaded_video = st.file_uploader("📼 上傳影片檔", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        with st.spinner("正在處理影片，請稍候..."):
            avg_resnet, avg_custom = process_video(uploaded_video)

        st.markdown(f"#### 🎬 ResNet50 平均預測: {'Deepfake' if avg_resnet > 0.5 else 'Real'} ({avg_resnet:.2%})")
        st.markdown(f"#### 🎬 Custom CNN 平均預測: {'Deepfake' if avg_custom > 0.5 else 'Real'} ({avg_custom:.2%})")
