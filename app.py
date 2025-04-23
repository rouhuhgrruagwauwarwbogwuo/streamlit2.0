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

# 🔹 ResNet50 模型建立
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(256, 256, 3))
resnet_classifier = Sequential([
    resnet_model,
    Dense(1, activation='sigmoid')
])
resnet_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 🔹 預處理函數 for both models
def preprocess_for_models(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256))

    # For ResNet50
    resnet_input = preprocess_input(np.expand_dims(img_resized, axis=0))

    # For Custom CNN (CLAHE gray enhancement)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    clahe_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    custom_input = np.expand_dims(clahe_rgb / 255.0, axis=0)

    return resnet_input, custom_input, img_rgb

# 🔹 預測圖片
def predict_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    resnet_input, custom_input, display_img = preprocess_for_models(img)

    # 預測
    resnet_pred = resnet_classifier.predict(resnet_input)[0][0]
    custom_pred = custom_model.predict(custom_input)[0][0]

    resnet_label = "Deepfake" if resnet_pred > 0.5 else "Real"
    custom_label = "Deepfake" if custom_pred > 0.5 else "Real"
    resnet_conf = resnet_pred if resnet_pred > 0.5 else 1 - resnet_pred
    custom_conf = custom_pred if custom_pred > 0.5 else 1 - custom_pred

    # 顯示圖片與結果
    st.image(display_img, caption="你上傳的圖片", use_container_width=True)
    st.markdown(f"### 🤖 ResNet50 預測: **{resnet_label}** ({resnet_conf:.2%})")
    st.markdown(f"### 🧠 自訂 CNN 預測: **{custom_label}** ({custom_conf:.2%})")

# 🔹 預測影片
def predict_video(uploaded_file):
    cap = cv2.VideoCapture(uploaded_file)

    if not cap.isOpened():
        st.error("❌ 影片無法開啟，請確認影片格式正確。")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每隔一定的幀來進行預測
        if frame_count % 30 == 0:
            resnet_input, custom_input, display_img = preprocess_for_models(frame)

            # 預測
            resnet_pred = resnet_classifier.predict(resnet_input)[0][0]
            custom_pred = custom_model.predict(custom_input)[0][0]

            resnet_label = "Deepfake" if resnet_pred > 0.5 else "Real"
            custom_label = "Deepfake" if custom_pred > 0.5 else "Real"
            resnet_conf = resnet_pred if resnet_pred > 0.5 else 1 - resnet_pred
            custom_conf = custom_pred if custom_pred > 0.5 else 1 - custom_pred

            # 顯示幀與結果
            st.image(display_img, caption=f"幀 {frame_count}", use_container_width=True)
            st.markdown(f"### 🤖 ResNet50 預測: **{resnet_label}** ({resnet_conf:.2%})")
            st.markdown(f"### 🧠 自訂 CNN 預測: **{custom_label}** ({custom_conf:.2%})")

        frame_count += 1

    cap.release()

# 🔹 Streamlit App
st.title("🕵️ Deepfake 偵測 App")

# 上傳圖片或影片
uploaded_file = st.file_uploader("📤 上傳圖片或影片", type=["jpg", "jpeg", "png", "mp4", "mov"])
if uploaded_file is not None:
    file_type = uploaded_file.type.split('/')[0]
    
    if file_type == "image":
        predict_image(uploaded_file)
    elif file_type == "video":
        predict_video(uploaded_file)
    else:
        st.error("❌ 無效的文件格式。請上傳圖片或影片。")
