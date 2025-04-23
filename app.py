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
import io

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

# 🔹 處理影片並生成結果
def process_video_and_generate_result(uploaded_file):
    video_bytes = uploaded_file.read()
    video_path = os.path.join(tempfile.gettempdir(), "uploaded_video.mp4")
    
    # 儲存影片至臨時檔案
    with open(video_path, "wb") as f:
        f.write(video_bytes)

    # 打開影片進行處理
    cap = cv2.VideoCapture(video_path)
    output_path = os.path.join(tempfile.gettempdir(), "processed_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 設定影片編碼
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))  # 設定輸出影片參數
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        resnet_input, custom_input, display_img = preprocess_for_models(frame)
        
        # 預測
        resnet_pred = resnet_classifier.predict(resnet_input)[0][0]
        custom_pred = custom_model.predict(custom_input)[0][0]

        # 合併結果
        combined_pred = (resnet_pred + custom_pred) / 2
        label = "Deepfake" if combined_pred > 0.5 else "Real"
        confidence = combined_pred if combined_pred > 0.5 else 1 - combined_pred

        # 繪製標籤到影像
        color = (0, 0, 255) if combined_pred > 0.5 else (0, 255, 0)
        cv2.putText(frame, f"{label} ({confidence:.2%})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 寫入影像
        out.write(frame)

    cap.release()
    out.release()

    return output_path

# 🔹 Streamlit App
st.title("🕵️ Deepfake 偵測 App")

uploaded_file = st.file_uploader("📤 上傳一張圖片或影片", type=["jpg", "jpeg", "png", "mp4", "mov"])
if uploaded_file is not None:
    # 若上傳的是圖片
    if uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)  # 解碼為圖片

        # 進行預處理並獲得模型輸入
        resnet_input, custom_input, display_img = preprocess_for_models(img)

        # 預測
        resnet_pred = resnet_classifier.predict(resnet_input)[0][0]
        custom_pred = custom_model.predict(custom_input)[0][0]

        # 合併結果
        combined_pred = (resnet_pred + custom_pred) / 2  # 這裡簡單取平均
        label = "Deepfake" if combined_pred > 0.5 else "Real"
        confidence = combined_pred if combined_pred > 0.5 else 1 - combined_pred

        # 顯示圖片與結果
        st.image(display_img, caption="你上傳的圖片", use_container_width=True)
        st.markdown(f"### 🧑‍⚖️ 最終預測結果: **{label}** ({confidence:.2%})")

    # 若上傳的是影片
    elif uploaded_file.type in ["video/mp4", "video/quicktime"]:
        st.markdown("### 📽️ 正在處理影片...")
        processed_video_path = process_video_and_generate_result(uploaded_file)

        # 顯示處理後的影片
        st.video(processed_video_path)
