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
resnet_base = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(256, 256, 3))
resnet_classifier = Sequential([
    resnet_base,
    Dense(1, activation='sigmoid')
])
resnet_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 🔹 預處理函數 for both models
def preprocess_for_models(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256))
    resnet_input = preprocess_input(np.expand_dims(img_resized, axis=0))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    clahe_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    custom_input = np.expand_dims(clahe_rgb / 255.0, axis=0)
    return resnet_input, custom_input, img_rgb

# 🔹 偵測影片函數
def process_video(video_file):
    st.video(video_file)
    temp_video = tempfile.NamedTemporaryFile(delete=False)
    temp_video.write(video_file.read())
    temp_video.close()
    cap = cv2.VideoCapture(temp_video.name)

    frame_count = 0
    resnet_scores, custom_scores = [], []

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
        if frame_count % 30 != 0:
            continue
        if frame is None or frame.size == 0:
            continue

        try:
            resnet_input, custom_input, _ = preprocess_for_models(frame)
            resnet_pred = resnet_classifier.predict(resnet_input)
            custom_pred = custom_model.predict(custom_input)

            if resnet_pred.shape[-1] == 1:
                resnet_score = resnet_pred[0][0]
            else:
                resnet_score = resnet_pred[0]

            if custom_pred.shape[-1] == 1:
                custom_score = custom_pred[0][0]
            else:
                custom_score = custom_pred[0]

            resnet_scores.append(resnet_score)
            custom_scores.append(custom_score)
        except Exception as e:
            st.warning(f"⚠️ 第 {frame_count} 幀處理失敗: {e}")
            continue

    cap.release()

    if len(resnet_scores) == 0 or len(custom_scores) == 0:
        st.error("❌ 無有效幀可分析")
        return

    avg_resnet = np.mean(resnet_scores)
    avg_custom = np.mean(custom_scores)

    label_resnet = "Deepfake" if avg_resnet > 0.5 else "Real"
    label_custom = "Deepfake" if avg_custom > 0.5 else "Real"

    st.markdown("### 📊 模型預測結果")
    st.markdown(f"- ResNet50: **{label_resnet}** ({avg_resnet:.2%})")
    st.markdown(f"- Custom CNN: **{label_custom}** ({avg_custom:.2%})")

# 🔹 Streamlit App
st.title("🕵️ Deepfake 偵測 App")

option = st.radio("請選擇要上傳的媒體類型:", ("圖片", "影片"))

if option == "圖片":
    uploaded_file = st.file_uploader("📤 上傳一張圖片", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        resnet_input, custom_input, display_img = preprocess_for_models(img)

        resnet_pred = resnet_classifier.predict(resnet_input)
        custom_pred = custom_model.predict(custom_input)

        resnet_score = resnet_pred[0][0] if resnet_pred.shape[-1] == 1 else resnet_pred[0]
        custom_score = custom_pred[0][0] if custom_pred.shape[-1] == 1 else custom_pred[0]

        combined_pred = (resnet_score + custom_score) / 2
        label = "Deepfake" if combined_pred > 0.5 else "Real"
        confidence = combined_pred if combined_pred > 0.5 else 1 - combined_pred

        st.image(display_img, caption="你上傳的圖片", use_container_width=True)
        st.markdown(f"### 🧑‍⚖️ 最終預測結果: **{label}** ({confidence:.2%})")
        st.markdown(f"- ResNet50: {resnet_score:.2%}\n- Custom CNN: {custom_score:.2%}")

elif option == "影片":
    uploaded_video = st.file_uploader("📤 上傳影片 (建議 .mp4)", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        process_video(uploaded_video)
