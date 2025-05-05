import streamlit as st
import numpy as np
import cv2
import tempfile
import requests
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 下載模型（只會下載一次）
@st.cache_resource
def download_model():
    model_url = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/deepfake_cnn_model.h5"
    model_path = "deepfake_cnn_model.h5"
    if not os.path.exists(model_path):
        with st.spinner("下載模型中..."):
            r = requests.get(model_url)
            with open(model_path, "wb") as f:
                f.write(r.content)
    return load_model(model_path)

# 圖像預處理（簡化）
def preprocess_pil_image(pil_img, target_size=(256, 256)):
    img = pil_img.resize(target_size)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# 單張圖片預測
def predict(model, pil_img):
    img_array = preprocess_pil_image(pil_img)
    pred = model.predict(img_array)[0][0]
    label = "Deepfake" if pred > 0.5 else "Real"
    confidence = pred if pred > 0.5 else 1 - pred
    return label, confidence

# 從影片擷取幀（每 10 幀）
def extract_frames(video_path, interval=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            frames.append((frame_idx, pil_img))
        frame_idx += 1
    cap.release()
    return frames

# 幀圖預測
def predict_frames(model, frames):
    results = []
    for idx, frame in frames:
        label, confidence = predict(model, frame)
        results.append((idx, frame, label, confidence))
    return results

# ==== Streamlit App 主體 ====
st.set_page_config(page_title="Deepfake 偵測器", layout="wide")
st.title("🧠 Deepfake 圖片與影片偵測器")

model = download_model()

tab1, tab2 = st.tabs(["🖼️ 圖片偵測", "🎥 影片偵測"])

# ---------- 圖片 ----------
with tab1:
    st.header("圖片偵測")
    uploaded_image = st.file_uploader("上傳圖片", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        pil_img = Image.open(uploaded_image).convert("RGB")
        st.image(pil_img, caption="原始圖片", use_column_width=True)
        label, confidence = predict(model, pil_img)
        st.subheader(f"✅ 預測結果：{label} ({confidence:.2%} 信心分數)")

# ---------- 影片 ----------
with tab2:
    st.header("影片偵測（每 10 幀抽圖）")
    uploaded_video = st.file_uploader("上傳影片", type=["mp4", "mov", "avi"])
    if uploaded_video:
        st.video(uploaded_video)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            video_path = tmp.name

        st.info("🎬 擷取影片幀與進行預測中...")
        frames = extract_frames(video_path, interval=10)
        results = predict_frames(model, frames)

        for idx, frame, label, confidence in results:
            st.image(frame, caption=f"第 {idx} 幀 - {label} ({confidence:.2%})", use_column_width=True)
