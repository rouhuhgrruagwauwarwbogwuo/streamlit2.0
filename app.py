import streamlit as st
import numpy as np
import cv2
import tempfile
import requests
import os
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# ==== 模型載入（下載一次）====
@st.cache_resource
def load_models():
    # 載入 ResNet50 並加上分類層
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(256, 256, 3))
    resnet_classifier = Sequential([
        base_model,
        Dense(1, activation='sigmoid')
    ])
    resnet_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 下載 Custom CNN 模型
    custom_model_url = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/deepfake_cnn_model.h5"
    custom_model_path = "deepfake_cnn_model.h5"
    if not os.path.exists(custom_model_path):
        with st.spinner("下載 Custom CNN 模型中..."):
            r = requests.get(custom_model_url)
            with open(custom_model_path, "wb") as f:
                f.write(r.content)
    custom_model = load_model(custom_model_path)

    return resnet_classifier, custom_model

# ==== 圖像預處理 ====
def preprocess_for_both_models(pil_img):
    img = pil_img.resize((256, 256))
    img_array = image.img_to_array(img)
    resnet_input = preprocess_input(np.expand_dims(img_array, axis=0))
    custom_input = np.expand_dims(img_array / 255.0, axis=0)
    return resnet_input, custom_input

# ==== 單張圖片預測 ====
def predict_with_both_models(pil_img):
    resnet_input, custom_input = preprocess_for_both_models(pil_img)
    resnet_pred = resnet_classifier.predict(resnet_input, verbose=0)[0][0]
    custom_pred = custom_model.predict(custom_input, verbose=0)[0][0]
    resnet_label = "Deepfake" if resnet_pred > 0.5 else "Real"
    custom_label = "Deepfake" if custom_pred > 0.5 else "Real"
    return resnet_label, resnet_pred, custom_label, custom_pred

# ==== 抽取影片幀 ====
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

# ==== 多幀預測 ====
def predict_frames(frames):
    results = []
    for idx, frame in frames:
        label, score, _, _ = predict_with_both_models(frame)
        confidence = score if label == "Deepfake" else 1 - score
        results.append((idx, frame, label, confidence))
    return results

# ==== 圖表視覺化 ====
def plot_confidence_bar(score):
    fig, ax = plt.subplots(figsize=(5, 1.5))
    ax.barh(['Confidence'], [score], color='red' if score > 0.5 else 'green')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Deepfake 機率')
    ax.set_title('ResNet50 預測信心')
    st.pyplot(fig)

# ==== App 主體 ====
st.set_page_config(page_title="Deepfake 偵測器", layout="wide")
st.title("🧠 Deepfake 圖片與影片偵測器")

# 載入模型
resnet_classifier, custom_model = load_models()

# Tab 分頁
tab1, tab2 = st.tabs(["🖼️ 圖片偵測", "🎥 影片偵測"])

# ---------- 圖片 ----------
with tab1:
    st.header("圖片偵測")
    uploaded_image = st.file_uploader("上傳圖片", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        pil_img = Image.open(uploaded_image).convert("RGB")
        st.image(pil_img, caption="原始圖片", use_container_width=True)

        resnet_label, resnet_score, custom_label, custom_score = predict_with_both_models(pil_img)

        st.subheader(f"✅ ResNet50 預測結果：{resnet_label} ({resnet_score:.2%})")
        plot_confidence_bar(resnet_score)
        st.caption(f"📌 Custom CNN 預測僅供參考：{custom_label} ({custom_score:.2%})")

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
        results = predict_frames(frames)

        for idx, frame, label, confidence in results:
            st.image(frame, caption=f"第 {idx} 幀 - {label} ({confidence:.2%})", use_container_width=True)
