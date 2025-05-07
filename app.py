import numpy as np
import streamlit as st
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from PIL import Image
from mtcnn import MTCNN
import tempfile
import os
import requests

# 🔹 頁面設定需放最上面
st.set_page_config(page_title="Deepfake 偵測器", layout="wide")
st.title("🧠 Deepfake 圖片與影片偵測器")

# 🔹 下載自訂 CNN 模型
def download_model():
    model_url = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/deepfake_cnn_model.h5"
    model_filename = "deepfake_cnn_model.h5"
    if not os.path.exists(model_filename):
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(model_filename, "wb") as f:
                f.write(response.content)
            print("模型已下載")
        else:
            print(f"模型下載失敗：{response.status_code}")
            return None
    return model_filename

# 🔹 載入 ResNet50 模型
resnet_base = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
resnet_classifier = Sequential([
    resnet_base,
    Dense(1, activation='sigmoid')
])
resnet_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 🔹 載入自訂 CNN 模型
model_path = download_model()
custom_model = load_model(model_path) if model_path else None

# 🔹 初始化人臉偵測器
detector = MTCNN()

# 🔹 擷取臉部區域
def extract_face(pil_img):
    img_array = np.array(pil_img)
    results = detector.detect_faces(img_array)
    if results:
        x, y, w, h = results[0]['box']
        face = img_array[y:y+h, x:x+w]
        face_pil = Image.fromarray(face).resize((224, 224))
        return face_pil
    return None

# 🔹 中心裁切
def center_crop(img, target_size=(224, 224)):
    width, height = img.size
    new_w, new_h = target_size
    left = (width - new_w) // 2
    top = (height - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))

# 🔹 預處理圖片
def preprocess_for_both_models(img):
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    img = center_crop(img, (224, 224))
    img_array = np.array(img)

    # 加上 Gaussian Blur（雖然讓圖片變藍，但偵測更準）
    img_array = cv2.GaussianBlur(img_array, (3, 3), 0)

    resnet_input = preprocess_input(np.expand_dims(img_array, axis=0))
    custom_input = np.expand_dims(img_array / 255.0, axis=0)

    return resnet_input, custom_input

# 🔹 預測
def predict_with_both_models(img):
    resnet_input, custom_input = preprocess_for_both_models(img)
    resnet_pred = resnet_classifier.predict(resnet_input)[0][0]
    resnet_label = "Deepfake" if resnet_pred > 0.5 else "Real"
    custom_pred = custom_model.predict(custom_input)[0][0] if custom_model else 0
    custom_label = "Deepfake" if custom_pred > 0.5 else "Real"
    return resnet_label, resnet_pred, custom_label, custom_pred

# 🔹 顯示預測
def show_prediction(img):
    resnet_label, resnet_conf, custom_label, custom_conf = predict_with_both_models(img)
    st.subheader(f"ResNet50：{resnet_label}（{resnet_conf:.2%}）")
    st.subheader(f"Custom CNN：{custom_label}（{custom_conf:.2%}）")

# 🔹 介面區塊
tab1, tab2 = st.tabs(["🖼️ 圖片偵測", "🎥 影片偵測"])

# ---------- 圖片 ----------
with tab1:
    st.header("圖片偵測")
    uploaded_image = st.file_uploader("上傳圖片", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        pil_img = Image.open(uploaded_image).convert("RGB")
        st.image(pil_img, caption="原始圖片", use_container_width=True)

        face_img = extract_face(pil_img)
        if face_img:
            st.image(face_img, caption="偵測到的人臉", width=300)
            show_prediction(face_img)
        else:
            st.info("未偵測到人臉，使用整張圖片預測")
            show_prediction(pil_img)

# ---------- 影片 ----------
with tab2:
    st.header("影片偵測（每 10 幀取 1 張）")
    uploaded_video = st.file_uploader("上傳影片", type=["mp4", "mov", "avi"])
    if uploaded_video:
        st.video(uploaded_video)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            video_path = tmp.name

        st.info("影片處理中，僅顯示第一個成功分析的幀")
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        found = False

        while cap.isOpened() and not found:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 10 == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                face_img = extract_face(frame_pil)
                if face_img:
                    st.image(face_img, caption=f"第 {frame_idx} 幀偵測到人臉", width=300)
                    show_prediction(face_img)
                    found = True
            frame_idx += 1
        cap.release()

        if not found:
            st.warning("影片中未偵測到可用人臉")
