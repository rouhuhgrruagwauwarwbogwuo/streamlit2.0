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
import h5py

# 檢查並下載模型檔案
def download_model():
    model_url = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/deepfake_cnn_model.h5"
    model_filename = "deepfake_cnn_model.h5"
    
    if not os.path.exists(model_filename):
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(model_filename, "wb") as f:
                f.write(response.content)
            print("模型檔案已成功下載！")
        else:
            print(f"下載失敗，狀態碼：{response.status_code}")
            return None
    else:
        print(f"模型檔案 {model_filename} 已存在")
    return model_filename

# 🔹 載入 ResNet50 模型
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
resnet_classifier = Sequential([
    resnet_model,
    Dense(1, activation='sigmoid')  
])
resnet_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 🔹 載入自訂 CNN 模型
model_path = download_model()
if model_path:
    try:
        custom_model = load_model(model_path)
        print("自訂 CNN 模型已成功載入")
    except Exception as e:
        print(f"載入自訂 CNN 模型時發生錯誤：{e}")
        custom_model = None
else:
    custom_model = None

# 🔹 初始化 MTCNN 人臉檢測器
detector = MTCNN()

# 🔹 中心裁切函數 - 避免高清圖片影響 ResNet50 預測
def center_crop(img, target_size=(224, 224)):
    width, height = img.size
    new_width, new_height = target_size
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    return img.crop((left, top, right, bottom))

# 🔹 預處理圖片，確保 ResNet 和 自訂 CNN 都能處理
def preprocess_for_both_models(img):
    # 1️⃣ **高清圖處理：LANCZOS 縮圖**
    img = img.resize((256, 256), Image.Resampling.LANCZOS)

    # 2️⃣ **ResNet50 必須 224x224**
    img = center_crop(img, (224, 224))

    img_array = np.array(img)  # 轉為 numpy array

    # 3️⃣ **可選：對 ResNet50 做 Gaussian Blur**
    apply_blur = True  # 🚀 這裡可以開關
    if apply_blur:
        img_array = cv2.GaussianBlur(img_array, (3, 3), 0)

    # 4️⃣ **ResNet50 特定預處理**
    resnet_input = preprocess_input(np.expand_dims(img_array, axis=0))

    # 5️⃣ **自訂 CNN 正規化 (0~1)**
    custom_input = np.expand_dims(img_array / 255.0, axis=0)

    return resnet_input, custom_input

# 🔹 進行預測
def predict_with_both_models(img):
    resnet_input, custom_input = preprocess_for_both_models(img)
    
    # ResNet50 預測
    resnet_prediction = resnet_classifier.predict(resnet_input)[0][0]
    resnet_label = "Deepfake" if resnet_prediction > 0.5 else "Real"
    
    # 自訂 CNN 預測
    custom_prediction = custom_model.predict(custom_input)[0][0] if custom_model else 0
    custom_label = "Deepfake" if custom_prediction > 0.5 else "Real"
    
    return resnet_label, resnet_prediction, custom_label, custom_prediction

# 🔹 顯示圖片和預測結果
def show_prediction(img):
    resnet_label, resnet_confidence, custom_label, custom_confidence = predict_with_both_models(img)
    
    # 顯示未經處理的圖片
    st.image(img, caption="原始圖片", use_container_width=True)
    
    # 顯示偵測到的人臉並縮小圖片
    st.image(img, caption="偵測到的人臉", use_container_width=False, width=300)
    
    # 顯示預測結果
    st.subheader(f"ResNet50: {resnet_label} ({resnet_confidence:.2%})\n"
                 f"Custom CNN: {custom_label} ({custom_confidence:.2%})")

# 🔹 擷取人臉函數
def extract_face(img):
    # 使用 MTCNN 偵測人臉
    faces = detector.detect_faces(np.array(img))
    
    if len(faces) > 0:
        # 假設我們只取第一個偵測到的人臉
        x, y, w, h = faces[0]['box']
        face_img = img.crop((x, y, x + w, y + h))  # 擷取人臉區域
        return face_img
    return None

# 🔹 Streamlit 主應用程式
st.set_page_config(page_title="Deepfake 偵測器", layout="wide")
st.title("🧠 Deepfake 圖片與影片偵測器")

tab1, tab2 = st.tabs(["🖼️ 圖片偵測", "🎥 影片偵測"])

# ---------- 圖片 ---------- 
with tab1:
    st.header("圖片偵測")
    uploaded_image = st.file_uploader("上傳圖片", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        pil_img = Image.open(uploaded_image).convert("RGB")
        st.image(pil_img, caption="原始圖片", use_container_width=True)

        # 嘗試擷取人臉區域
        face_img = extract_face(pil_img)
        if face_img:
            st.image(face_img, caption="偵測到的人臉", use_container_width=False, width=300)
            show_prediction(face_img)  
        else:
            st.write("未偵測到人臉，使用整體圖片進行預測")
            show_prediction(pil_img)

# ---------- 影片 ---------- 
with tab2:
    st.header("影片偵測（只顯示第一張預測結果）")
    uploaded_video = st.file_uploader("上傳影片", type=["mp4", "mov", "avi"])
    if uploaded_video:
        st.video(uploaded_video)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            video_path = tmp.name

        st.info("🎬 擷取影片幀與進行預測中...")
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 10 == 0:
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                face_img = extract_face(frame_pil)
                if face_img:
                    st.image(face_img, caption="偵測到的人臉", use_container_width=False, width=300)
                    show_prediction(face_img)
                    break  
            frame_idx += 1
        cap.release()
