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

# ⬝ 下載自訂 CNN 模型
@st.cache_resource
def download_model():
    model_url = "https://huggingface.co/wuwuwu123123/deepfakemodel2/resolve/main/deepfake_cnn_model.h5"
    model_filename = "deepfake_cnn_model.h5"
    if not os.path.exists(model_filename):
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(model_filename, "wb") as f:
                f.write(response.content)
        else:
            return None
    return model_filename

# ⬝ 加載 ResNet50 和自訂 CNN 模型
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
resnet_classifier = Sequential([
    resnet_model,
    Dense(1, activation='sigmoid')
])
resnet_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_path = download_model()
if model_path:
    try:
        custom_model = load_model(model_path)
    except Exception as e:
        custom_model = None
else:
    custom_model = None

# ⬝ 初始化 MTCNN
face_detector = MTCNN()

# ⬝ 中心補充 + 補充 CLAHE 與鋤化
@st.cache_data
def preprocess_for_both_models(img):
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    width, height = img.size
    left = (width - 224) // 2
    top = (height - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))

    img_array = np.array(img)

    # CLAHE + sharpen for custom CNN
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    img_array_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # sharpen
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img_array_clahe = cv2.filter2D(img_array_clahe, -1, sharpen_kernel)

    # preprocess input
    resnet_input = preprocess_input(np.expand_dims(img_array, axis=0))
    custom_input = np.expand_dims(img_array_clahe / 255.0, axis=0)

    return resnet_input, custom_input

# ⬝ 人臉擲取
@st.cache_data
def extract_face(pil_img):
    img_array = np.array(pil_img)
    results = face_detector.detect_faces(img_array)
    if results:
        x, y, w, h = results[0]['box']
        x, y = max(0, x), max(0, y)
        face = img_array[y:y+h, x:x+w]
        return Image.fromarray(face)
    return None

# ⬝ 預測
@st.cache_data
def predict_with_both_models(img):
    resnet_input, custom_input = preprocess_for_both_models(img)
    resnet_pred = resnet_classifier.predict(resnet_input)[0][0]
    custom_pred = custom_model.predict(custom_input)[0][0] if custom_model else 0
    return (
        ("Deepfake" if resnet_pred > 0.5 else "Real", resnet_pred),
        ("Deepfake" if custom_pred > 0.5 else "Real", custom_pred)
    )

# ⬝ 顯示預測結果
@st.cache_data
def show_prediction(img):
    (resnet_label, resnet_conf), (custom_label, custom_conf) = predict_with_both_models(img)
    st.image(img, caption="預測圖片", use_container_width=True)
    st.subheader(f"ResNet50: {resnet_label} ({resnet_conf:.2%})")
    st.subheader(f"Custom CNN: {custom_label} ({custom_conf:.2%})")

# ⬝ Streamlit App
st.set_page_config(page_title="Deepfake 偵測器", layout="wide")
st.title(":brain: Deepfake 圖片與影片偵測")

tab1, tab2 = st.tabs(["🖼️ 圖片偵測", "🎥 影片偵測"])

# 圖片偵測
with tab1:
    st.header("圖片偵測")
    uploaded_image = st.file_uploader("上傳圖片", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        pil_img = Image.open(uploaded_image).convert("RGB")
        st.image(pil_img, caption="原始圖片", use_container_width=True)

        if pil_img.width > 1000 or pil_img.height > 1000:
            st.warning("⚠️ 圖片解析度過高。ResNet50 偵測结果可能不出色，建議使用人臉區域。")

        face_img = extract_face(pil_img)
        if face_img:
            st.image(face_img, caption="擲取人臉", use_container_width=False, width=300)
            show_prediction(face_img)
        else:
            st.write("未偵測到人臉，使用整張圖片預測")
            show_prediction(pil_img)

# 影片偵測
with tab2:
    st.header("影片偵測 (只顯示第一張預測)")
    uploaded_video = st.file_uploader("上傳影片", type=["mp4", "mov", "avi"])
    if uploaded_video:
        st.video(uploaded_video)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            video_path = tmp.name

        st.info("🎮 擷取影片帶預測...")
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
                    st.image(face_img, caption="擲取人臉", use_container_width=False, width=300)
                    show_prediction(face_img)
                    break
            frame_idx += 1
        cap.release()
