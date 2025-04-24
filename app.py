import os
import numpy as np
import cv2
import tempfile
import requests
import h5py
import streamlit as st
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense

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
    except OSError:
        st.error("❌ 模型檔案無法讀取，可能是損壞或格式錯誤。")
        raise
    return load_model(model_path)

# 載入模型
try:
    custom_model = download_model()
except Exception as e:
    st.error(f"❌ 模型載入失敗: {e}")
    st.stop()

resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(256, 256, 3))
resnet_classifier = Sequential([
    resnet_model,
    Dense(1, activation='sigmoid')
])
resnet_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 🔧 高通濾波 + CLAHE + 銳化 + 人臉擷取 + 色彩空間轉換

def extract_face(img):
    detector = MTCNN()
    result = detector.detect_faces(img)
    if result:
        x, y, w, h = result[0]['box']
        face = img[y:y+h, x:x+w]
        return cv2.resize(face, (256, 256))
    return cv2.resize(img, (256, 256))

def high_pass_filter(img):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)

def enhance_image(img):
    img = high_pass_filter(img)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    img_sharp = cv2.filter2D(img_eq, -1, kernel)
    return img_sharp

def preprocess_for_models(img):
    face = extract_face(img)
    enhanced = enhance_image(face)
    resnet_input = preprocess_input(np.expand_dims(enhanced, axis=0))
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    clahe_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
    custom_input = np.expand_dims(clahe_rgb / 255.0, axis=0)
    return resnet_input, custom_input, face

def smooth_predictions(pred_list, window_size=5):
    if len(pred_list) < window_size:
        return pred_list
    return np.convolve(pred_list, np.ones(window_size)/window_size, mode='valid')

def plot_confidence(resnet_conf, custom_conf, combined_conf):
    fig, ax = plt.subplots()
    models = ['ResNet50', 'Custom CNN', 'Combined']
    confs = [resnet_conf, custom_conf, combined_conf]
    ax.bar(models, confs, color=['blue', 'green', 'purple'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Confidence')
    st.pyplot(fig)

def process_image(file_bytes):
    try:
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        resnet_input, custom_input, display_img = preprocess_for_models(img)
        resnet_pred = resnet_classifier.predict(resnet_input)[0][0]
        custom_pred = custom_model.predict(custom_input)[0][0]
        combined_pred = (resnet_pred + custom_pred) / 2
        label = "Deepfake" if combined_pred > 0.5 else "Real"
        confidence = combined_pred if combined_pred > 0.5 else 1 - combined_pred
        st.image(display_img, caption=f"預測結果：{label} ({confidence:.2%})", use_container_width=True)
        plot_confidence(resnet_pred, custom_pred, combined_pred)
    except Exception as e:
        st.error(f"❌ 圖片處理錯誤: {e}")

def process_video_and_generate_result(video_file):
    try:
        temp_video_path = os.path.join(tempfile.gettempdir(), "temp_video.mp4")
        with open(temp_video_path, "wb") as f:
            f.write(video_file.read())
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            st.error("❌ 無法打開影片檔案。")
            return
        frame_preds = []
        frames = []
        frame_interval = 10
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                resnet_input, custom_input, face = preprocess_for_models(frame)
                resnet_pred = resnet_classifier.predict(resnet_input)[0][0]
                custom_pred = custom_model.predict(custom_input)[0][0]
                combined_pred = (resnet_pred + custom_pred) / 2
                label = "Deepfake" if combined_pred > 0.5 else "Real"
                confidence = combined_pred if combined_pred > 0.5 else 1 - combined_pred
                cv2.putText(face, f"{label} ({confidence:.2%})", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                frames.append(face[:, :, ::-1])  # BGR to RGB
                frame_preds.append(combined_pred)
            frame_count += 1
        cap.release()
        st.success("🎉 偵測完成！")
        smoothed = smooth_predictions(frame_preds)
        st.line_chart(smoothed)
        for f in frames:
            st.image(f, use_container_width=True)
    except Exception as e:
        st.error(f"❌ 影片處理錯誤: {e}")

st.title("🕵️ Deepfake 偵測 App")
option = st.radio("請選擇檔案類型：", ("圖片", "影片"))

uploaded_file = st.file_uploader("📤 上傳檔案", type=["jpg", "jpeg", "png", "mp4", "mov"])

if uploaded_file is not None:
    try:
        if option == "圖片" and uploaded_file.type.startswith("image"):
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            process_image(file_bytes)
        elif option == "影片" and uploaded_file.type.startswith("video"):
            st.markdown("### 處理影片中...")
            process_video_and_generate_result(uploaded_file)
        else:
            st.warning("請確認上傳的檔案類型與選擇一致。")
    except Exception as e:
        st.error(f"❌ 發生錯誤: {e}")
