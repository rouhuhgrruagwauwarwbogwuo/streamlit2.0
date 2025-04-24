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
except Exception as e:
    st.error(f"❌ 模型載入失敗: {e}")
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
    img_resized = cv2.resize(img, (256, 256))

    # For ResNet50
    resnet_input = preprocess_input(np.expand_dims(img_resized, axis=0))

    # For Custom CNN (CLAHE gray enhancement)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    clahe_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    custom_input = np.expand_dims(clahe_rgb / 255.0, axis=0)

    return resnet_input, custom_input, img_resized

# 🔹 偵測影片並生成新影片
def process_video_and_generate_result(video_file):
    temp_video_path = os.path.join(tempfile.gettempdir(), "temp_video.mp4")
    with open(temp_video_path, "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture(temp_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video_path = os.path.join(tempfile.gettempdir(), "processed_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            resnet_input, custom_input, _ = preprocess_for_models(frame)
            resnet_pred = resnet_classifier.predict(resnet_input)[0][0]
            custom_pred = custom_model.predict(custom_input)[0][0]
            combined_pred = (resnet_pred + custom_pred) / 2
            label = "Deepfake" if combined_pred > 0.5 else "Real"
            confidence = combined_pred if combined_pred > 0.5 else 1 - combined_pred

            cv2.putText(frame, f"{label} ({confidence:.2%})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            out.write(frame)
        except Exception as e:
            st.error(f"❌ 處理影像幀時發生錯誤: {e}")
            break

    cap.release()
    out.release()

    return output_video_path

# 🔹 Streamlit App 主介面
st.title("🕵️ Deepfake 偵測 App")

file_type = st.radio("選擇要分析的檔案類型：", ["圖片", "影片"])
uploaded_file = st.file_uploader("📤 請上傳檔案", type=["jpg", "jpeg", "png", "mp4", "mov"])

if uploaded_file is not None:
    try:
        if file_type == "圖片" and uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
            st.markdown("### 🖼️ 圖片偵測結果")
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(img, caption="你上傳的圖片", use_container_width=True)

            resnet_input, custom_input, _ = preprocess_for_models(img)
            resnet_pred = resnet_classifier.predict(resnet_input)[0][0]
            custom_pred = custom_model.predict(custom_input)[0][0]
            combined_pred = (resnet_pred + custom_pred) / 2
            label = "Deepfake" if combined_pred > 0.5 else "Real"
            confidence = combined_pred if combined_pred > 0.5 else 1 - combined_pred

            st.markdown(f"### 🧑‍⚖️ 最終預測結果: **{label}** ({confidence:.2%})")

        elif file_type == "影片" and uploaded_file.type in ["video/mp4", "video/quicktime"]:
            st.markdown("### 📽️ 影片偵測中...")
            processed_video_path = process_video_and_generate_result(uploaded_file)
            st.video(processed_video_path)

        else:
            st.warning("請上傳符合所選類型的檔案。")

    except Exception as e:
        st.error(f"❌ 發生錯誤: {e}")
        st.write(f"錯誤詳情: {str(e)}")
