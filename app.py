import os
import tempfile
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense

# 🔹 模型下載 (假設已經下載)
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

    return load_model(model_path)

# 🔹 載入自訂模型
custom_model = download_model()

# 🔹 ResNet50 模型建立
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(256, 256, 3))
resnet_classifier = Sequential([
    resnet_model,
    Dense(1, activation='sigmoid')
])
resnet_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 🔹 預處理函數
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
    
    # 檢查影片是否能夠打開
    if not cap.isOpened():
        st.error("❌ 無法打開影片檔案！")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 設定影片編碼
    output_path = os.path.join(tempfile.gettempdir(), "processed_video.mp4")
    
    # 設定影片輸出格式
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))  # 20.0 是 FPS

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("❌ 讀取影片幀時出現錯誤或影片已結束。")
            break
        
        # 預處理每一幀圖像
        resnet_input, custom_input, display_img = preprocess_for_models(frame)
        
        # 預測
        resnet_pred = resnet_classifier.predict(resnet_input)[0][0]
        custom_pred = custom_model.predict(custom_input)[0][0]

        # 合併結果
        resnet_label = "Deepfake" if resnet_pred > 0.5 else "Real"
        custom_label = "Deepfake" if custom_pred > 0.5 else "Real"
        resnet_confidence = resnet_pred if resnet_pred > 0.5 else 1 - resnet_pred
        custom_confidence = custom_pred if custom_pred > 0.5 else 1 - custom_pred

        # 標註顯示
        color_resnet = (0, 0, 255) if resnet_pred > 0.5 else (0, 255, 0)
        color_custom = (0, 0, 255) if custom_pred > 0.5 else (0, 255, 0)
        
        # 在每一幀上加上預測結果文字
        cv2.putText(frame, f"ResNet: {resnet_label} ({resnet_confidence:.2%})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_resnet, 2)
        cv2.putText(frame, f"CNN: {custom_label} ({custom_confidence:.2%})", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color_custom, 2)

        # 寫入每一幀
        out.write(frame)

    # 釋放資源
    cap.release()
    out.release()

    return output_path

# 🔹 Streamlit App 顯示
st.title("🕵️ Deepfake 偵測 App")

uploaded_video = st.file_uploader("📤 上傳影片", type=["mp4", "mov", "avi"])
if uploaded_video is not None:
    output_video_path = process_video_and_generate_result(uploaded_video)
    
    if output_video_path:
        # 顯示處理後的影片
        st.video(output_video_path)
