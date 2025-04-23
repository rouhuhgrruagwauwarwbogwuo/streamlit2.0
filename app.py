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

    # 下載模型檔案
    if not os.path.exists(model_path):
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                f.write(response.content)
        else:
            st.error("❌ 模型下載失敗，請確認 Hugging Face 模型網址是否正確。")
            raise Exception("模型下載失敗。")

    # 檢查模型是否可讀取
    try:
        with h5py.File(model_path, 'r') as f:
            pass
    except OSError as e:
        st.error("❌ 模型檔案無法讀取，可能是損壞或格式錯誤。")
        raise

    return load_model(model_path)

# 🔹 載入自訂模型
try:
    custom_model = download_model()
except Exception:
    st.stop()

# 🔹 建立 ResNet50 模型
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

# 🔹 Streamlit App
st.title("🕵️ Deepfake 偵測 App")

# 上傳圖片或影片
uploaded_file = st.file_uploader("📤 上傳一張圖片或影片", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
if uploaded_file is not None:
    if uploaded_file.type in ["mp4", "avi", "mov"]:
        # 影片處理
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        video = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.video(uploaded_file)

        # OpenCV 影片讀取設定
        cap = cv2.VideoCapture(uploaded_file)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 5 == 0:  # 每 5 幀做一次預測
                # 預處理
                resnet_input, custom_input, display_img = preprocess_for_models(frame)

                # 預測
                resnet_pred = resnet_classifier.predict(resnet_input)[0][0]
                custom_pred = custom_model.predict(custom_input)[0][0]

                # 合併預測結果
                combined_pred = (resnet_pred + custom_pred) / 2
                label = "Deepfake" if combined_pred > 0.5 else "Real"
                confidence = combined_pred if combined_pred > 0.5 else 1 - combined_pred

                # 顯示預測結果
                cv2.putText(frame, f"{label} ({confidence:.2%})", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Deepfake Detection Video", frame)

                # 假如按 'q'，停止視頻播放
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    else:
        # 圖片處理
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        # 預處理
        resnet_input, custom_input, display_img = preprocess_for_models(img)

        # 預測結果
        resnet_pred = resnet_classifier.predict(resnet_input)[0][0]
        custom_pred = custom_model.predict(custom_input)[0][0]

        # 合併預測結果：可以根據需求加權兩個預測結果
        combined_pred = (resnet_pred + custom_pred) / 2  # 這裡簡單取平均
        label = "Deepfake" if combined_pred > 0.5 else "Real"
        confidence = combined_pred if combined_pred > 0.5 else 1 - combined_pred

        # 顯示結果
        st.image(display_img, caption="你上傳的圖片", use_container_width=True)
        st.markdown(f"### 🧑‍⚖️ 最終預測結果: **{label}** ({confidence:.2%})")
