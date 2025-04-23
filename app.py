import numpy as np
import cv2
import streamlit as st
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.resnet50 import preprocess_input

# 模型設定
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(256, 256, 3))
resnet_classifier = Sequential([
    resnet_model,
    Dense(1, activation='sigmoid')
])
resnet_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

@st.cache_resource
def load_custom_model():
    # 假設載入的自定義模型
    custom_model = load_model("deepfake_cnn_model.h5")
    return custom_model

custom_model = load_custom_model()

# 預處理影像
def preprocess_image(img):
    img_resized = cv2.resize(img, (256, 256))  # 確保影像尺寸是256x256
    resnet_input = preprocess_input(np.expand_dims(img_resized, axis=0))
    
    # 用 CLAHE 處理影像，轉為灰階再放回RGB
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    clahe_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    custom_input = np.expand_dims(clahe_rgb / 255.0, axis=0)
    
    return resnet_input, custom_input, img_resized

def process_video(uploaded_video):
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(temp_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        resnet_input, custom_input, display_img = preprocess_image(frame)
        
        # 預測
        resnet_pred = resnet_classifier.predict(resnet_input)[0][0]
        custom_pred = custom_model.predict(custom_input)[0][0]
        combined_pred = (resnet_pred + custom_pred) / 2
        label = "Deepfake" if combined_pred > 0.5 else "Real"
        confidence = combined_pred if combined_pred > 0.5 else 1 - combined_pred

        # 顯示每幀
        try:
            # 若影像過大則縮小
            display_img_resized = cv2.resize(display_img, (640, 480))
            st.image(display_img_resized, caption=f"Frame {frame_count}")
        except Exception as e:
            st.error(f"Error displaying frame {frame_count}: {str(e)}")
        
        frame_count += 1
    
    cap.release()

# Streamlit App
st.title("🕵️ Deepfake 偵測 App")

uploaded_file = st.file_uploader("📤 上傳一張圖片或影片", type=["jpg", "jpeg", "png", "mp4", "mov"])

if uploaded_file is not None:
    if uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
        # 讀取並顯示圖片
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        st.image(file_bytes, caption="你上傳的圖片", use_container_width=True)

        resnet_input, custom_input, display_img = preprocess_image(file_bytes)
        resnet_pred = resnet_classifier.predict(resnet_input)[0][0]
        custom_pred = custom_model.predict(custom_input)[0][0]
        combined_pred = (resnet_pred + custom_pred) / 2
        label = "Deepfake" if combined_pred > 0.5 else "Real"
        confidence = combined_pred if combined_pred > 0.5 else 1 - combined_pred

        st.markdown(f"### 🧑‍⚖️ 最終預測結果: **{label}** ({confidence:.2%})")

    elif uploaded_file.type in ["video/mp4", "video/quicktime"]:
        st.markdown("### 📽️ 正在逐幀處理影片...")
        process_video(uploaded_file)
