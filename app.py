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
from fpdf import FPDF

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
except Exception:
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

# 🔹 生成 PDF 報告
def generate_pdf(img, resnet_label, resnet_conf, custom_label, custom_conf):
    pdf = FPDF()
    pdf.add_page()

    # 設定字型為 Arial，防止 Unicode 問題
    pdf.set_font("Arial", size=12)

    # 文字內容，確保是基本的英文字符，不使用特殊符號
    pdf.cell(200, 10, txt=f"ResNet50 Prediction: {resnet_label} ({resnet_conf:.2%})", ln=True, align="C")
    pdf.cell(200, 10, txt=f"Custom CNN Prediction: {custom_label} ({custom_conf:.2%})", ln=True, align="C")

    # 儲存 PDF
    output_path = "/tmp/deepfake_report.pdf"
    pdf.output(output_path)
    
    return output_path

# 🔹 Streamlit App
st.title("🕵️ Deepfake 偵測 App")

uploaded_file = st.file_uploader("📤 上傳一張圖片", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    resnet_input, custom_input, display_img = preprocess_for_models(img)

    # 預測
    resnet_pred = resnet_classifier.predict(resnet_input)[0][0]
    custom_pred = custom_model.predict(custom_input)[0][0]

    resnet_label = "Deepfake" if resnet_pred > 0.5 else "Real"
    custom_label = "Deepfake" if custom_pred > 0.5 else "Real"
    resnet_conf = resnet_pred if resnet_pred > 0.5 else 1 - resnet_pred
    custom_conf = custom_pred if custom_pred > 0.5 else 1 - custom_pred

    # 顯示圖片與結果
    st.image(display_img, caption="你上傳的圖片", use_container_width=True)
    st.markdown(f"### 🤖 ResNet50 預測: **{resnet_label}** ({resnet_conf:.2%})")
    st.markdown(f"### 🧠 自訂 CNN 預測: **{custom_label}** ({custom_conf:.2%})")

    # 生成並下載 PDF 報告
    pdf_path = generate_pdf(display_img, resnet_label, resnet_conf, custom_label, custom_conf)
    with open(pdf_path, "rb") as f:
        st.download_button("📥 下載報告 PDF", f, file_name="deepfake_report.pdf", mime="application/pdf")
