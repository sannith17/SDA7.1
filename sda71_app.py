import streamlit as st
import numpy as np
import pandas as pd
import cv2
import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import joblib

# Set page
st.set_page_config(page_title="🌍 Satellite Image Calamity Detector", layout="wide")
st.title("🌍 Satellite Image Calamity Detector")

# Sidebar
st.sidebar.title("Upload Images")
before_image = st.sidebar.file_uploader("Upload BEFORE image", type=["jpg", "jpeg", "png"])
after_image = st.sidebar.file_uploader("Upload AFTER image", type=["jpg", "jpeg", "png"])

before_date = st.sidebar.date_input("Before Date", value=datetime.date(2022, 1, 1))
after_date = st.sidebar.date_input("After Date", value=datetime.date(2024, 1, 1))

# Load models
try:
    cnn_model = tf.keras.models.load_model("cnn_model.h5")
except:
    cnn_model = None
    st.warning("CNN model not found")

try:
    rf_model = joblib.load("rf_model.pkl")
except:
    rf_model = None
    st.warning("Random Forest model not found")

# Preprocess image
def preprocess_image(img, size=(64, 64)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img / 255.0

# PCA function
def apply_pca(img):
    img = cv2.resize(img, (128, 128))
    flat_img = img.reshape(-1, 3)
    pca = PCA(n_components=1)
    pca_img = pca.fit_transform(flat_img)
    return pca_img.reshape(128, 128)

# Random Forest mask generation
def rf_predict_mask(img):
    if rf_model:
        img_resized = cv2.resize(img, (64, 64))
        flat = img_resized.reshape(-1, 3)
        preds = rf_model.predict(flat)
        return preds.reshape(64, 64)
    return None

# CNN prediction
def cnn_predict(img):
    if cnn_model:
        input_img = preprocess_image(img).reshape(1, 64, 64, 3)
        pred = cnn_model.predict(input_img)
        return np.argmax(pred)
    return None

# Comparison heatmap
def generate_heatmap(before, after):
    before = cv2.resize(before, (128, 128))
    after = cv2.resize(after, (128, 128))
    diff = cv2.absdiff(before, after)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)
    return norm

# Main workflow
if before_image and after_image:
    before = cv2.imdecode(np.frombuffer(before_image.read(), np.uint8), 1)
    after = cv2.imdecode(np.frombuffer(after_image.read(), np.uint8), 1)

    st.subheader("PCA Visualization")
    pca_before = apply_pca(before)
    pca_after = apply_pca(after)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(pca_before, cmap='viridis')
    ax[0].set_title("PCA - Before")
    ax[1].imshow(pca_after, cmap='viridis')
    ax[1].set_title("PCA - After")
    st.pyplot(fig)

    # CNN Classification
    cnn_label_before = cnn_predict(before)
    cnn_label_after = cnn_predict(after)

    st.subheader("CNN Classification Results")
    st.write(f"BEFORE Image classified as: {cnn_label_before}")
    st.write(f"AFTER Image classified as: {cnn_label_after}")

    # RF Segmentation
    st.subheader("Random Forest Prediction Masks")
    rf_before = rf_predict_mask(before)
    rf_after = rf_predict_mask(after)

    if rf_before is not None and rf_after is not None:
        fig2, ax2 = plt.subplots(1, 2, figsize=(10, 5))
        ax2[0].imshow(rf_before, cmap='gray')
        ax2[0].set_title("RF Mask - Before")
        ax2[1].imshow(rf_after, cmap='gray')
        ax2[1].set_title("RF Mask - After")
        st.pyplot(fig2)

    # Final Heatmap
    st.subheader("Change Detection Heatmap")
    heatmap = generate_heatmap(before, after)
    st.image(heatmap, caption="Change Heatmap", use_container_width=True, clamp=True)

    # Calamity Alerts
    st.subheader("🛰️ Calamity Detection")
    date_diff = (after_date - before_date).days

    # Check for water increase / decrease from RF mask
    if rf_before is not None and rf_after is not None:
        inc_water = np.mean(rf_after == 1) - np.mean(rf_before == 1)
        dec_veg = np.mean(rf_before == 2) - np.mean(rf_after == 2)

        if inc_water > 0.1:
            if date_diff <= 10:
                st.error("⚠️ Possible Flood Detected")
            elif date_diff <= 60:
                st.warning("🌀 Seasonal Water Change Detected")
            else:
                st.info("🌊 Long-Term Water Body Expansion")

        if dec_veg > 0.1:
            if date_diff <= 30:
                st.error("🔥 Possible Deforestation Detected")
            else:
                st.warning("Vegetation Decline Over Time")
    else:
        st.warning("RF-based masks not available for calamity detection.")

else:
    st.info("Please upload both BEFORE and AFTER satellite images and select dates.")
