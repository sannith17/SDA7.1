# app.py
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os
import cv2
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# Load models
@st.cache_resource
def load_models():
    try:
        cnn_model = tf.keras.models.load_model("models/cnn_model.h5")
    except:
        cnn_model = None
        st.error("CNN model not found")

    try:
        rf_model = joblib.load("models/rf_model.pkl")
    except:
        rf_model = None
        st.error("Random Forest model not found")

    return cnn_model, rf_model

cnn_model, rf_model = load_models()

def preprocess_image(image):
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]
    return image_array

def apply_pca(image):
    img_flat = image.reshape(-1, 3)
    pca = PCA(n_components=1)
    img_pca = pca.fit_transform(img_flat)
    return img_pca.reshape(image.shape[0], image.shape[1])

def predict_cnn(image):
    if cnn_model:
        input_img = np.expand_dims(image, axis=0)
        preds = cnn_model.predict(input_img)[0]
        return preds
    return None

def predict_rf(image):
    if rf_model:
        flat = image.reshape(-1, 3)
        preds = rf_model.predict(flat)
        return preds.reshape(image.shape[:2])
    return None

def calculate_difference(before, after):
    diff = after.astype(float) - before.astype(float)
    return np.abs(diff) / 255.0

def display_comparison(before, after, before_date, after_date):
    col1, col2 = st.columns(2)
    with col1:
        st.image(before, caption=f"Before Image - {before_date}")
    with col2:
        st.image(after, caption=f"After Image - {after_date}")

    heatmap = calculate_difference(np.array(before), np.array(after))
    heatmap_mean = np.mean(heatmap, axis=2)
    fig, ax = plt.subplots()
    cax = ax.imshow(heatmap_mean, cmap='hot')
    fig.colorbar(cax)
    st.pyplot(fig)

    # Calamity Logic
    date_diff = (after_date - before_date).days
    inc_w = np.mean(heatmap_mean > 0.4)
    dec_v = np.mean(heatmap_mean < 0.2)

    if inc_w > 0.1:
        if date_diff <= 10:
            st.error("\u26A0\uFE0F Possible Flood Detected")
        elif date_diff <= 60:
            st.warning("\U0001F300 Seasonal Change Detected")
        else:
            st.info("\U0001F30A Urbanization or Long-Term Water Increase")

    if dec_v > 0.1:
        if date_diff <= 30:
            st.error("\U0001F525 Possible Deforestation")
        else:
            st.warning("Vegetation Decline Over Time")

# Page Logic
st.title("\U0001F30D Satellite Image Calamity Detector")
st.markdown("""
Upload two satellite images — before and after — and get:
- PCA preprocessing
- CNN and Random Forest based calamity detection
- Comparison heatmap
""")

with st.sidebar:
    before_img_file = st.file_uploader("Upload BEFORE image", type=["jpg", "jpeg", "png"], key="before")
    before_date = st.date_input("Before Date", value=datetime(2022, 1, 1))

    after_img_file = st.file_uploader("Upload AFTER image", type=["jpg", "jpeg", "png"], key="after")
    after_date = st.date_input("After Date", value=datetime(2023, 1, 1))

if before_img_file and after_img_file:
    before_img = Image.open(before_img_file).convert("RGB")
    after_img = Image.open(after_img_file).convert("RGB")

    before_prep = preprocess_image(before_img)
    after_prep = preprocess_image(after_img)

    st.subheader("PCA Visualization")
    pca_before = apply_pca(before_prep)
    pca_after = apply_pca(after_prep)

    col1, col2 = st.columns(2)
    with col1:
        st.image(pca_before, caption="PCA - Before", use_column_width=True)
    with col2:
        st.image(pca_after, caption="PCA - After", use_column_width=True)

    st.subheader("Random Forest & CNN Predictions")
    rf_pred = predict_rf(before_prep)
    cnn_pred = predict_cnn(after_prep)

    if rf_pred is not None:
        st.image(rf_pred, caption="Random Forest Prediction")
    if cnn_pred is not None:
        st.bar_chart(cnn_pred)

    st.subheader("Change Detection Heatmap & Calamity Warnings")
    display_comparison(before_img, after_img, before_date, after_date)
else:
    st.warning("Upload both BEFORE and AFTER images to continue.")
