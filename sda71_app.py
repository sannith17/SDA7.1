import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
import io
import base64
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load models (ensure these files exist)
cnn_model = tf.keras.models.load_model("models/cnn_model.h5")
rf_model = joblib.load("models/rf_model.pkl")

st.set_page_config(page_title="Satellite Calamity Detector", layout="wide")
PAGES = ["Upload Images", "Georeference", "Calamity Detection", "Final Visualization"]

if "current_page" not in st.session_state:
    st.session_state.current_page = PAGES[0]

def nav_buttons():
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("📤 Upload Images"):
            st.session_state.current_page = PAGES[0]
    with col2:
        if st.button("📌 Georeference"):
            st.session_state.current_page = PAGES[1]
    with col3:
        if st.button("🌊 Calamity Detection"):
            st.session_state.current_page = PAGES[2]
    with col4:
        if st.button("📊 Final Visualization"):
            st.session_state.current_page = PAGES[3]

def preprocess_image(image):
    img = image.resize((128, 128)).convert("RGB")
    return np.array(img) / 255.0

def calculate_ndwi(image):
    green = image[:, :, 1].astype(float)
    nir = image[:, :, 0].astype(float)
    ndwi = (green - nir) / (green + nir + 1e-5)
    return ndwi

def calculate_ndvi(image):
    nir = image[:, :, 0].astype(float)
    red = image[:, :, 2].astype(float)
    ndvi = (nir - red) / (nir + red + 1e-5)
    return ndvi

def upload_images():
    st.subheader("📥 Upload BEFORE and AFTER Satellite Images")
    before_img = st.file_uploader("Upload BEFORE Image", type=["jpg", "png", "jpeg"], key="before")
    after_img = st.file_uploader("Upload AFTER Image", type=["jpg", "png", "jpeg"], key="after")

    before_date = st.date_input("Date of BEFORE image")
    after_date = st.date_input("Date of AFTER image")

    if before_img and after_img:
        st.image([before_img, after_img], caption=["BEFORE", "AFTER"], width=300)
        st.success("Images successfully uploaded.")

        st.session_state.before_image = Image.open(before_img)
        st.session_state.after_image = Image.open(after_img)
        st.session_state.before_date = before_date
        st.session_state.after_date = after_date

def georeference_images():
    st.subheader("🌍 Georeferencing Interface (Prototype)")
    st.info("This is a placeholder. You can integrate raster geo-align libraries (e.g., `rasterio`, `gdal`) here.")
    if "before_image" in st.session_state and "after_image" in st.session_state:
        st.image(st.session_state.before_image, caption="Before Image", width=300)
        st.image(st.session_state.after_image, caption="After Image", width=300)
    else:
        st.warning("Please upload images first.")

def calamity_detection():
    st.subheader("🌊 Calamity Detection using NDWI, NDVI, CNN & Random Forest")
    if "before_image" in st.session_state and "after_image" in st.session_state:
        before = preprocess_image(st.session_state.before_image)
        after = preprocess_image(st.session_state.after_image)

        # NDWI/NDVI Changes
        ndwi_before = calculate_ndwi(before)
        ndwi_after = calculate_ndwi(after)
        ndvi_before = calculate_ndvi(before)
        ndvi_after = calculate_ndvi(after)

        water_change = np.mean(ndwi_after - ndwi_before)
        vegetation_change = np.mean(ndvi_after - ndvi_before)

        date_diff = (st.session_state.after_date - st.session_state.before_date).days

        # CNN Prediction
        cnn_pred_before = cnn_model.predict(np.expand_dims(before, axis=0))[0]
        cnn_pred_after = cnn_model.predict(np.expand_dims(after, axis=0))[0]

        # Random Forest Prediction
        flat_before = before.flatten().reshape(1, -1)
        flat_after = after.flatten().reshape(1, -1)
        rf_pred_before = rf_model.predict(flat_before)[0]
        rf_pred_after = rf_model.predict(flat_after)[0]

        # Display model outputs
        st.markdown(f"📌 **CNN Before Prediction:** {np.argmax(cnn_pred_before)} | After: {np.argmax(cnn_pred_after)}")
        st.markdown(f"📌 **RF Before Prediction:** {rf_pred_before} | After: {rf_pred_after}")

        st.markdown(f"🧮 **NDWI Increase:** `{water_change:.3f}`")
        st.markdown(f"🧮 **NDVI Decrease:** `{vegetation_change:.3f}`")

        # Decision Logic
        if water_change > 0.1:
            if date_diff <= 10:
                st.error("⚠️ Possible Flood Detected")
            elif date_diff <= 60:
                st.warning("🌀 Seasonal Change Detected")
            else:
                st.info("🌊 Urbanization or Long-Term Water Increase")

        if vegetation_change < -0.1:
            if date_diff <= 30:
                st.error("🔥 Possible Deforestation")
            else:
                st.warning("🍂 Vegetation Decline Over Time")

    else:
        st.warning("Please upload both BEFORE and AFTER images.")

def final_visualization():
    st.subheader("📊 Final Visualization: Comparison Heatmap")
    if "before_image" in st.session_state and "after_image" in st.session_state:
        before = preprocess_image(st.session_state.before_image)
        after = preprocess_image(st.session_state.after_image)

        diff = np.mean(after - before, axis=2)  # RGB diff mean
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(diff, cmap='coolwarm', ax=ax, cbar=True)
        ax.set_title("Heatmap: AFTER vs BEFORE Image Difference")
        st.pyplot(fig)
    else:
        st.warning("Please upload both images for visualization.")

# Routing
st.title("🛰️ Satellite Data Calamity Detection Dashboard")
st.sidebar.title("Navigation")
selected = st.sidebar.radio("Go to", PAGES, index=PAGES.index(st.session_state.current_page))
st.session_state.current_page = selected

# Main Page Dispatcher
if selected == "Upload Images":
    upload_images()
elif selected == "Georeference":
    georeference_images()
elif selected == "Calamity Detection":
    calamity_detection()
elif selected == "Final Visualization":
    final_visualization()

nav_buttons()
