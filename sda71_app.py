import streamlit as st
import numpy as np
import cv2
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from tensorflow.keras.models import load_model
from utils.image_utils import preprocess_image, calculate_ndwi, calculate_ndvi, compare_images, plot_comparison_heatmap

st.set_page_config(page_title="Satellite Water & Calamity Detector", layout="wide")

# Load pretrained models
cnn_model = load_model("models/cnn_model.h5")
rf_model = RandomForestClassifier()
rf_model.load("models/rf_model.pkl")  # Custom method or joblib.load

# Session states
if 'before_image' not in st.session_state:
    st.session_state.before_image = None
    st.session_state.after_image = None
    st.session_state.before_date = None
    st.session_state.after_date = None

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📤 Upload Images", "🧠 Analysis", "🌊 Water Detection", "🗺️ Georeferencing", "📊 Final Visualization"])

# Upload Page
if page == "📤 Upload Images":
    st.title("Upload Satellite Images")
    before = st.file_uploader("Upload BEFORE image", type=['jpg', 'png', 'tif'])
    after = st.file_uploader("Upload AFTER image", type=['jpg', 'png', 'tif'])
    before_date = st.date_input("BEFORE Image Date")
    after_date = st.date_input("AFTER Image Date")

    if before and after:
        st.session_state.before_image = preprocess_image(before)
        st.session_state.after_image = preprocess_image(after)
        st.session_state.before_date = before_date
        st.session_state.after_date = after_date
        st.success("Images and dates successfully uploaded.")

# Analysis Page
elif page == "🧠 Analysis":
    st.title("Calamity Detection Analysis")
    if st.session_state.before_image is not None and st.session_state.after_image is not None:
        b_img, a_img = st.session_state.before_image, st.session_state.after_image
        ndwi_b = calculate_ndwi(b_img)
        ndwi_a = calculate_ndwi(a_img)
        ndvi_b = calculate_ndvi(b_img)
        ndvi_a = calculate_ndvi(a_img)

        inc_w, dec_v = compare_images(ndwi_b, ndwi_a), compare_images(ndvi_b, ndvi_a)
        date_diff = (st.session_state.after_date - st.session_state.before_date).days

        if inc_w > 0.1:
            if date_diff <= 10:
                st.error("⚠️ Possible Flood Detected")
            elif date_diff <= 60:
                st.warning("🌀 Seasonal Change Detected")
            else:
                st.info("🌊 Urbanization or Long-Term Water Increase")

        if dec_v > 0.1:
            if date_diff <= 30:
                st.error("🔥 Possible Deforestation")
            else:
                st.warning("Vegetation Decline Over Time")

        st.metric("Water Increase %", f"{inc_w*100:.2f}%")
        st.metric("Vegetation Decrease %", f"{dec_v*100:.2f}%")
    else:
        st.warning("Please upload both BEFORE and AFTER images along with their dates.")

# Water Detection Page
elif page == "🌊 Water Detection":
    st.title("Water Body Detection using NDWI")
    if st.session_state.after_image is not None:
        ndwi = calculate_ndwi(st.session_state.after_image)
        st.image(ndwi, caption="NDWI - Water Detection", use_column_width=True, clamp=True)
    else:
        st.warning("Upload AFTER image to detect water bodies.")

# Georeferencing Page
elif page == "🗺️ Georeferencing":
    st.title("Georeferencing Support (Overlay Placeholder)")
    st.markdown("You can extend this section using rasterio or folium for geospatial overlays with coordinates.")

# Final Visualization Page
elif page == "📊 Final Visualization":
    st.title("Final Visualization and Comparison")
    if st.session_state.before_image is not None and st.session_state.after_image is not None:
        b_img, a_img = st.session_state.before_image, st.session_state.after_image

        # PCA + CNN + RF
        st.subheader("CNN & RF Classification")
        flat_b = PCA(n_components=50).fit_transform(b_img.reshape(-1, 3))
        cnn_pred = cnn_model.predict(a_img.reshape(1, *a_img.shape))[0]
        rf_pred = rf_model.predict(flat_b)

        st.write("CNN Result Sample:", np.round(cnn_pred[:5], 2))
        st.write("RF Class Count:", np.unique(rf_pred, return_counts=True))

        # Comparison Heatmap
        st.subheader("Comparison Heatmap")
        heatmap = plot_comparison_heatmap(b_img, a_img)
        st.image(heatmap, caption="Change Heatmap (After vs Before)", use_column_width=True, clamp=True)
    else:
        st.warning("Upload both images to proceed.")
