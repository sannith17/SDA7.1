import streamlit as st
import numpy as np
import pandas as pd
import cv2
import os
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from PIL import Image
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(layout="wide")

# Load models
@st.cache_resource

def load_models():
    try:
        cnn_model = tf.keras.models.load_model("cnn_model.h5")
    except:
        cnn_model = None
    try:
        rf_model = joblib.load("rf_model.pkl")
    except:
        rf_model = None
    return cnn_model, rf_model

cnn_model, rf_model = load_models()

# Session state setup
if 'page' not in st.session_state:
    st.session_state.page = 1
if 'analysis_type' not in st.session_state:
    st.session_state.analysis_type = None

# Navigation functions
def next_page():
    st.session_state.page += 1

def reset():
    st.session_state.page = 1

# Image preprocessing

def load_and_preprocess(image_file):
    image = Image.open(image_file)
    image_np = np.array(image)
    return image_np

# PCA visualization if image > 5MB

def pca_visualization(image_np):
    img_resized = cv2.resize(image_np, (128, 128))
    pixels = img_resized.reshape(-1, 3)
    pca = PCA(n_components=3)
    scaled = StandardScaler().fit_transform(pixels)
    pca_img = pca.fit_transform(scaled)
    pca_img = pca_img.reshape(128, 128, 3)
    pca_img = (pca_img - pca_img.min()) / (pca_img.max() - pca_img.min())
    return pca_img

# Random Forest prediction

def predict_rf(image_np):
    img_resized = cv2.resize(image_np, (128, 128))
    pixels = img_resized.reshape(-1, 3)
    prediction = rf_model.predict(pixels)
    segmented_img = prediction.reshape(128, 128)
    return segmented_img

# CNN prediction

def predict_cnn(image_np):
    img_resized = cv2.resize(image_np, (128, 128)) / 255.0
    input_array = np.expand_dims(img_resized, axis=0)
    predictions = cnn_model.predict(input_array)
    return predictions

# Difference map

def difference_heatmap(before_mask, after_mask):
    diff = after_mask != before_mask
    return diff.astype(np.uint8) * 255

# Calamity Detection

def detect_calamity(date1, date2, mask1, mask2):
    diff_mask = mask2 != mask1
    change_percentage = np.sum(diff_mask) / diff_mask.size
    date_diff = (date2 - date1).days

    if change_percentage > 0.15:
        if date_diff <= 30:
            return "🔥 Possible Deforestation"
        elif date_diff <= 10:
            return "⚠️ Possible Flood"
        else:
            return "🌊 Urbanization or Seasonal Change"
    return "No Significant Calamity Detected"

# Layout pages

if st.session_state.page == 1:
    st.title("🌍 Satellite Image Calamity Detector")
    st.subheader("Step 1: Choose Analysis Type")
    st.session_state.analysis_type = st.radio("Select Type of Analysis:", ["Land Body", "Water Body"])
    st.button("Next", on_click=next_page)

elif st.session_state.page == 2:
    st.title("Upload Satellite Images")
    col1, col2 = st.columns(2)
    with col1:
        before_image = st.file_uploader("Upload BEFORE image", type=['jpg', 'jpeg', 'png'], key="before")
        before_date = st.date_input("Before Date")
    with col2:
        after_image = st.file_uploader("Upload AFTER image", type=['jpg', 'jpeg', 'png'], key="after")
        after_date = st.date_input("After Date")
    if before_image and after_image:
        st.session_state.before_image = before_image
        st.session_state.after_image = after_image
        st.session_state.before_date = before_date
        st.session_state.after_date = after_date
        st.button("Next", on_click=next_page)

elif st.session_state.page == 3:
    st.title("Georeferencing & Alignment")
    b_img_np = load_and_preprocess(st.session_state.before_image)
    a_img_np = load_and_preprocess(st.session_state.after_image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(b_img_np, caption="Before Image")
    with col2:
        st.image(a_img_np, caption="After Image")

    st.session_state.b_np = b_img_np
    st.session_state.a_np = a_img_np
    st.button("Next", on_click=next_page)

elif st.session_state.page == 4:
    st.title("Calamity Detection and Visualization")
    b_np = st.session_state.b_np
    a_np = st.session_state.a_np
    before_date = st.session_state.before_date
    after_date = st.session_state.after_date

    if cnn_model and rf_model:
        b_mask = predict_rf(b_np)
        a_mask = predict_rf(a_np)
        diff = difference_heatmap(b_mask, a_mask)
        calamity_result = detect_calamity(before_date, after_date, b_mask, a_mask)

        col1, col2 = st.columns(2)
        with col1:
            st.image(b_mask, caption="Random Forest - Before Mask")
        with col2:
            st.image(a_mask, caption="Random Forest - After Mask")

        st.subheader("Heatmap of Changes")
        st.image(diff, caption="Change Heatmap", use_container_width=True)

        unique_b, count_b = np.unique(b_mask, return_counts=True)
        unique_a, count_a = np.unique(a_mask, return_counts=True)

        df = pd.DataFrame({
            "Element": [f"Class {i}" for i in unique_b],
            "Before %": [round((c/np.sum(count_b))*100, 2) for c in count_b],
            "After %": [round((count_a[i]/np.sum(count_a))*100 if i < len(count_a) else 0, 2) for i in range(len(unique_b))]
        })
        st.dataframe(df)

        fig1, ax1 = plt.subplots()
        ax1.pie(df["After %"], labels=df["Element"], autopct='%1.1f%%')
        st.pyplot(fig1)

        st.success(f"Prediction: {calamity_result}")

        if st.session_state.before_image.size > 5e6:
            pca_b = pca_visualization(b_np)
            st.image(pca_b, caption="PCA Visualization (Before)", use_container_width=True)
        if st.session_state.after_image.size > 5e6:
            pca_a = pca_visualization(a_np)
            st.image(pca_a, caption="PCA Visualization (After)", use_container_width=True)

    else:
        st.error("Models not found. Please ensure 'cnn_model.h5' and 'rf_model.pkl' are in the same directory.")

    st.button("Restart", on_click=reset)
