import streamlit as st
import cv2
import numpy as np
import plotly.express as px
import pandas as pd
from PIL import Image
from io import BytesIO

st.set_page_config(layout="wide")
st.title("Satellite Data Analysis")

# Helper function to read uploaded images (supports TIFF)
def read_image(file):
    if file is None:
        return None
    if file.name.lower().endswith(('.tif', '.tiff')):
        image = Image.open(file).convert("RGB")
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        file.seek(0)  # Reset file pointer
        return cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

# Convert NumPy array to download-ready image bytes (cache to avoid recomputation)
@st.cache_data
def convert_to_image_bytes(img_array):
    img = Image.fromarray(img_array)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer

# Initialize page number in session state
if 'page' not in st.session_state:
    st.session_state.page = 1

def nav_buttons():
    cols = st.columns([1,6,1])
    with cols[0]:
        if st.session_state.page > 1:
            if st.button("⬅️ Back"):
                st.session_state.page -= 1
    with cols[2]:
        if st.session_state.page < 3:
            if st.button("Next ➡️"):
                st.session_state.page += 1

# Page 1: Select analysis type
if st.session_state.page == 1:
    st.header("Step 1: Choose Analysis Type")
    st.session_state.analysis_type = st.radio("Select analysis type:", ["Land-based", "Water-based"])
    nav_buttons()

# Page 2: Upload images and input dates (both before and after on same page)
elif st.session_state.page == 2:
    st.header("Step 2: Upload Images and Enter Dates")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("BEFORE Image & Date")
        before_img = st.file_uploader("Upload BEFORE image", type=['jpg', 'png', 'tif'], key="before")
        before_date = st.date_input("Date of BEFORE image", key="before_date")
        # Store in session state
        if before_img is not None:
            st.session_state.before_img = before_img
        if before_date is not None:
            st.session_state.before_date = before_date

    with col2:
        st.subheader("AFTER Image & Date")
        after_img = st.file_uploader("Upload AFTER image", type=['jpg', 'png', 'tif'], key="after")
        after_date = st.date_input("Date of AFTER image", key="after_date")
        # Store in session state
        if after_img is not None:
            st.session_state.after_img = after_img
        if after_date is not None:
            st.session_state.after_date = after_date

    # Show preview thumbnails if available
    if 'before_img' in st.session_state and st.session_state.before_img is not None:
        st.image(st.session_state.before_img, caption="Before Image Preview", width=250)
    if 'after_img' in st.session_state and st.session_state.after_img is not None:
        st.image(st.session_state.after_img, caption="After Image Preview", width=250)

    nav_buttons()

# Page 3: Visualization and Calamity Analysis
elif st.session_state.page == 3:
    st.header("Step 3: Change Detection and Visualization")

    # Check if both images and dates are uploaded
    if ('before_img' in st.session_state and st.session_state.before_img is not None and
        'after_img' in st.session_state and st.session_state.after_img is not None and
        'before_date' in st.session_state and
        'after_date' in st.session_state):

        before = read_image(st.session_state.before_img)
        after = read_image(st.session_state.after_img)

        # Resize AFTER image to BEFORE image size for alignment
        after_resized = cv2.resize(after, (before.shape[1], before.shape[0]))

        st.image([before, after_resized], caption=["Before Image (aligned)", "After Image (aligned)"], width=300)

        # Masks generation function
        def generate_mask(image, lower, upper):
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            return mask // 255  # Normalize mask to 0 and 1

        # Overlay creation
        def create_overlay(before_mask, after_mask, pixel_area=0.0001):
            diff = after_mask.astype(int) - before_mask.astype(int)
            overlay = np.zeros((*diff.shape, 3), dtype=np.uint8)
            overlay[:,:,0] = (diff == 1) * 255  # Red: Increase
            overlay[:,:,1] = (diff == -1) * 255  # Green: Decrease
            inc_area = np.sum(diff == 1) * pixel_area
            dec_area = np.sum(diff == -1) * pixel_area
            return overlay, inc_area, dec_area

        # Generate masks for land, water, vegetation based on HSV thresholds
        land_mask_b = generate_mask(before, [10, 0, 100], [25, 255, 255])
        land_mask_a = generate_mask(after_resized, [10, 0, 100], [25, 255, 255])

        water_mask_b = generate_mask(before, [90, 50, 50], [140, 255, 255])
        water_mask_a = generate_mask(after_resized, [90, 50, 50], [140, 255, 255])

        veg_mask_b = generate_mask(before, [35, 50, 50], [85, 255, 255])
        veg_mask_a = generate_mask(after_resized, [35, 50, 50], [85, 255, 255])

        st.subheader("Heat Maps of Changes")
        tab1, tab2, tab3 = st.tabs(["Water", "Vegetation", "Land"])

        with tab1:
            overlay_w, inc_w, dec_w = create_overlay(water_mask_b, water_mask_a)
            st.image(overlay_w, caption="Water Change Map (Red=Increase, Green=Decrease)", channels="RGB")
            st.download_button("Download Water Overlay", data=convert_to_image_bytes(overlay_w), file_name="water_overlay.png", mime="image/png")

        with tab2:
            overlay_v, inc_v, dec_v = create_overlay(veg_mask_b, veg_mask_a)
            st.image(overlay_v, caption="Vegetation Change Map (Red=Increase, Green=Decrease)", channels="RGB")
            st.download_button("Download Vegetation Overlay", data=convert_to_image_bytes(overlay_v), file_name="vegetation_overlay.png", mime="image/png")

        with tab3:
            overlay_l, inc_l, dec_l = create_overlay(land_mask_b, land_mask_a)
            st.image(overlay_l, caption="Land Change Map (Red=Increase, Green=Decrease)", channels="RGB")
            st.download_button("Download Land Overlay", data=convert_to_image_bytes(overlay_l), file_name="land_overlay.png", mime="image/png")

        st.subheader("Change Summary in Area (sq km)")
        data = pd.DataFrame({
            "Category": ["Water Increase", "Water Decrease", "Vegetation Increase", "Vegetation Decrease", "Land Increase", "Land Decrease"],
            "Area (sq km)": [inc_w, dec_w, inc_v, dec_v, inc_l, dec_l]
        })

        fig = px.bar(data, x="Category", y="Area (sq km)", color="Category", title="Change in Area")
        st.plotly_chart(fig)

        pie_fig = px.pie(data, values="Area (sq km)", names="Category", title="Percentage Change by Category",
                         color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(pie_fig)

        st.subheader("Change Comparison Table")
        st.dataframe(data.style.highlight_max(axis=0, color='lightgreen'))

        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Summary CSV", data=csv, file_name="change_summary.csv", mime="text/csv")

        st.subheader("Calamity Analysis")
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

    else:
        st.warning("Please upload both BEFORE and AFTER images along with their dates on the previous page.")

    nav_buttons()
