import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO

st.set_page_config(layout="wide")
st.title("Satellite Data Analysis")

# Helper: read image (support jpg, png, tiff)
def read_image(file):
    if file is None:
        return None
    try:
        if file.name.lower().endswith(('.tif', '.tiff')):
            image = Image.open(file).convert("RGB")
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            file.seek(0)
            file_bytes = file.read()
            img_np = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            return img
    except Exception as e:
        st.error(f"Error reading image: {e}")
        return None

# Convert NumPy array to PNG bytes for download
@st.cache_data
def convert_to_image_bytes(img_array):
    img = Image.fromarray(img_array)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer

# Initialize session state variables
if "page" not in st.session_state:
    st.session_state.page = 1
if "before_img" not in st.session_state:
    st.session_state.before_img = None
if "after_img" not in st.session_state:
    st.session_state.after_img = None
if "before_date" not in st.session_state:
    st.session_state.before_date = None
if "after_date" not in st.session_state:
    st.session_state.after_date = None
if "analysis_type" not in st.session_state:
    st.session_state.analysis_type = None

def nav_buttons():
    cols = st.columns([1, 6, 1])
    with cols[0]:
        if st.session_state.page > 1:
            if st.button("⬅️ Back"):
                st.session_state.page -= 1
                st.experimental_rerun()
    with cols[2]:
        if st.session_state.page < 3:
            if st.button("Next ➡️"):
                if st.session_state.page == 1:
                    if st.session_state.analysis_type is not None:
                        st.session_state.page += 1
                        st.experimental_rerun()
                    else:
                        st.warning("Please select an analysis type to proceed.")
                elif st.session_state.page == 2:
                    if (st.session_state.before_img is not None and
                        st.session_state.after_img is not None and
                        st.session_state.before_date is not None and
                        st.session_state.after_date is not None):
                        st.session_state.page += 1
                        st.experimental_rerun()
                    else:
                        st.warning("Please upload both BEFORE and AFTER images and their dates to proceed.")

# Page 1: Choose analysis type
if st.session_state.page == 1:
    st.header("Step 1: Choose Analysis Type")
    st.session_state.analysis_type = st.radio("Select analysis type:", ["Land-based", "Water-based"], index=0)
    nav_buttons()

# Page 2: Upload images and dates
elif st.session_state.page == 2:
    st.header("Step 2: Upload Images and Enter Dates")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("BEFORE Image & Date")
        before_file = st.file_uploader("Upload BEFORE image", type=['jpg', 'png', 'jpeg', 'tif', 'tiff'], key="before_uploader")
        before_date = st.date_input("Date of BEFORE image", key="before_date_input")
        if before_file is not None:
            st.session_state.before_img = before_file
        if before_date is not None:
            st.session_state.before_date = before_date
        if st.session_state.before_img is not None:
            img = read_image(st.session_state.before_img)
            if img is not None:
                st.image(img, caption="BEFORE Image Preview", use_container_width=True)

    with col2:
        st.subheader("AFTER Image & Date")
        after_file = st.file_uploader("Upload AFTER image", type=['jpg', 'png', 'jpeg', 'tif', 'tiff'], key="after_uploader")
        after_date = st.date_input("Date of AFTER image", key="after_date_input")
        if after_file is not None:
            st.session_state.after_img = after_file
        if after_date is not None:
            st.session_state.after_date = after_date
        if st.session_state.after_img is not None:
            img = read_image(st.session_state.after_img)
            if img is not None:
                st.image(img, caption="AFTER Image Preview", use_container_width=True)

    nav_buttons()

# Page 3: Visualization & Calamity Detection
elif st.session_state.page == 3:
    st.header("Step 3: Change Detection and Visualization")

    # Validate inputs
    if (st.session_state.before_img is not None and st.session_state.after_img is not None and
        st.session_state.before_date is not None and st.session_state.after_date is not None):

        before = read_image(st.session_state.before_img)
        after = read_image(st.session_state.after_img)

        if before is None or after is None:
            st.error("Error reading one or both images. Please go back and re-upload.")
            nav_buttons()
            st.stop()

        # Resize AFTER image to BEFORE image size
        after_resized = cv2.resize(after, (before.shape[1], before.shape[0]))

        st.image([before, after_resized], caption=["Before Image", "After Image"], width=300)

        # Mask generator (HSV thresholds)
        def generate_mask(image, lower, upper):
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            return mask // 255

        # Overlay & area calculation
        def create_overlay(before_mask, after_mask, pixel_area=0.0001):
            diff = after_mask.astype(int) - before_mask.astype(int)
            overlay = np.zeros((*diff.shape, 3), dtype=np.uint8)
            overlay[:, :, 0] = (diff == 1) * 255  # Red = Increase
            overlay[:, :, 1] = (diff == -1) * 255  # Green = Decrease
            inc_area = np.sum(diff == 1) * pixel_area
            dec_area = np.sum(diff == -1) * pixel_area
            return overlay, inc_area, dec_area

        # Masks for categories
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
        st.table(data)

        # Calculate date difference in days
        date_diff = (st.session_state.after_date - st.session_state.before_date).days

        st.subheader("Calamity Possibility Analysis")

        # Flood detection based on water increase and short time diff
        if inc_w > 0.1:
            if date_diff <= 10:
                st.error("⚠️ Possible Flood Detected")
            elif date_diff <= 60:
                st.warning("🌀 Seasonal Change Detected")
            else:
                st.info("🌊 Urbanization or Long-Term Water Increase")

        # Deforestation detection based on vegetation decrease
        if dec_v > 0.1:
            if date_diff <= 30:
                st.error("🔥 Possible Deforestation")
            else:
                st.warning("🌿 Vegetation Decline Over Time")

    else:
        st.warning("Please upload both BEFORE and AFTER images along with their dates on the previous page.")

    nav_buttons()
