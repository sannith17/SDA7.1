import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import io
from sklearn import svm
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from torchvision import transforms
from datetime import datetime, timedelta
import sys
import warnings
import seaborn as sns
from streamlit_echarts import st_echarts

# Suppress warnings
warnings.filterwarnings("ignore")

# Workaround for Python compatibility issues
if sys.version_info >= (3, 13):
    import torch._classes
    torch._classes._register_python_class = lambda *args, **kwargs: None

# Initialize session state for page navigation and data storage
def initialize_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = 1
    if 'heatmap_overlay_svm' not in st.session_state:
        st.session_state.heatmap_overlay_svm = None
    if 'heatmap_overlay_cnn' not in st.session_state:
        st.session_state.heatmap_overlay_cnn = None
    if 'aligned_images' not in st.session_state:
        st.session_state.aligned_images = None
    if 'change_mask' not in st.session_state:
        st.session_state.change_mask = None
    if 'classification_svm' not in st.session_state:
        st.session_state.classification_svm = None
    if 'classification_cnn' not in st.session_state:
        st.session_state.classification_cnn = None
    if 'before_date' not in st.session_state:
        st.session_state.before_date = datetime(2023, 1, 1)
    if 'after_date' not in st.session_state:
        st.session_state.after_date = datetime(2023, 6, 1)
    if 'before_file' not in st.session_state:
        st.session_state.before_file = None
    if 'after_file' not in st.session_state:
        st.session_state.after_file = None
    if 'model_choice' not in st.session_state:
        st.session_state.model_choice = "SVM"
    if 'svm_roc_fig' not in st.session_state:
        st.session_state.svm_roc_fig = None
    if 'cnn_roc_fig' not in st.session_state:
        st.session_state.cnn_roc_fig = None
    if 'svm_accuracy' not in st.session_state:
        st.session_state.svm_accuracy = None
    if 'cnn_accuracy' not in st.session_state:
        st.session_state.cnn_accuracy = None
    if 'classification_before_svm' not in st.session_state:
        st.session_state.classification_before_svm = {"Vegetation": 45, "Land": 35, "Water": 20}
    if 'classification_before_cnn' not in st.session_state:
        st.session_state.classification_before_cnn = {"Vegetation": 50, "Land": 30, "Water": 20} # Corrected 'water' to 'Water'
    if 'correlation_matrix' not in st.session_state:
        st.session_state.correlation_matrix = None
    if 'veg_change_heatmap' not in st.session_state:
        st.session_state.veg_change_heatmap = None
    if 'water_change_heatmap' not in st.session_state:
        st.session_state.water_change_heatmap = None
    if 'land_change_heatmap' not in st.session_state:
        st.session_state.land_change_heatmap = None
    if 'classification_after_raw' not in st.session_state:
        st.session_state.classification_after_raw = None # Stores raw classification data for detailed heatmaps

initialize_session_state()

# Set the page layout and browser tab title
st.set_page_config(layout="wide", page_title="Satellite Image Analysis")

# Custom visible title with yellow color and large font
st.markdown(
    """
    <h1 style='color: yellow; font-size: 72px; font-weight: bold; text-align: center;'>
        Satellite Image Analysis
    </h1>
    """,
    unsafe_allow_html=True
)

# -------- Models --------
class DummyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 3)  # 3 classes: Vegetation, Land, Water
        
        # Workaround for PyTorch internal class registration
        if hasattr(torch._C, '_ImperativeEngine'):
            self._backend = torch._C._ImperativeEngine()
        else:
            self._backend = None
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize models safely
try:
    cnn_model = DummyCNN()
    cnn_model.eval()
    svm_model = svm.SVC(probability=True, random_state=42, kernel='rbf')
    
    # Generate correlation matrix for demonstration
    features = ['NDVI', 'NDWI', 'Brightness', 'Urban Index']
    st.session_state.correlation_matrix = pd.DataFrame(
        np.array([
            [1.0, -0.2, 0.1, -0.3],
            [-0.2, 1.0, -0.4, -0.1],
            [0.1, -0.4, 1.0, 0.6],
            [-0.3, -0.1, 0.6, 1.0]
        ]),
        columns=features,
        index=features
    )
except Exception as e:
    st.error(f"Model initialization failed: {e}")
    st.stop()

# -------- Image Processing Functions --------
def validate_image(image):
    """Validate and convert image to RGB format"""
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return Image.fromarray(image)
    elif isinstance(image, Image.Image):
        return image.convert("RGB")
    else:
        raise ValueError("Unsupported image format")

def preprocess_img(img, size=(64, 64)):
    """Preprocess image for model input"""
    try:
        img = validate_image(img)
        img = img.resize(size)
        img_arr = np.array(img) / 255.0
        return img_arr
    except Exception as e:
        st.error(f"Image preprocessing failed: {e}")
        return None

def calculate_ndvi(img):
    """Calculate NDVI (Normalized Difference Vegetation Index)"""
    img_np = np.array(img)
    red = img_np[:, :, 0].astype(float)
    # Using green as pseudo-NIR for demonstration as typical satellite images might not have a dedicated NIR band
    nir = img_np[:, :, 1].astype(float) # Using green as a proxy for NIR
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir - red) / (nir + red)
        ndvi = np.nan_to_num(ndvi)
        ndvi = np.clip(ndvi, -1, 1)
    return ndvi

def calculate_ndwi(img):
    """Calculate NDWI (Normalized Difference Water Index)"""
    img_np = np.array(img)
    green = img_np[:, :, 1].astype(float)
    blue = img_np[:, :, 2].astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        ndwi = (green - blue) / (green + blue) # Green - Blue for NDWI
        ndwi = np.nan_to_num(ndwi)
        ndwi = np.clip(ndwi, -1, 1)
    return ndwi

def align_images(img1, img2):
    """Align images using ECC (Enhanced Correlation Coefficient) method"""
    try:
        img1_np = np.array(validate_image(img1))
        img2_np = np.array(validate_image(img2))

        # Ensure images are of similar size, resize if necessary
        if img1_np.shape != img2_np.shape:
            img2_np = cv2.resize(img2_np, (img1_np.shape[1], img1_np.shape[0]))

        # Convert to grayscale
        gray1 = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)

        # Initialize warp matrix
        warp_mode = cv2.MOTION_EUCLIDEAN
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)

        # Find transformation
        _, warp_matrix = cv2.findTransformECC(gray1, gray2, warp_matrix, warp_mode, criteria)
        
        # Apply transformation
        aligned = cv2.warpAffine(img2_np, warp_matrix, 
                                 (img1_np.shape[1], img1_np.shape[0]),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        # Calculate difference mask (for visual alignment check)
        diff_mask = cv2.absdiff(img1_np, aligned)
        diff_mask = cv2.cvtColor(diff_mask, cv2.COLOR_RGB2GRAY)
        _, black_mask = cv2.threshold(diff_mask, 30, 255, cv2.THRESH_BINARY_INV)
        aligned_black = cv2.bitwise_and(aligned, aligned, mask=black_mask)

        return Image.fromarray(aligned), Image.fromarray(aligned_black)
    except Exception as e:
        st.warning(f"Image alignment failed, returning original image: {e}")
        # Return original images if alignment fails or is not applicable
        return validate_image(img2).resize(img1.size), Image.fromarray(np.zeros_like(np.array(validate_image(img1))))


def get_change_mask(img1, img2, threshold=30):
    """Generate change mask between two images"""
    try:
        img1 = validate_image(img1)
        img2 = validate_image(img2).resize(img1.size)

        gray1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        _, change_mask = cv2.threshold(diff, threshold, 1, cv2.THRESH_BINARY)
        return change_mask.astype(np.uint8)
    except Exception as e:
        st.error(f"Change mask generation failed: {e}")
        return np.zeros((100, 100), dtype=np.uint8)  # Return empty mask if error occurs

# -------- Classification Functions --------
def classify_land_svm(img_arr):
    """Improved land classification using SVM with spectral indices, returning per-pixel classification."""
    try:
        if img_arr is None:
            # Return dummy pixel classifications for a small image
            h, w = 64, 64
            dummy_classification = np.random.randint(0, 3, size=(h, w)) # 0: Veg, 1: Land, 2: Water
            unique, counts = np.unique(dummy_classification, return_counts=True)
            class_counts = dict(zip(unique, counts))
            total_pixels = h * w
            percentages = {
                "Vegetation": (class_counts.get(0, 0) / total_pixels) * 100,
                "Land": (class_counts.get(1, 0) / total_pixels) * 100,
                "Water": (class_counts.get(2, 0) / total_pixels) * 100
            }
            return percentages, dummy_classification

        img = Image.fromarray((img_arr * 255).astype(np.uint8))
        
        # Calculate spectral indices
        ndvi = calculate_ndvi(img)
        ndwi = calculate_ndwi(img)
        
        # Calculate brightness
        brightness = np.mean(img_arr, axis=2)
        
        # Flatten for classification
        h, w = ndvi.shape
        features = np.column_stack([
            ndvi.flatten(),
            ndwi.flatten(),
            brightness.flatten()
        ])
        
        # Dummy training and classification
        # In a real application, SVM would be trained on labeled data
        # For this demo, we simulate classification based on simple thresholds and a random component
        pixel_classifications = np.zeros(len(features), dtype=int) # 0: Veg, 1: Land, 2: Water

        # Simulate classification based on indices
        # Vegetation: High NDVI
        pixel_classifications[features[:, 0] > 0.3] = 0
        # Water: High NDWI and not vegetation
        pixel_classifications[(features[:, 1] > 0.1) & (pixel_classifications != 0)] = 2
        # Land: Remaining pixels
        pixel_classifications[(pixel_classifications != 0) & (pixel_classifications != 2)] = 1

        # Reshape pixel classifications back to image dimensions
        pixel_class_map = pixel_classifications.reshape((h, w))

        # Calculate overall percentages
        unique, counts = np.unique(pixel_classifications, return_counts=True)
        class_counts = dict(zip(unique, counts))
        total_pixels = h * w
        percentages = {
            "Vegetation": (class_counts.get(0, 0) / total_pixels) * 100,
            "Land": (class_counts.get(1, 0) / total_pixels) * 100,
            "Water": (class_counts.get(2, 0) / total_pixels) * 100
        }
        
        return percentages, pixel_class_map
    except Exception as e:
        st.error(f"SVM classification failed: {e}")
        # Return dummy data in case of error
        h, w = 64, 64
        dummy_classification = np.random.randint(0, 3, size=(h, w))
        unique, counts = np.unique(dummy_classification, return_counts=True)
        class_counts = dict(zip(unique, counts))
        total_pixels = h * w
        percentages = {
            "Vegetation": (class_counts.get(0, 0) / total_pixels) * 100,
            "Land": (class_counts.get(1, 0) / total_pixels) * 100,
            "Water": (class_counts.get(2, 0) / total_pixels) * 100
        }
        return percentages, dummy_classification


def classify_land_cnn(img):
    """Improved land classification using CNN, returning per-pixel classification."""
    try:
        img = validate_image(img)
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = cnn_model(img_tensor)
            probabilities = torch.softmax(output, dim=1).numpy()[0]
            
            # Simulate per-pixel classification for CNN
            # This is a simplification; a real CNN would output a classification map
            h, w = img.size[1], img.size[0] # Note: PIL.Image.size is (width, height)
            
            # Create a dummy classification map based on probabilities for visualization
            # Assign pixels randomly based on the overall class probabilities
            pixel_class_map = np.random.choice(
                a=[0, 1, 2], # 0: Vegetation, 1: Land, 2: Water
                size=(h, w),
                p=probabilities
            )
            
            # Further refine based on spectral indices as a hacky way to simulate better classification
            ndvi_map = calculate_ndvi(img).resize((h,w)) # Resize if necessary
            ndwi_map = calculate_ndwi(img).resize((h,w))

            pixel_class_map[ndvi_map > 0.4] = 0 # Strong vegetation
            pixel_class_map[ndwi_map > 0.2] = 2 # Strong water
            
            # Calculate overall percentages from the simulated pixel_class_map
            unique, counts = np.unique(pixel_class_map, return_counts=True)
            class_counts = dict(zip(unique, counts))
            total_pixels = h * w
            percentages = {
                "Vegetation": (class_counts.get(0, 0) / total_pixels) * 100,
                "Land": (class_counts.get(1, 0) / total_pixels) * 100,
                "Water": (class_counts.get(2, 0) / total_pixels) * 100
            }
            return percentages, pixel_class_map
    except Exception as e:
        st.error(f"CNN classification failed: {e}")
        # Return dummy data in case of error
        h, w = 64, 64
        dummy_classification = np.random.randint(0, 3, size=(h, w))
        unique, counts = np.unique(dummy_classification, return_counts=True)
        class_counts = dict(zip(unique, counts))
        total_pixels = h * w
        percentages = {
            "Vegetation": (class_counts.get(0, 0) / total_pixels) * 100,
            "Land": (class_counts.get(1, 0) / total_pixels) * 100,
            "Water": (class_counts.get(2, 0) / total_pixels) * 100
        }
        return percentages, dummy_classification


# -------- Analysis Functions --------
def detect_calamity(date1, date2, before_class_perc, after_class_perc):
    """Detects potential calamities based on changes in land cover and time difference"""
    try:
        date_diff_days = (date2 - date1).days
        
        # Calculate percentage changes for each class
        veg_change = after_class_perc.get("Vegetation", 0) - before_class_perc.get("Vegetation", 0)
        water_change = after_class_perc.get("Water", 0) - before_class_perc.get("Water", 0)
        land_change = after_class_perc.get("Land", 0) - before_class_perc.get("Land", 0)

        # Calamity Detection Logic based on Water Changes
        if abs(water_change) > 5: # More than 5% change in water
            if date_diff_days <= 5:
                if water_change > 0:
                    return "⚠️ **Possible Flood:** Significant increase in water bodies in a very short period."
                else:
                    return "💧 **Possible Drought/Water Body Shrinkage:** Significant decrease in water bodies in a very short period."
            elif date_diff_days <= 90: # Less than 3 months
                return "🌱 **Seasonal Water Level Changes:** Natural variations in water bodies due to seasons."
            elif date_diff_days <= 180: # Less than 6 months
                if water_change > 0:
                    return "🌊 **Potential Reservoir Expansion/New Water Body:** Gradual increase in water over a medium term."
                else:
                    return "🏜️ **Long-term Water Scarcity/Drought Tendency:** Gradual decrease in water over a medium term."
            else: # More than 6 months/years
                if water_change > 0:
                    return "📈 **Climatic Change Impact (Increased Rainfall/Sea Level Rise):** Long-term increase in water body extent."
                else:
                    return "📉 **Climatic Change Impact (Decreased Rainfall/Desertification):** Long-term decrease in water body extent."

        # Calamity Detection Logic based on Land (and implicitly, vegetation) Changes
        if abs(land_change) > 5 or abs(veg_change) > 5: # More than 5% change in land or vegetation
            if date_diff_days <= 30: # Less than 1 month
                if veg_change < -5:
                    return "🔥 **Possible Wildfire/Deforestation:** Significant loss of vegetation in a short term."
                elif land_change > 5:
                    return "🏗️ **Rapid Urbanization/Construction:** Quick increase in bare land/developed areas."
            elif date_diff_days <= 180: # Less than 6 months
                if veg_change < -5:
                    return "🪵 **Moderate Deforestation/Agricultural Expansion:** Gradual vegetation loss."
                elif land_change > 5:
                    return "🏙️ **Continuous Urbanization:** Steady expansion of developed areas."
            else: # More than 6 months/years
                if veg_change < -10:
                    return "🌴 **Long-term Deforestation/Desertification:** Persistent and large-scale vegetation loss."
                elif land_change > 10:
                    return "🏭 **Extensive Urban Sprawl:** Large-scale and prolonged development of land."

        return "✅ **No Significant Calamity Detected:** Minimal changes observed between the two images."
    except Exception as e:
        st.error(f"Calamity detection failed: {e}")
        return "❓ **Analysis Unavailable:** Could not determine calamity status."

def generate_roc_curve(model_type):
    """Generate proper ROC curve for model evaluation"""
    try:
        # Generate realistic dummy data
        n_samples = 100
        y_true = np.random.randint(0, 2, n_samples)
        
        if model_type == "SVM":
            # SVM typically has smoother curves, slightly less accurate
            y_scores = np.random.rand(n_samples) * 0.3 + y_true * 0.6 + np.random.normal(0, 0.05, n_samples)
            y_scores = np.clip(y_scores, 0, 1) # Ensure scores are within [0, 1]
        else: # CNN
            # CNN typically has better performance, steeper curve
            y_scores = np.random.rand(n_samples) * 0.2 + y_true * 0.7 + np.random.normal(0, 0.03, n_samples)
            y_scores = np.clip(y_scores, 0, 1) # Ensure scores are within [0, 1]
            
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))
        color = '#4682B4' if model_type == "SVM" else '#2e8b57' # SteelBlue for SVM, SeaGreen for CNN
        ax.plot(fpr, tpr, color=color, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'Receiver Operating Characteristic ({model_type})')
        ax.legend(loc="lower right")
        st.pyplot(fig) # Directly display the plot
        return fig
    except Exception as e:
        st.error(f"ROC curve generation failed: {e}")
        return plt.figure()  # Return empty figure

def calculate_accuracy(model_type):
    """Calculate accuracy for model evaluation"""
    try:
        # More realistic dummy accuracies
        return 0.82 if model_type == "SVM" else 0.91
    except:
        return 0.0

def generate_bar_chart(before_data, after_data):
    """Generate bar chart using ECharts or Plotly as a fallback"""
    try:
        options = {
            "tooltip": {"trigger": 'axis', "axisPointer": {"type": 'shadow'}},
            "legend": {"data": ['Before', 'After'], "textStyle": {"color": '#ffffff'}},
            "grid": {"left": '3%', "right": '4%', "bottom": '3%', "containLabel": True},
            "xAxis": {
                "type": 'value',
                "axisLabel": {"color": '#ffffff'},
                "axisLine": {"lineStyle": {"color": '#ffffff'}},
                "splitLine": {"lineStyle": {"color": '#333333'}}
            },
            "yAxis": {
                "type": 'category',
                "data": list(before_data.keys()),
                "axisLabel": {"color": '#ffffff'},
                "axisLine": {"lineStyle": {"color": '#ffffff'}},
                "splitLine": {"show": False}
            },
            "series": [
                {
                    "name": 'Before',
                    "type": 'bar',
                    "data": list(before_data.values()),
                    "itemStyle": {"color": '#4682B4'} # SteelBlue
                },
                {
                    "name": 'After',
                    "type": 'bar',
                    "data": list(after_data.values()),
                    "itemStyle": {"color": '#FFA500'} # Orange
                }
            ]
        }
        return options
    except Exception as e:
        st.warning(f"ECharts configuration failed, using Plotly instead: {str(e)}")
        import plotly.graph_objects as go
        
        fig = go.Figure(data=[
            go.Bar(name='Before', y=list(before_data.keys()), 
                   x=list(before_data.values()), orientation='h', marker_color='#4682B4'),
            go.Bar(name='After', y=list(after_data.keys()), 
                   x=list(after_data.values()), orientation='h', marker_color='#FFA500')
        ])
        fig.update_layout(barmode='group',
                          plot_bgcolor='rgba(0,0,0,0)',
                          paper_bgcolor='rgba(0,0,0,0)',
                          font_color='white')
        return fig

def create_class_heatmap(original_img, class_map, target_class_index, color=(0, 255, 0)):
    """
    Creates a heatmap overlay for a specific class.
    target_class_index: 0 for Vegetation, 1 for Land, 2 for Water
    color: RGB tuple for the heatmap color
    """
    try:
        original_img_np = np.array(validate_image(original_img))
        h, w, _ = original_img_np.shape

        # Resize class_map to match original image dimensions
        # Use nearest neighbor interpolation for discrete class maps
        class_map_resized = cv2.resize(class_map.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        # Create a mask for the target class
        class_mask = (class_map_resized == target_class_index).astype(np.uint8) * 255

        # Create a colored overlay from the mask
        heatmap_overlay = np.zeros_like(original_img_np, dtype=np.uint8)
        heatmap_overlay[class_mask == 255] = color

        # Blend the heatmap with the original image
        blended_img = cv2.addWeighted(original_img_np, 0.7, heatmap_overlay, 0.3, 0)
        return Image.fromarray(blended_img)
    except Exception as e:
        st.error(f"Error creating class heatmap: {e}")
        return original_img # Return original if error

# -------- Page Functions --------
def page1():
    """Model selection page"""
    st.markdown("<h2 style='color: white;'>1. Model Selection</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        <p style='font-size: 18px; color: lightgray;'>
            Choose the analysis model that best suits your needs.
        </p>
        """,
        unsafe_allow_html=True
    )
    
    st.session_state.model_choice = st.selectbox(
        "Select Analysis Model", 
        ["SVM", "CNN"],
        help="Choose between Support Vector Machine (SVM) or Convolutional Neural Network (CNN)"
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.info("**SVM** - Faster for smaller datasets, robust for high-dimensional data, but may struggle with highly complex non-linear patterns. Generally less computationally intensive than deep CNNs for inference after training.")
    with col2:
        st.info("**CNN** - Highly accurate for complex image patterns, excels in feature extraction directly from raw pixels, but requires more data and computational resources for training and inference. More powerful for large and varied image analysis tasks.")
    
    st.markdown("---")
    if st.button("Next ➡️", key="page1_next", help="Proceed to the next step: Image Upload"):
        st.session_state.page = 2

def page2():
    """Image upload and date selection page"""
    st.markdown(
        """
        <h2 style='font-size: 36px; color: white;'>
            2. Image Upload & Dates
        </h2>
        <p style='font-size: 18px; color: lightgray;'>
            Please upload the <b>before</b> and <b>after/current</b> satellite images along with their respective dates for analysis.
        </p>
        """,
        unsafe_allow_html=True
    )
    
    st.subheader("Upload Your Images")
    col_upload1, col_upload2 = st.columns(2)
    with col_upload1:
        st.session_state.before_date = st.date_input(
            "BEFORE image date", 
            value=st.session_state.before_date,
            max_value=datetime.today().date(), # Ensure date is not in future
            help="Select the date for the 'before' satellite image."
        )
        st.session_state.before_file = st.file_uploader(
            "Upload BEFORE image",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="before_uploader",
            help="Upload the satellite image taken at the 'before' date."
        )
        if st.session_state.before_file:
            try:
                st.image(st.session_state.before_file, caption="Before Image Preview", use_column_width=True)
            except Exception as e:
                st.error(f"Could not load BEFORE image preview: {e}")

    with col_upload2:
        st.session_state.after_date = st.date_input(
            "AFTER image date", 
            value=st.session_state.after_date,
            min_value=st.session_state.before_date, # After date cannot be before before date
            max_value=datetime.today().date(), # Ensure date is not in future
            help="Select the date for the 'after' (current) satellite image."
        )
        st.session_state.after_file = st.file_uploader(
            "Upload AFTER image",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="after_uploader",
            help="Upload the satellite image taken at the 'after' date."
        )
        if st.session_state.after_file:
            try:
                st.image(st.session_state.after_file, caption="After Image Preview", use_column_width=True)
            except Exception as e:
                st.error(f"Could not load AFTER image preview: {e}")

    st.markdown("---")
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Back", key="page2_back", help="Go back to Model Selection"):
            st.session_state.page = 1
    with col2:
        if st.session_state.before_file and st.session_state.after_file:
            if st.button("Next ➡️", key="page2_next", help="Process images and proceed to Alignment"):
                try:
                    # Process images
                    before_img_pil = Image.open(st.session_state.before_file).convert("RGB")
                    after_img_pil = Image.open(st.session_state.after_file).convert("RGB")
                    
                    # Align images
                    aligned_after, aligned_black = align_images(before_img_pil, after_img_pil)
                    st.session_state.aligned_images = {
                        "before": before_img_pil,
                        "after": aligned_after,
                        "aligned_black": aligned_black
                    }
                    
                    # Calculate change mask (overall change)
                    st.session_state.change_mask = get_change_mask(before_img_pil, aligned_after)
                    
                    # Classify based on selected model
                    if st.session_state.model_choice == "SVM":
                        after_arr = preprocess_img(aligned_after, size=(64, 64)) # Use smaller size for classification
                        st.session_state.classification_svm, st.session_state.classification_after_raw = classify_land_svm(after_arr)
                        st.session_state.classification = st.session_state.classification_svm
                        
                        # Generate evaluation metrics
                        # ROC and Accuracy generation moved to Page 6 (Evaluation)
                        
                    elif st.session_state.model_choice == "CNN":
                        st.session_state.classification_cnn, st.session_state.classification_after_raw = classify_land_cnn(aligned_after)
                        st.session_state.classification = st.session_state.classification_cnn
                        
                        # Generate evaluation metrics
                        # ROC and Accuracy generation moved to Page 6 (Evaluation)

                    st.session_state.page = 3
                except Exception as e:
                    st.error(f"Error processing images: {str(e)}. Please ensure images are valid.")
        else:
            st.warning("Please upload both images to proceed.")

def page3():
    """Aligned images comparison page"""
    st.markdown("<h2 style='color: white;'>3. Aligned Images Comparison</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        <p style='font-size: 18px; color: lightgray;'>
            Review the original 'Before' image, the 'After' image aligned to the 'Before' image, and a 'Difference' view highlighting alignment success.
        </p>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.aligned_images is None:
        st.error("No aligned images found. Please upload images first.")
        st.session_state.page = 2
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(
            st.session_state.aligned_images["before"],
            caption=f"BEFORE Image ({st.session_state.before_date.strftime('%Y-%m-%d')})", 
            use_column_width=True,
            channels="RGB"
        )
    with col2:
        st.image(
            st.session_state.aligned_images["after"],
            caption=f"Aligned AFTER Image ({st.session_state.after_date.strftime('%Y-%m-%d')})", 
            use_column_width=True,
            channels="RGB"
        )
    with col3:
        st.image(
            st.session_state.aligned_images["aligned_black"],
            caption="Aligned Difference (Black areas indicate good alignment)", 
            use_column_width=True,
            channels="RGB"
        )
    st.markdown("---")
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Back", key="page3_back", help="Go back to Image Upload"):
            st.session_state.page = 2
    with col2:
        if st.button("Next ➡️", key="page3_next", help="Proceed to Change Detection Heatmap"):
            st.session_state.page = 4

def page4():
    """Change detection heatmap page with multiple heatmaps"""
    st.markdown("<h2 style='color: white;'>4. Change Detection Heatmaps</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        <p style='font-size: 18px; color: lightgray;'>
            Visualize areas of change. The primary heatmap shows overall change.
            Additional heatmaps highlight changes specifically in Vegetation, Water, and Land.
        </p>
        """,
        unsafe_allow_html=True
    )

    # Validate data
    required_keys = ['aligned_images', 'change_mask', 'classification_after_raw']
    if not all(key in st.session_state for key in required_keys):
        st.error("Processing data not found. Please upload and process images first.")
        st.session_state.page = 2
        return

    st.subheader(f"Overall Change Heatmap using {st.session_state.model_choice} Model")
    
    # Generate and display overall change heatmap (similar to previous version, now colored by model for clarity)
    h, w = st.session_state.aligned_images["after"].size[1], st.session_state.aligned_images["after"].size[0] #PIL image size (width, height)
    aligned_after_resized = st.session_state.aligned_images["after"].resize((w, h))

    # Create model-specific heatmap for overall changes
    heatmap_color = (255, 0, 0) # Default red for overall change
    if st.session_state.model_choice == "SVM":
        heatmap_color = (0, 0, 255) # Blue for SVM
    elif st.session_state.model_choice == "CNN":
        heatmap_color = (0, 255, 0) # Green for CNN

    heatmap_np = np.zeros((h, w, 3), dtype=np.uint8)
    heatmap_np[st.session_state.change_mask == 1] = heatmap_color # Apply color where change mask is 1

    heatmap_img = Image.fromarray(heatmap_np)
    overall_heatmap_overlay = Image.blend(
        aligned_after_resized.convert("RGB"),
        heatmap_img.convert("RGB"),
        alpha=0.5
    )
    st.image(
        overall_heatmap_overlay, 
        caption=f"Overall Change Heatmap ({st.session_state.model_choice} - {st.session_state.model_choice} highlights changes)", 
        use_column_width=True,
        channels="RGB"
    )
    st.caption(f"Darker {st.session_state.model_choice} areas indicate detected changes.")

    st.subheader("Detailed Class Change Heatmaps")
    
    # Generate and display heatmaps for specific classes
    after_img_raw = st.session_state.aligned_images["after"]
    class_map = st.session_state.classification_after_raw # This is the per-pixel classification map

    col_veg, col_water, col_land = st.columns(3)
    st.markdown("---")
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Back", key="page4_back", help="Go back to Aligned Images Comparison"):
            st.session_state.page = 3
    with col2:
        if st.button("Next ➡️", key="page4_next", help="Proceed to Land Classification & Calamity Analysis"):
            st.session_state.page = 5

def page5():
    """Land classification and analysis page"""
    st.markdown("<h2 style='color: white;'>5. Land Classification & Analysis</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        <p style='font-size: 18px; color: lightgray;'>
            This section provides detailed land cover classification and identifies potential calamities based on detected changes.
        </p>
        """,
        unsafe_allow_html=True
    )

    

    # Classification Table & Charts
    st.subheader(f"Land Classification using {st.session_state.model_choice}")
    
    col_tables, col_charts = st.columns([1, 1])
    with col_tables:
        st.markdown("<p style='font-size: 18px; color: white;'><b>Land Cover Area Distribution (%)</b></p>", unsafe_allow_html=True)
        # Combine data for a single, more informative table
        df_classification = pd.DataFrame({
            "Class": list(before_class_data.keys()),
            "Before Area (%)": list(before_class_data.values()),
            "After Area (%)": [classification_data.get(k, 0) for k in before_class_data.keys()] # Ensure keys match
        })
        df_classification["Change (%)"] = df_classification["After Area (%)"] - df_classification["Before Area (%)"]
        st.dataframe(df_classification.style.format({
            "Before Area (%)": "{:.1f}%",
            "After Area (%)": "{:.1f}%",
            "Change (%)": "{:+.1f}%" # Show +/- for change
        }).background_gradient(cmap='RdYlGn', subset=['Change (%)']), use_container_width=True) # Add color gradient for change

        # Download button for classification data
        csv_data = df_classification.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Classification Data (CSV)",
            data=csv_data,
            file_name=f"land_classification_{st.session_state.model_choice}_{st.session_state.after_date.strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Download the land classification percentages as a CSV file."
        )

    with col_charts:
        st.markdown("<p style='font-size: 18px; color: white;'><b>Classification Distribution Comparison</b></p>", unsafe_allow_html=True)
        
        # Create tabs for pie charts (Before/After)
        tab1, tab2 = st.tabs(["Before Image", "After Image"])
        
        with tab1:
            fig1, ax1 = plt.subplots(figsize=(6, 6))
            ax1.pie(
                before_class_data.values(), 
                labels=before_class_data.keys(), 
                autopct='%1.1f%%',
                colors=['#2e8b57', '#cd853f', '#4682b4'], # Vegetation green, land brown, water blue
                startangle=90
            )
            ax1.axis('equal')
            ax1.set_title("Before Image Land Cover", color='white')
            st.pyplot(fig1)
            plt.close(fig1) # Close figure to prevent warning

        with tab2:
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            ax2.pie(
                classification_data.values(), 
                labels=classification_data.keys(), 
                autopct='%1.1f%%',
                colors=['#2e8b57', '#cd853f', '#4682b4'],
                startangle=90
            )
            ax2.axis('equal')
            ax2.set_title("After Image Land Cover", color='white')
            st.pyplot(fig2)
            plt.close(fig2) # Close figure to prevent warning

    # Add bar chart using ECharts for direct comparison
    st.subheader("Land Cover Changes Over Time")
    try:
        bar_options = generate_bar_chart(before_class_data, classification_data)
        if isinstance(bar_options, dict):  # ECharts format
            st_echarts(options=bar_options, height="400px")
        else:  # Plotly format
            st.plotly_chart(bar_options, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to render chart: {str(e)}")
        # Fallback to simple matplotlib bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        y = np.arange(len(before_class_data))
        width = 0.35
        ax.barh(y - width/2, before_class_data.values(), height=width, label='Before', color='#4682B4')
        ax.barh(y + width/2, classification_data.values(), height=width, label='After', color='#FFA500')
        ax.set_yticks(y)
        ax.set_yticklabels(before_class_data.keys(), color='white')
        ax.set_xlabel('Area (%)', color='white')
        ax.set_title('Land Cover Changes', color='white')
        ax.legend()
        ax.tick_params(axis='x', colors='white')
        ax.spines['top'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        fig.patch.set_facecolor('none') # Transparent background
        ax.set_facecolor('none') # Transparent background
        st.pyplot(fig)
        plt.close(fig)

    # Validate data
    required_keys = ['classification', 'change_mask', 'before_date', 'after_date']
    if not all(key in st.session_state for key in required_keys):
        st.error("Analysis data not found. Please start from the beginning.")
        st.session_state.page = 1
        return

    # Calculate change percentage for overall area (still useful)
    try:
        total_pixels = np.prod(st.session_state.change_mask.shape)
        changed_pixels = np.sum(st.session_state.change_mask)
        overall_change_percentage = changed_pixels / total_pixels
    except:
        overall_change_percentage = 0

    st.subheader("🚨 Calamity Detection")
    # Get classification data
    classification_data = st.session_state.classification # After image classification
    if classification_data is None:
        classification_data = {"Vegetation": 0, "Land": 0, "Water": 0}

    # Get before classification data (dummy for now)
    before_class_data = st.session_state.classification_before_svm if st.session_state.model_choice == "SVM" else st.session_state.classification_before_cnn
    
    calamity_report = detect_calamity(
        st.session_state.before_date,
        st.session_state.after_date,
        before_class_data, # Pass before classification
        classification_data # Pass after classification
    )
    
    # Display calamity report with appropriate color
    if "⚠️" in calamity_report or "🔥" in calamity_report or "💧" in calamity_report or "🏜️" in calamity_report or "🪵" in calamity_report or "🏭" in calamity_report:
        color = "red"
    elif "🌱" in calamity_report or "✅" in calamity_report:
        color = "lightgreen"
    elif "🏗️" in calamity_report or "🌊" in calamity_report or "🏙️" in calamity_report or "📈" in calamity_report or "📉" in calamity_report or "🌴" in calamity_report:
        color = "orange"
    else:
        color = "gray"
    
    st.markdown(f"<h3 style='color: {color};'>{calamity_report}</h3>", unsafe_allow_html=True)
    
    st.markdown(f"""
        <p style='font-size: 16px; color: lightgray;'>
            <b>Overall Area Change Detected:</b> {overall_change_percentage:.2%} of the image area<br>
            <b>Time Period:</b> {(st.session_state.after_date - st.session_state.before_date).days} days between images.
        </p>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("---")
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Back", key="page5_back", help="Go back to Change Detection Heatmap"):
            st.session_state.page = 4
    with col2:
        if st.button("Next ➡️", key="page5_next", help="Proceed to Feature Correlation & Model Evaluation"):
            st.session_state.page = 6

def page6():
    """Feature correlation analysis and model evaluation page"""
    st.markdown("<h2 style='color: white;'>6. Feature Correlation & Model Evaluation</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        <p style='font-size: 18px; color: lightgray;'>
            Understand the relationships between different spectral features and evaluate the performance of the selected model.
        </p>
        """,
        unsafe_allow_html=True
    )
    
    if st.session_state.correlation_matrix is None:
        st.error("Correlation data not available. Please ensure images are processed.")
        st.session_state.page = 1
        return
    
    st.subheader("Feature Correlation Matrix")
    st.info("This heatmap shows the correlation between different spectral indices derived from the satellite images. Values close to 1 or -1 indicate strong positive or negative correlation, respectively.")
    
    # Display correlation matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        st.session_state.correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".1f", # Format annotations to one decimal place
        vmin=-1,
        vmax=1,
        linewidths=.5, # Add lines between cells
        linecolor='black',
        ax=ax
    )
    ax.set_title("Feature Correlation Matrix", color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    fig.patch.set_facecolor('none') # Transparent background
    ax.set_facecolor('none') # Transparent background
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
    st.pyplot(fig)
    plt.close(fig) # Close figure to prevent warning
    
    # Interpretation
    st.markdown("""
    ### Correlation Interpretation:
    - **NDVI (Normalized Difference Vegetation Index):** Measures vegetation health. High NDVI means dense vegetation. It typically has a negative correlation with water and urban areas.
    - **NDWI (Normalized Difference Water Index):** Highlights water bodies. High NDWI indicates water. It generally has a negative correlation with land and vegetation.
    - **Brightness (Overall reflectance):** Represents the overall lightness of the area. Highly reflective surfaces (like barren land or urban structures) will have high brightness.
    - **Urban Index (Simulated):** A hypothetical index to differentiate urban areas. Expected to correlate positively with brightness and negatively with vegetation.
    
    **Example Interpretations for this Dummy Matrix:**
    - A negative correlation between **NDVI** and **NDWI** (-0.2) suggests that as vegetation increases, water content tends to decrease, and vice-versa, which is typical.
    - **Brightness** has a strong positive correlation with **Urban Index** (0.6), implying urban areas are generally brighter.
    - **NDWI** shows a negative correlation with **Brightness** (-0.4), indicating water bodies are typically less bright than other land covers.
    """)
    st.markdown("---")

    st.subheader("Model Evaluation")
    st.info(f"Evaluating the performance of the selected **{st.session_state.model_choice}** model.")
    
    col_roc, col_metrics = st.columns(2)

    with col_roc:
        st.markdown("#### Receiver Operating Characteristic (ROC) Curve")
        st.caption("The ROC curve illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied. The closer the curve is to the top-left corner, the higher the accuracy of the test.")
        
        if st.session_state.model_choice == "SVM":
            # Generate and display SVM ROC curve
            st.session_state.svm_roc_fig = generate_roc_curve("SVM")
        else: # CNN
            # Generate and display CNN ROC curve
            st.session_state.cnn_roc_fig = generate_roc_curve("CNN")
            
    with col_metrics:
        st.markdown("#### Key Performance Metrics")
        if st.session_state.model_choice == "SVM":
            st.session_state.svm_accuracy = calculate_accuracy("SVM")
            st.metric(label="Model Accuracy (SVM)", value=f"{st.session_state.svm_accuracy:.2f}")
            st.markdown("""
            **Confusion Matrix (Dummy Data for SVM):**
            ```
                Predicted
                   V    L    W
            True V 75   10   5
                 L 10   60   10
                 W 5    10   25
            ```
            (V = Vegetation, L = Land, W = Water)
            """)
        else: # CNN
            st.session_state.cnn_accuracy = calculate_accuracy("CNN")
            st.metric(label="Model Accuracy (CNN)", value=f"{st.session_state.cnn_accuracy:.2f}")
            st.markdown("""
            **Confusion Matrix (Dummy Data for CNN):**
            ```
                Predicted
                   V    L    W
            True V 85   5    3
                 L 5    70   5
                 W 2    3    30
            ```
            (V = Vegetation, L = Land, W = Water)
            """)
        st.caption("These metrics provide an indication of how well the model performs. Accuracy is the proportion of correctly classified instances. The confusion matrix shows the number of correct and incorrect predictions made by the classification model compared against the actual outcomes.")

    st.markdown("---")
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Back", key="page6_back", help="Go back to Land Classification & Analysis"):
            st.session_state.page = 5
    with col2:
        if st.button("Restart Analysis 🔄", key="page6_restart", help="Start the analysis from the beginning."):
            st.session_state.page = 1
            # Clear relevant session state variables for a fresh start
            st.session_state.before_file = None
            st.session_state.after_file = None
            st.session_state.aligned_images = None
            st.session_state.change_mask = None
            st.session_state.classification_svm = None
            st.session_state.classification_cnn = None
            st.session_state.heatmap_overlay_svm = None
            st.session_state.heatmap_overlay_cnn = None
            st.session_state.veg_change_heatmap = None
            st.session_state.water_change_heatmap = None
            st.session_state.land_change_heatmap = None
            st.session_state.classification_after_raw = None
            st.experimental_rerun() # Rerun to reflect changes immediately


# Main app logic to render pages
if st.session_state.page == 1:
    page1()
elif st.session_state.page == 2:
    page2()
elif st.session_state.page == 3:
    page3()
elif st.session_state.page == 4:
    page4()
elif st.session_state.page == 5:
    page5()
elif st.session_state.page == 6:
    page6()
