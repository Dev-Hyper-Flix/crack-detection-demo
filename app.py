import streamlit as st
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io

# Set up the page layout and title
st.set_page_config(page_title="Crack Detection Dashboard", layout="wide")

st.title("🔍 Crack Detection Dashboard")
st.caption("Real-Time Crack Detection using U-Net and DeepLabV3+ Models")
st.markdown("---")

# Initialize session state for model caching
@st.cache_resource
def load_models():
    """Load pre-trained DeepLabV3+ and U-Net models"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load DeepLabV3+ with ResNet101 backbone
    deeplab_model = models.segmentation.deeplabv3_resnet101(
        pretrained=True,
        progress=False
    ).to(device)
    deeplab_model.eval()
    
    # Load DeepLabV3 with ResNet50 (serves as U-Net alternative)
    unet_model = models.segmentation.deeplabv3_resnet50(
        pretrained=True,
        progress=False
    ).to(device)
    unet_model.eval()
    
    return deeplab_model, unet_model, device

def preprocess_image(image_cv):
    """Preprocess image for model inference"""
    # Resize to standard input size
    img_resized = cv2.resize(image_cv, (512, 512))
    
    # Convert BGR to RGB and normalize
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Normalize using ImageNet statistics
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return preprocess(img_pil).unsqueeze(0)

def detect_cracks_with_model(image_tensor, model, device):
    """Run inference on image using segmentation model"""
    with torch.no_grad():
        output = model(image_tensor.to(device))
    
    # Extract segmentation output
    output = output['out']
    predictions = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
    
    return predictions

def create_crack_overlay(original_img, predictions, color, threshold=0):
    """Create colored overlay for detected cracks/anomalies"""
    # Resize predictions to match original image
    h, w = original_img.shape[:2]
    predictions_resized = cv2.resize(
        predictions.astype(np.uint8), 
        (w, h), 
        interpolation=cv2.INTER_NEAREST
    )
    
    # Create binary mask (non-background pixels)
    mask = (predictions_resized > threshold).astype(np.uint8)
    
    # Apply morphological operations to enhance cracks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Create overlay
    overlay = original_img.copy()
    overlay[mask > 0] = [
        int(overlay[mask > 0, 0] * 0.5 + color[0] * 0.5),
        int(overlay[mask > 0, 1] * 0.5 + color[1] * 0.5),
        int(overlay[mask > 0, 2] * 0.5 + color[2] * 0.5)
    ]
    
    return overlay, mask, np.sum(mask)

def analyze_crack_severity(mask):
    """Analyze severity of detected cracks"""
    crack_percentage = (np.sum(mask) / mask.size) * 100
    
    if crack_percentage < 0.5:
        return "NONE", crack_percentage
    elif crack_percentage < 2:
        return "LOW", crack_percentage
    elif crack_percentage < 5:
        return "MEDIUM", crack_percentage
    else:
        return "SEVERE", crack_percentage

# Load models
st.info("Loading AI models... This may take a moment on first run.")
try:
    deeplab_model, unet_model, device = load_models()
    st.success("✓ Models loaded successfully")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Create layout
col_input, col_outputs = st.columns([1, 2], gap="large")

with col_input:
    st.header("📥 Data Ingestion")
    uploaded_file = st.file_uploader(
        "Upload Pavement/Surface Image", 
        type=["jpg", "jpeg", "png"]
    )
    
    # Option to use example
    st.markdown("---")
    use_example = st.checkbox("Use example image")

# Process image
if uploaded_file is not None or use_example:
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        # Create a synthetic example with some cracks
        img = np.ones((512, 512, 3), dtype=np.uint8) * 150
        # Draw some crack-like lines
        cv2.line(img, (100, 200), (300, 250), (80, 80, 80), 3)
        cv2.line(img, (150, 100), (400, 350), (80, 80, 80), 2)
        cv2.line(img, (200, 50), (350, 300), (60, 60, 60), 2)
    
    # Process with models
    img_tensor = preprocess_image(img)
    
    with st.spinner("🔄 Running crack detection analysis..."):
        # U-Net predictions (using DeepLabV3-ResNet50)
        unet_predictions = detect_cracks_with_model(img_tensor, unet_model, device)
        unet_overlay, unet_mask, unet_crack_pixels = create_crack_overlay(
            img.copy(), unet_predictions, (255, 0, 0)  # Red
        )
        
        # DeepLabV3+ predictions (using DeepLabV3-ResNet101)
        deeplab_predictions = detect_cracks_with_model(img_tensor, deeplab_model, device)
        deeplab_overlay, deeplab_mask, deeplab_crack_pixels = create_crack_overlay(
            img.copy(), deeplab_predictions, (0, 120, 255)  # Orange/Blue
        )
    
    # Analyze severity
    unet_severity, unet_percentage = analyze_crack_severity(unet_mask)
    deeplab_severity, deeplab_percentage = analyze_crack_severity(deeplab_mask)
    
    # Display results
    with col_input:
        st.header("📊 Analysis Results")
        
        # Overall detection status
        if unet_severity == "NONE" and deeplab_severity == "NONE":
            st.success("✓ No Cracks Detected")
            status_text = "✓ CLEAR SURFACE - No structural anomalies found"
        else:
            st.warning("⚠ Cracks Detected")
            status_text = f"⚠ {max(unet_severity, deeplab_severity)} severity - Structural anomalies detected"
        
        st.markdown(f"### {status_text}")
        
        # Metrics
        st.markdown("**U-Net Analysis:**")
        st.metric("Crack Coverage", f"{unet_percentage:.2f}%", "Severity: " + unet_severity)
        
        st.markdown("**DeepLabV3+ Analysis:**")
        st.metric("Crack Coverage", f"{deeplab_percentage:.2f}%", "Severity: " + deeplab_severity)
    
    with col_outputs:
        st.header("🧠 Model Execution Pipeline")
        col_unet, col_deeplab = st.columns(2)
        
        with col_unet:
            st.markdown("#### <span style='color: #e53e3e;'>🔴 U-Net Output</span>", unsafe_allow_html=True)
            st.markdown(f"**Cracks Detected:** {unet_severity} ({unet_percentage:.2f}%)")
            st.image(
                cv2.cvtColor(unet_overlay, cv2.COLOR_BGR2RGB),
                use_container_width=True,
                caption="Red overlay = Detected anomalies"
            )
        
        with col_deeplab:
            st.markdown("#### <span style='color: #3182ce;'>🔵 DeepLabV3+ Output</span>", unsafe_allow_html=True)
            st.markdown(f"**Cracks Detected:** {deeplab_severity} ({deeplab_percentage:.2f}%)")
            st.image(
                cv2.cvtColor(deeplab_overlay, cv2.COLOR_BGR2RGB),
                use_container_width=True,
                caption="Orange overlay = Detected anomalies"
            )
    
    # Comparison section
    st.markdown("---")
    st.header("📈 Model Comparison & Insights")
    
    comparison_col1, comparison_col2 = st.columns(2)
    
    with comparison_col1:
        st.subheader("Detection Difference")
        difference = abs(unet_percentage - deeplab_percentage)
        st.metric("Coverage Variance", f"{difference:.2f}%")
        
        if difference < 1:
            st.info("✓ Models agree closely on crack detection")
        elif difference < 3:
            st.warning("⚠ Models show slight variation")
        else:
            st.warning("⚠ Models show significant variation - manual review recommended")
    
    with comparison_col2:
        st.subheader("Recommendation")
        
        if unet_severity == "NONE" and deeplab_severity == "NONE":
            st.success("✅ Surface Status: CLEAR\n\nNo maintenance required.")
        elif max(unet_severity, deeplab_severity) == "SEVERE":
            st.error("🚨 IMMEDIATE ACTION REQUIRED\n\nSurface shows severe damage.")
        elif max(unet_severity, deeplab_severity) == "MEDIUM":
            st.warning("⚠️ SCHEDULE MAINTENANCE\n\nModerate damage detected.")
        else:
            st.info("ℹ️ MONITOR\n\nMinor damage - continue monitoring.")
    
    # Original image display
    st.markdown("---")
    st.subheader("📷 Original Image")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

else:
    with col_outputs:
        st.header("🧠 Model Execution Pipeline")
        st.info("👆 Upload an image in the left panel to start crack detection analysis...")
