import streamlit as st
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
from pathlib import Path

# Set up the page layout and title
st.set_page_config(page_title="Crack Detection Dashboard", layout="wide")

st.title("🔍 Crack Detection Dashboard")
st.caption("Real-Time Crack Detection using Custom-Trained Deep Learning Models")
st.markdown("---")

# Initialize session state for model caching
@st.cache_resource
def load_custom_models():
    """Load custom-trained crack detection models"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint_dir = Path('./checkpoints')
    
    # Load custom trained model
    model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=2)
    
    # Try to load best model checkpoint
    best_model_path = checkpoint_dir / 'best_model.pth'
    
    if best_model_path.exists():
        try:
            checkpoint = torch.load(best_model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            st.success("✓ Custom trained model loaded successfully!")
        except Exception as e:
            st.warning(f"Could not load custom model: {e}")
            st.info("Using pre-trained ImageNet weights as fallback")
            model = models.segmentation.deeplabv3_resnet50(pretrained=True, num_classes=2)
    else:
        st.info("ℹ️ Custom model not found. Using pre-trained weights.")
        st.info("Run `python train.py` to train on Kaggle crack dataset.")
        model = models.segmentation.deeplabv3_resnet50(pretrained=True, num_classes=2)
    
    model.to(device)
    model.eval()
    
    # Load comparison model
    deeplab_model = models.segmentation.deeplabv3_resnet101(pretrained=True, num_classes=2)
    deeplab_model.to(device)
    deeplab_model.eval()
    
    return model, deeplab_model, device

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
    mask = cv2.dilate(kernel, iterations=1)
    
    # Create overlay
    overlay = original_img.copy().astype(float)
    overlay[mask > 0] = [
        overlay[mask > 0, 0] * 0.5 + color[0] * 0.5,
        overlay[mask > 0, 1] * 0.5 + color[1] * 0.5,
        overlay[mask > 0, 2] * 0.5 + color[2] * 0.5
    ]
    overlay = overlay.astype(np.uint8)
    
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
with st.spinner("Loading AI models... This may take a moment on first run."):
    try:
        custom_model, deeplab_model, device = load_custom_models()
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
        # Custom trained model predictions
        custom_predictions = detect_cracks_with_model(img_tensor, custom_model, device)
        custom_overlay, custom_mask, custom_crack_pixels = create_crack_overlay(
            img.copy(), custom_predictions, (255, 0, 0)  # Red
        )
        
        # DeepLabV3+ predictions (comparison)
        deeplab_predictions = detect_cracks_with_model(img_tensor, deeplab_model, device)
        deeplab_overlay, deeplab_mask, deeplab_crack_pixels = create_crack_overlay(
            img.copy(), deeplab_predictions, (0, 120, 255)  # Orange
        )
    
    # Analyze severity
    custom_severity, custom_percentage = analyze_crack_severity(custom_mask)
    deeplab_severity, deeplab_percentage = analyze_crack_severity(deeplab_mask)
    
    # Display results
    with col_input:
        st.header("📊 Analysis Results")
        
        # Overall detection status
        if custom_severity == "NONE" and deeplab_severity == "NONE":
            st.success("✓ No Cracks Detected")
            status_text = "✓ CLEAR SURFACE - No structural anomalies found"
        else:
            st.warning("⚠ Cracks Detected")
            status_text = f"⚠ {max(custom_severity, deeplab_severity)} severity - Structural anomalies detected"
        
        st.markdown(f"### {status_text}")
        
        # Metrics
        st.markdown("**Custom Trained Model:**")
        st.metric("Crack Coverage", f"{custom_percentage:.2f}%", "Severity: " + custom_severity)
        
        st.markdown("**DeepLabV3+ (Reference):**")
        st.metric("Crack Coverage", f"{deeplab_percentage:.2f}%", "Severity: " + deeplab_severity)
    
    with col_outputs:
        st.header("🧠 Model Execution Pipeline")
        col_custom, col_deeplab = st.columns(2)
        
        with col_custom:
            st.markdown("#### <span style='color: #e53e3e;'>🔴 Custom Trained Model</span>", unsafe_allow_html=True)
            st.markdown(f"**Cracks Detected:** {custom_severity} ({custom_percentage:.2f}%)")
            st.image(
                cv2.cvtColor(custom_overlay, cv2.COLOR_BGR2RGB),
                use_container_width=True,
                caption="Red overlay = Detected anomalies"
            )
        
        with col_deeplab:
            st.markdown("#### <span style='color: #3182ce;'>🔵 DeepLabV3+ (Reference)</span>", unsafe_allow_html=True)
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
        difference = abs(custom_percentage - deeplab_percentage)
        st.metric("Coverage Variance", f"{difference:.2f}%")
        
        if difference < 1:
            st.info("✓ Models agree closely on crack detection")
        elif difference < 3:
            st.warning("⚠ Models show slight variation")
        else:
            st.warning("⚠ Models show significant variation - manual review recommended")
    
    with comparison_col2:
        st.subheader("Recommendation")
        
        if custom_severity == "NONE" and deeplab_severity == "NONE":
            st.success("✅ Surface Status: CLEAR\n\nNo maintenance required.")
        elif max(custom_severity, deeplab_severity) == "SEVERE":
            st.error("🚨 IMMEDIATE ACTION REQUIRED\n\nSurface shows severe damage.")
        elif max(custom_severity, deeplab_severity) == "MEDIUM":
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
        
        st.markdown("---")
        st.info("""
        **Model Information:**
        - 🔴 Custom Trained Model: Fine-tuned on Kaggle Crack Segmentation Dataset
        - 🔵 DeepLabV3+: Pre-trained reference model for comparison
        
        **Training Details:**
        To train the custom model on your dataset, run:
        ```
        python train.py
        ```
        """)
