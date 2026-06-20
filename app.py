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
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Create overlay with better blending
    overlay = original_img.copy().astype(float)
    overlay[mask > 0] = [
        overlay[mask > 0, 0] * 0.4 + color[0] * 0.6,
        overlay[mask > 0, 1] * 0.4 + color[1] * 0.6,
        overlay[mask > 0, 2] * 0.4 + color[2] * 0.6
    ]
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
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

# Load models with status
with st.spinner("Loading AI models... This may take a moment on first run."):
    try:
        custom_model, deeplab_model, device = load_custom_models()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# Create layout
col_input, col_info = st.columns([2, 1], gap="medium")

with col_input:
    st.header("📥 Image Upload")
    st.markdown("Upload a pavement image to detect cracks")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

with col_info:
    st.header("ℹ️ About")
    st.markdown("""
    **Custom Trained Model**
    - DeepLabV3-ResNet50
    - Trained on Kaggle dataset
    
    **Reference Model**
    - DeepLabV3-ResNet101
    - Pre-trained on ImageNet
    """)

# Process image
if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if original_img is None:
        st.error("❌ Could not read the image file. Please try another image.")
    else:
        # Store original image for later display
        original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Preprocess for models
        img_tensor = preprocess_image(original_img)
        
        # Show processing status
        with st.spinner("🔄 Running crack detection analysis on both models..."):
            # Custom trained model predictions
            custom_predictions = detect_cracks_with_model(img_tensor, custom_model, device)
            custom_overlay, custom_mask, custom_crack_pixels = create_crack_overlay(
                original_img.copy(), custom_predictions, (255, 0, 0)  # Red
            )
            custom_overlay_rgb = cv2.cvtColor(custom_overlay, cv2.COLOR_BGR2RGB)
            
            # DeepLabV3+ predictions (comparison)
            deeplab_predictions = detect_cracks_with_model(img_tensor, deeplab_model, device)
            deeplab_overlay, deeplab_mask, deeplab_crack_pixels = create_crack_overlay(
                original_img.copy(), deeplab_predictions, (0, 120, 255)  # Orange
            )
            deeplab_overlay_rgb = cv2.cvtColor(deeplab_overlay, cv2.COLOR_BGR2RGB)
        
        # Analyze severity
        custom_severity, custom_percentage = analyze_crack_severity(custom_mask)
        deeplab_severity, deeplab_percentage = analyze_crack_severity(deeplab_mask)
        
        # Display results section
        st.markdown("---")
        st.header("📊 Analysis Results")
        
        # Overall detection status
        col_status1, col_status2 = st.columns(2)
        
        with col_status1:
            st.subheader("🔴 Custom Trained Model")
            if custom_severity == "NONE":
                st.success(f"✓ {custom_severity}")
            elif custom_severity == "LOW":
                st.info(f"⚠ {custom_severity}")
            elif custom_severity == "MEDIUM":
                st.warning(f"⚠ {custom_severity}")
            else:
                st.error(f"🚨 {custom_severity}")
            
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric("Coverage", f"{custom_percentage:.2f}%")
            with col_metric2:
                st.metric("Severity", custom_severity)
        
        with col_status2:
            st.subheader("🔵 DeepLabV3+ Reference")
            if deeplab_severity == "NONE":
                st.success(f"✓ {deeplab_severity}")
            elif deeplab_severity == "LOW":
                st.info(f"⚠ {deeplab_severity}")
            elif deeplab_severity == "MEDIUM":
                st.warning(f"⚠ {deeplab_severity}")
            else:
                st.error(f"🚨 {deeplab_severity}")
            
            col_metric3, col_metric4 = st.columns(2)
            with col_metric3:
                st.metric("Coverage", f"{deeplab_percentage:.2f}%")
            with col_metric4:
                st.metric("Severity", deeplab_severity)
        
        # Display model outputs side by side
        st.markdown("---")
        st.header("🧠 Model Detection Output")
        
        col_output1, col_output2 = st.columns(2, gap="large")
        
        with col_output1:
            st.markdown("### <span style='color: #e53e3e;'>🔴 Custom Trained Model Output</span>", unsafe_allow_html=True)
            st.markdown(f"**Red Overlay** indicates detected cracks")
            st.image(
                custom_overlay_rgb,
                use_container_width=True,
                caption=f"Crack Coverage: {custom_percentage:.2f}% | Severity: {custom_severity}"
            )
        
        with col_output2:
            st.markdown("### <span style='color: #3182ce;'>🔵 DeepLabV3+ Reference Output</span>", unsafe_allow_html=True)
            st.markdown(f"**Orange Overlay** indicates detected cracks")
            st.image(
                deeplab_overlay_rgb,
                use_container_width=True,
                caption=f"Crack Coverage: {deeplab_percentage:.2f}% | Severity: {deeplab_severity}"
            )
        
        # Comparison and recommendations
        st.markdown("---")
        st.header("📈 Model Comparison & Recommendations")
        
        comparison_col1, comparison_col2 = st.columns(2, gap="medium")
        
        with comparison_col1:
            st.subheader("🔍 Detection Variance")
            difference = abs(custom_percentage - deeplab_percentage)
            st.metric("Coverage Difference", f"{difference:.2f}%")
            
            if difference < 1:
                st.success("✓ Models agree closely - high confidence in detection")
            elif difference < 3:
                st.info("⚠ Models show minor variation - good agreement")
            else:
                st.warning("⚠ Models show significant variation - manual review recommended")
        
        with comparison_col2:
            st.subheader("✅ Maintenance Recommendation")
            
            max_severity = max(custom_severity, deeplab_severity)
            
            if custom_severity == "NONE" and deeplab_severity == "NONE":
                st.success("✅ **CLEAR SURFACE**\n\nNo maintenance required.")
            elif max_severity == "SEVERE":
                st.error("🚨 **IMMEDIATE ACTION REQUIRED**\n\nSurface shows severe damage. Schedule urgent repairs.")
            elif max_severity == "MEDIUM":
                st.warning("⚠️ **SCHEDULE MAINTENANCE**\n\nModerate damage detected. Plan repairs within 30 days.")
            else:
                st.info("ℹ️ **MONITOR**\n\nMinor damage detected. Continue monitoring and schedule routine maintenance.")
        
        # Display original image
        st.markdown("---")
        st.header("📷 Original Image")
        st.image(
            original_img_rgb,
            use_container_width=True,
            caption=f"Uploaded Image (Size: {original_img_rgb.shape[1]}x{original_img_rgb.shape[0]}px)"
        )

else:
    # Show placeholder when no image is uploaded
    col_empty1, col_empty2 = st.columns(2)
    
    with col_empty1:
        st.info("👆 **Upload an image in the left panel** to start crack detection analysis")
    
    with col_empty2:
        if st.checkbox("Use example image for demo", key="use_example"):
            # Create a synthetic example with some cracks
            example_img = np.ones((512, 512, 3), dtype=np.uint8) * 150
            # Draw some crack-like lines
            cv2.line(example_img, (100, 200), (300, 250), (80, 80, 80), 3)
            cv2.line(example_img, (150, 100), (400, 350), (80, 80, 80), 2)
            cv2.line(example_img, (200, 50), (350, 300), (60, 60, 60), 2)
            cv2.line(example_img, (250, 400), (450, 100), (70, 70, 70), 2)
            
            # Preprocess for models
            img_tensor = preprocess_image(example_img)
            
            with st.spinner("🔄 Running crack detection on example image..."):
                # Custom trained model predictions
                custom_predictions = detect_cracks_with_model(img_tensor, custom_model, device)
                custom_overlay, custom_mask, custom_crack_pixels = create_crack_overlay(
                    example_img.copy(), custom_predictions, (255, 0, 0)  # Red
                )
                custom_overlay_rgb = cv2.cvtColor(custom_overlay, cv2.COLOR_BGR2RGB)
                
                # DeepLabV3+ predictions (comparison)
                deeplab_predictions = detect_cracks_with_model(img_tensor, deeplab_model, device)
                deeplab_overlay, deeplab_mask, deeplab_crack_pixels = create_crack_overlay(
                    example_img.copy(), deeplab_predictions, (0, 120, 255)  # Orange
                )
                deeplab_overlay_rgb = cv2.cvtColor(deeplab_overlay, cv2.COLOR_BGR2RGB)
            
            # Analyze severity
            custom_severity, custom_percentage = analyze_crack_severity(custom_mask)
            deeplab_severity, deeplab_percentage = analyze_crack_severity(deeplab_mask)
            
            st.markdown("---")
            st.header("📊 Example Analysis Results")
            
            col_status1, col_status2 = st.columns(2)
            
            with col_status1:
                st.subheader("🔴 Custom Trained Model")
                st.metric("Coverage", f"{custom_percentage:.2f}%")
                st.metric("Severity", custom_severity)
            
            with col_status2:
                st.subheader("🔵 DeepLabV3+ Reference")
                st.metric("Coverage", f"{deeplab_percentage:.2f}%")
                st.metric("Severity", deeplab_severity)
            
            st.markdown("---")
            st.header("🧠 Example Model Detection Output")
            
            col_output1, col_output2 = st.columns(2, gap="large")
            
            with col_output1:
                st.markdown("### <span style='color: #e53e3e;'>🔴 Custom Trained Model</span>", unsafe_allow_html=True)
                st.image(custom_overlay_rgb, use_container_width=True, caption=f"Coverage: {custom_percentage:.2f}%")
            
            with col_output2:
                st.markdown("### <span style='color: #3182ce;'>🔵 DeepLabV3+ Reference</span>", unsafe_allow_html=True)
                st.image(deeplab_overlay_rgb, use_container_width=True, caption=f"Coverage: {deeplab_percentage:.2f}%")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    <p>Crack Detection Dashboard | Powered by DeepLabV3 and PyTorch</p>
    <p>For training on custom datasets, run: <code>python train.py</code></p>
</div>
""", unsafe_allow_html=True)
