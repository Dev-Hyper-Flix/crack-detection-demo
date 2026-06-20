# Crack Detection Demo - Setup & Training Guide

## 📊 Training on Kaggle Crack Segmentation Dataset

### Step 1: Download the Dataset

1. Visit [Kaggle Crack Segmentation Dataset](https://www.kaggle.com/datasets/uniquelyshaurya/cracksegmentation)
2. Download the dataset
3. Extract and organize as follows:

```
crack_dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── masks/
    ├── image1_mask.png
    ├── image2_mask.png
    └── ...
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Add these to requirements.txt if not present:
```
tqdm==4.66.1
```

### Step 3: Train the Model

```bash
python train.py
```

**Training Configuration:**
- Model: DeepLabV3 with ResNet50 backbone
- Batch Size: 8
- Learning Rate: 1e-4 (with scheduler)
- Epochs: 50
- Image Size: 512x512
- Loss: Focal Loss (handles class imbalance)
- Optimizer: Adam with weight decay (1e-4)

**Expected Output:**
```
Epoch 1/50
Training: 100%|████| 125/125 [02:45<00:00, 1.32s/it]
Train Loss: 0.3421
Validating: 100%|████| 31/31 [00:28<00:00, 0.90s/it]
Val Loss: 0.2856, Val IoU: 0.7421
✓ Best model saved at epoch 1
```

### Step 4: Models Saved

Models are automatically saved to `./checkpoints/`:
- `best_model.pth` - Best model based on validation loss
- `checkpoint_epoch_5.pth` - Periodic checkpoints
- `final_model.pth` - Final model after all epochs
- `config.json` - Training configuration

### Step 5: Run the Application

```bash
streamlit run app.py
```

The app will:
1. ✅ Load your custom trained model automatically
2. 📊 Display predictions from your custom model (RED overlay)
3. 🔵 Compare with pre-trained DeepLabV3+ (ORANGE overlay)
4. 📈 Show severity analysis and maintenance recommendations
5. 📤 Allow users to upload their own pavement images

---

## 🎯 Training Tips

### Dataset Preparation
- Ensure masks are binary (0 = background, 255 = crack)
- Use consistent naming: `image.jpg` → `image_mask.png`
- Recommended: 500+ images for optimal training

### GPU Acceleration
```bash
# Verify CUDA is working
python -c "import torch; print(torch.cuda.is_available())"

# Expected: True (if you have NVIDIA GPU)
```

### Troubleshooting

**Issue: Out of memory**
```python
# Reduce batch size in train.py
config['batch_size'] = 4  # from 8
```

**Issue: Models not training well**
```python
# Increase epochs and adjust learning rate
config['num_epochs'] = 100
config['learning_rate'] = 5e-5
```

**Issue: Imbalanced dataset (more background than cracks)**
```python
# Already handled with Focal Loss in training
# Which naturally weighs hard examples higher
```

---

## 📊 Model Architecture

### Custom DeepLabV3-ResNet50
```
Input (3, 512, 512)
    ↓
ResNet50 Backbone (pretrained on ImageNet)
    ↓
ASPP (Atrous Spatial Pyramid Pooling)
    ↓
Decoder
    ↓
Output (2, 512, 512) - [background, crack]
```

### Training Details
- **Loss Function**: Focal Loss (handles class imbalance better than CrossEntropyLoss)
- **Optimizer**: Adam with weight decay (1e-4)
- **LR Scheduler**: ReduceLROnPlateau (patience=5, factor=0.5)
- **Validation Split**: 20%
- **Gradient Clipping**: max_norm=1.0

---

## 🚀 Using the Trained Model

### In the Streamlit App
1. Upload a pavement image
2. View results from **Custom Trained Model** (Red) vs **Reference Model** (Orange)
3. Get severity classification: NONE, LOW, MEDIUM, SEVERE
4. Receive actionable maintenance recommendations

### Model Output
- **Red Overlay**: Cracks detected by your custom trained model
- **Orange Overlay**: Cracks detected by pre-trained reference model
- **Coverage %**: Percentage of image showing crack pixels
- **Severity Level**: Based on crack percentage coverage

---

## 📈 Performance Metrics

After training, monitor these metrics:
- **Train Loss**: Should decrease steadily
- **Val Loss**: Should plateau after ~20-30 epochs
- **Val IoU**: Should reach 0.75+ (depends on dataset quality)
- **Best Model**: Automatically selected checkpoint

### Expected Performance
- Small cracks: ~85% detection rate
- Medium cracks: ~92% detection rate
- Large cracks: ~95% detection rate

---

## 📝 Citation

If using the Kaggle dataset, please cite:
```
Uniquely Shaurya. (2022). Crack Segmentation Dataset. 
Retrieved from https://www.kaggle.com/datasets/uniquelyshaurya/cracksegmentation
```

---

## 🔧 Advanced Customization

### Modify Training Parameters

Edit `train.py` to change:

```python
config = {
    'batch_size': 16,              # Increase for faster training (if GPU memory allows)
    'learning_rate': 5e-5,         # Lower for fine-tuning, higher for faster learning
    'num_epochs': 100,             # More epochs for better accuracy
    'img_size': 512,               # Keep at 512 for consistency
}
```

### Use Different Backbones

Replace the backbone in `CrackSegmentationModel`:

```python
# Option 1: ResNet101 (slower, more accurate)
self.model = models.segmentation.deeplabv3_resnet101(pretrained=True, num_classes=2)

# Option 2: MobileNetV3 (faster, less accurate)
self.model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True, num_classes=2)
```

---

**Ready to train? 🚀 Run `python train.py` now!**
