import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import json
from datetime import datetime

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CrackSegmentationDataset(Dataset):
    """
    Dataset class for crack segmentation from Kaggle dataset
    Expected structure:
    dataset/
    ├── images/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── masks/
        ├── image1_mask.png
        ├── image2_mask.png
        └── ...
    """
    
    def __init__(self, image_dir, mask_dir, transform=None, img_size=512):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.transform = transform
        
        # Get list of image files
        self.image_files = sorted([f for f in self.image_dir.glob('*') 
                                   if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        print(f"Found {len(self.image_files)} images in {image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_dir / (img_path.stem + '_mask.png')
        
        # Read image
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Read mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")
        
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        # Normalize mask to 0 and 1
        mask = (mask > 127).astype(np.uint8)
        
        # Convert to tensors
        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(mask).long()
        
        # Normalize image using ImageNet statistics
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        image = normalize(image)
        
        return image, mask

class CrackSegmentationModel(nn.Module):
    """
    Custom U-Net style model for crack segmentation
    """
    
    def __init__(self, num_classes=2):
        super(CrackSegmentationModel, self).__init__()
        self.num_classes = num_classes
        
        # Use DeepLabV3 with ResNet50 backbone
        self.model = models.segmentation.deeplabv3_resnet50(
            pretrained=True,
            num_classes=num_classes
        )
    
    def forward(self, x):
        output = self.model(x)
        return output['out']

def dice_loss(pred, target, smooth=1e-5):
    """
    Dice loss for binary segmentation
    """
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal loss to handle class imbalance
    """
    ce_loss = nn.functional.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()

class CrackSegmentationTrainer:
    """
    Trainer class for crack segmentation model
    """
    
    def __init__(self, model, train_loader, val_loader, device, learning_rate=1e-4, 
                 weight_decay=1e-4, checkpoint_dir='./checkpoints'):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute loss (combination of focal loss and dice loss)
            loss = focal_loss(outputs, masks)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating")
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute loss
                loss = focal_loss(outputs, masks)
                total_loss += loss.item()
                
                # Compute IoU
                preds = torch.argmax(outputs, dim=1)
                iou = self.compute_iou(preds, masks)
                total_iou += iou
                
                pbar.set_postfix({'loss': loss.item(), 'iou': iou})
        
        avg_loss = total_loss / len(self.val_loader)
        avg_iou = total_iou / len(self.val_loader)
        self.val_losses.append(avg_loss)
        
        return avg_loss, avg_iou
    
    def compute_iou(self, pred, target, num_classes=2):
        """Compute Intersection over Union"""
        ious = []
        for c in range(num_classes):
            pred_c = (pred == c).float()
            target_c = (target == c).float()
            
            intersection = (pred_c * target_c).sum().item()
            union = (pred_c + target_c - pred_c * target_c).sum().item()
            
            iou = intersection / (union + 1e-6)
            ious.append(iou)
        
        return np.mean(ious)
    
    def train(self, num_epochs):
        """Train the model for num_epochs"""
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss, val_iou = self.validate()
            print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True)
                print(f"✓ Best model saved at epoch {epoch + 1}")
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)
        
        print(f"\n✓ Training completed!")
        print(f"Best model at epoch {self.best_epoch + 1} with loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pth'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pth'
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print(f"Checkpoint loaded from {checkpoint_path}")

def main():
    """Main training script"""
    
    # Configuration
    config = {
        'dataset_path': './crack_dataset',  # Update with your Kaggle dataset path
        'image_dir': './crack_dataset/images',
        'mask_dir': './crack_dataset/masks',
        'batch_size': 8,
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'img_size': 512,
        'train_split': 0.8,
        'val_split': 0.2,
        'checkpoint_dir': './checkpoints'
    }
    
    print("=" * 60)
    print("Crack Segmentation Model Training")
    print("=" * 60)
    print(f"Config: {json.dumps(config, indent=2)}")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists(config['image_dir']) or not os.path.exists(config['mask_dir']):
        print(f"\n❌ Dataset not found!")
        print(f"Expected structure:")
        print(f"  {config['dataset_path']}/")
        print(f"  ├── images/")
        print(f"  └── masks/")
        print(f"\nPlease download the Kaggle Crack Segmentation dataset:")
        print(f"  https://www.kaggle.com/datasets/uniquelyshaurya/cracksegmentation")
        return
    
    # Create dataset
    print("\n📦 Loading dataset...")
    dataset = CrackSegmentationDataset(
        image_dir=config['image_dir'],
        mask_dir=config['mask_dir'],
        img_size=config['img_size']
    )
    
    # Split dataset
    train_size = int(len(dataset) * config['train_split'])
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    print("\n🧠 Creating model...")
    model = CrackSegmentationModel(num_classes=2)
    print(f"✓ Model created on device: {device}")
    
    # Create trainer
    trainer = CrackSegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config['learning_rate'],
        checkpoint_dir=config['checkpoint_dir']
    )
    
    # Train model
    print("\n🚀 Starting training...")
    trainer.train(num_epochs=config['num_epochs'])
    
    # Save final model
    print("\n💾 Saving final model...")
    torch.save(model.state_dict(), Path(config['checkpoint_dir']) / 'final_model.pth')
    
    # Save config
    with open(Path(config['checkpoint_dir']) / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Training complete! Models saved to {config['checkpoint_dir']}")

if __name__ == '__main__':
    main()
