"""
Optimized Segmentation Training Script
Enhanced version with all best practices for maximum accuracy
Improvements:
- Larger batch size with gradient accumulation
- AdamW optimizer with cosine annealing
- Synchronized data augmentation
- Improved segmentation head architecture
- Mixed precision training
- Early stopping and checkpointing
- Enhanced metrics and logging
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import os
import torchvision
from tqdm import tqdm
import random
from torch.cuda.amp import autocast, GradScaler
import json
from datetime import datetime

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# ============================================================================
# Utility Functions
# ============================================================================

def save_image(img, filename):
    """Save an image tensor to file after denormalizing."""
    img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = (img * std + mean) * 255
    cv2.imwrite(filename, img[:, :, ::-1])


# ============================================================================
# Mask Conversion
# ============================================================================

# Mapping from raw pixel values to new class IDs
value_map = {
    0: 0,        # background
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    700: 6,      # Logs
    800: 7,      # Rocks
    7100: 8,     # Landscape
    10000: 9     # Sky
}
n_classes = len(value_map)


def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


# ============================================================================
# Synchronized Augmentation Transform
# ============================================================================

class SynchronizedAugmentation:
    """Apply the same random augmentation to both image and mask."""
    
    def __init__(self, h, w, is_train=True):
        self.h = h
        self.w = w
        self.is_train = is_train
        
    def __call__(self, image, mask):
        # Resize
        image = TF.resize(image, [self.h, self.w])
        mask = TF.resize(mask, [self.h, self.w], interpolation=InterpolationMode.NEAREST)
        
        if self.is_train:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # Random rotation (-15 to 15 degrees)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
            
            # Random vertical flip (for outdoor scenes)
            if random.random() > 0.7:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            
            # Color jitter (only for image)
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            if random.random() > 0.5:
                image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
            if random.random() > 0.5:
                image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))
        
        # Convert to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask) * 255
        
        # Normalize image
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return image, mask


# ============================================================================
# Dataset with Synchronized Augmentation
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, h, w, is_train=True):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = SynchronizedAugmentation(h, w, is_train=is_train)
        self.data_ids = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = convert_mask(mask)

        image, mask = self.transform(image, mask)

        return image, mask


# ============================================================================
# Improved Segmentation Head Architecture
# ============================================================================

class ImprovedSegmentationHead(nn.Module):
    """Enhanced segmentation head with BatchNorm, residual connections, and dropout."""
    
    def __init__(self, in_channels, out_channels, tokenW, tokenH, dropout=0.1):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        # Initial projection with more capacity
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Dropout2d(dropout)
        )

        # First depthwise separable block
        self.block1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Dropout2d(dropout)
        )
        
        # Second depthwise separable block
        self.block2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Dropout2d(dropout)
        )
        
        # Final refinement
        self.refine = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        # Classification head
        self.classifier = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        
        x = self.stem(x)
        
        # Residual connections for better gradient flow
        identity = x
        x = self.block1(x)
        x = x + identity  # Skip connection
        
        identity = x
        x = self.block2(x)
        x = x + identity  # Skip connection
        
        x = self.refine(x)
        x = self.classifier(x)
        
        return x


# ============================================================================
# Advanced Loss Functions
# ============================================================================

class CombinedLoss(nn.Module):
    """Combines Cross Entropy with Dice Loss for better boundary segmentation."""
    
    def __init__(self, weight_ce=1.0, weight_dice=0.5, smooth=1e-6):
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.smooth = smooth
        self.ce_loss = nn.CrossEntropyLoss()
    
    def dice_loss(self, pred, target, num_classes=10):
        """Compute Dice loss."""
        pred = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        
        intersection = (pred * target_onehot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()
    
    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target, num_classes=n_classes)
        return self.weight_ce * ce + self.weight_dice * dice


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10, ignore_index=255):
    """Compute IoU for each class and return mean IoU."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    iou_per_class = []
    for class_id in range(num_classes):
        if class_id == ignore_index:
            continue

        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().numpy())

    return np.nanmean(iou_per_class), iou_per_class


def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    """Compute Dice coefficient (F1 Score) per class and return mean Dice Score."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    dice_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        dice_score = (2. * intersection + smooth) / (pred_inds.sum().float() + target_inds.sum().float() + smooth)

        dice_per_class.append(dice_score.cpu().numpy())

    return np.mean(dice_per_class), dice_per_class


def compute_pixel_accuracy(pred, target):
    """Compute pixel accuracy."""
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().cpu().numpy()


def evaluate_metrics(model, backbone, data_loader, device, num_classes=10, show_progress=True):
    """Evaluate all metrics on a dataset."""
    iou_scores = []
    dice_scores = []
    pixel_accuracies = []
    per_class_iou = [[] for _ in range(num_classes)]

    model.eval()
    backbone.eval()
    
    loader = tqdm(data_loader, desc="Evaluating", leave=False, unit="batch") if show_progress else data_loader
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            with autocast():
                output = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = model(output)
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

            labels = labels.squeeze(dim=1).long()

            iou, iou_per_class = compute_iou(outputs, labels, num_classes=num_classes)
            dice, _ = compute_dice(outputs, labels, num_classes=num_classes)
            pixel_acc = compute_pixel_accuracy(outputs, labels)

            iou_scores.append(iou)
            dice_scores.append(dice)
            pixel_accuracies.append(pixel_acc)
            
            for i, iou_val in enumerate(iou_per_class):
                if not np.isnan(iou_val):
                    per_class_iou[i].append(iou_val)

    model.train()
    
    # Calculate per-class averages
    per_class_avg = [np.mean(scores) if scores else 0.0 for scores in per_class_iou]
    
    return np.mean(iou_scores), np.mean(dice_scores), np.mean(pixel_accuracies), per_class_avg


# ============================================================================
# Plotting Functions
# ============================================================================

def save_training_plots(history, output_dir):
    """Save all training metric plots to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Loss curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train', linewidth=2)
    plt.plot(history['val_loss'], label='Validation', linewidth=2)
    plt.title('Loss Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_iou'], label='Train IoU', linewidth=2)
    plt.plot(history['val_iou'], label='Val IoU', linewidth=2)
    plt.title('IoU Score Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('IoU Score', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['train_pixel_acc'], label='Train Acc', linewidth=2)
    plt.plot(history['val_pixel_acc'], label='Val Acc', linewidth=2)
    plt.title('Pixel Accuracy Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Dice score
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_dice'], label='Train Dice', linewidth=2, marker='o', markersize=4)
    plt.plot(history['val_dice'], label='Val Dice', linewidth=2, marker='s', markersize=4)
    plt.title('Dice Score Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Dice Score', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dice_score.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Learning rate schedule
    if 'learning_rate' in history:
        plt.figure(figsize=(10, 6))
        plt.plot(history['learning_rate'], linewidth=2, color='red')
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'learning_rate.png'), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Saved training plots to '{output_dir}'")


def save_history_to_file(history, output_dir, class_names=None):
    """Save training history to a text file with detailed statistics."""
    filepath = os.path.join(output_dir, 'training_history.txt')
    
    if class_names is None:
        class_names = ['Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
                      'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky']
    
    with open(filepath, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("OPTIMIZED SEGMENTATION TRAINING RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total epochs trained: {len(history['train_loss'])}\n\n")
        
        f.write("BEST VALIDATION METRICS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Best Val IoU:      {max(history['val_iou']):.4f} (Epoch {np.argmax(history['val_iou']) + 1})\n")
        f.write(f"  Best Val Dice:     {max(history['val_dice']):.4f} (Epoch {np.argmax(history['val_dice']) + 1})\n")
        f.write(f"  Best Val Accuracy: {max(history['val_pixel_acc']):.4f} (Epoch {np.argmax(history['val_pixel_acc']) + 1})\n")
        f.write(f"  Lowest Val Loss:   {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss']) + 1})\n")
        f.write("=" * 80 + "\n\n")
        
        # Per-class IoU for best epoch
        if 'per_class_iou' in history and history['per_class_iou']:
            best_epoch = np.argmax(history['val_iou'])
            f.write("PER-CLASS IoU AT BEST EPOCH:\n")
            f.write("-" * 50 + "\n")
            for i, (name, iou) in enumerate(zip(class_names, history['per_class_iou'][best_epoch])):
                f.write(f"  {name:<20}: {iou:.4f}\n")
            f.write("\n")

        f.write("PER-EPOCH HISTORY:\n")
        f.write("-" * 100 + "\n")
        headers = ['Epoch', 'Train Loss', 'Val Loss', 'Train IoU', 'Val IoU',
                   'Train Dice', 'Val Dice', 'Train Acc', 'Val Acc', 'LR']
        f.write("{:<8} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}\n".format(*headers))
        f.write("-" * 100 + "\n")

        n_epochs = len(history['train_loss'])
        for i in range(n_epochs):
            lr = history['learning_rate'][i] if 'learning_rate' in history else 0.0
            f.write("{:<8} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.2e}\n".format(
                i + 1,
                history['train_loss'][i],
                history['val_loss'][i],
                history['train_iou'][i],
                history['val_iou'][i],
                history['train_dice'][i],
                history['val_dice'][i],
                history['train_pixel_acc'][i],
                history['val_pixel_acc'][i],
                lr
            ))

    # Also save as JSON for easier parsing
    json_path = os.path.join(output_dir, 'training_history.json')
    with open(json_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Saved training history to '{filepath}' and '{json_path}'")


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("OPTIMIZED SEGMENTATION TRAINING")
    print("=" * 80)
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("=" * 80 + "\n")

    # Hyperparameters (OPTIMIZED)
    batch_size = 8  # Increased from 2 for stable gradients
    accumulation_steps = 2  # Effective batch size = 8 * 2 = 16
    w = int(((960 / 2) // 14) * 14)
    h = int(((540 / 2) // 14) * 14)
    lr = 3e-4  # Optimized for AdamW with this batch size
    n_epochs = 50  # Increased from 10 for full convergence
    weight_decay = 0.01
    dropout = 0.15
    
    # Early stopping parameters
    patience = 15
    min_delta = 0.0001  # Minimum improvement to reset patience

    print("TRAINING CONFIGURATION:")
    print("-" * 80)
    print(f"  Batch size:            {batch_size}")
    print(f"  Accumulation steps:    {accumulation_steps}")
    print(f"  Effective batch size:  {batch_size * accumulation_steps}")
    print(f"  Image size:            {h} x {w}")
    print(f"  Initial learning rate: {lr}")
    print(f"  Weight decay:          {weight_decay}")
    print(f"  Dropout:               {dropout}")
    print(f"  Max epochs:            {n_epochs}")
    print(f"  Early stopping patience: {patience}")
    print("=" * 80 + "\n")

    # Output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'train_stats_optimized')
    os.makedirs(output_dir, exist_ok=True)

    # Dataset paths
    data_dir = os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir = os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'val')

    # Create datasets with synchronized augmentation
    print("Loading datasets...")
    trainset = MaskDataset(data_dir=data_dir, h=h, w=w, is_train=True)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True, persistent_workers=True)

    valset = MaskDataset(data_dir=val_dir, h=h, w=w, is_train=False)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True, persistent_workers=True)

    print(f"Training samples: {len(trainset)}")
    print(f"Validation samples: {len(valset)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}\n")

    # Load DINOv2 backbone
    print("Loading DINOv2 backbone...")
    BACKBONE_SIZE = "small"  # Can be changed to "base", "large", or "giant"
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"

    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone_model.eval()
    backbone_model.to(device)
    
    # Freeze backbone
    for param in backbone_model.parameters():
        param.requires_grad = False
    
    print(f"Backbone: {backbone_name} (frozen)")

    # Get embedding dimension
    imgs, _ = next(iter(train_loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
    n_embedding = output.shape[2]
    print(f"Embedding dimension: {n_embedding}")
    print(f"Patch tokens shape: {output.shape}\n")

    # Create improved segmentation head
    print("Creating improved segmentation head...")
    classifier = ImprovedSegmentationHead(
        in_channels=n_embedding,
        out_channels=n_classes,
        tokenW=w // 14,
        tokenH=h // 14,
        dropout=dropout
    )
    classifier = classifier.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    # Loss function (Combined CE + Dice)
    loss_fct = CombinedLoss(weight_ce=1.0, weight_dice=0.5)
    
    # Optimizer (AdamW - better than SGD)
    optimizer = optim.AdamW(
        classifier.parameters(), 
        lr=lr, 
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler (Cosine Annealing with Warmup)
    warmup_epochs = 3
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=n_epochs,
        steps_per_epoch=len(train_loader) // accumulation_steps,
        pct_start=warmup_epochs/n_epochs,
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100
    )
    
    # Mixed precision training
    scaler = GradScaler()

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_iou': [],
        'val_iou': [],
        'train_dice': [],
        'val_dice': [],
        'train_pixel_acc': [],
        'val_pixel_acc': [],
        'learning_rate': [],
        'per_class_iou': []
    }

    # Early stopping variables
    best_val_iou = 0.0
    patience_counter = 0
    best_epoch = 0
    best_model_path = os.path.join(output_dir, 'best_model.pth')

    # Training loop
    print("\nStarting training...")
    print("=" * 80)

    try:
        epoch_pbar = tqdm(range(n_epochs), desc="Training Progress", unit="epoch", position=0)
        for epoch in epoch_pbar:
            # Training phase
            classifier.train()
            train_losses = []

            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", 
                             leave=False, unit="batch", position=1)
            
            optimizer.zero_grad()
            for batch_idx, (imgs, labels) in enumerate(train_pbar):
                imgs, labels = imgs.to(device), labels.to(device)

                # Mixed precision forward pass
                with autocast():
                    with torch.no_grad():
                        output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]

                    logits = classifier(output)
                    outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
                    labels_long = labels.squeeze(dim=1).long()
                    loss = loss_fct(outputs, labels_long)
                    loss = loss / accumulation_steps  # Scale loss for accumulation

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

                # Update weights every accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                train_losses.append(loss.item() * accumulation_steps)
                train_pbar.set_postfix(loss=f"{loss.item() * accumulation_steps:.4f}")

            # Handle remaining gradients
            if (batch_idx + 1) % accumulation_steps != 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Validation phase
            classifier.eval()
            val_losses = []

            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]", 
                           leave=False, unit="batch", position=1)
            with torch.no_grad():
                for imgs, labels in val_pbar:
                    imgs, labels = imgs.to(device), labels.to(device)

                    with autocast():
                        output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
                        logits = classifier(output)
                        outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
                        labels_long = labels.squeeze(dim=1).long()
                        loss = loss_fct(outputs, labels_long)

                    val_losses.append(loss.item())
                    val_pbar.set_postfix(loss=f"{loss.item():.4f}")

            # Calculate metrics
            print(f"\n  Computing training metrics...", end=" ")
            train_iou, train_dice, train_pixel_acc, _ = evaluate_metrics(
                classifier, backbone_model, train_loader, device, num_classes=n_classes, show_progress=False
            )
            print("✓")
            
            print(f"  Computing validation metrics...", end=" ")
            val_iou, val_dice, val_pixel_acc, per_class_iou = evaluate_metrics(
                classifier, backbone_model, val_loader, device, num_classes=n_classes, show_progress=False
            )
            print("✓")

            # Store history
            epoch_train_loss = np.mean(train_losses)
            epoch_val_loss = np.mean(val_losses)
            current_lr = optimizer.param_groups[0]['lr']

            history['train_loss'].append(epoch_train_loss)
            history['val_loss'].append(epoch_val_loss)
            history['train_iou'].append(train_iou)
            history['val_iou'].append(val_iou)
            history['train_dice'].append(train_dice)
            history['val_dice'].append(val_dice)
            history['train_pixel_acc'].append(train_pixel_acc)
            history['val_pixel_acc'].append(val_pixel_acc)
            history['learning_rate'].append(current_lr)
            history['per_class_iou'].append(per_class_iou)

            # Print epoch summary
            print(f"\n  Epoch {epoch+1} Summary:")
            print(f"    Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
            print(f"    Train IoU:  {train_iou:.4f} | Val IoU:  {val_iou:.4f}")
            print(f"    Train Dice: {train_dice:.4f} | Val Dice: {val_dice:.4f}")
            print(f"    Train Acc:  {train_pixel_acc:.4f} | Val Acc:  {val_pixel_acc:.4f}")
            print(f"    Learning Rate: {current_lr:.2e}")

            # Update progress bar
            epoch_pbar.set_postfix(
                val_loss=f"{epoch_val_loss:.3f}",
                val_iou=f"{val_iou:.3f}",
                val_acc=f"{val_pixel_acc:.3f}",
                best_iou=f"{best_val_iou:.3f}"
            )

            # Early stopping and model checkpointing
            if val_iou > best_val_iou + min_delta:
                improvement = val_iou - best_val_iou
                best_val_iou = val_iou
                best_epoch = epoch + 1
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_iou': best_val_iou,
                    'history': history,
                }, best_model_path)
                print(f"    ✓ New best model! Val IoU improved by {improvement:.4f} → Saved to '{best_model_path}'")
            else:
                patience_counter += 1
                print(f"    ⚠ No improvement for {patience_counter}/{patience} epochs")
                
                if patience_counter >= patience:
                    print(f"\n  Early stopping triggered! Best Val IoU: {best_val_iou:.4f} at epoch {best_epoch}")
                    break

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'history': history,
                }, checkpoint_path)
                print(f"    Checkpoint saved: {checkpoint_path}")

            print()  # Empty line for readability

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print(f"Best model saved at epoch {best_epoch} with Val IoU: {best_val_iou:.4f}")

    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(best_model_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} with Val IoU: {checkpoint['best_val_iou']:.4f}")

    # Save plots and history
    print("\nGenerating training visualizations...")
    save_training_plots(history, output_dir)
    save_history_to_file(history, output_dir)

    # Save final model
    final_model_path = os.path.join(script_dir, "segmentation_head_optimized.pth")
    torch.save(classifier.state_dict(), final_model_path)
    print(f"Saved final model to '{final_model_path}'")

    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nBest Results (Epoch {best_epoch}):")
    print(f"  Val IoU:      {best_val_iou:.4f}")
    print(f"  Val Dice:     {max(history['val_dice']):.4f}")
    print(f"  Val Accuracy: {max(history['val_pixel_acc']):.4f}")
    print(f"  Lowest Val Loss: {min(history['val_loss']):.4f}")
    
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"Best model saved to: {best_model_path}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
