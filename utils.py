import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

def get_device():
    """Check and return the appropriate device for computation"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"CUDA Available: True")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU Device: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        device = torch.device('cpu')
        print("CUDA Available: False, using CPU")
    return device

def get_transforms(phase='train', image_size=224):
    """Get data transforms for different phases"""
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else: # 'val' or 'test'
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def create_data_loaders(data_dir, batch_size=32, num_workers=4, image_size=224):
    """Create data loaders for train, validation, and test sets"""
    
    train_transform = get_transforms('train', image_size)
    val_test_transform = get_transforms('val', image_size) # Same for val and test
    
    train_dataset = ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    val_dataset = ImageFolder(
        root=os.path.join(data_dir, 'valid'), # Assuming 'valid' for validation
        transform=val_test_transform
    )
    test_dataset = ImageFolder(
        root=os.path.join(data_dir, 'test'),
        transform=val_test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    if train_dataset.classes:
        print(f"Classes: {train_dataset.classes}")
        print(f"Class to index mapping: {train_dataset.class_to_idx}")
    
    return train_loader, val_loader, test_loader, train_dataset.class_to_idx

def calculate_metrics(y_true, y_pred, class_names=None): # Added class_names parameter
    """Calculate and return various evaluation metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    # Specify zero_division=0 to handle cases where a class might not be predicted
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm, class_names=None, title='Confusion Matrix'):
    """Plot confusion matrix"""
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])] # Default class names if None

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def save_model(model, optimizer, epoch, loss, path, scheduler=None): # Added scheduler
    """Save model checkpoint, including optimizer and scheduler state"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(checkpoint, path)
    print(f"Model checkpoint saved to {path} (Epoch: {epoch}, Loss: {loss:.4f})")

def load_checkpoint(model, optimizer, path, device, scheduler=None): # Renamed and added scheduler
    """Load model checkpoint, including optimizer and scheduler state"""
    if not os.path.exists(path):
        print(f"Checkpoint file not found: {path}")
        return None, None, None # Return None for epoch, optimizer, scheduler if file not found

    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    epoch = checkpoint.get('epoch', -1) # Default to -1 if epoch not in checkpoint
    loss = checkpoint.get('loss', float('inf')) # Default to infinity if loss not in checkpoint
    
    print(f"Model checkpoint loaded from {path}")
    print(f"Resuming from Epoch: {epoch}, Last Validation Loss: {loss:.4f}")
    
    return epoch, optimizer, scheduler
