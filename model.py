import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights # New import for ViT

class AIDetectionCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(AIDetectionCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second convolutional block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third convolutional block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fourth convolutional block
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        # Third block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        
        # Fourth block
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool4(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1) # Flatten
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x) # Output logits
        
        return x

# Alternative lightweight model for faster training
class AIDetectionResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AIDetectionResNet, self).__init__()
        from torchvision.models import resnet18, ResNet18_Weights # Updated for newer torchvision
        
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) # Use pre-trained weights
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

# --- New Ensemble Model ---
class EnsembleModel(nn.Module):
    def __init__(self, cnn_model, vit_model, num_classes=2):
        super(EnsembleModel, self).__init__()
        self.cnn_model = cnn_model
        self.vit_model = vit_model
        # Both sub-models should be configured to output `num_classes` logits.
        # We will average their logits in the forward pass.

    def forward(self, x):
        # Get logits from CNN
        # Make a copy of x if transforms are different or models modify input in-place,
        # but standard models shouldn't.
        # For ViT, input size must match. Ensure transforms handle this.
        # Your current get_transforms resizes to image_size (e.g., 224), which is standard for ViT.

        logits_cnn = self.cnn_model(x)
        
        # Get logits from ViT
        logits_vit = self.vit_model(x)
        
        # Simple averaging of logits
        # Ensure both logits_cnn and logits_vit have the same shape, e.g., (batch_size, num_classes)
        avg_logits = (logits_cnn + logits_vit) / 2.0
        
        return avg_logits

