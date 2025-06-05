# AI-image-detector-trainer
# a robust deep learning model to classify images as authentic or AI-generated, The core of the system is an ensemble architecture that fuses predictions from a custom CNN
 and a fine-tuned Vision Transformer (ViT-B/16).
 • Utilized PyTorch for model development, with a focus on advanced data augmentation techniques, learning rate
 scheduling, and comprehensive model evaluation to ensure generalization
 
# Core Libraries

PyTorch (for deep learning)

torchvision (for datasets, transforms, pretrained models including ViT)

Pillow (for image processing)

numpy (for numerical operations)

# Training and Utilities

tqdm (for progress bars)

matplotlib (for plotting, e.g., confusion matrices)

scikit-learn (for metrics like accuracy, precision, recall, F1-score)

# File structure
archive/
  Midjourney_Exp2/
    train/
      FAKE/
      REAL/
    valid/
      FAKE/
      REAL/
    test/
      FAKE/
      REAL/
model.py

predict.py

train.py

utils.py
