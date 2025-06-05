import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
import argparse # For command-line arguments

from model import AIDetectionCNN, AIDetectionResNet, EnsembleModel 
from torchvision.models import vit_b_16, ViT_B_16_Weights
# Updated import for the modified load_checkpoint function
from utils import get_device, create_data_loaders, calculate_metrics, save_model, load_checkpoint, plot_confusion_matrix

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for data, targets in progress_bar:
            data, targets = data.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * data.size(0) # Weighted by batch size
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}', # Current batch loss
                'Acc': f'{100. * correct_predictions / total_samples:.2f}%'
            })
        
        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in tqdm(self.val_loader, desc="Validation"):
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item() * data.size(0) # Weighted by batch size
                _, predicted = torch.max(outputs.data, 1)
                total_samples += targets.size(0)
                correct_predictions += (predicted == targets).sum().item()
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        return epoch_loss, epoch_acc, all_predictions, all_targets
    
    # Modified train method to accept start_epoch and num_epochs
    def train(self, num_epochs, save_path='best_model.pth', early_stopping_patience=10, start_epoch=0):
        best_val_loss = float('inf') # Track best validation loss for saving
        patience_counter = 0
        
        print(f"Starting training on {self.device} from epoch {start_epoch + 1}")
        print("-" * 50)
        
        # Adjust epochs based on start_epoch
        for epoch_idx in range(num_epochs): # Iterate for the number of additional epochs
            current_epoch = start_epoch + epoch_idx # Actual epoch number
            start_time = time.time()
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, _, _ = self.validate_epoch()
            
            if self.scheduler:
                # For ReduceLROnPlateau, step with validation loss
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else: # For other schedulers, step per epoch
                    self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch [{current_epoch + 1}/{start_epoch + num_epochs}] - Time: {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Current Learning Rate: {current_lr:.6f}")
            
            if val_loss < best_val_loss: # Save based on validation loss
                best_val_loss = val_loss
                # Pass the scheduler to save_model
                save_model(self.model, self.optimizer, current_epoch, val_loss, save_path, scheduler=self.scheduler)
                patience_counter = 0
                print(f"New best validation loss: {best_val_loss:.4f}. Model saved.")
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {current_epoch + 1} epochs due to no improvement in validation loss.")
                break
            print("-" * 50)
        
        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        return self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies
    
    def plot_training_history(self, start_epoch=0):
        epochs_range = range(start_epoch, start_epoch + len(self.train_losses))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(epochs_range, self.train_losses, label='Training Loss')
        ax1.plot(epochs_range, self.val_losses, label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(epochs_range, self.train_accuracies, label='Training Accuracy')
        ax2.plot(epochs_range, self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="AI Image Detection Model Training")
    parser.add_argument('--data_dir', type=str, default='archive/Midjourney_Exp2', help='Directory for the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--initial_lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train for a new run')
    parser.add_argument('--image_size', type=int, default=224, help='Image size for transforms')
    parser.add_argument('--model_type', type=str, default='cnn', choices=['cnn', 'resnet', 'ensemble_cnn_vit'], help='Type of model architecture')
    parser.add_argument('--save_path', type=str, default='best_ai_detection_model.pth', help='Path to save the best model')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Patience for early stopping')
    
    # Arguments for resuming training
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to checkpoint to load and continue training')
    parser.add_argument('--continue_epochs', type=int, default=20, help='Number of additional epochs to train when resuming')
    parser.add_argument('--new_lr', type=float, default=None, help='New learning rate to use when resuming training')

    args = parser.parse_args()
    
    device = get_device()
    
    train_loader, val_loader, test_loader, class_to_idx = create_data_loaders(
        args.data_dir, args.batch_size, image_size=args.image_size
    )
    num_classes = len(class_to_idx)

    # --- Model Initialization ---
    if args.model_type == 'cnn':
        model = AIDetectionCNN(num_classes=num_classes)
    elif args.model_type == 'resnet':
        model = AIDetectionResNet(num_classes=num_classes)
    elif args.model_type == 'ensemble_cnn_vit':
        cnn_part = AIDetectionCNN(num_classes=num_classes)
        vit_part = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        vit_part.heads.head = nn.Linear(vit_part.heads.head.in_features, num_classes)
        model = EnsembleModel(cnn_model=cnn_part, vit_model=vit_part, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")
    model = model.to(device)

    # --- Optimizer and Scheduler ---
    # Use initial_lr if not resuming or if new_lr is not specified for resume
    current_learning_rate = args.initial_lr 
    optimizer = optim.Adam(model.parameters(), lr=current_learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)

    start_epoch = 0 # Default start epoch

    # --- Load Checkpoint if specified ---
    if args.load_checkpoint:
        print(f"Attempting to load checkpoint from: {args.load_checkpoint}")
        # Pass model, optimizer, and scheduler to load_checkpoint
        loaded_epoch, optimizer, scheduler = load_checkpoint(
            model, optimizer, args.load_checkpoint, device, scheduler=scheduler
        )
        if loaded_epoch is not None: # Check if loading was successful
             start_epoch = loaded_epoch + 1 # Start from the next epoch
             # If a new LR is specified for resuming, set it AFTER loading optimizer state
             if args.new_lr is not None:
                 print(f"Overriding learning rate to: {args.new_lr}")
                 for param_group in optimizer.param_groups:
                     param_group['lr'] = args.new_lr
                 current_learning_rate = args.new_lr # Update for logging
             else: # Use the LR from the loaded optimizer
                 current_learning_rate = optimizer.param_groups[0]['lr']
             print(f"Resuming training with LR: {current_learning_rate}")
        else:
            print(f"Could not load checkpoint. Starting training from scratch with LR: {args.initial_lr}")
            # Ensure optimizer and scheduler are re-initialized if loading failed
            optimizer = optim.Adam(model.parameters(), lr=args.initial_lr, weight_decay=1e-5)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)
            start_epoch = 0 # Reset start_epoch

    elif args.new_lr is not None: # If not loading checkpoint but new_lr is specified (for a new run)
        print(f"Starting new training with specified LR: {args.new_lr}")
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.new_lr
        current_learning_rate = args.new_lr
    
    print(f"Initial/Current Learning Rate for this run: {optimizer.param_groups[0]['lr']:.6f}")


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model_type}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device, scheduler)
    
    # Determine number of epochs for this run
    epochs_to_run = args.continue_epochs if args.load_checkpoint and loaded_epoch is not None else args.num_epochs

    # --- Train the model ---
    trainer.train(
        num_epochs=epochs_to_run,
        save_path=args.save_path, # Use the same save path or a new one
        early_stopping_patience=args.early_stopping_patience,
        start_epoch=start_epoch
    )
    
    trainer.plot_training_history(start_epoch=start_epoch if args.load_checkpoint and loaded_epoch is not None else 0)
    
    # --- Evaluation on test set (using the best model saved during training) ---
    print("\nLoading best model for final evaluation on test set...")
    # Re-initialize model structure before loading state_dict
    if args.model_type == 'cnn':
        eval_model = AIDetectionCNN(num_classes=num_classes)
    elif args.model_type == 'resnet':
        eval_model = AIDetectionResNet(num_classes=num_classes)
    elif args.model_type == 'ensemble_cnn_vit':
        cnn_part_eval = AIDetectionCNN(num_classes=num_classes)
        vit_part_eval = vit_b_16() 
        vit_part_eval.heads.head = nn.Linear(vit_part_eval.heads.head.in_features, num_classes)
        eval_model = EnsembleModel(cnn_model=cnn_part_eval, vit_model=vit_part_eval, num_classes=num_classes)
    
    # Load the best model saved by the trainer
    # The load_checkpoint function can be reused if it only loads model_state_dict when optimizer is None
    # For simplicity, directly load model_state_dict here for evaluation model
    best_model_checkpoint = torch.load(args.save_path, map_location=device)
    eval_model.load_state_dict(best_model_checkpoint['model_state_dict'])
    eval_model = eval_model.to(device)
    eval_model.eval()
    
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for data, targets in tqdm(test_loader, desc="Testing best model"):
            data, targets = data.to(device), targets.to(device)
            outputs = eval_model(data)
            _, predicted = torch.max(outputs.data, 1)
            test_predictions.extend(predicted.cpu().numpy())
            test_targets.extend(targets.cpu().numpy())
    
    test_metrics = calculate_metrics(test_targets, test_predictions, class_names=list(class_to_idx.keys()))
    
    print("\nTest Results (using best model from training run):")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1-Score: {test_metrics['f1_score']:.4f}")
    
    plot_confusion_matrix(test_metrics['confusion_matrix'], class_names=list(class_to_idx.keys()))

if __name__ == "__main__":
    main()
