import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import argparse
from pathlib import Path

# Updated model imports
from model import AIDetectionCNN, AIDetectionResNet, EnsembleModel
from torchvision.models import vit_b_16 # ViT_B_16_Weights not needed here as weights come from checkpoint
from utils import get_device, get_transforms,nn

class AIDetector:
    def __init__(self, model_path, model_type='cnn', device=None, num_classes=2): # Added num_classes
        """Initialize the AI detector with a trained model"""
        self.device = device if device else get_device()
        self.transform = get_transforms('val', image_size=224) # Assuming image_size is consistent
        self.num_classes = num_classes
        
        # --- Updated Model Loading ---
        if model_type == 'cnn':
            self.model = AIDetectionCNN(num_classes=self.num_classes)
        elif model_type == 'resnet':
            self.model = AIDetectionResNet(num_classes=self.num_classes)
        elif model_type == 'ensemble_cnn_vit':
            cnn_part = AIDetectionCNN(num_classes=self.num_classes)
            
            vit_part = vit_b_16() # Initialize ViT structure
            # Adjust the head to match the number of classes, as done during training
            vit_part.heads.head = nn.Linear(vit_part.heads.head.in_features, self.num_classes)
            
            self.model = EnsembleModel(cnn_model=cnn_part, vit_model=vit_part, num_classes=self.num_classes)
            print("Loading Ensemble (CNN + ViT) Model for prediction.")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        # The state_dict from the ensemble model will have keys like 'cnn_model.conv1...' and 'vit_model.encoder...'
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Class names (assuming binary classification: FAKE, REAL)
        # You might want to make this more dynamic if num_classes changes
        if self.num_classes == 2:
            self.class_names = ['FAKE', 'REAL'] 
        else:
            self.class_names = [f'CLASS_{i}' for i in range(self.num_classes)]
        
        print(f"AI Detector (type: {model_type}) loaded successfully on {self.device}")
    
    def predict_single_image(self, image_path, return_probabilities=False):
        """Predict whether a single image is AI-generated or real"""
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor) # Forward pass through the ensemble
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_class_idx = torch.max(probabilities, dim=1)
                predicted_class_idx = predicted_class_idx.item()
                confidence = confidence.item()

            result = {
                'prediction': self.class_names[predicted_class_idx],
                'confidence': confidence,
            }
            # Assuming class 'FAKE' is at index 0 for 'is_ai_generated'
            if self.class_names[predicted_class_idx] == 'FAKE':
                 result['is_ai_generated'] = True
            else:
                 result['is_ai_generated'] = False


            if return_probabilities:
                probs_dict = {}
                for i, class_name in enumerate(self.class_names):
                    probs_dict[class_name] = probabilities[0][i].item()
                result['probabilities'] = probs_dict
            
            return result
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None
    
    def predict_batch(self, image_folder, output_file=None):
        """Predict on a batch of images in a folder"""
        results = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(image_folder).glob(f'*{ext}'))
            image_files.extend(Path(image_folder).glob(f'*{ext.upper()}'))
        
        print(f"Found {len(image_files)} images to process")
        
        for image_path in tqdm(image_files, desc="Predicting Batch"): # Added tqdm for progress
            result = self.predict_single_image(str(image_path), return_probabilities=True)
            if result:
                result['filename'] = image_path.name
                results.append(result)
                # print(f"{image_path.name}: {result['prediction']} (confidence: {result['confidence']:.3f})") # Optional: print during batch
        
        if output_file:
            import csv
            # Dynamically create fieldnames based on class_names for probabilities
            fieldnames = ['filename', 'prediction', 'confidence', 'is_ai_generated']
            if results and 'probabilities' in results[0]:
                 for class_name in self.class_names:
                    fieldnames.append(f'{class_name.lower()}_probability')

            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for result in results:
                    row_data = {
                        'filename': result['filename'],
                        'prediction': result['prediction'],
                        'confidence': result['confidence'],
                        'is_ai_generated': result.get('is_ai_generated', 'N/A')
                    }
                    if 'probabilities' in result:
                        for class_name, prob in result['probabilities'].items():
                            row_data[f'{class_name.lower()}_probability'] = prob
                    writer.writerow(row_data)
            print(f"Results saved to {output_file}")
        return results
    
    def predict_with_visualization(self, image_path):
        """Predict and visualize the result"""
        import matplotlib.pyplot as plt
        
        result = self.predict_single_image(image_path, return_probabilities=True)
        
        if result:
            image = Image.open(image_path).convert('RGB')
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            ax1.imshow(image)
            ax1.set_title(f"Input Image\n{Path(image_path).name}")
            ax1.axis('off')
            
            categories = list(result['probabilities'].keys())
            probabilities_values = list(result['probabilities'].values())
            
            # Set colors based on prediction (e.g., predicted class green, others red)
            bar_colors = ['grey'] * len(categories)
            try:
                predicted_idx = categories.index(result['prediction'])
                bar_colors[predicted_idx] = 'green' if result['prediction'] == 'REAL' else 'red'
            except ValueError:
                pass # Predicted class not in categories somehow

            bars = ax2.bar(categories, probabilities_values, color=bar_colors, alpha=0.7)
            ax2.set_ylim(0, 1)
            ax2.set_ylabel('Probability')
            ax2.set_title(f"Prediction: {result['prediction']}\nConfidence: {result['confidence']:.3f}")
            
            for bar_val, prob_val in zip(bars, probabilities_values):
                height = bar_val.get_height()
                ax2.text(bar_val.get_x() + bar_val.get_width()/2., height + 0.01,
                        f'{prob_val:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()
            return result
        else:
            print("Failed to process image")
            return None

def main():
    parser = argparse.ArgumentParser(description='AI-Generated Image Detection')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--image_path', type=str, help='Path to a single image')
    parser.add_argument('--image_folder', type=str, help='Path to a folder of images')
    # --- Updated model_type choices ---
    parser.add_argument('--model_type', type=str, default='cnn', 
                       choices=['cnn', 'resnet', 'ensemble_cnn_vit'], 
                       help='Type of model architecture')
    parser.add_argument('--output_file', type=str, help='Output CSV file for batch predictions')
    parser.add_argument('--visualize', action='store_true', help='Show visualization for single image prediction')
    
    args = parser.parse_args()
    
    # Initialize detector (assuming 2 classes for FAKE/REAL)
    # If you have a way to get num_classes from the model checkpoint or config, use that.
    # For now, hardcoding to 2 for this example.
    num_classes = 2 
    detector = AIDetector(args.model_path, args.model_type, num_classes=num_classes)
    
    if args.image_path:
        if args.visualize:
            result = detector.predict_with_visualization(args.image_path)
        else:
            result = detector.predict_single_image(args.image_path, return_probabilities=True)
            
        if result:
            print(f"\nPrediction Results for {args.image_path}:")
            print(f"  Prediction: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            if 'is_ai_generated' in result: # Check if key exists
                 print(f"  Is AI Generated: {result['is_ai_generated']}")
            if 'probabilities' in result:
                for class_name, prob in result['probabilities'].items():
                    print(f"  {class_name} probability: {prob:.3f}")
    
    elif args.image_folder:
        results = detector.predict_batch(args.image_folder, args.output_file)
        if results: # Ensure results is not empty
            total_images = len(results)
            # Filter out results where 'is_ai_generated' might be missing if prediction failed for some
            ai_generated = sum(1 for r in results if r.get('is_ai_generated', False))
            real_images = total_images - ai_generated
            
            print(f"\nBatch Prediction Summary:")
            print(f"  Total images processed: {total_images}")
            if total_images > 0:
                print(f"  AI-generated images: {ai_generated} ({ai_generated/total_images*100:.1f}%)")
                print(f"  Real images: {real_images} ({real_images/total_images*100:.1f}%)")
            else:
                print("  No images were successfully processed.")
    else:
        print("Please provide either --image_path for single image or --image_folder for batch prediction")

if __name__ == "__main__":
    main()
