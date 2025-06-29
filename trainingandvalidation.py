import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix
import time
import json
from pathlib import Path

# Import the CNN model
from CNN import BirdCNN

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BirdTrainer")

class BirdTrainer:
    """
    Trainer class for the Bird CNN model with comprehensive training, 
    validation, and evaluation capabilities.
    """
    
    def __init__(self, data_dir='Clean/', 
                 batch_size=32, 
                 learning_rate=0.001,
                 device=None,
                 model_save_dir='models/',
                 results_dir='results/'):
        """
        Initialize the trainer with configurable parameters.
        
        Args:
            data_dir (str): Directory containing training data organized in class folders
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for optimizer
            device (str): Device to train on ('cuda' or 'cpu')
            model_save_dir (str): Directory to save model checkpoints
            results_dir (str): Directory to save training results and plots
        """
        # Set configuration
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model_save_dir = Path(model_save_dir)
        self.results_dir = Path(results_dir)
        
        # Create directories if they don't exist
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Select device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Define transformations
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomAffine(10, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load and split dataset
        self._prepare_data()
        
        # Initialize model
        self.model = BirdCNN().to(self.device)
        
        # Create optimizer and loss function
        self.criterion = nn.BCELoss()  # Binary Cross Entropy Loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'best_val_acc': 0.0,
            'best_epoch': 0
        }
    
    def _prepare_data(self):
        """
        Load and prepare datasets for training and validation.
        """
        try:
            # Load dataset
            full_dataset = datasets.ImageFolder(root=self.data_dir, transform=self.train_transform)
            
            # Record class mapping
            self.class_to_idx = full_dataset.class_to_idx
            logger.info(f"Class to index mapping: {self.class_to_idx}")
            
            # Split data into train and validation sets
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            
            # Create random splits with a fixed seed for reproducibility
            generator = torch.Generator().manual_seed(42)
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_size, val_size], generator=generator
            )
            
            # Apply different transform to validation set
            self.val_dataset.dataset.transform = self.val_transform
            
            # Create data loaders
            self.train_loader = DataLoader(
                self.train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                num_workers=4,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            self.val_loader = DataLoader(
                self.val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False,
                num_workers=4,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            logger.info(f"Dataset loaded: {train_size} training samples, {val_size} validation samples")
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def train(self, epochs=50, early_stopping_patience=10):
        """
        Train the model with validation and early stopping.
        
        Args:
            epochs (int): Number of training epochs
            early_stopping_patience (int): Patience for early stopping
        
        Returns:
            dict: Training history with metrics
        """
        logger.info(f"Starting training for {epochs} epochs")
        
        # Initialize tracking variables
        best_val_acc = 0.0
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train one epoch
            train_loss, train_acc = self._train_epoch(epoch + 1, epochs)
            
            # Validate
            val_loss, val_acc = self._validate_epoch()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            epoch_time = time.time() - epoch_start
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Time: {epoch_time:.2f}s - "
                f"Train Loss: {train_loss:.4f} - "
                f"Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Val Acc: {val_acc:.4f}"
            )
            
            # Check for improvement
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.history['best_val_acc'] = best_val_acc
                self.history['best_epoch'] = epoch + 1
                patience_counter = 0
                
                # Save best model
                self._save_model(f"bird_cnn_best.pth")
                logger.info(f"New best model saved with validation accuracy: {best_val_acc:.4f}")
            else:
                patience_counter += 1
                
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self._save_model(f"bird_cnn_epoch_{epoch+1}.pth")
                
            # Check early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
                
        # Save final model
        self._save_model("bird_cnn_final.pth")
        
        # Save history
        self._save_history()
        
        # Generate and save training plots
        self._plot_training_curves()
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/60:.2f} minutes")
        logger.info(f"Best validation accuracy: {best_val_acc:.4f} at epoch {self.history['best_epoch']}")
        
        return self.history
    
    def _train_epoch(self, current_epoch, total_epochs):
        """
        Train for one epoch.
        
        Args:
            current_epoch (int): Current epoch number
            total_epochs (int): Total number of epochs
            
        Returns:
            tuple: (epoch_loss, epoch_accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Create progress bar
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {current_epoch}/{total_epochs} [Train]",
            leave=False,
            ncols=100
        )
        
        for inputs, labels in progress_bar:
            # Move data to device
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).float()
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct/total:.4f}"
            })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = correct / total
        
        return epoch_loss, epoch_accuracy
    
    def _validate_epoch(self):
        """
        Validate the model on the validation set.
        
        Returns:
            tuple: (validation_loss, validation_accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Create progress bar
        progress_bar = tqdm(
            self.val_loader,
            desc="Validating",
            leave=False,
            ncols=100
        )
        
        with torch.no_grad():
            for inputs, labels in progress_bar:
                # Move data to device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).float()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Track statistics
                running_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{correct/total:.4f}"
                })
        
        # Calculate validation metrics
        val_loss = running_loss / len(self.val_loader)
        val_accuracy = correct / total
        
        return val_loss, val_accuracy
    
    def _save_model(self, filename):
        """
        Save the model state dictionary.
        
        Args:
            filename (str): Filename for the model checkpoint
        """
        path = self.model_save_dir / filename
        torch.save(self.model.state_dict(), path)
    
    def _save_history(self):
        """
        Save training history to a JSON file.
        """
        history_file = self.results_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=4)
    
    def _plot_training_curves(self):
        """
        Generate and save training curves showing loss and accuracy.
        """
        # Plot training and validation loss
        plt.figure(figsize=(12, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Training Accuracy')
        plt.plot(self.history['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "training_curves.png", dpi=300)
    
    def find_optimal_threshold(self):
        """
        Find the optimal classification threshold using precision-recall curve.
        
        Returns:
            float: Optimal threshold value
        """
        logger.info("Finding optimal classification threshold...")
        
        # Load the best model
        best_model_path = self.model_save_dir / "bird_cnn_best.pth"
        if not best_model_path.exists():
            logger.warning("Best model not found. Using current model state.")
        else:
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        
        self.model.eval()
        
        # Collect predictions and labels
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Evaluating", ncols=100):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = outputs.cpu().numpy()
                
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())
        
        # Convert to numpy arrays
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
        
        # Calculate F1 score for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Find threshold with best F1 score
        best_idx = np.argmax(f1_scores)
        if best_idx >= len(thresholds):  # Handle edge case
            best_threshold = 0.5
        else:
            best_threshold = thresholds[best_idx]
        
        best_f1 = f1_scores[best_idx]
        
        logger.info(f"Optimal threshold: {best_threshold:.4f}, F1 score: {best_f1:.4f}")
        
        # Plot precision-recall curve
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, label=f'Precision-Recall Curve (F1={best_f1:.4f})')
        plt.axvline(x=recall[best_idx], color='r', linestyle='--', alpha=0.5, 
                   label=f'Best threshold: {best_threshold:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.results_dir / "precision_recall_curve.png", dpi=300)
        
        # Generate classification report at optimal threshold
        pred_labels = (all_probs >= best_threshold).astype(int)
        report = classification_report(all_labels, pred_labels, target_names=['Not Bird', 'Bird'])
        
        logger.info(f"\nClassification Report at threshold {best_threshold:.4f}:\n{report}")
        
        # Save optimal threshold
        threshold_file = self.results_dir / "optimal_threshold.json"
        with open(threshold_file, 'w') as f:
            json.dump({
                'threshold': float(best_threshold),
                'f1_score': float(best_f1)
            }, f, indent=4)
        
        return best_threshold
    
    def evaluate(self, threshold=None):
        """
        Evaluate the model on the validation set.
        
        Args:
            threshold (float): Classification threshold (if None, uses 0.5)
            
        Returns:
            dict: Evaluation metrics
        """
        if threshold is None:
            # Try to load optimal threshold
            threshold_file = self.results_dir / "optimal_threshold.json"
            if threshold_file.exists():
                with open(threshold_file, 'r') as f:
                    threshold = json.load(f)['threshold']
            else:
                threshold = 0.5
        
        logger.info(f"Evaluating model with threshold: {threshold:.4f}")
        
        # Load the best model
        best_model_path = self.model_save_dir / "bird_cnn_best.pth"
        if best_model_path.exists():
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            logger.info(f"Loaded best model from {best_model_path}")
        
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Evaluating", ncols=100):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                preds = (outputs >= threshold).float().cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds, dtype=int)
        all_labels = np.array(all_labels, dtype=int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Calculate metrics
        true_pos = cm[1, 1]
        false_pos = cm[0, 1]
        true_neg = cm[0, 0]
        false_neg = cm[1, 0]
        
        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'confusion_matrix': cm.tolist()
        }
        
        # Save evaluation results
        eval_file = self.results_dir / "evaluation.json"
        with open(eval_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        classes = ['Not Bird', 'Bird']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(self.results_dir / "confusion_matrix.png", dpi=300)
        
        logger.info(f"Evaluation results: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        return metrics

# Example usage
if __name__ == "__main__":
    # Create trainer instance
    trainer = BirdTrainer(
        data_dir='Clean/',
        batch_size=32,
        learning_rate=0.001,
        model_save_dir='models/',
        results_dir='results/'
    )
    
    # Train model
    trainer.train(epochs=50, early_stopping_patience=10)
    
    # Find optimal threshold
    optimal_threshold = trainer.find_optimal_threshold()
    
    # Evaluate model with optimal threshold
    trainer.evaluate(threshold=optimal_threshold)