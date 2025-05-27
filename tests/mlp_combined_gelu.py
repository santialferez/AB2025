#!/usr/bin/env python3
"""
MLP Neural Network with Combined GELU Activation Function
Training and Validation on Digits Dataset

Combined GELU: GELU(x) + GELU(-x)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time


class CombinedGELU(nn.Module):
    """
    Combined GELU activation function: GELU(x) + GELU(-x)
    """
    def __init__(self):
        super(CombinedGELU, self).__init__()
        
    def forward(self, x):
        return F.gelu(x) + F.gelu(-x)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with Combined GELU activation
    """
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.2):
        super(MLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(CombinedGELU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def load_and_preprocess_data():
    """
    Load and preprocess the digits dataset
    """
    print("Loading digits dataset...")
    digits = load_digits()
    X, y = digits.data, digits.target
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    y_test = torch.LongTensor(y_test)
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_data_loaders(train_data, val_data, test_data, batch_size=32):
    """
    Create PyTorch DataLoaders
    """
    train_dataset = TensorDataset(train_data[0], train_data[1])
    val_dataset = TensorDataset(val_data[0], val_data[1])
    test_dataset = TensorDataset(test_data[0], test_data[1])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def validate_epoch(model, val_loader, criterion, device):
    """
    Validate the model for one epoch
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def test_model(model, test_loader, device):
    """
    Test the model and return detailed metrics
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    accuracy = accuracy_score(all_targets, all_predictions)
    
    return accuracy, all_predictions, all_targets


def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Plot training and validation metrics
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accuracies, label='Training Accuracy', color='blue')
    ax2.plot(val_accuracies, label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """
    Main training and validation procedure
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    config = {
        'hidden_sizes': [128, 64, 32],
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'batch_size': 32,
        'num_epochs': 100,
        'patience': 10,  # Early stopping patience
        'weight_decay': 1e-4
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Load and preprocess data
    train_data, val_data, test_data = load_and_preprocess_data()
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data, config['batch_size']
    )
    
    # Model configuration
    input_size = train_data[0].shape[1]  # 64 features for digits dataset
    num_classes = len(torch.unique(train_data[1]))  # 10 classes
    
    # Initialize model
    model = MLP(
        input_size=input_size,
        hidden_sizes=config['hidden_sizes'],
        num_classes=num_classes,
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    print(f"\nModel Architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print("\nStarting training...")
    start_time = time.time()
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{config['num_epochs']}]:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Load best model for testing
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_accuracy, test_predictions, test_targets = test_model(model, test_loader, device)
    
    print(f"\nFinal Results:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    
    # Detailed classification report
    class_names = [str(i) for i in range(num_classes)]
    print("\nClassification Report:")
    report = classification_report(test_targets, test_predictions, target_names=class_names)
    print(report)
    
    # Save classification report to file
    with open('classification_report.txt', 'w') as f:
        f.write("MLP with Combined GELU - Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Best Validation Loss: {best_val_loss:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot confusion matrix
    plot_confusion_matrix(test_targets, test_predictions, class_names)
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Test the combined GELU activation function
    print("\nTesting Combined GELU activation function:")
    combined_gelu = CombinedGELU()
    test_input = torch.tensor([0.0, 1.0, -1.0, 2.0, -2.0])
    output = combined_gelu(test_input)
    print(f"Input: {test_input}")
    print(f"Combined GELU output: {output}")
    print(f"Regular GELU(x): {F.gelu(test_input)}")
    print(f"Regular GELU(-x): {F.gelu(-test_input)}")
    
    # Save activation function test results
    with open('activation_function_test.txt', 'w') as f:
        f.write("Combined GELU Activation Function Test\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Input: {test_input}\n")
        f.write(f"Combined GELU output: {output}\n")
        f.write(f"Regular GELU(x): {F.gelu(test_input)}\n")
        f.write(f"Regular GELU(-x): {F.gelu(-test_input)}\n")
    
    print("\nTraining completed successfully!")
    print("Results saved to:")
    print("- classification_report.txt")
    print("- training_history.png")
    print("- confusion_matrix.png")
    print("- activation_function_test.txt")


if __name__ == "__main__":
    main() 