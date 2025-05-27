#!/usr/bin/env python3
"""
Simple Comparison: MLP with Regular GELU vs Combined GELU Activation Function
Training and Validation on Digits Dataset

Comparison: GELU(x) vs GELU(x) + GELU(-x)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
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
    Multi-Layer Perceptron with configurable activation function
    """
    def __init__(self, input_size, hidden_sizes, num_classes, activation_type='gelu', dropout_rate=0.2):
        super(MLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Choose activation function
            if activation_type == 'combined_gelu':
                layers.append(CombinedGELU())
            elif activation_type == 'relu':
                layers.append(nn.ReLU())
            else:  # regular gelu
                layers.append(nn.GELU())
                
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        self.activation_type = activation_type
    
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


def train_and_evaluate_model(activation_type, config, train_loader, val_loader, test_loader, device):
    """
    Train and evaluate a model with specified activation function
    """
    print(f"\n{'='*60}")
    print(f"Training MLP with {activation_type.upper()} activation")
    print(f"{'='*60}")
    
    # Model configuration
    input_size = 64  # digits dataset features
    num_classes = 10  # digit classes
    
    # Initialize model
    model = MLP(
        input_size=input_size,
        hidden_sizes=config['hidden_sizes'],
        num_classes=num_classes,
        activation_type=activation_type,
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    print(f"Model: {model.activation_type}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if config['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=config['learning_rate'],
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 1e-4)
        )
    else:  # adam
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )
    
    # Training loop
    print(f"Starting training with {config['optimizer'].upper()} optimizer...")
    start_time = time.time()
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(config['num_epochs']):
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{config['num_epochs']}] - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Final evaluation on test set
    test_accuracy, test_predictions, test_targets = test_model(model, test_loader, device)
    
    print(f"\nFinal Results:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Final Train Accuracy: {train_accuracies[-1]:.2f}%")
    print(f"Final Validation Accuracy: {val_accuracies[-1]:.2f}%")
    
    return {
        'activation_type': activation_type,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'final_train_acc': train_accuracies[-1],
        'final_val_acc': val_accuracies[-1]
    }


def plot_comparison(results_gelu, results_combined, results_relu):
    """
    Plot comparison between GELU, Combined GELU, and ReLU
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    epochs = range(1, len(results_gelu['train_losses']) + 1)
    
    # Training Loss
    ax1.plot(epochs, results_gelu['train_losses'], label='Regular GELU', color='blue', linewidth=2)
    ax1.plot(epochs, results_combined['train_losses'], label='Combined GELU', color='red', linewidth=2)
    ax1.plot(epochs, results_relu['train_losses'], label='ReLU', color='green', linewidth=2)
    ax1.set_title('Training Loss Comparison')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation Loss
    ax2.plot(epochs, results_gelu['val_losses'], label='Regular GELU', color='blue', linewidth=2)
    ax2.plot(epochs, results_combined['val_losses'], label='Combined GELU', color='red', linewidth=2)
    ax2.plot(epochs, results_relu['val_losses'], label='ReLU', color='green', linewidth=2)
    ax2.set_title('Validation Loss Comparison')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Training Accuracy
    ax3.plot(epochs, results_gelu['train_accuracies'], label='Regular GELU', color='blue', linewidth=2)
    ax3.plot(epochs, results_combined['train_accuracies'], label='Combined GELU', color='red', linewidth=2)
    ax3.plot(epochs, results_relu['train_accuracies'], label='ReLU', color='green', linewidth=2)
    ax3.set_title('Training Accuracy Comparison')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Validation Accuracy
    ax4.plot(epochs, results_gelu['val_accuracies'], label='Regular GELU', color='blue', linewidth=2)
    ax4.plot(epochs, results_combined['val_accuracies'], label='Combined GELU', color='red', linewidth=2)
    ax4.plot(epochs, results_relu['val_accuracies'], label='ReLU', color='green', linewidth=2)
    ax4.set_title('Validation Accuracy Comparison')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('activation_comparison_three.png', dpi=300, bbox_inches='tight')
    plt.show()


def test_activation_functions():
    """
    Test and compare the activation functions
    """
    print("\n" + "="*60)
    print("ACTIVATION FUNCTION COMPARISON")
    print("="*60)
    
    test_input = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    regular_gelu = F.gelu(test_input)
    combined_gelu = CombinedGELU()(test_input)
    
    print(f"Input:           {test_input}")
    print(f"Regular GELU:    {regular_gelu}")
    print(f"Combined GELU:   {combined_gelu}")
    print(f"Difference:      {combined_gelu - regular_gelu}")
    
    # Save to file
    with open('activation_comparison_adam.txt', 'w') as f:
        f.write("Activation Function Comparison\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Input:           {test_input}\n")
        f.write(f"Regular GELU:    {regular_gelu}\n")
        f.write(f"Combined GELU:   {combined_gelu}\n")
        f.write(f"Difference:      {combined_gelu - regular_gelu}\n")


def main():
    """
    Main comparison procedure
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configuration - Simple setup
    config = {
        'hidden_sizes': [128, 64, 32],
        'dropout_rate': 0.2,
        'learning_rate': 0.001,  # Lower learning rate for Adam
        'batch_size': 32,
        'num_epochs': 50,
        'optimizer': 'adam',  # Use Adam optimizer
        'momentum': 0.9,
        'weight_decay': 1e-4
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Test activation functions first
    test_activation_functions()
    
    # Load and preprocess data
    train_data, val_data, test_data = load_and_preprocess_data()
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data, config['batch_size']
    )
    
    # Train model with regular GELU
    results_gelu = train_and_evaluate_model(
        'gelu', config, train_loader, val_loader, test_loader, device
    )
    
    # Train model with Combined GELU
    results_combined = train_and_evaluate_model(
        'combined_gelu', config, train_loader, val_loader, test_loader, device
    )
    
    # Train model with ReLU
    results_relu = train_and_evaluate_model(
        'relu', config, train_loader, val_loader, test_loader, device
    )
    
    # Compare results
    print("\n" + "="*80)
    print("FINAL THREE-WAY COMPARISON SUMMARY")
    print("="*80)
    
    print(f"{'Metric':<25} {'Regular GELU':<15} {'Combined GELU':<15} {'ReLU':<15} {'Best':<10}")
    print("-" * 85)
    
    # Test Accuracy comparison
    test_accs = [results_gelu['test_accuracy'], results_combined['test_accuracy'], results_relu['test_accuracy']]
    best_test = max(test_accs)
    best_test_idx = test_accs.index(best_test)
    best_test_name = ['GELU', 'Combined', 'ReLU'][best_test_idx]
    print(f"{'Test Accuracy':<25} {results_gelu['test_accuracy']:<15.4f} {results_combined['test_accuracy']:<15.4f} {results_relu['test_accuracy']:<15.4f} {best_test_name:<10}")
    
    # Training Accuracy comparison
    train_accs = [results_gelu['final_train_acc'], results_combined['final_train_acc'], results_relu['final_train_acc']]
    best_train = max(train_accs)
    best_train_idx = train_accs.index(best_train)
    best_train_name = ['GELU', 'Combined', 'ReLU'][best_train_idx]
    print(f"{'Final Train Acc (%)':<25} {results_gelu['final_train_acc']:<15.2f} {results_combined['final_train_acc']:<15.2f} {results_relu['final_train_acc']:<15.2f} {best_train_name:<10}")
    
    # Validation Accuracy comparison
    val_accs = [results_gelu['final_val_acc'], results_combined['final_val_acc'], results_relu['final_val_acc']]
    best_val = max(val_accs)
    best_val_idx = val_accs.index(best_val)
    best_val_name = ['GELU', 'Combined', 'ReLU'][best_val_idx]
    print(f"{'Final Val Acc (%)':<25} {results_gelu['final_val_acc']:<15.2f} {results_combined['final_val_acc']:<15.2f} {results_relu['final_val_acc']:<15.2f} {best_val_name:<10}")
    
    # Training Time comparison
    times = [results_gelu['training_time'], results_combined['training_time'], results_relu['training_time']]
    best_time = min(times)
    best_time_idx = times.index(best_time)
    best_time_name = ['GELU', 'Combined', 'ReLU'][best_time_idx]
    print(f"{'Training Time (s)':<25} {results_gelu['training_time']:<15.2f} {results_combined['training_time']:<15.2f} {results_relu['training_time']:<15.2f} {best_time_name:<10}")
    
    # Save detailed summary to file
    with open('three_way_comparison_summary.txt', 'w') as f:
        f.write("GELU vs Combined GELU vs ReLU Comparison Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Configuration:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        f.write(f"{'Metric':<25} {'Regular GELU':<15} {'Combined GELU':<15} {'ReLU':<15} {'Best':<10}\n")
        f.write("-" * 85 + "\n")
        f.write(f"{'Test Accuracy':<25} {results_gelu['test_accuracy']:<15.4f} {results_combined['test_accuracy']:<15.4f} {results_relu['test_accuracy']:<15.4f} {best_test_name:<10}\n")
        f.write(f"{'Final Train Acc (%)':<25} {results_gelu['final_train_acc']:<15.2f} {results_combined['final_train_acc']:<15.2f} {results_relu['final_train_acc']:<15.2f} {best_train_name:<10}\n")
        f.write(f"{'Final Val Acc (%)':<25} {results_gelu['final_val_acc']:<15.2f} {results_combined['final_val_acc']:<15.2f} {results_relu['final_val_acc']:<15.2f} {best_val_name:<10}\n")
        f.write(f"{'Training Time (s)':<25} {results_gelu['training_time']:<15.2f} {results_combined['training_time']:<15.2f} {results_relu['training_time']:<15.2f} {best_time_name:<10}\n")
        f.write("\n")
        f.write("Detailed Results:\n")
        f.write("-" * 20 + "\n")
        f.write(f"GELU - Test: {results_gelu['test_accuracy']:.4f}, Train: {results_gelu['final_train_acc']:.2f}%, Val: {results_gelu['final_val_acc']:.2f}%, Time: {results_gelu['training_time']:.2f}s\n")
        f.write(f"Combined GELU - Test: {results_combined['test_accuracy']:.4f}, Train: {results_combined['final_train_acc']:.2f}%, Val: {results_combined['final_val_acc']:.2f}%, Time: {results_combined['training_time']:.2f}s\n")
        f.write(f"ReLU - Test: {results_relu['test_accuracy']:.4f}, Train: {results_relu['final_train_acc']:.2f}%, Val: {results_relu['final_val_acc']:.2f}%, Time: {results_relu['training_time']:.2f}s\n")
    
    # Plot three-way comparison
    plot_comparison(results_gelu, results_combined, results_relu)
    
    print("\nThree-way comparison completed!")
    print("Files generated:")
    print("- activation_comparison_three.png")
    print("- three_way_comparison_summary.txt")
    print("- activation_comparison_adam.txt")


if __name__ == "__main__":
    main() 