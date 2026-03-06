import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from models.ensemble import EnsembleModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from src.augmentation import augment_data

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# 2. Load Data
def load_data():
    if not os.path.exists('data/processed_X.npy') or not os.path.exists('data/processed_y.npy'):
        raise FileNotFoundError("Processed data not found. Run src/prepare_data.py first.")
    
    X = np.load('data/processed_X.npy')
    y = np.load('data/processed_y.npy')
    
    # Check shapes
    print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y

# 3. Training Function
def train_model():
    try:
        X, y = load_data()
    except FileNotFoundError as e:
        print(e)
        return
    
    # Split Data (80% Train, 20% Test)
    # Stratify is important for imbalanced data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create Datasets
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize Model
    # Determine input size from data if possible, but we know it's 2 features (ECG, EDA)
    model = EnsembleModel(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0.0
    
    print("Starting Training...")
    epochs = 20
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f} - Val Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_ensemble_model.pth")
            # print("  -> Saved Best Model")
            
    print("\nFinal Evaluation on Test Set:")
    print(classification_report(all_labels, all_preds, target_names=['Baseline', 'Stress', 'Amusement']))
    print(f"Best Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    train_model()