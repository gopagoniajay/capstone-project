import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from models.ensemble import EnsembleModel

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating on: {device}")

# 2. Load Data and Recreate Split
def load_test_data():
    if not os.path.exists('data/processed_X.npy') or not os.path.exists('data/processed_y.npy'):
        raise FileNotFoundError("Processed data not found.")
    
    X = np.load('data/processed_X.npy')
    y = np.load('data/processed_y.npy')
    
    # Replicate the split used in training
    # random_state=42 ensures we get the exact same test set
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_test, y_test

def evaluate_model():
    X_test, y_test = load_test_data()
    
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize Model and Load Weights
    model = EnsembleModel(num_classes=3).to(device)
    model_path = "best_ensemble_model.pth"
    
    if not os.path.exists(model_path):
        print("Model file not found. Please train the model first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Running Inference...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {acc:.4f}")
    
    class_names = ['Baseline', 'Stress', 'Amusement']
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - WESAD Emotion Recognition')
    
    save_path = 'confusion_matrix.png'
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    # plt.show() # Uncomment if running locally with display

if __name__ == "__main__":
    evaluate_model()
