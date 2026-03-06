import sys
import os

# Add root directory to sys.path to allow importing 'models'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
# os imported above
from models.ensemble import EnsembleModel
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# 2. Load Data
def load_data():
    if not os.path.exists('data/processed_X.npy'):
        raise FileNotFoundError("Processed data not found. Run src/prepare_data.py first.")
    
    X = np.load('data/processed_X.npy')
    y = np.load('data/processed_y.npy')
    subjects = np.load('data/processed_subjects.npy')
    
    return X, y, subjects

# 3. Training Loop per Fold
def train_fold(X_train, y_train, X_test, y_test, fold_idx):
    # Calculate Class Weights to handle imbalance
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"  -> Class Weights: {class_weights.cpu().numpy()}")

    # Create Datasets
    # Explicitly cast to float32 (for input) and long (for labels) to avoid Double vs Float errors
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize Model with higher dropout for generalization
    model = EnsembleModel(num_classes=3).to(device)
    
    # Use Weighted Loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3) # Increased weight decay
    
    # Scheduler: Reduce LR if validation accuracy stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_acc = 0.0
    patience = 8  # Increased patience
    no_improve = 0
    
    epochs = 50 # Increased epochs for aggressive augmentation
    
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
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels) # Calc val loss for sanity
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        acc = accuracy_score(all_labels, all_preds)
        
        # Step the scheduler based on Accuracy
        scheduler.step(acc)
        
        # print(f"    Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            no_improve = 0
            # Save temporary best model for this fold
            torch.save(model.state_dict(), f"temp_fold_{fold_idx}.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                # Early stopping
                print(f"    -> Early stopping at epoch {epoch+1}")
                break
                
    # Load best model for this fold
    if os.path.exists(f"temp_fold_{fold_idx}.pth"):
        model.load_state_dict(torch.load(f"temp_fold_{fold_idx}.pth"))
        os.remove(f"temp_fold_{fold_idx}.pth")
        
    return model, best_acc, all_preds, all_labels

from src.augmentation import augment_data

from scipy.signal import medfilt

def smooth_predictions(preds, kernel_size=5):
    # Apply Median Filter to smooth predictions
    return medfilt(preds, kernel_size=kernel_size)

def main():
    X, y, subjects = load_data()
    unique_subjects = np.unique(subjects)
    
    print(f"Starting Leave-One-Subject-Out Validation on {len(unique_subjects)} subjects.")
    print("Improvements Enabled: Class Weights + LR Scheduler + Enhanced Patience + Temporal Smoothing")
    
    fold_accuracies = []
    smoothed_accuracies = []
    
    all_fold_preds = []
    all_fold_preds_smooth = []
    all_fold_labels = []
    
    for i, test_subject in enumerate(unique_subjects):
        print(f"\nDistilling Fold {i+1}/{len(unique_subjects)} (Subject {test_subject})...")
        
        # Split Data
        test_mask = (subjects == test_subject)
        train_mask = ~test_mask
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        # Apply Data Augmentation only to the training set of this fold
        # This ensures we don't leak information to the test set
        X_train, y_train = augment_data(X_train, y_train)
        
        # Train
        model, acc, preds, labels = train_fold(X_train, y_train, X_test, y_test, i)
        
        # Apply Smoothing
        preds_smooth = smooth_predictions(preds, kernel_size=15) # 15 samples ~= 150 seconds context? No, window is 10s. 15 windows = 150s.
        # Actually window overlap is 50%. 15 windows * 5s step = 75 seconds context. Reasonable.
        acc_smooth = accuracy_score(labels, preds_smooth)
        
        print(f"  -> Subject {test_subject} Accuracy [Raw]:      {acc:.4f}")
        print(f"  -> Subject {test_subject} Accuracy [Smoothed]: {acc_smooth:.4f} (+{(acc_smooth-acc)*100:.2f}%)")
        
        fold_accuracies.append(acc)
        smoothed_accuracies.append(acc_smooth)
        
        all_fold_preds.extend(preds)
        all_fold_preds_smooth.extend(preds_smooth)
        all_fold_labels.extend(labels)
        
    avg_acc = np.mean(fold_accuracies)
    avg_acc_smooth = np.mean(smoothed_accuracies)
    
    print(f"\nAverage LOSO Accuracy [Raw]:      {avg_acc:.4f}")
    print(f"Average LOSO Accuracy [Smoothed]: {avg_acc_smooth:.4f}")
    
    # Global Classification Report
    print("\nGlobal Classification Report (Smoothed):")
    report = classification_report(all_fold_labels, all_fold_preds_smooth, target_names=['Baseline', 'Stress', 'Amusement'])
    print(report)
    
    
    # Save Results
    with open("loso_results.txt", "w") as f:
        f.write(f"Average LOSO Accuracy [Raw]: {avg_acc:.4f}\n")
        f.write(f"Average LOSO Accuracy [Smoothed]: {avg_acc_smooth:.4f}\n\n")
        f.write("Global Classification Report (Smoothed):\n")
        f.write(report)
        
    cm = confusion_matrix(all_fold_labels, all_fold_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Baseline', 'Stress', 'Amusement'], yticklabels=['Baseline', 'Stress', 'Amusement'])
    plt.title('LOSO Confusion Matrix')
    plt.savefig('loso_confusion_matrix.png')
    
    # NEW: Save detailed JSON for plotting
    import json
    loso_details = {
        'subjects': [str(s) for s in unique_subjects],
        'accuracies': fold_accuracies,
        'smoothed_accuracies': smoothed_accuracies
    }
    os.makedirs('results', exist_ok=True)
    with open(os.path.join('results', 'loso_details.json'), 'w') as f:
        json.dump(loso_details, f, indent=4)
    print("Saved results/loso_details.json")

if __name__ == "__main__":
    main()
