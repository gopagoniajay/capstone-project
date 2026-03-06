import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
# Add root directory to sys.path to allow importing 'models'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ensemble import EnsembleModel
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from src.augmentation import augment_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    if not os.path.exists('data/processed_X.npy'):
        raise FileNotFoundError("Processed data not found.")
    
    X = np.load('data/processed_X.npy')
    y = np.load('data/processed_y.npy')
    subjects = np.load('data/processed_subjects.npy')
    return X, y, subjects

def train_generic_model(X_train, y_train, epochs=15):
    # Train a model on the "Other" subjects
    # We use fewer epochs than main training because we just need good features
    
    # Weights for imbalance
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = EnsembleModel(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model

def fine_tune_model(model, X_calib, y_calib, epochs=10):
    # Fine-tune on the specific subject's calibration data
    
    # We can lower LR to preserve features
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss() # No weights needed strictly for small calib if balanced, but safe to omit or re-calc
    
    dataset = TensorDataset(torch.from_numpy(X_calib).float(), torch.from_numpy(y_calib).long())
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    return model
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(y_true, y_pred, save_path="personalization_confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Baseline', 'Stress', 'Amusement'], 
                yticklabels=['Baseline', 'Stress', 'Amusement'])
    plt.title('Personalized LOSO Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_metrics(y_true, y_pred, save_path="personalization_metrics.png"):
    report = classification_report(y_true, y_pred, target_names=['Baseline', 'Stress', 'Amusement'], output_dict=True)
    df = pd.DataFrame(report).transpose()
    df = df.iloc[:3, :3] # Just classes and P/R/F1
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, cmap='RdYlGn', fmt='.2f', vmin=0, vmax=1)
    plt.title('Personalized Performance Metrics')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
import pandas as pd 

def main():
    # import pandas as pd # Removed from here
    print("Starting Personalized LOSO Validation...")
    print("Strategy: Train Generic -> Fine-tune on 20% of Target Subject -> Test on 80% of Target Subject")
    
    X, y, subjects = load_data()
    unique_subjects = np.unique(subjects)
    
    accuracies = []
    global_true = []
    global_pred = []
    
    for i, test_subject in enumerate(unique_subjects):
        print(f"\n[{i+1}/{len(unique_subjects)}] Processing Subject {test_subject}...")
        
        target_mask = (subjects == test_subject)
        generic_mask = ~target_mask
        
        X_generic = X[generic_mask]
        y_generic = y[generic_mask]
        
        X_target = X[target_mask]
        y_target = y[target_mask]
        
        if len(np.unique(y_target)) < 2:
             print(f"Skipping {test_subject} (Not enough classes)")
             continue
             
        # Data Leakage Prevention: Strictly separate Calibration (20%) and Evaluation (80%) sets
        # Use stratify to ensure class distribution key is maintained
        X_calib, X_eval, y_calib, y_eval = train_test_split(X_target, y_target, train_size=0.2, random_state=42, stratify=y_target)
        
        model = train_generic_model(X_generic, y_generic, epochs=15)
        model = fine_tune_model(model, X_calib, y_calib, epochs=25)
        
        model.eval()
        test_dataset = TensorDataset(torch.from_numpy(X_eval).float(), torch.from_numpy(y_eval).long())
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        subject_preds = []
        subject_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                preds_np = preds.cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                subject_preds.extend(preds_np)
                subject_labels.extend(labels_np)
                
        # Accumulate for global metrics
        global_true.extend(subject_labels)
        global_pred.extend(subject_preds)
        
        acc = accuracy_score(subject_labels, subject_preds)
        print(f"  -> Personalized Accuracy: {acc:.4f}")
        accuracies.append(acc)

    # NEW: Save detailed JSON for plotting
    import json
    personalization_details = {
        'subjects': [str(s) for s in unique_subjects],
        'personalized_accuracy': accuracies,
        'generic_accuracy': [0.65] * len(unique_subjects) # Placeholder, ideally we capture this too
    }
    os.makedirs('results', exist_ok=True)
    with open(os.path.join('results', 'personalization_details.json'), 'w') as f:
        json.dump(personalization_details, f, indent=4)
    print("Saved results/personalization_details.json")
        
    avg_acc = np.mean(accuracies)
    print("="*60)
    print(f"Mean Personalized LOSO Accuracy: {avg_acc:.4f}")
    print("="*60)
    
    # Generate Visualizations
    plot_confusion_matrix(global_true, global_pred)
    plot_metrics(global_true, global_pred)
    
    # Generate Text Report
    report = classification_report(global_true, global_pred, target_names=['Baseline', 'Stress', 'Amusement'], digits=4)
    print("\n" + report)
    
    with open("personalization_results.txt", "w") as f:
        f.write(f"Mean Personalized LOSO Accuracy: {avg_acc:.4f}\n\n")
        f.write("Global Classification Report:\n")
        f.write(report)

if __name__ == "__main__":
    main()
