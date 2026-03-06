import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from models.ensemble import EnsembleModel
from evaluate import load_test_data

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating on: {device}")

def visualize_metrics():
    # Load Data (re-using function from evaluate.py if possible, or re-implementing)
    # Since we imported load_test_data from evaluate, we use it directly
    try:
        X_test, y_test = load_test_data()
    except FileNotFoundError as e:
        print(e)
        return

    # Initialize Model and Load Weights
    # We need to make sure models/ensemble.py is importable. 
    # Assumes visualize_evaluation.py is in the root directory same as evaluate.py
    
    model = EnsembleModel(num_classes=3).to(device)
    model_path = "best_ensemble_model.pth"
    
    if not os.path.exists(model_path):
        print("Model file not found. Please train the model first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Run Inference
    all_preds = []
    # Convert entire test set to tensor (be mindful of memory if dataset is huge, but here it seems fine)
    # Processing in batches is safer as per evaluate.py
    from torch.utils.data import DataLoader, TensorDataset
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    all_labels = y_test # Ground truth
    
    print("Running Inference for Visualization...")
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            
    # Calculate Metrics
    class_names = ['Baseline', 'Stress', 'Amusement']
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    overall_acc = report['accuracy']
    
    # Prepare Data for Plotting
    metrics = ['precision', 'recall', 'f1-score']
    data = {
        'Class': [],
        'Metric': [],
        'Score': []
    }
    
    for class_name in class_names:
        for metric in metrics:
            data['Class'].append(class_name)
            data['Metric'].append(metric.capitalize())
            data['Score'].append(report[class_name][metric])
            
    # plotting
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Create grouped bar chart
    ax = sns.barplot(x='Class', y='Score', hue='Metric', data=data, palette='viridis')
    
    # Add accuracy line
    plt.axhline(y=overall_acc, color='r', linestyle='--', linewidth=2, label=f'Overall Accuracy ({overall_acc:.2f})')
    
    plt.title('Performance Metrics by Class', fontsize=16)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
        
    plt.tight_layout()
    save_path = 'evaluation_metrics.png'
    plt.savefig(save_path)
    print(f"Metrics visualization saved to {save_path}")
    # plt.show()

if __name__ == "__main__":
    visualize_metrics()
