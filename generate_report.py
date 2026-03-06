import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import classification_report, confusion_matrix
from models.ensemble import EnsembleModel
from evaluate import load_test_data

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Generating report on: {device}")

def generate_pdf_report():
    try:
        X_test, y_test = load_test_data()
    except FileNotFoundError as e:
        print(e)
        return

    # Initialize Model
    model = EnsembleModel(num_classes=3).to(device)
    model_path = "best_ensemble_model.pth"
    
    if not os.path.exists(model_path):
        print("Model file not found. Please train the model first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Run Inference
    from torch.utils.data import DataLoader, TensorDataset
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    all_preds = []
    all_labels = y_test
    
    print("Running Inference...")
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            
    # --- Start PDF Generation ---
    pdf_path = 'evaluation_report.pdf'
    with PdfPages(pdf_path) as pdf:
        
        # Page 1: Text Summary
        plt.figure(figsize=(8.5, 11))
        class_names = ['Baseline', 'Stress', 'Amusement']
        report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
        report_str = classification_report(all_labels, all_preds, target_names=class_names)
        
        acc = report_dict['accuracy']
        
        text_content = (
            f"Evaluation Report\n"
            f"=================\n\n"
            f"Overall Accuracy: {acc:.4f}\n\n"
            f"Classification Report:\n"
            f"{report_str}"
        )
        
        plt.axis('off')
        plt.text(0.1, 0.9, text_content, transform=plt.gca().transAxes, fontsize=12, family='monospace', va='top')
        plt.title("Model Evaluation Summary")
        pdf.savefig()
        plt.close()
        
        # Page 2: Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        pdf.savefig()
        plt.close()
        
        # Page 3: Metrics Visualization
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
                data['Score'].append(report_dict[class_name][metric])
                
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Class', y='Score', hue='Metric', data=data, palette='viridis')
        
        # Accuracy line
        plt.axhline(y=acc, color='r', linestyle='--', linewidth=2, label=f'Overall Accuracy ({acc:.2f})')
        
        plt.title('Performance Metrics by Class', fontsize=16)
        plt.ylabel('Score', fontsize=12)
        plt.ylim(0, 1.0)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f')
            
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
    print(f"PDF report generated successfully: {pdf_path}")

if __name__ == "__main__":
    generate_pdf_report()
