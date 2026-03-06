import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_overall_comparison():
    # Load data
    with open('results/personalization_details.json', 'r') as f:
        data = json.load(f)
    
    # Calculate Means
    mean_generic = np.mean(data['generic_accuracy'])
    mean_personalized = np.mean(data['personalized_accuracy'])
    
    # Prepare Data for Plotting
    categories = ['Generic (LOSO)', 'Personalized (Target)']
    values = [mean_generic, mean_personalized]
    colors = ['#3498db', '#2ecc71'] # Blue for Generic, Green for Personalized
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(categories, values, color=colors, width=0.5)
    
    # Add Values on Top
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2%}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Formatting
    plt.title('Overall Model Evaluation: Generic vs Personalized', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Save
    save_dir = 'personalization loso results'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'Figure_Overall_Accuracy_Comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}")

if __name__ == "__main__":
    plot_overall_comparison()
