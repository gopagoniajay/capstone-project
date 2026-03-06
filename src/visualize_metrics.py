import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_accuracy_comparison():
    # Data from our experiments
    scenarios = ['Baseline (Schmidt et al.)', 'Global (Augmented)', 'Generalization (LOSO)', 'Personalized (S6)']
    accuracies = [80.0, 91.10, 81.0, 99.72]
    colors = ['#bdc3c7', '#3498db', '#e67e22', '#2ecc71'] # Grey, Blue, Orange, Green

    plt.figure(figsize=(10, 6))
    bars = plt.bar(scenarios, accuracies, color=colors)
    
    # Add values on top
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.ylim(0, 110)
    plt.title('Model Accuracy Comparison: Impact of Methodology', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')
    print("Saved accuracy_comparison.png")

def plot_f1_scores():
    # Global Augmented Model Class Performance
    classes = ['Baseline', 'Stress', 'Amusement']
    precision = [0.92, 0.93, 0.85]
    recall = [0.95, 0.94, 0.74]
    f1_score = [0.93, 0.94, 0.79]

    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(10, 6))
    r1 = plt.bar(x - width, precision, width, label='Precision', color='#3498db')
    r2 = plt.bar(x, recall, width, label='Recall', color='#e74c3c')
    r3 = plt.bar(x + width, f1_score, width, label='F1-Score', color='#2ecc71')

    plt.xlabel('Emotion Class', fontsize=12)
    plt.ylabel('Score (0-1)', fontsize=12)
    plt.title('Class-wise Performance Metrics (Global Augmented Model)', fontsize=14)
    plt.xticks(x, classes)
    plt.legend()
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add F1 values
    for i, v in enumerate(f1_score):
        plt.text(i + width, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('f1_score_analysis.png')
    print("Saved f1_score_analysis.png")

if __name__ == "__main__":
    sns.set_style("whitegrid")
    plot_accuracy_comparison()
    plot_f1_scores()
