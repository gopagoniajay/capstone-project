import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

def plot_personalized_accuracy():
    # Load data
    with open('results/personalization_details.json', 'r') as f:
        data = json.load(f)
    
    subjects = data['subjects']
    personalized_acc = data['personalized_accuracy']
    
    # Create DataFrame
    df = pd.DataFrame({
        'Subject': subjects,
        'Accuracy': personalized_acc
    })
    
    # Calculate Mean
    mean_acc = np.mean(personalized_acc)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Create the bar plot
    palette = sns.color_palette("viridis", len(subjects))
    ax = sns.barplot(x='Subject', y='Accuracy', data=df, palette=palette)
    
    # Add Mean Line
    plt.axhline(y=mean_acc, color='r', linestyle='--', label=f'Mean Accuracy ({mean_acc:.2%})')
    
    # Annotate Bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2%}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=10, fontweight='bold')
    
    # Formatting
    plt.title('Personalized LOSO Accuracy per Subject', fontsize=16, fontweight='bold')
    plt.xlabel('Subjects', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save Plot
    save_dir = 'personalization loso results'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'Figure_Personalized_LOSO_Accuracy.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    
    # Show Plot (optional, usually for notebooks)
    # plt.show()

if __name__ == "__main__":
    plot_personalized_accuracy()
