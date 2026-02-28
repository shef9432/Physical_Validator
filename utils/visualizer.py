import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_report():
    csv_path = "data/audit_report.csv"
    if not os.path.exists(csv_path): return

    df = pd.read_csv(csv_path)
    plt.style.use('dark_background')
    cats = df['Category'].unique()
    
    fig, axes = plt.subplots(len(cats), 1, figsize=(12, 5 * len(cats)))
    if len(cats) == 1: axes = [axes]

    for i, cat in enumerate(cats):
        subset = df[df['Category'] == cat].sort_values('Param_Value')
        sns.lineplot(ax=axes[i], x='Param_Value', y='Confidence', data=subset, 
                     color='#FF9F1C', marker='o', markersize=3, linewidth=2)
        
        axes[i].axhline(y=0.8, color='#2EC4B6', linestyle='--', alpha=0.8, label='Reliability Threshold')
        axes[i].fill_between(subset['Param_Value'], subset['Confidence'], color='#FF9F1C', alpha=0.1)
        axes[i].set_title(f"Robustness Analytics: {cat}", fontsize=16)
        axes[i].set_ylim(0, 1.05)
        axes[i].grid(True, linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig("data/robustness_curves.png", dpi=300)
    print("📈 Visualization Report Exported: data/robustness_curves.png")

if __name__ == "__main__":
    generate_report()