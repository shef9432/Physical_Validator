import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_survivability():
    df = pd.read_csv("data/audit_report.csv")
    plt.style.use('dark_background')
    categories = df['Category'].unique()
    
    fig, axes = plt.subplots(len(categories), 1, figsize=(12, 5 * len(categories)))

    for i, cat in enumerate(categories):
        data = df[df['Category'] == cat].sort_values('Param_Value')
        sns.lineplot(ax=axes[i], x='Param_Value', y='Validated_Conf', data=data, 
                     color='#00d2ff', linewidth=2.5, label='Validated Confidence')
        
        axes[i].axhline(y=0.0, color='red', linestyle='-', alpha=0.3, label='Blind Point')
        axes[i].set_title(f"Model Survivability Boundary: {cat}", fontsize=16)
        axes[i].set_ylim(-0.05, 1.05)
        axes[i].grid(True, linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig("data/audit_results_summary.png", dpi=300)
    print("📈 Audit report generated in data/ folder.")

if __name__ == "__main__":
    plot_survivability()
