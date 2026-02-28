import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_audit_report():
    df = pd.read_csv("data/audit_report.csv")
    plt.style.use('dark_background')
    categories = df['Category'].unique()
    
    # Automatically detect confidence columns added during audit
    conf_cols = [col for col in df.columns if col.startswith('Conf_')]
    colors = sns.color_palette("husl", len(conf_cols))

    fig, axes = plt.subplots(len(categories), 1, figsize=(12, 5 * len(categories)))

    for i, cat in enumerate(categories):
        data = df[df['Category'] == cat].sort_values('Param_Value')
        for j, col in enumerate(conf_cols):
            sns.lineplot(ax=axes[i], x='Param_Value', y=col, data=data, 
                         label=col.replace('Conf_', 'Model '), color=colors[j], linewidth=2.5)
        
        axes[i].axhline(y=0.0, color='red', linestyle='-', alpha=0.3, label='Blind Point')
        axes[i].set_title(f"Model Survivability: {cat}", fontsize=16)
        axes[i].set_ylim(-0.05, 1.05)
        axes[i].grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.savefig("data/survivability_report.png", dpi=300)
    print("📈 Report exported to data/survivability_report.png")

if __name__ == "__main__":
    plot_audit_report()
