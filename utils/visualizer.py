import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class PhysicalVisualizer:
    def __init__(self, csv_path="data/audit_report.csv"):
        self.csv_path = csv_path
        # 设置高端工业深色主题
        plt.style.use('dark_background') 
        self.accent_color = "#FF9F1C"  # 工业警戒橙
        self.safety_color = "#2EC4B6"  # 工业安全青
        self.danger_color = "#E71D36"  # 错误红

    def plot_robustness(self, save_path="data/robustness_curves.png"):
        if not os.path.exists(self.csv_path):
            print("❌ Error: Audit report not found.")
            return

        df = pd.read_csv(self.csv_path)
        categories = df['Category'].unique()
        
        fig, axes = plt.subplots(len(categories), 1, figsize=(12, 6 * len(categories)))
        if len(categories) == 1: axes = [axes]

        for i, cat in enumerate(categories):
            subset = df[df['Category'] == cat].sort_values('Param_Value')
            
            # 绘制带有发光效果的平滑曲线
            sns.lineplot(ax=axes[i], data=subset, x='Param_Value', y='Confidence', 
                         color=self.accent_color, marker='o', markersize=6, 
                         linewidth=3, label=f'{cat} Sensitivity Curve')
            
            # 阴影填充：标定 SOA (Safe Operating Area)
            axes[i].fill_between(subset['Param_Value'], subset['Confidence'], 
                                 color=self.accent_color, alpha=0.1)

            # 绘制阈值红线 (80% 置信度红线)
            axes[i].axhline(y=0.8, color=self.safety_color, linestyle='--', alpha=0.6, label='Reliability Threshold (80%)')
            
            # 装饰细节
            axes[i].set_title(f"Industrial AI Robustness: {cat} Analysis", fontsize=16, pad=20, color='white')
            axes[i].set_ylim(0, 1.1)
            axes[i].grid(True, linestyle=':', alpha=0.3)
            axes[i].set_xlabel(f"Physical Stress Value ({cat})", fontsize=12)
            axes[i].set_ylabel("AI Confidence Score", fontsize=12)
            axes[i].legend(loc='lower left', frameon=True, facecolor='#222222')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 [SUCCESS] Professional Report Generated: {save_path}")