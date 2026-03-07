import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class AuditReporter:
    def __init__(self, data: list):
        self.df = pd.DataFrame(data)

    def to_csv(self, filepath: str):
        self.df.to_csv(filepath, index=False)

    def plot_trend(self, filepath: str):
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=self.df, x="level", y="conf", hue="dimension", marker="o")
        plt.title("AI Model Robustness Trend")
        plt.grid(True)
        plt.savefig(filepath)
        plt.close()