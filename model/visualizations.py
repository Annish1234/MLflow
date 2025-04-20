# model/visualizations.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def generate_visual_report():
    df = pd.read_csv("creditcard_10k_balanced.csv")
    df.columns = df.columns.str.strip()

    with PdfPages("visual_report.pdf") as pdf:
        # Plot 1: Class distribution
        sns.countplot(x='Class', data=df)
        plt.title('Class Distribution')
        pdf.savefig()
        plt.close()

        # Plot 2: Heatmap
        sns.heatmap(df.corr(), cmap='coolwarm')
        plt.title('Correlation Heatmap')
        pdf.savefig()
        plt.close()

        # Plot 3: Amount by class
        sns.boxplot(x='Class', y='Amount', data=df)
        plt.title('Transaction Amount by Class')
        pdf.savefig()
        plt.close()

        # Plot 4: Amount histogram
        sns.histplot(df['Amount'], bins=30, kde=True)
        plt.title('Transaction Amount Distribution')
        pdf.savefig()
        plt.close()
