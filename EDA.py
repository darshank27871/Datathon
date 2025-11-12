import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define file paths
CLEANED_DATA_PATH = 'features_telco_churn.csv'
PLOT_OUTPUT_DIR = os.path.join('output', 'plots')


def run_eda():
    """
    Loads the cleaned data and generates all EDA plots,
    saving them to the 'output/plots' directory.
    """
    print("Starting Exploratory Data Analysis (EDA)...")

    # Ensure the plot directory exists
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

    try:
        df = pd.read_csv(CLEANED_DATA_PATH)
        print(f"Loaded '{CLEANED_DATA_PATH}'.")
    except FileNotFoundError:
        print(f"Error: The file '{CLEANED_DATA_PATH}' was not found.")
        print("Please run 'python 01_data_cleaning.py' first.")
        return

    sns.set_style("whitegrid")

    # 1. Churn Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Churn')
    plt.title('Customer Churn Distribution')
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, '01_churn_distribution.png'))
    plt.close()

    # 2. Contract vs. Churn
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Contract', hue='Churn', palette='pastel')
    plt.title('Churn Rate by Contract Type')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, '02_contract_churn.png'))
    plt.close()

    # 3. Internet Service vs. Churn
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='InternetService', hue='Churn', palette='pastel')
    plt.title('Churn Rate by Internet Service')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, '03_internet_service_churn.png'))
    plt.close()

    # 4. Numerical Features Histograms
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(df['tenure'], kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('Distribution of Tenure')
    sns.histplot(df['MonthlyCharges'], kde=True, ax=axes[1], color='salmon')
    axes[1].set_title('Distribution of Monthly Charges')
    sns.histplot(df['TotalCharges'], kde=True, ax=axes[2], color='lightgreen')
    axes[2].set_title('Distribution of Total Charges')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, '04_numerical_histograms.png'))
    plt.close()

    # 5. Numerical Features vs. Churn (Box Plots)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sns.boxplot(data=df, x='Churn', y='tenure', ax=axes[0], palette='coolwarm')
    axes[0].set_title('Tenure vs. Churn')
    sns.boxplot(data=df, x='Churn', y='MonthlyCharges', ax=axes[1], palette='coolwarm')
    axes[1].set_title('Monthly Charges vs. Churn')
    sns.boxplot(data=df, x='Churn', y='TotalCharges', ax=axes[2], palette='coolwarm')
    axes[2].set_title('Total Charges vs. Churn')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, '05_numerical_vs_churn_boxplots.png'))
    plt.close()

    # 6. Correlation Heatmap
    df_corr = df.copy()
    df_corr['Churn'] = df_corr['Churn'].map({'No': 0, 'Yes': 1})
    numerical_cols = df_corr.select_dtypes(include=np.number).columns
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_corr[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, '06_correlation_heatmap.png'))
    plt.close()

    print(f"✅ Success! All EDA plots saved to '{PLOT_OUTPUT_DIR}'.")


if __name__ == "__main__":
    run_eda()