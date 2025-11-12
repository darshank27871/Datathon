import pandas as pd
import numpy as np
import os

# Define file paths
CLEANED_DATA_PATH = 'cleaned_telco_churn.csv'
FEATURE_DATA_PATH = 'features_telco_churn.csv'


def engineer_features():
    """
    Loads the cleaned data, engineers new features,
    and saves the enriched data.
    """
    print(f"Starting feature engineering...")

    try:
        df = pd.read_csv(CLEANED_DATA_PATH)
        print(f"Loaded '{CLEANED_DATA_PATH}'. Original shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: The file '{CLEANED_DATA_PATH}' was not found.")
        print("Please run 'python 01_data_cleaning.py' first.")
        return

    # --- 1. Bin 'tenure' into groups ---
    # Binning a numerical feature can help the model see non-linear relationships.
    # 0-12 months: High-risk new customers
    # 13-36 months: Established customers
    # 37-60 months: Loyal customers
    # 61+ months: Veterans
    tenure_bins = [0, 12, 36, 60, 72]
    tenure_labels = ['New (0-12m)', 'Established (13-36m)', 'Loyal (37-60m)', 'Veteran (61m+)']
    df['Tenure_Group'] = pd.cut(df['tenure'], bins=tenure_bins, labels=tenure_labels, right=True)
    print("Engineered feature: 'Tenure_Group'")

    # --- 2. Aggregate Add-On Services ---
    # Instead of 6 separate 'Yes/No' columns, let's count how many add-ons
    # a customer has. This simplifies 6 features into one powerful one.
    addon_cols = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    # Count how many of these columns have the value 'Yes'
    df['TotalAddon_Count'] = df[addon_cols].apply(lambda row: (row == 'Yes').sum(), axis=1)
    print("Engineered feature: 'TotalAddon_Count'")

    # --- 3. Simplify Payment Method ---
    # The key difference in payment is often 'automatic' vs 'manual'.
    # Manual payments (mailed check, electronic check) require effort,
    # making it easier for a customer to "forget" or decide to churn.
    df['Is_Automatic_Payment'] = df['PaymentMethod'].str.contains('automatic', case=False, na=False).astype(int)
    print("Engineered feature: 'Is_Automatic_Payment'")

    # --- 4. Create Interaction Feature: Charge vs. Tenure ---
    # 'TotalCharges' is roughly 'tenure * MonthlyCharges'.
    # What if the current 'MonthlyCharges' are much higher than their
    # historical average? This could signal a price hike or promo ending.

    # Calculate historical average monthly charge
    # (df['tenure'] >= 1 is guaranteed from cleaning step)
    df['Avg_Monthly_Charge'] = df['TotalCharges'] / df['tenure']

    # Calculate the difference. A high positive number is a "red flag".
    df['Monthly_vs_Average_Diff'] = df['MonthlyCharges'] - df['Avg_Monthly_Charge']
    print("Engineered feature: 'Monthly_vs_Average_Diff'")

    # --- Save the Enriched File ---
    df.to_csv(FEATURE_DATA_PATH, index=False)

    print("\n-------------------------------------------------")
    print(f"✅ Success! Enriched data saved as: {FEATURE_DATA_PATH}")
    print(f"New dataset shape: {df.shape}")
    print("\n--- New Columns Info ---")
    print(df[['tenure', 'Tenure_Group', 'TotalAddon_Count', 'Is_Automatic_Payment', 'Monthly_vs_Average_Diff']].head())
    print("\nNew Data Info:")
    df.info()
    print("-------------------------------------------------")


if __name__ == "__main__":
    engineer_features()