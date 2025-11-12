import pandas as pd
import numpy as np
import os

# Define file paths
# Load the file created by the *previous* feature engineering step
FEATURE_DATA_PATH = 'features_telco_churn.csv'
# Define the new, final file path
FINAL_DATA_PATH = 'final_features_telco_churn.csv'


def create_clv_feature():
    """
    Loads the feature-engineered data, adds a CLV tier,
    and saves the final dataset for modeling.
    """
    print(f"Loading data from '{FEATURE_DATA_PATH}'...")

    try:
        df = pd.read_csv(FEATURE_DATA_PATH)
        print(f"Successfully loaded. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: The file '{FEATURE_DATA_PATH}' was not found.")
        print("Please run 'python create_features.py' first.")
        return

    # --- Engineer Customer Value Tier ---
    # We use 'MonthlyCharges' as it's the best proxy for *future* value
    # that we are trying to save.

    # We use pd.qcut (Quantile-based cut) to split the customers
    # into 4 equal-sized groups (quartiles).

    clv_labels = ['Low_Value', 'Medium_Value', 'High_Value', 'Top_Value']
    df['Customer_Value_Tier'] = pd.qcut(df['MonthlyCharges'],
                                        q=4,
                                        labels=clv_labels,
                                        duplicates='drop')  # Handle if quantiles overlap

    print("Engineered feature: 'Customer_Value_Tier'")

    # --- Save the Final File ---
    df.to_csv(FINAL_DATA_PATH, index=False)

    print("\n-------------------------------------------------")
    print(f"✅ Success! Final feature data saved as: {FINAL_DATA_PATH}")
    print(f"New dataset shape: {df.shape}")
    print("\n--- New Column Info ---")
    print(df[['MonthlyCharges', 'Customer_Value_Tier']].head())
    print("\nValue Tier Distribution:")
    print(df['Customer_Value_Tier'].value_counts())
    print("-------------------------------------------------")


if __name__ == "__main__":
    create_clv_feature()