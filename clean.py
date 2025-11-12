import pandas as pd
import numpy as np
import os

# --- THIS IS THE SECTION YOU UPDATED ---
# Use a raw string (r'...') to fix the UnicodeError from backslashes '\'
FOLDER_PATH = r'C:\Users\Darshan kagadal\Downloads\customer_retention_telecom'
RAW_DATA_PATH = os.path.join(FOLDER_PATH, 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
CLEANED_DATA_PATH = os.path.join(FOLDER_PATH, 'cleaned_telco_churn.csv')


# --- END OF UPDATE ---

def clean_data():
    """
    Loads the raw Telco churn data from your specified path,
    cleans it, and saves the cleaned data back to that same folder.
    """
    print(f"Starting data cleaning...")
    print(f"Loading data from: {RAW_DATA_PATH}")

    try:
        # 1. Load the dataset
        df = pd.read_csv(RAW_DATA_PATH)
        print(f"Successfully loaded file. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: The file was not found at '{RAW_DATA_PATH}'")
        print("Please double-check that 'WA_Fn-UseC_-Telco-Customer-Churn.csv' is inside that folder.")
        return
    except Exception as e:
        print(f"An error occurred loading the file: {e}")
        return

    # --- Data Cleaning ---

    # 2. Drop customerID
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        print("Dropped 'customerID' column.")

    # 3. Convert 'TotalCharges' to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # 4. Handle missing values (the NaNs created from 'TotalCharges')
    initial_rows = df.shape[0]
    df = df.dropna(subset=['TotalCharges'])
    rows_dropped = initial_rows - df.shape[0]
    print(f"Dropped {rows_dropped} rows with missing 'TotalCharges'.")

    # 5. Convert 'SeniorCitizen' from 0/1 to No/Yes
    if 'SeniorCitizen' in df.columns:
        df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
        print("Converted 'SeniorCitizen' to 'No'/'Yes'.")

    # --- Save the Cleaned File ---
    try:
        df.to_csv(CLEANED_DATA_PATH, index=False)
        print("\n-------------------------------------------------")
        print(f"✅ Success! Cleaned data saved as: {CLEANED_DATA_PATH}")
        print(f"Final dataset shape: {df.shape}")
        print("-------------------------------------------------")
    except Exception as e:
        print(f"Error saving file: {e}")
        print(f"Please check if you have write permissions for the folder '{FOLDER_PATH}'.")


if __name__ == "__main__":
    clean_data()