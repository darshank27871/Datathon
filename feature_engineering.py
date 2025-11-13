import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    """Load the CSV data."""
    try:
        df = pd.read_csv(file_path)
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        if 'Churn' in df.columns and df['Churn'].dtype == object:
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).fillna(df['Churn'])
        logger.info(f"Data loaded successfully from {file_path}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def perform_eda(df):
    """Perform exploratory data analysis and save summary."""
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'describe': df.describe(),
        'missing_values': df.isnull().sum()
    }
    logger.info("EDA summary generated")
    # Save EDA to file for reproducibility
    with open('eda_summary.txt', 'w') as f:
        for key, value in summary.items():
            f.write(f"{key}:\n{value}\n\n")
    return summary

def handle_missing_outliers(df):
    """Handle missing values and outliers."""
    # Handle missing values (assuming numerical imputation with median, categorical with mode)
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Handle outliers (simple clipping to 1.5 IQR)
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)
    
    logger.info("Missing values and outliers handled")
    return df

def create_derived_features(df):
    """Create derived features."""
    if 'tenure' in df.columns:
        df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 36, np.inf], labels=['Low', 'Medium', 'High'])
    if 'MonthlyCharges' in df.columns:
        df['MonthlyChargesGroup'] = pd.cut(df['MonthlyCharges'], bins=[0, 50, 80, np.inf], labels=['Low', 'Medium', 'High'])
    if {'tenure', 'TotalCharges', 'MonthlyCharges'}.issubset(df.columns):
        df['AvgMonthlyCharge'] = np.where(df['tenure'] > 0, df['TotalCharges'] / df['tenure'], df['MonthlyCharges'])
    elif 'MonthlyCharges' in df.columns:
        df['AvgMonthlyCharge'] = df['MonthlyCharges']
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    existing_service_cols = [c for c in service_cols if c in df.columns]
    if existing_service_cols:
        df['NumServices'] = df[existing_service_cols].apply(lambda x: (x == 'Yes')).sum(axis=1)
    else:
        df['NumServices'] = 0
    if 'Contract' in df.columns:
        contract_map = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
        df['ContractMonths'] = df['Contract'].map(contract_map).fillna(0).astype(int)
    elif 'ContractMonths' not in df.columns:
        df['ContractMonths'] = 0
    logger.info("Derived features created")
    return df

def feature_transformations(df):
    """Apply scaling and encoding."""
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyCharge', 'NumServices', 'ContractMonths']
    cat_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
                'TenureGroup', 'MonthlyChargesGroup']
    
    pre_num = ('num', Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), [c for c in num_cols if c in df.columns])
    pre_cat = ('cat', Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), [c for c in cat_cols if c in df.columns])
    transformers = []
    if pre_num[2]:
        transformers.append(pre_num)
    if pre_cat[2]:
        transformers.append(pre_cat)
    preprocessor = ColumnTransformer(transformers=transformers)
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_transformed = preprocessor.fit_transform(X)
    
    # Get feature names
    feature_names = preprocessor.get_feature_names_out()
    
    # Since sparse_threshold=0.0, it should be dense
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
    
    logger.info("Feature transformations applied")
    return X_transformed_df, y, preprocessor

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    stratify_opt = y if pd.Series(y).nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_opt)
    logger.info("Data split into train and test sets")
    return X_train, X_test, y_train, y_test

def main():
    file_path = r'c:\Users\varma\Videos\Datathon\datastuff\cleaned_telco_churn.csv'
    df = load_data(file_path)
    perform_eda(df)
    df = handle_missing_outliers(df)
    df = create_derived_features(df)
    X_transformed, y, preprocessor = feature_transformations(df)
    X_train, X_test, y_train, y_test = split_data(X_transformed, y)
    
    # Save intermediate outputs
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)
    
    import joblib
    joblib.dump(preprocessor, 'preprocessor.pkl')
    
    logger.info("Feature engineering completed")

# Unit tests
import unittest

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'tenure': [1, 34, 0],
            'TotalCharges': [29.85, 1889.5, 0],
            'MonthlyCharges': [29.85, 56.95, 20],
            'Churn': [0, 0, 1]
        })

    def test_create_derived_features(self):
        df_new = create_derived_features(self.df)
        self.assertIn('AvgMonthlyCharge', df_new.columns)
        self.assertEqual(df_new['AvgMonthlyCharge'].iloc[0], 29.85)

    def test_handle_missing_outliers(self):
        self.df.loc[0, 'tenure'] = np.nan
        df_new = handle_missing_outliers(self.df)
        self.assertFalse(df_new.isnull().any().any())

if __name__ == '__main__':
    main()
    unittest.main(argv=[''], exit=False)