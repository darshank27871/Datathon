import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# Define file paths
CLEANED_DATA_PATH = 'features_telco_churn.csv'
MODEL_OUTPUT_DIR = os.path.join('output', 'models')
MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, 'random_forest_model.joblib')


def train_model():
    """
    Loads cleaned data, trains a Random Forest classifier,
    evaluates it, and saves the trained pipeline.
    """
    print("Starting model training...")

    # Ensure the model directory exists
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    try:
        df = pd.read_csv(CLEANED_DATA_PATH)
        print(f"Loaded '{CLEANED_DATA_PATH}'.")
    except FileNotFoundError:
        print(f"Error: The file '{CLEANED_DATA_PATH}' was not found.")
        print("Please run 'python 01_data_cleaning.py' first.")
        return

    # --- 1. Define Features (X) and Target (y) ---
    X = df.drop('Churn', axis=1)
    y = df['Churn'].map({'No': 0, 'Yes': 1})

    # --- 2. Identify Numerical and Categorical Features ---
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()
    print(f"Numerical features: {numerical_features}")
    print(f"Categorical features: {categorical_features}")

    # --- 3. Create Preprocessing Pipeline ---
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # --- 4. Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # --- 5. Create Random Forest Model Pipeline ---
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ))
    ])

    # --- 6. Train the Model ---
    print("Training Random Forest model...")
    rf_pipeline.fit(X_train, y_train)
    print("Training complete.")

    # --- 7. Evaluate the Model ---
    print("\n--- Model Evaluation (on Test Set) ---")
    y_pred = rf_pipeline.predict(X_test)
    y_prob = rf_pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # --- 8. Save the Model ---
    joblib.dump(rf_pipeline, MODEL_PATH)
    print("\n-------------------------------------------------")
    print(f"✅ Success! Model pipeline saved to: {MODEL_PATH}")
    print("-------------------------------------------------")


if __name__ == "__main__":
    train_model()