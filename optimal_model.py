import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from scipy.stats import randint

# Define file paths
CLEANED_DATA_PATH = 'cleaned_telco_churn.csv'
MODEL_OUTPUT_DIR = os.path.join('output', 'models')
MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, 'random_forest_model_optimized.joblib')


def optimize_model():
    """
    Loads cleaned data, tunes a Random Forest classifier using
    RandomizedSearchCV, evaluates it, and saves the best pipeline.
    """
    print("Starting model optimization with RandomizedSearchCV...")

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

    # --- 5. Create Random Forest Pipeline ---
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ))
    ])

    # --- 6. Define the Hyperparameter Grid for RandomizedSearchCV ---
    param_dist = {
        'classifier__n_estimators': randint(100, 500),  # Number of trees
        'classifier__max_depth': [None] + list(randint(10, 30).rvs(5)),  # Max depth
        'classifier__min_samples_split': randint(2, 11),  # Min samples to split
        'classifier__min_samples_leaf': randint(1, 11),  # Min samples at a leaf
        'classifier__bootstrap': [True, False]  # Whether to bootstrap
    }

    # --- 7. Set up RandomizedSearchCV ---
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=25,  # Try 25 different combinations
        cv=3,  # 3-fold cross-validation
        scoring='roc_auc',
        random_state=42,
        n_jobs=-1  # Use all available cores
    )

    # --- 8. Train the Model ---
    print("\nTraining and tuning model... (This may take a few minutes)")
    random_search.fit(X_train, y_train)
    print("Optimization complete.")

    # --- 9. Show Best Parameters ---
    print("\n--- Best Hyperparameters Found ---")
    print(random_search.best_params_)

    # --- 10. Evaluate the Best Model Found ---
    print("\n--- Best Model Evaluation (on Test Set) ---")
    best_model = random_search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Optimized Accuracy: {accuracy:.4f}")
    print(f"Optimized ROC AUC Score: {roc_auc:.4f}")
    print("\nOptimized Classification Report:")
    print(classification_report(y_test, y_pred))

    # --- 11. Save the Optimized Model ---
    joblib.dump(best_model, MODEL_PATH)
    print("\n-------------------------------------------------")
    print(f"✅ Success! Optimized model pipeline saved to: {MODEL_PATH}")
    print("-------------------------------------------------")


if __name__ == "__main__":
    optimize_model()