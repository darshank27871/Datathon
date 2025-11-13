import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data_and_report():
    """Load test data, model, and report for visualization."""
    try:
        X_test = pd.read_csv('X_test.csv')
        y_test = pd.read_csv('y_test.csv').values.ravel()
        model = joblib.load('xgboost_model.pkl')
    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e}. Please run feature_engineering.py and model_training.py first.")
        raise

    # Create a dataframe for plotting distributions
    df = X_test.copy()
    df['Churn'] = y_test
    df['PredictedChurn'] = model.predict(X_test)

    # Load feature importance, with a fallback to the model's importances
    try:
        with open('evaluation_report.json', 'r') as f:
            report = json.load(f)
        feat_imp = pd.DataFrame(report['feature_importance'])
    except (FileNotFoundError, KeyError):
        logger.warning("evaluation_report.json not found or malformed. Falling back to model's feature importances.")
        importance = model.feature_importances_
        feature_names = X_test.columns
        feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importance}).sort_values('importance', ascending=False)

    logger.info("Data and report loaded for visualization")
    return df, feat_imp, X_test, model

def plot_churn_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.countplot(x='Churn', data=df, ax=axes[0])
    axes[0].set_title('Actual Churn Distribution')
    sns.countplot(x='PredictedChurn', data=df, ax=axes[1])
    axes[1].set_title('Predicted Churn Distribution')
    plt.savefig('churn_distribution.png', dpi=300)
    plt.close()
    logger.info("Churn distribution plot saved")

def plot_feature_importance(feat_imp):
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feat_imp.head(20))
    plt.title('Top 20 Feature Importances')
    plt.savefig('feature_importance.png', dpi=300)
    plt.close()
    logger.info("Feature importance plot saved")

def plot_probability_distribution(model, X_test):
    """Plot the distribution of churn probabilities."""
    probs = model.predict_proba(X_test)[:, 1]

    plt.figure(figsize=(8, 6))
    sns.histplot(probs, kde=True)
    plt.title('Churn Probability Distribution')
    plt.xlabel('Probability of Churn')
    plt.savefig('probability_distribution.png', dpi=300)
    plt.close()
    logger.info("Probability distribution plot saved")

def main():
    df, feat_imp, X_test, model = load_data_and_report()
    plot_churn_distribution(df)
    plot_feature_importance(feat_imp)
    plot_probability_distribution(model, X_test)

import unittest

class TestVisualization(unittest.TestCase):
    def test_plot_churn_distribution(self):
        df = pd.DataFrame({'Churn': [0, 1, 0], 'PredictedChurn': [0, 1, 1]})
        plot_churn_distribution(df)

if __name__ == '__main__':
    main()
    unittest.main(argv=[''], exit=False)