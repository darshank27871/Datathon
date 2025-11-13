import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import logging
import json 
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_test_data_and_model():
    ''' Load test data and trained model.'''
    try:
        X_test = pd.read_csv('X_test.csv')
        y_test = pd.read_csv('y_test.csv').values.ravel()
        model = joblib.load('xgboost_model.pkl')
        logger.info("Test data and model loaded")
        return X_test, y_test, model
    except FileNotFoundError:
        logger.error("Files not found")
        raise
    except Exception as e:
        logger.error(f"Error loading: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = classification_report(y_test, y_pred, output_dict=True)
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        auc = None
    metrics['auc'] = auc

    cm  = confusion_matrix(y_test, y_pred)

    importance = model.feature_importances_
    feature_names = X_test.columns
    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importance}).sort_values('importance', ascending=False)

    logger.info("Evaluation completed")
    return metrics, cm, feat_imp

def visualize_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()
    logger.info("Confusion matrix saved")

def save_report(metrics, feat_imp):
    report = {'metrics': metrics, 'feature_importance': feat_imp.to_dict(orient='records')}
    with open('evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    logger.info("Evaluation report saved")

def main():
    X_test, y_test, model = load_test_data_and_model()
    metrics, cm, feat_imp = evaluate_model(model, X_test, y_test)
    visualize_confusion_matrix(cm)
    save_report(metrics, feat_imp)

import unittest

class TestEvaluation(unittest.TestCase):
    def test_evaluate_model(self):
        model = xgb.XGBClassifier()
        # Fit with a minimal two-class dataset
        model.fit([[0], [1]], [0, 1])
        X_test = pd.DataFrame([[0], [1]])
        y_test = [0, 1]
        metrics, cm, feat_imp = evaluate_model(model, X_test, y_test)
        self.assertIsInstance(metrics, dict)
        self.assertIsNotNone(cm)
        self.assertIsNotNone(feat_imp)

if __name__ == '__main__':
    main()
    unittest.main(argv=[''], exit=False)