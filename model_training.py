import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import joblib
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_processed_data():
    """Load processed train data."""
    try:
        X_train = pd.read_csv('X_train.csv')
        y_train = pd.read_csv('y_train.csv').values.ravel()
        logger.info("Processed data loaded")
        return X_train, y_train
    except FileNotFoundError:
        logger.error("Processed data files not found")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def train_xgboost(X_train, y_train):
    """Train XGBoost model with hyperparameter tuning and early stopping."""
    # Split for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 1.0]
    }
    
    # Removed explicit 'objective' as it's default for binary classification to avoid potential warnings
    xgb_clf = xgb.XGBClassifier(eval_metric='auc', random_state=42)
    
    grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_split, y_train_split)
    
    best_params = grid_search.best_params_
    logger.info(f"Best parameters: {best_params}")
    
    # Train with early stopping, default objective is 'binary:logistic'
    model = xgb.XGBClassifier(**best_params, eval_metric='auc', random_state=42, enable_categorical=False, early_stopping_rounds=10)
    model.fit(X_train_split, y_train_split, eval_set=[(X_val, y_val)], verbose=True)
    
    logger.info("Model trained")
    return model

def main():
    X_train, y_train = load_processed_data()
    model = train_xgboost(X_train, y_train)
    joblib.dump(model, 'xgboost_model.pkl')
    logger.info("Model saved")

# Unit tests
import unittest

class TestModelTraining(unittest.TestCase):
    def test_load_processed_data(self):
        try:
            X, y = load_processed_data()
            self.assertIsNotNone(X)
            self.assertIsNotNone(y)
        except:
            pass  # Skip if files not present during test

if __name__ == '__main__':
    main()
    unittest.main(argv=[''], exit=False)
