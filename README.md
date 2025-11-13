# Customer Churn Prediction Pipeline Using XGBoost ML Algorithm

## Project Overview

This project implements a machine learning pipeline to predict customer
churn based on historical customer data using the XGBoost algorithm.
The primary objective is to identify customers who are likely to 
discontinue their service, allowing for proactive retention efforts.
This pipeline covers the entire workflow, from data loading to feature
engineering, model training, evaluation, and visualization.

- **Key Objectives**: Predict customer churn, identify key drivers of
churn, and provide actionable insights for retention strategies.
- **Problem Domain**: Customer relationship management, retention
strategy.
- **Expected Inputs**: A CSV file containing customer data ('cleaned_telco_churn.csv')
- **Expected Outputs**: A trained XGBoost model, evaluation metrics, 
feature importance, and visualizations.

## File Descriptions

- **feature_engineering.py**: This script handles data preparation,
including:
  - Loading the raw data.
  - Performing exploratory data analysis (EDA).
  - Handling missing values and outliers.
  - Creating derived features to enhance predictive power.
  - Applying feature transformations such as scaling and one-hot encoding.

- **model_training.py**: This script focuses on training the prediction
model:
  - It uses the processed data from feature engineering step.
  - An XGBoost classifier is trained with hyperparameter tuning using
    `GridSearchCV`.
  - The best-trained model is saved to a file (`xgboost_model.pkl`).

- **evaluation.py**: This script evaluates the performance of the trained model:
  - It loads the test data and the saved model.
  - It generates a classification report, confusion matrix, and AUC score.
  - Feature importances are calculated and saved.
  - The results are in `evaluation_report.json` and `confusion_matrix.png`.

- **visualization.py**: This script creates visualizations to interpret
the model's predictions and behavior:
  - It plots the distribution of actual vs predicted churn.
  - It visualizes the top feature importances.
  - It generates a distribution plot of churn probabilities.

- **app.py**: This script implements a Streamlit web application for
interactive customer churn prediction.
  - Input real-time customer data to get predictions on whether the
    customer is likely to churn.
  - Displays the prediction results and probability of churn.


## Installation instructions

To set up the environment, install the required Python packages:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
```

No special configuration is needed.

## Usage Guide

Execute the scripts in the following order:

1. **feature_engineering.py**:
   ```bash
   python feature_engineering.py
   ```

2. **model_training.py**:
   ```bash
   python model_training.py
   ```

3. **evaluation.py**:
   ```bash
   python evaluation.py
   ```

4. **visualization.py**:
   ```bash
   python visualization.py
   ```

5. **app.py**:
   ```bash
   streamlit run app.py
   ```

## Results Interpretation

- **`xgboost_model.pkl`**: The trained and saved machine learning model.

- **`evaluation_report.json`**: Contains the detailed performance metrics
(precision, recall, F1-score, AUC) and a list of features ranked by
importance.

- **`confusion_matrix.png`**: A visual representation of the model's 
performance, showing correct and incorrect predictions.

- **`churn_distribution.png`**: A bar chart comparing the distribution 
of actual vs predicted churn.

- **`feature_importance.png`**: A bar chart showing the most influential
features in the model.

- **`probability_distribution.png`**: A histogram of the predicted churn probabilites, which can be useful for setting decision thresholds.