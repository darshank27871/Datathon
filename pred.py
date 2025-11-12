import pandas as pd
import joblib
import os

# Define file path
MODEL_PATH = os.path.join('output', 'models', 'random_forest_model.joblib')


def predict_single_customer(customer_data):
    """
    Loads the saved model pipeline and makes a prediction
    on a single customer's data.

    Args:
    customer_data (dict): A dictionary with all the features
                          for a single customer.
    """

    try:
        # Load the trained model pipeline
        model_pipeline = joblib.load(MODEL_PATH)
        print(f"Loaded model pipeline from '{MODEL_PATH}'.")
    except FileNotFoundError:
        print(f"Error: The model file '{MODEL_PATH}' was not found.")
        print("Please run 'python 03_model_training.py' first.")
        return

    # Convert the dictionary to a DataFrame
    customer_df = pd.DataFrame([customer_data])

    # Make prediction
    prediction = model_pipeline.predict(customer_df)

    # Get probabilities
    probabilities = model_pipeline.predict_proba(customer_df)

    churn_probability = probabilities[0][1]
    prediction_label = "Yes" if prediction[0] == 1 else "No"

    print("\n--- Prediction Result ---")
    print(f"Customer Profile: {customer_data['Contract']} contract, {customer_data['tenure']} months tenure.")
    print(f"Predicted Churn: {prediction_label}")
    print(f"Probability of Churn: {churn_probability:.2%}")
    print("--------------------------")


if __name__ == "__main__":
    # Define a sample customer (high-risk profile)
    sample_customer = {
        'gender': 'Female',
        'SeniorCitizen': 'No',
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 2,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 70.70,
        'TotalCharges': 151.65
    }

    print("--- Predicting for High-Risk Customer ---")
    predict_single_customer(sample_customer)

    # Example of a low-risk customer
    sample_customer_loyal = {
        'gender': 'Male',
        'SeniorCitizen': 'No',
        'Partner': 'Yes',
        'Dependents': 'Yes',
        'tenure': 70,
        'PhoneService': 'Yes',
        'MultipleLines': 'Yes',
        'InternetService': 'DSL',
        'OnlineSecurity': 'Yes',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'Yes',
        'TechSupport': 'Yes',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Two year',
        'PaperlessBilling': 'No',
        'PaymentMethod': 'Credit card (automatic)',
        'MonthlyCharges': 89.80,
        'TotalCharges': 6500.00
    }

    print("\n--- Predicting for Low-Risk (Loyal) Customer ---")
    predict_single_customer(sample_customer_loyal)