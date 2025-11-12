import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide"
)

# --- Model Loading ---
# Path to the model
MODEL_PATH = os.path.join('output', 'models', 'random_forest_model_optimized.joblib')


@st.cache_resource
def load_model(model_path):
    """Loads the saved model pipeline from disk."""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}.")
        st.error("Please run the '03_model_training_optimized.py' script to train and save the model.")
        return None
    except Exception as e:
        st.error(f"An error occurred loading the model: {e}")
        return None


model = load_model(MODEL_PATH)


# --- Feature Engineering Function ---
# This function MUST be identical to the one used in training
def engineer_features(df):
    """
    Engineers the 5 new features from the raw user inputs.
    This must match the training script's logic exactly.
    """

    # 1. Tenure_Group
    tenure_bins = [0, 12, 36, 60, 72]
    tenure_labels = ['New (0-12m)', 'Established (13-36m)', 'Loyal (37-60m)', 'Veteran (61m+)']
    df['Tenure_Group'] = pd.cut(df['tenure'], bins=tenure_bins, labels=tenure_labels, right=True).astype(str)

    # 2. TotalAddon_Count
    addon_cols = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    df['TotalAddon_Count'] = df[addon_cols].apply(lambda row: (row == 'Yes').sum(), axis=1)

    # 3. Is_Automatic_Payment
    df['Is_Automatic_Payment'] = df['PaymentMethod'].str.contains('automatic', case=False, na=False).astype(int)

    # 4. Monthly_vs_Average_Diff
    # Handle potential division by zero if tenure is 0 (though slider min is 1)
    df['tenure'] = df['tenure'].replace(0, 1)  # Safety clamp
    df['Avg_Monthly_Charge'] = df['TotalCharges'] / df['tenure']
    df['Monthly_vs_Average_Diff'] = df['MonthlyCharges'] - df['Avg_Monthly_Charge']
    df = df.drop(columns=['Avg_Monthly_Charge'])

    # 5. Customer_Value_Tier
    # We use qcut's logic. Since we only have one value, we'll assign it
    # based on the quartiles from the original data.
    # Original data quartiles: [25%: 35.59, 50%: 70.35, 75%: 89.86]
    mc = df['MonthlyCharges'].iloc[0]
    if mc <= 35.59:
        df['Customer_Value_Tier'] = 'Low_Value'
    elif mc <= 70.35:
        df['Customer_Value_Tier'] = 'Medium_Value'
    elif mc <= 89.86:
        df['Customer_Value_Tier'] = 'High_Value'
    else:
        df['Customer_Value_Tier'] = 'Top_Value'

    return df


# --- Streamlit UI ---
st.title("📡 Telco Customer Churn Prediction")
st.markdown(
    "This app uses a machine learning model to predict the likelihood of a customer churning (leaving the company).")
st.markdown("Adjust the customer's details in the sidebar to see the prediction.")

# --- Sidebar for Inputs ---
st.sidebar.header("Customer Details")


# Helper function for inputs
def get_user_inputs():
    # --- Demographic ---
    st.sidebar.subheader("Demographics")
    gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
    SeniorCitizen = st.sidebar.selectbox("Senior Citizen", ['No', 'Yes'])
    Partner = st.sidebar.selectbox("Partner", ['No', 'Yes'])
    Dependents = st.sidebar.selectbox("Dependents", ['No', 'Yes'])

    # --- Account Info ---
    st.sidebar.subheader("Account Information")
    tenure = st.sidebar.slider("Tenure (months)", 1, 72, 12)
    Contract = st.sidebar.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ['No', 'Yes'])
    PaymentMethod = st.sidebar.selectbox("Payment Method",
                                         ['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
                                          'Credit card (automatic)'])
    MonthlyCharges = st.sidebar.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=70.7,
                                             step=0.05)
    TotalCharges = st.sidebar.number_input("Total Charges ($)", min_value=18.0, value=151.65, step=1.0)

    # --- Service Info ---
    st.sidebar.subheader("Services Subscribed")
    PhoneService = st.sidebar.selectbox("Phone Service", ['No', 'Yes'])
    MultipleLines = st.sidebar.selectbox("Multiple Lines", ['No', 'Yes', 'No phone service'])
    InternetService = st.sidebar.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])

    # --- Add-on Services ---
    st.sidebar.subheader("Add-on Services")
    OnlineSecurity = st.sidebar.selectbox("Online Security", ['No', 'Yes', 'No internet service'])
    OnlineBackup = st.sidebar.selectbox("Online Backup", ['No', 'Yes', 'No internet service'])
    DeviceProtection = st.sidebar.selectbox("Device Protection", ['No', 'Yes', 'No internet service'])
    TechSupport = st.sidebar.selectbox("Tech Support", ['No', 'Yes', 'No internet service'])
    StreamingTV = st.sidebar.selectbox("Streaming TV", ['No', 'Yes', 'No internet service'])
    StreamingMovies = st.sidebar.selectbox("Streaming Movies", ['No', 'Yes', 'No internet service'])

    # --- Create Dictionary from inputs ---
    input_data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
    }
    return pd.DataFrame([input_data])


# Get inputs
input_df = get_user_inputs()

# --- Prediction and Display ---
if model is not None:
    # 1. Engineer features from the user's input
    try:
        final_input_df = engineer_features(input_df.copy())

        # 2. Make prediction
        prediction_proba = model.predict_proba(final_input_df)[0]
        churn_probability = prediction_proba[1]  # Probability of "Yes" (Churn)

        # 3. Display the results
        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Churn Probability",
                value=f"{churn_probability:.1%}",
                delta=f"{churn_probability - 0.2658:.1%} vs. average"  # 0.2658 is the avg churn rate
            )

            # Show a colored bar
            st.progress(churn_probability)

        with col2:
            if churn_probability > 0.6:
                st.error("🔴 HIGH RISK")
                st.markdown(
                    "This customer is **very likely** to churn. Recommend immediate proactive retention action.")
            elif churn_probability > 0.3:
                st.warning("🟠 MEDIUM RISK")
                st.markdown("This customer shows some risk of churning. Consider targeted promotions or a check-in.")
            else:
                st.success("🟢 LOW RISK")
                st.markdown("This customer is **unlikely** to churn. No immediate action required.")

        # --- Show Engineered Features (for transparency) ---
        with st.expander("Show Engineered Features (What the model *actually* sees)"):
            st.dataframe(final_input_df.drop(columns=[col for col in input_df.columns if
                                                      col in final_input_df.columns and col not in ['tenure',
                                                                                                    'MonthlyCharges',
                                                                                                    'TotalCharges']]))

    except Exception as e:
        st.error(f"An error occurred during feature engineering or prediction: {e}")
        st.error("Please check your inputs and the model file.")

else:
    st.info("Model is not loaded. Please ensure the model file exists and the app is restarted.")