import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time

#Load the trained model and preprocessor
try:
    model = joblib.load('xgboost_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    st.success("Model and preprocessor loaded successfully!")
except FileNotFoundError:
    st.error("Model or preprocessor not found. Please run the training pipeline first as described in the README.md")
    st.stop()

st.title("Customer Churn Prediction")
st.write("Enter the customer details to predict churn.")

# Create input fields
st.sidebar.header("Customer Data")

def user_input_features():
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    senior_citizen_option = st.sidebar.selectbox('Senior Citizen', ('No', 'Yes'))
    senior_citizen = 1 if senior_citizen_option == 'Yes' else 0
    partner = st.sidebar.selectbox('Partner', ('Yes', 'No'))
    dependents = st.sidebar.selectbox('Dependents', ('Yes', 'No'))
    tenure = st.sidebar.slider('Tenure (months)', 0, 72, 24)
    phone_service = st.sidebar.selectbox('Phone Service', ('Yes', 'No'))
    multiple_lines = st.sidebar.selectbox('Multiple Lines', ('No', 'Yes', 'No phone service'))
    internet_service = st.sidebar.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
    online_security = st.sidebar.selectbox('Online Security', ('No', 'Yes', 'No internet service'))
    online_backup = st.sidebar.selectbox('Online Backup', ('No', 'Yes', 'No internet service'))
    device_protection = st.sidebar.selectbox('Device Protection', ('No', 'Yes', 'No internet service'))
    tech_support = st.sidebar.selectbox('Tech Support', ('No', 'Yes', 'No internet service'))
    streaming_tv = st.sidebar.selectbox('Streaming TV', ('No', 'Yes', 'No internet service'))
    streaming_movies = st.sidebar.selectbox('Streaming Movies', ('No', 'Yes', 'No internet service'))
    contract = st.sidebar.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    paperless_billing = st.sidebar.selectbox('Paperless Billing', ('Yes', 'No'))
    payment_method = st.sidebar.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    monthly_charges = st.sidebar.slider('Monthly Charges', 18.0, 120.0, 70.0)
    total_charges = st.sidebar.slider('Total Charges', 18.0, 8684.0, 1400.0)

    data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader("User Input")
st.write(input_df)

# Placeholder for real-time prediction
prediction_placeholder = st.empty()

def make_prediction(input_df):

    try:
        with st.spinner("Analyzing Customer Data"):

            time.sleep(0.3) #Debounce delay to prevent rapid predictions

            df = input_df.copy()

            if 'tenure' in df.columns:
                df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 36, np.inf], labels=['Low', 'Medium', 'High'])

            if 'MonthlyCharges' in df.columns:
                df['MonthlyChargesGroup'] = pd.cut(df['MonthlyCharges'], bins=[0, 50, 80, np.inf], labels=['Low', 'Medium', 'High'])

            if {'tenure', 'TotalCharges', 'MonthlyCharges'}.issubset(df.columns):
                df['AvgMonthlyCharge'] = np.where(df['tenure'] > 0, df['TotalCharges'] / df['tenure'], df['MonthlyCharges'])
            elif 'MonthlyCharges' in df.columns:
                df['AvgMonthlyCharge'] = df['MonthlyCharges']

            service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
            existing_service_cols = [c for c in service_cols if c in df.columns]
            if existing_service_cols:
                df['NumServices'] = df[existing_service_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)
            else:
                df['NumServices'] = 0

            if 'Contract' in df.columns:
                contract_map = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
                df['ContractMonths'] = df['Contract'].map(contract_map).fillna(0).astype(int)

            elif 'ContractMonths' not in df.columns:
                df['ContractMonths'] = 0

            # Ensure all columns expected by the preprocessor are present
            for col in preprocessor.feature_names_in_:
                if col not in df.columns:
                    df[col] = 0

            # Apply the transformation
            input_transformed = preprocessor.transform(df)

            feature_names = preprocessor.get_feature_names_out()

            input_transformed_df = pd.DataFrame(input_transformed, columns=feature_names)

            #Make prediction
            prediction = model.predict(input_transformed_df)
            prediction_proba = model.predict_proba(input_transformed_df)

        with prediction_placeholder.container():
            st.subheader("Prediction Result")
            churn_status = "Yes" if prediction[0] == 1 else "No"
            st.metric(label="Customer Will Churn", value=churn_status)

            st.subheader("Prediction Probability")
            st.write(f"Probability of Churn: {prediction_proba[0][1]:.2f}")
            st.progress(float(prediction_proba[0][1]))

    except Exception as e:
        with prediction_placeholder.container():
            st.error(f"An error occurred during prediction: {e}")

make_prediction(input_df)

st.write("---")
