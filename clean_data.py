import pandas as pd
import numpy as np
df = pd.read_excel('datastuff/WA_Fn-UseC_-Telco-Customer-Churn.xlsx')

df = df.drop('customerID', axis=1)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

for col in service_cols:
    df[col] = df[col].replace({'No phone service': 'No', 'No internet service': 'No'})

df = df.drop_duplicates()

def tenure_group(tenure):
    if tenure <= 12:
        return 'Low'
    elif tenure <= 24:
        return 'Medium'
    else:
        return 'High'

df['TenureGroup'] = df['tenure'].apply(tenure_group)

df['MontlhyChargesGroup'] = pd.cut(df['MonthlyCharges'], bins=3, labels=['Low', 'Medium', 'High'])

df['AvgMonthlyCharge'] = df['TotalCharges'] / (df['tenure']+1)

df['NumServices'] = df[service_cols].apply(lambda x: (x != 'No').sum(), axis=1)

contract_map = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
df['ContractMonths'] = df['Contract'].map(contract_map)

df.to_csv('cleaned_telco_churn.csv', index=False)
print("Cleaned dataset saved to 'cleaned_telco_churn.csv'")
print(df.head())
print("\nChurn distribustion:\n", df['Churn'].value_counts(normalize=True))