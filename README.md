# 📡 Telco Customer Churn Prediction Engine

This project is a complete end-to-end data science solution designed to predict customer churn for a telecommunications company. It uses a machine learning model to proactively identify customers who are at high risk of leaving, allowing the business to take targeted retention actions.

The final model is deployed in an interactive **Streamlit web application** for real-time, non-technical use.

## ✨ Key Features

* **Advanced Feature Engineering:** Creates 5 new, high-impact features from the raw data, including:
    * **`Tenure_Group`**: Bins tenure into risk categories (e.g., "New", "Loyal").
    * **`TotalAddon_Count`**: A "stickiness" score counting all services.
    * **`Is_Automatic_Payment`**: A binary flag for "set-it-and-forget-it" payments.
    * **`Monthly_vs_Average_Diff`**: A "Bill Shock" feature to detect recent price hikes.
    * **`Customer_Value_Tier`**: A quartile-based CLV proxy.
* **Model Optimization:** Uses `RandomizedSearchCV` to tune the Random Forest model for the best possible performance, focusing on the **ROC AUC score** and **Recall**.
  
  **Interactive UI:** A user-friendly web app built with Streamlit that allows anyone to input customer details and get an instant churn probability and risk assessment.



## ⚙️ Project Workflow

The project is structured as a modular pipeline. The scripts are numbered to be run in order:

1.  **`Clean.py`**:
    * **Input:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
    * **Action:** Handles missing values, drops irrelevant columns, and corrects data types.
    * **Output:** `cleaned_telco_churn.csv`

2.  **`EDA.py`**:
    * **Input:** `cleaned_telco_churn.csv`
    * **Action:** Generates and saves all EDA plots (distributions, correlations, etc.).
    * **Output:** All plots saved to `output/plots/`

3.  **`create.py`**:
    * **Input:** `cleaned_telco_churn.csv`
    * **Action:** Engineers the 5 new features described above.
    * **Output:** `final_features_telco_churn.csv`

4.  **`03_model_training_optimized.py`**:
    * **Input:** `final_features_telco_churn.csv`
    * **Action:** Builds a `scikit-learn` pipeline (preprocessing + model), performs hyperparameter tuning with `RandomizedSearchCV`, and evaluates the best model.
    * **Output:** The final, trained model saved to `output/models/random_forest_model_optimized.joblib`

5.  **`app.py`**:
    * **Input:** Loads the saved model from `output/models/`.
    * **Action:** Runs the Streamlit web server, providing a UI for real-time predictions.

---

## 🚀 How to Run

### 1. Setup

First, clone the repository and install the required libraries.

```bash
# Clone this repository
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

# Install all required packages
pip install -r requirements.txt
