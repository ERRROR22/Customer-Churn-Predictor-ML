import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide"
)

# ─── Load & Preprocess Data ────────────────────────────────
@st.cache_data
def load_and_train():
    # Use a relative path that works locally and on Cloud
    data_path = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    if not os.path.exists(data_path):
        st.error(f"Data file not found at {data_path}. Please ensure the 'data' folder exists.")
        st.stop()

    df = pd.read_csv(data_path)

    # Preprocessing
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df.drop(columns=['customerID'], inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                   'PaperlessBilling', 'MultipleLines']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0,
                               'Male': 1, 'Female': 0,
                               'No phone service': 0})

    multi_cols = ['InternetService', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport', 'StreamingTV',
                  'StreamingMovies', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Build SHAP explainer - using the newer Explainer API for better Waterfall support
    explainer = shap.LinearExplainer(model, X_train)

    return model, explainer, X_train.columns.tolist()

# Initialize data and model
model, explainer, model_columns = load_and_train()

# ─── Header ────────────────────────────────────────────────
st.title("🔮 Customer Churn Predictor")
st.markdown("### Predict whether a customer will churn — and understand *why*")
st.divider()

# ─── Sidebar Inputs ────────────────────────────────────────
st.sidebar.header("👤 Customer Details")

# Personal Info
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months)", 1, 72, 12)

# Services
st.sidebar.markdown("---")
st.sidebar.markdown("**📡 Services**")
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No"])

# Billing
st.sidebar.markdown("---")
st.sidebar.markdown("**💳 Billing**")
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.sidebar.selectbox("Payment Method", 
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
total_charges = st.sidebar.slider("Total Charges ($)", 0.0, 9000.0, float(tenure * monthly_charges))

# ─── Preprocess Input ──────────────────────────────────────
def preprocess_input():
    data = {
        'gender': 1 if gender == 'Male' else 0,
        'SeniorCitizen': 1 if senior == 'Yes' else 0,
        'Partner': 1 if partner == 'Yes' else 0,
        'Dependents': 1 if dependents == 'Yes' else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone_service == 'Yes' else 0,
        'MultipleLines': 1 if multiple_lines == 'Yes' else 0,
        'PaperlessBilling': 1 if paperless == 'Yes' else 0,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'InternetService_Fiber optic': 1 if internet_service == 'Fiber optic' else 0,
        'InternetService_No': 1 if internet_service == 'No' else 0,
        'OnlineSecurity_Yes': 1 if online_security == 'Yes' else 0,
        'OnlineBackup_Yes': 1 if online_backup == 'Yes' else 0,
        'DeviceProtection_Yes': 1 if device_protection == 'Yes' else 0,
        'TechSupport_Yes': 1 if tech_support == 'Yes' else 0,
        'StreamingTV_Yes': 1 if streaming_tv == 'Yes' else 0,
        'StreamingMovies_Yes': 1 if streaming_movies == 'Yes' else 0,
        'Contract_One year': 1 if contract == 'One year' else 0,
        'Contract_Two year': 1 if contract == 'Two year' else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment == 'Credit card (automatic)' else 0,
        'PaymentMethod_Electronic check': 1 if payment == 'Electronic check' else 0,
        'PaymentMethod_Mailed check': 1 if payment == 'Mailed check' else 0,
    }
    
    input_df = pd.DataFrame([data])
    
    # Critical: Match training columns exactly
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    return input_df[model_columns]

# ─── Execution ─────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🔮 Predict Churn", use_container_width=True, type="primary")

st.divider()

if predict_btn:
    input_df = preprocess_input()
    pred_prob = model.predict_proba(input_df)[0][1]
    pred_label = 1 if pred_prob > 0.5 else 0

    # UI Metrics
    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        st.markdown("### 🎯 Prediction")
        if pred_label == 1:
            st.error("🔴 WILL CHURN")
        else:
            st.success("🟢 WON'T CHURN")

    with res_col2:
        st.markdown("### 📊 Churn Probability")
        st.metric(label="Probability", value=f"{pred_prob*100:.1f}%")
        st.progress(float(pred_prob))

    with res_col3:
        st.markdown("### ⚠️ Risk Level")
        if pred_prob < 0.3:
            st.success("🟢 LOW RISK")
        elif pred_prob < 0.6:
            st.warning("🟡 MEDIUM RISK")
        else:
            st.error("🔴 HIGH RISK")

    st.divider()

    # SHAP Explanation Logic Fix
    st.markdown("### 🔍 Why this prediction? (SHAP Explanation)")
    
    # Generate SHAP values for the single input row
    shap_values = explainer(input_df)

    fig, ax = plt.subplots(figsize=(10, 6))
    # Correctly access the first row of SHAP explanation
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
    plt.close()

    st.divider()

    # Risk Factor Analysis
    st.markdown("### 🚨 Top Factors")
    # Extract values from the SHAP explanation object
    shap_importance = pd.DataFrame({
        'Feature': model_columns,
        'SHAP Value': shap_values.values[0]
    }).sort_values('SHAP Value', ascending=False)

    top_risk = shap_importance.head(3)
    top_protect = shap_importance.tail(3)

    f_col1, f_col2 = st.columns(2)
    with f_col1:
        st.markdown("**🔴 Increasing Churn Risk:**")
        for _, row in top_risk.iterrows():
            if row['SHAP Value'] > 0:
                st.write(f"- {row['Feature']}")

    with f_col2:
        st.markdown("**🟢 Reducing Churn Risk:**")
        for _, row in top_protect.iterrows():
            if row['SHAP Value'] < 0:
                st.write(f"- {row['Feature']}")