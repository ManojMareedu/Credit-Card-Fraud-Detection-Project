import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from data_ingestion import load_data
from models import (
    train_test_split_data,
    CreditFraud_logreg_model,
    CreditFraud_knn_model,
    CreditFraud_decision_tree_model,
    CreditFraud_random_forest_model,
)
from EDA import eda_step

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("💳 Credit Card Fraud Detection Demo")
st.markdown("""
This interactive dashboard demonstrates a full MLOps pipeline for credit card fraud detection, including data ingestion, EDA, model training, and prediction.
""")

# --- Sidebar for file upload and model selection ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your transaction data (CSV, JSON, ZIP, XLSX)", type=["csv", "json", "zip", "xlsx"])

st.sidebar.header("2. Choose Model")
model_choice = st.sidebar.selectbox(
    "Select a model for prediction:",
    ("Logistic Regression", "KNN", "Decision Tree", "Random Forest")
)

if uploaded_file:
    # --- Data Ingestion ---
    st.subheader("Raw Data Preview")
    df = load_data(uploaded_file)
    st.dataframe(df.head())

    # --- EDA ---
    st.subheader("Exploratory Data Analysis")
    # Run EDA step and get processed DataFrame
    df_eda = eda_step(df)
    st.write("Basic Statistics:")
    st.dataframe(df_eda.describe())
    st.write("Null Values:")
    st.write(df_eda.isnull().sum())

    # Visualizations
    st.write("Feature Distributions and Correlations:")
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    df_eda[['distance_from_home', 'distance_from_last_transaction']].plot(ax=axs[0, 0])
    axs[0, 0].set_title('Distance Features Over Transactions')
    axs[0, 0].set_xlabel('Index')
    axs[0, 0].set_ylabel('Distance')

    axs[0, 1].boxplot(df_eda)
    axs[0, 1].set_title('Boxplot to Identify Outliers')
    axs[0, 1].set_xlabel('Feature Index')
    axs[0, 1].set_ylabel('Value')

    columns = ['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order', 'fraud']
    axs[1, 0].hist([df_eda[col] for col in columns], histtype='bar', bins=15, label=columns)
    axs[1, 0].legend()
    axs[1, 0].set_xticks([0, 1])
    axs[1, 0].set_xlabel('Binary Feature Value')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].set_title('Distribution of Binary Features')

    cor_coef = df_eda.corr()
    sns.heatmap(cor_coef, annot=True, cmap='coolwarm', ax=axs[1, 1])
    axs[1, 1].set_title('Correlation Heatmap')

    plt.tight_layout()
    st.pyplot(fig)

    # --- Model Training & Evaluation ---
    st.subheader("Model Training & Evaluation")
    X_train, X_test, y_train, y_test = train_test_split_data(df_eda)

    if model_choice == "Logistic Regression":
        model = CreditFraud_logreg_model(X_train, y_train, X_test, y_test)
    elif model_choice == "KNN":
        model = CreditFraud_knn_model(X_train, y_train, X_test, y_test)
    elif model_choice == "Decision Tree":
        model = CreditFraud_decision_tree_model(X_train, y_train, X_test, y_test)
    elif model_choice == "Random Forest":
        model = CreditFraud_random_forest_model(X_train, y_train, X_test, y_test)

    st.success(f"{model_choice} training and evaluation complete! Check your terminal or MLflow for detailed metrics and ROC curves.")

    # --- Prediction Demo ---
    st.subheader("Try a Prediction")
    st.markdown("Enter transaction details to predict fraud:")

    input_data = {}
    input_data['distance_from_home'] = st.number_input("Distance from Home", min_value=0.0, value=10.0)
    input_data['distance_from_last_transaction'] = st.number_input("Distance from Last Transaction", min_value=0.0, value=1.0)
    input_data['ratio_to_median_purchase_price'] = st.number_input("Ratio to Median Purchase Price", min_value=0.0, value=1.0)
    input_data['repeat_retailer'] = st.selectbox("Repeat Retailer", [0, 1])
    input_data['used_chip'] = st.selectbox("Used Chip", [0, 1])
    input_data['used_pin_number'] = st.selectbox("Used PIN Number", [0, 1])
    input_data['online_order'] = st.selectbox("Online Order", [0, 1])

    if st.button("Predict Fraud"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0][1]
        else:
            proba = None
        st.write(f"**Prediction:** {'Fraud' if prediction[0] == 1 else 'Legitimate'}")
        if proba is not None:
            st.write(f"**Fraud Probability:** {proba:.2%}")

else:
    st.info("Please upload a transaction data file to begin.")