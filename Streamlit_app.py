import streamlit as st
import pandas as pd
import json
import mlflow.sklearn
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="Credit Card Fraud Detection - ML Dashboard",
    layout="wide"
)

st.markdown(
    """
    # 💳 Credit Card Fraud Detection
    ### End-to-End ML Engineering System
    ---
    """
)

# ------------------------------------------------------
# Load Production Model
# ------------------------------------------------------

@st.cache_resource
def load_model():
    return mlflow.sklearn.load_model("exported_model")

model = load_model()

# ------------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Model Comparison",
        "Feature Importance",
        "Live Prediction"
    ]
)

# ------------------------------------------------------
# Overview Page
# ------------------------------------------------------
if page == "Overview":
    st.title("💳 Credit Card Fraud Detection System")
    st.markdown("### Production Model: Random Forest")

    st.success("Selected as best model based on weighted evaluation score.")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("AUC", "1.00")
    col2.metric("Precision", "1.00")
    col3.metric("Recall", "0.9999")
    col4.metric("F1 Score", "0.9999")

    st.markdown("---")

    st.info("""
    ⚠ **Note:** The dataset used in this project is synthetically generated for demonstration purposes.  
    The near-perfect performance metrics reflect characteristics of synthetic data and should not be interpreted as real-world fraud detection performance.
    """)

# ------------------------------------------------------
# Model Comparison Page
# ------------------------------------------------------
elif page == "Model Comparison":
    st.title("📊 Model Performance Comparison")

    data = {
        "Model": ["Logistic Regression", "KNN", "Decision Tree", "Random Forest"],
        "AUC": [0.966, 0.989, 0.9999, 1.0],
        "Precision": [0.899, 0.893, 0.9999, 1.0],
        "Recall": [0.570, 0.929, 0.9998, 0.9999],
        "F1": [0.698, 0.911, 0.9999, 0.9999],
    }

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

    if os.path.exists("assets/roc_comparison.png"):
        st.image("assets/roc_comparison.png", caption="ROC Curve Comparison")

# ------------------------------------------------------
# Feature Importance Page
# ------------------------------------------------------
elif page == "Feature Importance":
    st.title("📈 Feature Importance (Random Forest)")

    features = [
        "distance_from_home",
        "distance_from_last_transaction",
        "ratio_to_median_purchase_price",
        "repeat_retailer",
        "used_chip",
        "used_pin_number",
        "online_order"
    ]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        st.warning("Feature importance not available for this model.")
        importances = []

    fig, ax = plt.subplots()
    ax.barh(features, importances)
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance - Random Forest")
    st.pyplot(fig)

# ------------------------------------------------------
# Live Prediction Page
# ------------------------------------------------------
elif page == "Live Prediction":
    st.title("🔍 Fraud Prediction")

    st.markdown("Enter transaction details below:")

    input_data = {}

    input_data['distance_from_home'] = st.number_input("Distance from Home", 0.0, 10000.0, 10.0)
    input_data['distance_from_last_transaction'] = st.number_input("Distance from Last Transaction", 0.0, 10000.0, 1.0)
    input_data['ratio_to_median_purchase_price'] = st.number_input("Ratio to Median Purchase Price", 0.0, 10.0, 1.0)
    input_data['repeat_retailer'] = st.selectbox("Repeat Retailer", [0, 1])
    input_data['used_chip'] = st.selectbox("Used Chip", [0, 1])
    input_data['used_pin_number'] = st.selectbox("Used PIN Number", [0, 1])
    input_data['online_order'] = st.selectbox("Online Order", [0, 1])

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"🚨 Fraud Detected (Probability: {probability:.2%})")
        else:
            st.success(f"✅ Legitimate Transaction (Fraud Probability: {probability:.2%})")
