import streamlit as st
import pandas as pd
import mlflow.sklearn
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="Credit Card Fraud Detection - ML Dashboard",
    layout="wide"
)

# ------------------------------------------------------
# Header
# ------------------------------------------------------
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
    st.title("Production Model Overview")
    st.markdown("### Selected Model: Random Forest")

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
    st.title("Model Performance Comparison")

    data = {
        "Model": ["Logistic Regression", "KNN", "Decision Tree", "Random Forest"],
        "AUC": [0.966, 0.989, 0.9999, 1.0],
        "Precision": [0.899, 0.893, 0.9999, 1.0],
        "Recall": [0.570, 0.929, 0.9998, 0.9999],
        "F1 Score": [0.698, 0.911, 0.9999, 0.9999],
    }

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

    if os.path.exists("assets/roc_comparison.png"):
        st.image("assets/roc_comparison.png", caption="ROC Curve Comparison")

# ------------------------------------------------------
# Feature Importance Page
# ------------------------------------------------------
elif page == "Feature Importance":
    st.title("Feature Importance - Random Forest")

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

        fig, ax = plt.subplots()
        ax.barh(features, importances)
        ax.set_xlabel("Importance Score")
        ax.set_title("Feature Importance")
        st.pyplot(fig)
    else:
        st.warning("Feature importance not available for this model.")

# ------------------------------------------------------
# Live Prediction Page
# ------------------------------------------------------
elif page == "Live Prediction":
    st.title("Live Fraud Risk Prediction")

    st.markdown("### Transaction Features")
    st.caption("Hover over the ⓘ icon beside each feature to understand what it represents.")

    input_data = {}

    input_data['distance_from_home'] = st.number_input(
        "Distance from Home (km)",
        0.0, 10000.0, 10.0,
        help="Distance between transaction location and customer's registered home location."
    )

    input_data['distance_from_last_transaction'] = st.number_input(
        "Distance from Last Transaction (km)",
        0.0, 10000.0, 1.0,
        help="Distance between this transaction and the customer's previous transaction."
    )

    input_data['ratio_to_median_purchase_price'] = st.number_input(
        "Ratio to Median Purchase Price",
        0.0, 10.0, 1.0,
        help="Transaction amount divided by customer's historical median purchase amount."
    )

    input_data['repeat_retailer'] = st.selectbox(
        "Repeat Retailer",
        [0, 1],
        help="1 if the customer has previously transacted with this retailer, otherwise 0."
    )

    input_data['used_chip'] = st.selectbox(
        "Used Chip",
        [0, 1],
        help="1 if the card chip was used during the transaction, otherwise 0."
    )

    input_data['used_pin_number'] = st.selectbox(
        "Used PIN Number",
        [0, 1],
        help="1 if a PIN was entered during the transaction, otherwise 0."
    )

    input_data['online_order'] = st.selectbox(
        "Online Order",
        [0, 1],
        help="1 if the transaction was conducted online, otherwise 0."
    )

    if st.button("Predict Fraud Risk"):
        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.markdown("---")
        st.subheader("Prediction Result")

        # Risk Interpretation
        if probability < 0.3:
            risk_level = "Low Risk"
        elif probability < 0.7:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        if prediction == 1:
            st.error(f"🚨 Fraud Detected")
        else:
            st.success(f"✅ Legitimate Transaction")

        st.metric("Fraud Probability", f"{probability:.2%}")
        st.info(f"Risk Level: **{risk_level}**")

