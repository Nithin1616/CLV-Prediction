import streamlit as st
import numpy as np
import pandas as pd
import pickle

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="CLV Predictor", page_icon="💰", layout="wide")

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    with open("random_forest_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# -------------------------------
# Title
# -------------------------------
st.title("💰 Customer Lifetime Value Predictor")
st.markdown("### Predict future customer value using Machine Learning")

st.markdown("---")

# -------------------------------
# Layout (2 Columns)
# -------------------------------
col1, col2 = st.columns(2)

# -------------------------------
# Column 1 - Numerical
# -------------------------------
with col1:
    st.subheader("📊 Customer Financial Details")

    income = st.number_input("Income", value=50000)
    monthly_premium = st.number_input("Monthly Premium Auto", value=100)
    total_claim = st.number_input("Total Claim Amount", value=500)

    st.subheader("📈 Customer Behavior")

    months_since_claim = st.number_input("Months Since Last Claim", value=5)
    num_policies = st.number_input("Number of Policies", value=2)
    open_complaints = st.number_input("Number of Open Complaints", value=0)
    months_policy = st.number_input("Months Since Policy Inception", value=12)

# -------------------------------
# Column 2 - Categorical
# -------------------------------
with col2:
    st.subheader("👤 Customer Profile")

    employment = st.selectbox("Employment Status", ["Employed", "Unemployed"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    education = st.selectbox("Education", ["High School", "Bachelor", "Master", "Doctor"])

    st.subheader("🚗 Policy Details")

    policy_type = st.selectbox("Policy Type", ["Corporate Auto", "Personal Auto", "Special Auto"])
    policy = st.selectbox("Policy", ["Corporate L1", "Corporate L2", "Personal L1", "Personal L2"])
    coverage = st.selectbox("Coverage", ["Basic", "Extended", "Premium"])
    renew_offer = st.selectbox("Renew Offer Type", ["Offer1", "Offer2", "Offer3", "Offer4"])

    st.subheader("📍 Additional Info")

    location_code = st.selectbox("Location Code", ["Suburban", "Urban", "Rural"])
    vehicle_class = st.selectbox("Vehicle Class", ["Two-Door Car", "Four-Door Car", "SUV"])
    vehicle_size = st.selectbox("Vehicle Size", ["Small", "Medsize", "Large"])
    sales_channel = st.selectbox("Sales Channel", ["Agent", "Branch", "Call Center", "Web"])
    response = st.selectbox("Response", ["Yes", "No"])

# -------------------------------
# Feature Engineering
# -------------------------------
claim_to_premium = total_claim / (monthly_premium + 1)
income_to_premium = income / (monthly_premium + 1)

st.markdown("---")

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("🚀 Predict Customer Lifetime Value"):

    try:
        input_df = pd.DataFrame([{
            "Income": income,
            "Monthly Premium Auto": monthly_premium,
            "Months Since Last Claim": months_since_claim,
            "Number of Policies": num_policies,
            "Total Claim Amount": total_claim,

            "EmploymentStatus": employment,
            "Gender": gender,
            "Marital Status": marital,
            "Policy Type": policy_type,
            "Vehicle Class": vehicle_class,
            "Vehicle Size": vehicle_size,
            "Sales Channel": sales_channel,

            # FIXED MISSING FEATURES
            "Location Code": location_code,
            "Policy": policy,
            "Renew Offer Type": renew_offer,
            "Number of Open Complaints": open_complaints,
            "Education": education,
            "Coverage": coverage,
            "Months Since Policy Inception": months_policy,
            "Response": response,

            # Engineered
            "claim_to_premium": claim_to_premium,
            "income_to_premium": income_to_premium
        }])

        # Prediction
        pred_log = model.predict(input_df)
        prediction = np.expm1(pred_log[0])

        # Display Result
        st.success(f"💡 Predicted CLV: ₹ {prediction:,.2f}")

        # Extra Insight
        if prediction > 50000:
            st.info("🔥 High Value Customer")
        elif prediction > 20000:
            st.info("⚡ Medium Value Customer")
        else:
            st.info("💤 Low Value Customer")

    except Exception as e:
        st.error(f"❌ Error: {e}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Machine Learning CLV Project")