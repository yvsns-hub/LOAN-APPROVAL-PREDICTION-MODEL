import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 2rem;
    }
    .approved {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .rejected {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Load the model
try:
    with open('loan_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please run 'train.py' first to train the model.")
    st.stop()

# Header
st.title("🏦 Loan Approval Prediction System")
st.markdown("---")

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("👤 Personal Details")
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
    education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed?", ["Yes", "No"])
    
    st.subheader("💰 Financial Details")
    income_annum = st.number_input("Annual Income (₹)", min_value=10000, value=500000, step=10000)
    loan_amount = st.number_input("Requested Loan Amount (₹)", min_value=10000, value=1000000, step=10000)
    loan_term = st.number_input("Loan Term (Years)", min_value=1, max_value=20, value=10)
    cibil_score = st.slider("CIBIL Score", 300, 900, 700)

with col2:
    st.subheader("🏠 Asset Valuation")
    residential_assets_value = st.number_input("Residential Assets Value (₹)", min_value=0, value=500000, step=10000)
    commercial_assets_value = st.number_input("Commercial Assets Value (₹)", min_value=0, value=200000, step=10000)
    luxury_assets_value = st.number_input("Luxury Assets Value (₹)", min_value=0, value=100000, step=10000)
    bank_asset_value = st.number_input("Bank Assets Value (₹)", min_value=0, value=300000, step=10000)

    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("🚀 Predict Loan Status"):
        # Preprocessing inputs
        # Convert term to months for consistency with training data (if needed)
        # Assuming training data uses years based on loan.py logic
        
        edu_val = 1 if education == "Graduate" else 0
        se_val = 1 if self_employed == "Yes" else 0
        
        # Creating features array
        # Order: [no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score, residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value]
        # Note: loan_id was dropped in training
        features = np.array([[no_of_dependents, edu_val, se_val, income_annum, 
                             loan_amount, loan_term, cibil_score, 
                             residential_assets_value, commercial_assets_value, 
                             luxury_assets_value, bank_asset_value]])
        
        prediction = model.predict(features)
        
        if prediction[0] == 1:
            st.markdown("""
                <div class="result-box approved">
                    <h2>✅ Loan Approved</h2>
                    <p>Congratulations! Based on our analysis, your loan application is likely to be approved.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="result-box rejected">
                    <h2>❌ Loan Rejected</h2>
                    <p>We're sorry. Based on our analysis, your loan application is likely to be rejected at this time.</p>
                </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Disclaimer: This tool is for educational purposes and provides an automated prediction based on a machine learning model.")
