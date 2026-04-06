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
        background: linear-gradient(135deg, #0f172a 0%, #1a1f3a 100%);
        color: #e2e8f0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #0ea5e9 0%, #10b981 100%);
        color: white;
        border: none;
        border-radius: 8px;
        height: 3.5em;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(14, 165, 233, 0.3);
    }
    .result-box {
        padding: 2.5rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 2rem;
        border-width: 2px;
        border-style: solid;
    }
    .approved {
        background-color: rgba(16, 185, 129, 0.1);
        color: #10b981;
        border-color: #10b981;
    }
    .rejected {
        background-color: rgba(239, 68, 68, 0.1);
        color: #ef4444;
        border-color: #ef4444;
    }
    .stNumberInput > label, .stSelectbox > label, .stSlider > label {
        color: #0ea5e9 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# Load the model
try:
    with open('loan_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please run 'train.py' first.")
    st.stop()

# Header
st.title("🏦 Loan Approval Intelligence System")
st.markdown("---")

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("👤 Personal Details")
    loan_id = st.number_input("Loan ID", min_value=1, value=1001)
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
    education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed?", ["Yes", "No"])
    
    st.subheader("💰 Financial Details")
    income_annum = st.number_input("Annual Income (₹)", min_value=10000, value=500000, step=10000)
    loan_amount = st.number_input("Requested Loan Amount (₹)", min_value=10000, value=1000000, step=10000)
    loan_term = st.number_input("Loan Term (Years)", min_value=1, max_value=25, value=10)
    cibil_score = st.slider("CIBIL Score", 300, 900, 700)

with col2:
    st.subheader("🏠 Assets Information")
    residential_assets_value = st.number_input("Residential Assets Value (₹)", min_value=0, value=500000, step=10000)
    commercial_assets_value = st.number_input("Commercial Assets Value (₹)", min_value=0, value=0, step=10000)
    luxury_assets_value = st.number_input("Luxury Assets Value (₹)", min_value=0, value=0, step=10000)
    bank_asset_value = st.number_input("Bank Assets Value (₹)", min_value=0, value=0, step=10000)

    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("🚀 EXECUTE PREDICTION ANALYSIS"):
        # Map values
        edu_val = 1 if education == "Graduate" else 0
        se_val = 1 if self_employed == "Yes" else 0
        
        # Prepare Feature Vector
        # Sequence must be: [loan_id, dependents, edu, self_employed, income, loan_amt, term, cibil, res_assets, com_assets, lux_assets, bank_assets]
        features = np.array([[loan_id, no_of_dependents, edu_val, se_val, income_annum, 
                             loan_amount, loan_term, cibil_score, 
                             residential_assets_value, commercial_assets_value, 
                             luxury_assets_value, bank_asset_value]])
        
        prediction = model.predict(features)
        
        if prediction[0] == 1:
            st.markdown("""
                <div class="result-box approved">
                    <h2 style="color: #10b981;">✅ LOAN APPROVED</h2>
                    <p>Based on our analysis, the applicant meets all eligibility criteria for this loan.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="result-box rejected">
                    <h2 style="color: #ef4444;">❌ LOAN REJECTED</h2>
                    <p>Analysis identifies significant risk. The application does not meet the minimum requirements at this time.</p>
                </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Disclaimer: This tool provides predictions based on a mathematical model and is intended for illustrative purposes.")
