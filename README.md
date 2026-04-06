# Loan Approval Prediction System 🏦

**Live Application:** [View App Here](https://loan-approval-prediction-model-dtnwx5mjwdf8gusdbdzscp.streamlit.app/)

## 📖 Overview
The **Loan Approval Prediction System** is a sophisticated machine learning application designed to streamline and automate the financial assessment process. By leveraging historical loan data, the system predicts whether a loan application is likely to be **Approved** or **Rejected** based on a multi-dimensional analysis of applicant profiles.

This tool is built for financial enthusiasts and institutions who want an intelligent, data-driven perspective on creditworthiness and risk assessment.

## 🚀 Key Features
- **Intelligent Prediction**: Uses a Logistic Regression model trained on thousands of loan records.
- **Financial Ratios**: Calculates critical metrics like Loan-to-Income, Debt Coverage, and Asset-to-Loan ratios.
- **Asset Valuation**: Comprehensive assessment of residential, commercial, luxury, and bank assets.
- **Interactive UI**: A clean, modern dashboard built with Streamlit for a seamless user experience.

## 🛠️ Technical Stack
- **Languages**: Python
- **Framework**: Streamlit (Frontend)
- **Data Science**: Pandas, NumPy, Scikit-Learn
- **Model Storage**: Pickle

## 📊 Data Insights
The model analyzes several key factors to determine eligibility:
- **CIBIL Score**: The most significant factor in credit assessment.
- **Income & Loan Amount**: Evaluates the applicant's ability to repay.
- **Assets**: Considers the collateral value to mitigate lending risk.

## 🚀 Installation & Running Locally
1. Clone the repository:
   ```bash
   git init
   git remote add origin https://github.com/yvsns-hub/LOAN-APPROVAL-PREDICTION-MODEL.git
   git pull origin main
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## 📜 License
This project is licensed under the MIT License.
