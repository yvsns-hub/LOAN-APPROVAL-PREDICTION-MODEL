---
title: Loan Approval Prediction Model
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# Loan Approval Prediction System 🏦

Predict loan approval status based on applicant demographics and financial data.

## Features
- Interactive Streamlit Dashboard
- Logistic Regression Model for status prediction
- Real-time prediction based on CIBIL score, income, and assets

## Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Train the model (optional, if `loan_model.pkl` is missing):
   ```bash
   python train.py
   ```
2. Run the application:
   ```bash
   streamlit run app.py
   ```

## Dataset
This project uses the `loan_approval_dataset.csv`, which includes features like annual income, loan amount, CIBIL score, and asset valuations.

## License
MIT License
