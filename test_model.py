import pickle
import numpy as np

# Load the model
try:
    with open('c:/Users/yvsns/OneDrive/Desktop/ml projects/models/Loan-Approval-Prediction-System/loan_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Test Case: Low CIBIL
# [loan_id, dependents, edu, self_employed, income, loan_amt, term, cibil, res_assets, com_assets, lux_assets, bank_assets]
test_input = np.array([[1001, 2, 1, 0, 5000000, 2000000, 10, 300, 1000000, 0, 0, 0]])
prediction = model.predict(test_input)
print(f"Input: CIBIL 300, Inc 5M, Loan 2M, Term 10")
print(f"Prediction: {'Approved' if prediction[0] == 1 else 'Rejected'} ({prediction[0]})")

# Test Case: High CIBIL
test_input_high = np.array([[1001, 2, 1, 0, 5000000, 2000000, 10, 800, 1000000, 0, 0, 0]])
prediction_high = model.predict(test_input_high)
print(f"Input: CIBIL 800, Inc 5M, Loan 2M, Term 10")
print(f"Prediction: {'Approved' if prediction_high[0] == 1 else 'Rejected'} ({prediction_high[0]})")
