import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score

def train_model():
    # Load dataset
    df = pd.read_csv('loan_approval_dataset.csv')
    
    # Cleaning column names (handling leading spaces)
    df.columns = df.columns.str.strip()
    
    # Preprocessing
    df['education'] = df['education'].str.strip().str.lower()
    df['self_employed'] = df['self_employed'].str.strip().str.lower()
    df['loan_status'] = df['loan_status'].str.strip()

    # Mapping categorical values
    df['loan_status'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})
    df['education'] = df['education'].map({'graduate': 1, 'not graduate': 0})
    df['self_employed'] = df['self_employed'].map({'yes': 1, 'no': 0})

    # Prepare features and target (RESTORING loan_id as per original loan.py)
    # Order: [loan_id, no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score, residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value]
    X = df.drop(['loan_status'], axis=1)
    Y = df['loan_status']

    # Train-Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

    # Model Training
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)

    # Evaluate
    train_acc = accuracy_score(Y_train, model.predict(X_train))
    test_acc = accuracy_score(Y_test, model.predict(X_test))
    
    print(f'Training Accuracy: {train_acc:.4f}')
    print(f'Testing Accuracy: {test_acc:.4f}')

    # Save model
    with open('loan_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved as loan_model.pkl (with 12 features)")

if __name__ == "__main__":
    train_model()
