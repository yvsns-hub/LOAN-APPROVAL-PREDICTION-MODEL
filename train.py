import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model():
    # Load dataset
    df = pd.read_csv('loan_approval_dataset.csv')
    
    # Aggressive data cleaning
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    
    # Preprocessing
    df['education'] = df['education'].str.lower()
    df['self_employed'] = df['self_employed'].str.lower()
    df['loan_status'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})
    
    # 🚨 CRITICAL FIX: ENFORCING BANKING RULE
    # Any data point with CIBIL score < 500 should be REJECTED (Target = 0)
    # This overrides the "Approved" noise in the original dataset for low CIBIL cases.
    df.loc[df['cibil_score'] < 500, 'loan_status'] = 0
    
    df['education'] = df['education'].map({'graduate': 1, 'not graduate': 0})
    df['self_employed'] = df['self_employed'].map({'yes': 1, 'no': 0})
    
    df = df.dropna()

    # Features and Target
    X = df.drop(['loan_status'], axis=1)
    Y = df['loan_status']

    # Train-Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

    # TRAINING HIGH-PRECISION RANDOM FOREST
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, Y_train)

    # Evaluate
    test_acc = accuracy_score(Y_test, model.predict(X_test))
    print(f'Testing Accuracy with Hard Threshold: {test_acc:.4f}')

    # Save model
    with open('loan_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Fixed model saved as loan_model.pkl (Enforced CIBIL Rules)")

if __name__ == "__main__":
    train_model()
