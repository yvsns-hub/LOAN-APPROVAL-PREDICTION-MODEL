import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score

def train_model():
    # Load dataset
    df = pd.read_csv('loan_approval_dataset.csv')
    
    # Aggressive column name cleaning
    df.columns = df.columns.str.strip()
    
    # Aggressive value cleaning (stripping leading/trailing spaces from all object columns)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    
    # Preprocessing
    df['education'] = df['education'].str.lower()
    df['self_employed'] = df['self_employed'].str.lower()

    # Mapping categorical values
    df['loan_status'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})
    df['education'] = df['education'].map({'graduate': 1, 'not graduate': 0})
    df['self_employed'] = df['self_employed'].map({'yes': 1, 'no': 0})

    # Drop any rows with NaN if mapping failed
    initial_count = len(df)
    df = df.dropna()
    if len(df) < initial_count:
        print(f"Warning: Dropped {initial_count - len(df)} rows due to mapping issues.")

    # Prepare features and target
    # Sequence: [loan_id, no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score, residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value]
    X = df.drop(['loan_status'], axis=1)
    Y = df['loan_status'].astype(int)

    # Train-Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

    # Model Training - Using a more balanced approach
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, Y_train)

    # Evaluate
    train_acc = accuracy_score(Y_train, model.predict(X_train))
    test_acc = accuracy_score(Y_test, model.predict(X_test))
    
    print(f'Training Accuracy: {train_acc:.4f}')
    print(f'Testing Accuracy: {test_acc:.4f}')

    # Save model
    with open('loan_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved as loan_model.pkl")

if __name__ == "__main__":
    train_model()
