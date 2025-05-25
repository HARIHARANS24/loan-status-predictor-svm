# streamlit_app.py

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Title
st.title("🏦 Loan Status Prediction App")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('dataset.csv')  # Change this path if needed
    df.dropna(inplace=True)
    df.replace({"Loan_Status": {'N': 0, 'Y': 1}}, inplace=True)
    df['Dependents'].replace('3+', 4, inplace=True)
    df.replace({
        'Married': {'No': 0, 'Yes': 1},
        'Gender': {'Male': 1, 'Female': 0},
        'Self_Employed': {'No': 0, 'Yes': 1},
        'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
        'Education': {'Graduate': 1, 'Not Graduate': 0}
    }, inplace=True)
    return df

loan_dataset = load_data()

# Data and label separation
X = loan_dataset.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
Y = loan_dataset['Loan_Status']

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)

# Train the model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Evaluation
train_accuracy = accuracy_score(Y_train, classifier.predict(X_train))
test_accuracy = accuracy_score(Y_test, classifier.predict(X_test))

st.subheader("Model Performance")
st.write(f"✅ Training Accuracy: {train_accuracy:.2f}")
st.write(f"✅ Test Accuracy: {test_accuracy:.2f}")

# Sidebar input form
st.sidebar.header("Enter Applicant Information")

def user_input_features():
    Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    Married = st.sidebar.selectbox("Married", ["Yes", "No"])
    Dependents = st.sidebar.selectbox("Dependents", [0, 1, 2, 4])
    Education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
    ApplicantIncome = st.sidebar.number_input("Applicant Income", min_value=0)
    CoapplicantIncome = st.sidebar.number_input("Coapplicant Income", min_value=0)
    LoanAmount = st.sidebar.number_input("Loan Amount (in thousands)", min_value=0)
    Loan_Amount_Term = st.sidebar.number_input("Loan Amount Term", min_value=0)
    Credit_History = st.sidebar.selectbox("Credit History", [1.0, 0.0])
    Property_Area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    # Encoding inputs
    data = {
        "Gender": 1 if Gender == "Male" else 0,
        "Married": 1 if Married == "Yes" else 0,
        "Dependents": Dependents,
        "Education": 1 if Education == "Graduate" else 0,
        "Self_Employed": 1 if Self_Employed == "Yes" else 0,
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": Credit_History,
        "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2}[Property_Area]
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Prediction
if st.button("Predict Loan Status"):
    prediction = classifier.predict(input_df)
    result = "Loan Approved ✅" if prediction[0] == 1 else "Loan Not Approved ❌"
    st.subheader("Prediction Result")
    st.success(result)
