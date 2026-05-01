import json
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

import nba
from nba import generate_nba


# LOAD MODEL + FEATURE COLUMNS


model = XGBClassifier()
model.load_model("xgboost_churn_model.json")

with open("feature_columns.json") as f:
    feature_columns = json.load(f)


# USER INPUT FUNCTION


def get_user_input():
    print("\nEnter Customer Details:\n")

    data = {
        "gender": input("Gender (Male/Female): ").lower(),
        "SeniorCitizen": int(input("Senior Citizen (0/1): ")),
        "Partner": input("Partner (Yes/No): ").lower(),
        "Dependents": input("Dependents (Yes/No): ").lower(),
        "tenure": int(input("Tenure (months): ")),
        "PhoneService": input("Phone Service (Yes/No): ").lower(),
        "MultipleLines": input("Multiple Lines (Yes/No/No phone service): ").lower(),
        "InternetService": input("Internet Service (DSL/Fiber optic/No): ").lower(),
        "OnlineSecurity": input("Online Security (Yes/No/No internet service): ").lower(),
        "OnlineBackup": input("Online Backup (Yes/No/No internet service): ").lower(),
        "DeviceProtection": input("Device Protection (Yes/No/No internet service): ").lower(),
        "TechSupport": input("Tech Support (Yes/No/No internet service): ").lower(),
        "StreamingTV": input("Streaming TV (Yes/No/No internet service): ").lower(),
        "StreamingMovies": input("Streaming Movies (Yes/No/No internet service): ").lower(),
        "Contract": input("Contract (Month-to-month/One year/Two year): ").lower(),
        "PaperlessBilling": input("Paperless Billing (Yes/No): ").lower(),
        "PaymentMethod": input("Payment Method (Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic)): ").lower(),
        "MonthlyCharges": float(input("Monthly Charges: ")),
        "TotalCharges": float(input("Total Charges: "))
    }

    return pd.DataFrame([data])


# PREPROCESSING FUNCTION

def preprocess(df):

    df['gender'] = df['gender'].replace({'female': 1, 'male': 0})

    yes_no_columns = [
        'Partner','Dependents','PhoneService','MultipleLines',
        'OnlineSecurity','OnlineBackup','DeviceProtection',
        'TechSupport','StreamingTV','StreamingMovies','PaperlessBilling'
    ]

    for col in yes_no_columns:
        df[col] = df[col].replace({
            'yes': 1,
            'no': 0,
            'no internet service': 0,
            'no phone service': 0
        })

    df = pd.get_dummies(df, columns=[
        'InternetService','Contract','PaymentMethod'
    ])

    # Align columns with training data
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df


# PREDICTION FUNCTION


def predict(df):

    prob = model.predict_proba(df)[0][1]
    pred = model.predict(df)[0]

    
    print(f"Churn Probability: {prob:.4f}")
    print(f"Prediction: {'Churn' if pred == 1 else 'No Churn'}")

    return prob, pred


# SHAP EXPLANATION


def shap_explain(df):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)

    values = shap_values[0]
    feature_names = df.columns

    # Top 5 important features
    top_idx = np.argsort(np.abs(values))[-5:]

    print("Top Factors Influencing Prediction:\n")
    for i in reversed(top_idx):
        print(f"{feature_names[i]}: {values[i]:.4f}")


# XGBOOST FEATURE IMPORTANCE


def feature_importance():

    importance = model.get_booster().get_score(importance_type='weight')

    print("\nGlobal Feature Importance (Top 10):\n")

    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

    for feature, score in sorted_imp:
        print(f"{feature}: {score}")


# MAIN

if __name__ == "__main__":

    user_df = get_user_input()
    processed_df = preprocess(user_df)

    prob, pred = predict(processed_df)

    shap_explain(processed_df)

    feature_importance()
    top_features = shap_explain(processed_df)

    # nba_output = generate_nba(
    # customer_data=user_df.iloc[0],
    # shap_features=top_features,
    # churn_prob=prob,
    # prediction=pred
    # )

    # print("\n========== GENAI RETENTION STRATEGY ==========\n")
    # print(nba_output)