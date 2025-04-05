import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load model-related objects
def preprocess_input(df):
    binary_cols = ['Gender', 'Complain']
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    multi_class_cols = ['PreferredLoginDevice', 'PreferredPaymentMode', 'PreferedOrderCat', 'MaritalStatus']
    df = pd.get_dummies(df, columns=multi_class_cols)

    # Scale numeric columns
    numerical_cols = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
                      'SatisfactionScore', 'NumberOfAddress', 'OrderAmountHikeFromlastYear',
                      'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 'CashbackAmount']
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Add engineered features
    df['AverageOrderValue'] = df['CashbackAmount'] / (df['OrderCount'] + 1e-5)
    df['EngagementScore'] = df['HourSpendOnApp'] * df['NumberOfDeviceRegistered']
    
    return df

@st.cache_data
def load_model():
    # Load a pre-trained model (retrain here for demo purposes)
    data = pd.read_excel('E Commerce Dataset.xlsx', sheet_name='E Comm')
    numerical_missing = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 
                         'OrderAmountHikeFromlastYear', 'CouponUsed', 
                         'OrderCount', 'DaySinceLastOrder']
    for col in numerical_missing:
        data[col].fillna(data[col].median(), inplace=True)

    y = data['Churn']
    X = preprocess_input(data.drop(['CustomerID', 'Churn'], axis=1))

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model, X.columns

# Web app UI
st.title("E-Commerce Customer Churn Predictor")
st.markdown("Upload a CSV with customer data to predict churn probability.")

uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    model, feature_cols = load_model()
    input_processed = preprocess_input(input_df)

    # Align columns with training features
    missing_cols = set(feature_cols) - set(input_processed.columns)
    for col in missing_cols:
        input_processed[col] = 0  # Add missing columns with default 0
    input_processed = input_processed[feature_cols]  # Match column order

    # Make predictions
    predictions = model.predict(input_processed)
    probabilities = model.predict_proba(input_processed)[:, 1]

    input_df['Churn Prediction'] = predictions
    input_df['Churn Probability'] = probabilities

    st.success("Prediction complete!")
    st.dataframe(input_df)

    st.download_button(
        label="Download Results as CSV",
        data=input_df.to_csv(index=False).encode('utf-8'),
        file_name='churn_predictions_result.csv',
        mime='text/csv'
    )
