import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
def load_data():
    data = pd.read_excel('E Commerce Dataset.xlsx', sheet_name='E Comm')
    numerical_missing = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 
                         'OrderAmountHikeFromlastYear', 'CouponUsed', 
                         'OrderCount', 'DaySinceLastOrder']
    for col in numerical_missing:
        data[col].fillna(data[col].median(), inplace=True)
    return data

@st.cache_data
def train_model(data):
    y = data['Churn']
    X = preprocess_input(data.drop(['CustomerID', 'Churn'], axis=1))
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model, X.columns

# Web app UI
st.title("📊 E-Commerce Churn Analytics Dashboard")
st.markdown("Explore customer data and predict churn.")

# Load and display raw data
data = load_data()
st.subheader("Raw Dataset")
st.dataframe(data.head())

# Visualizations
st.subheader("Data Visualizations")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    sns.countplot(x='Churn', data=data, ax=ax1)
    ax1.set_title("Churn Distribution")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='Churn', y='CashbackAmount', data=data, ax=ax2)
    ax2.set_title("Cashback Amount by Churn")
    st.pyplot(fig2)

fig3, ax3 = plt.subplots()
sns.histplot(data['Tenure'], bins=20, kde=True, ax=ax3)
ax3.set_title("Distribution of Customer Tenure")
st.pyplot(fig3)

# Preprocessing and model training
model, feature_cols = train_model(data)

# Upload CSV for prediction
st.subheader("Predict Customer Churn")
st.markdown("Upload a CSV to get churn predictions.")

uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    input_processed = preprocess_input(input_df)

    missing_cols = set(feature_cols) - set(input_processed.columns)
    for col in missing_cols:
        input_processed[col] = 0
    input_processed = input_processed[feature_cols]

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
