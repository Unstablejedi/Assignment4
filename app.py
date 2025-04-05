import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Set Seaborn style
sns.set(style="whitegrid", palette="pastel")

# Load model-related objects
def preprocess_input(df):
    binary_cols = ['Gender', 'Complain']
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    multi_class_cols = ['PreferredLoginDevice', 'PreferredPaymentMode', 'PreferedOrderCat', 'MaritalStatus']
    df = pd.get_dummies(df, columns=multi_class_cols)

    numerical_cols = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
                      'SatisfactionScore', 'NumberOfAddress', 'OrderAmountHikeFromlastYear',
                      'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 'CashbackAmount']
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    df['AverageOrderValue'] = df['CashbackAmount'] / (df['OrderCount'] + 1e-5)
    df['EngagementScore'] = df['HourSpendOnApp'] * df['NumberOfDeviceRegistered']
    return df

@st.cache_data
def load_data(uploaded_excel=None):
    if uploaded_excel is not None:
        data = pd.read_excel(uploaded_excel, sheet_name='E Comm')
    else:
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

st.title("🌈 E-Commerce Churn Analytics Dashboard")
st.markdown("Explore trends, visualize behavior, and predict customer churn with style.")

# Upload dataset option
uploaded_excel = st.file_uploader("Upload an updated dataset (Excel)", type=["xlsx"])
data = load_data(uploaded_excel)

st.subheader("🧾 Raw Dataset")
st.dataframe(data.head())

# Visualizations
st.subheader("📊 Data Visualizations")

col1, col2 = st.columns(2)
with col1:
    fig1, ax1 = plt.subplots()
    sns.countplot(x='Churn', data=data, ax=ax1, palette='rainbow')
    ax1.set_title("Churn Distribution")
    st.pyplot(fig1)
    st.markdown("Most customers are retained, but there's a non-trivial churn segment worth investigating.")

with col2:
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='Churn', y='CashbackAmount', data=data, ax=ax2, palette='pastel')
    ax2.set_title("Cashback Amount by Churn")
    st.pyplot(fig2)
    st.markdown("Churned customers seem to receive less cashback on average.")

fig3, ax3 = plt.subplots()
sns.histplot(data['Tenure'], bins=20, kde=True, ax=ax3, color='mediumorchid')
ax3.set_title("Distribution of Customer Tenure")
st.pyplot(fig3)
st.markdown("Tenure is right-skewed, indicating many newer customers — retention efforts may be critical early.")

fig4, ax4 = plt.subplots()
sns.violinplot(x='Churn', y='SatisfactionScore', data=data, ax=ax4, palette='coolwarm')
ax4.set_title("Satisfaction Score by Churn")
st.pyplot(fig4)
st.markdown("Customers with low satisfaction scores are more likely to churn.")

fig5, ax5 = plt.subplots()
sns.heatmap(data.corr(numeric_only=True), annot=True, fmt=".2f", cmap='Spectral', ax=ax5)
ax5.set_title("Correlation Heatmap")
st.pyplot(fig5)

fig6, ax6 = plt.subplots()
sns.countplot(data=data, x='CityTier', hue='Churn', palette='Set2', ax=ax6)
ax6.set_title("Churn by City Tier")
st.pyplot(fig6)

model, feature_cols = train_model(data)

st.subheader("🔮 Predict Customer Churn")
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

    st.success("✅ Prediction complete!")
    st.dataframe(input_df)

    st.download_button(
        label="📥 Download Results as CSV",
        data=input_df.to_csv(index=False).encode('utf-8'),
        file_name='churn_predictions_result.csv',
        mime='text/csv'
    )
