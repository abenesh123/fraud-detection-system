import streamlit as st
import joblib
import numpy as np
import pandas as pd


st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)


@st.cache_resource
def load_model():
    model=joblib.load("fraud_model.pkl")
    return model
model=load_model()


st.title("🔍 Financial fraud detection")
st.markdown("Enter the transaction details below to check if it's fraudulent")
st.divider()


st.subheader("Transaction details")


col1,col2,col3=st.columns(3)


with col1:
    time    = st.number_input("Time (seconds)", min_value=0.0, value=50000.0)
    amount  = st.number_input("Transaction Amount (₹)", min_value=0.0, value=100.0)
    V1      = st.number_input("V1", value=0.0)
    V2      = st.number_input("V2", value=0.0)
    V3      = st.number_input("V3", value=0.0)
    V4      = st.number_input("V4", value=0.0)
    V5      = st.number_input("V5", value=0.0)
    V6      = st.number_input("V6", value=0.0)
    V7      = st.number_input("V7", value=0.0)
    V8      = st.number_input("V8", value=0.0)

with col2:
    V9      = st.number_input("V9",  value=0.0)
    V10     = st.number_input("V10", value=0.0)
    V11     = st.number_input("V11", value=0.0)
    V12     = st.number_input("V12", value=0.0)
    V13     = st.number_input("V13", value=0.0)
    V14     = st.number_input("V14", value=0.0)
    V15     = st.number_input("V15", value=0.0)
    V16     = st.number_input("V16", value=0.0)
    V17     = st.number_input("V17", value=0.0)
    V18     = st.number_input("V18", value=0.0)

with col3:
    V19     = st.number_input("V19", value=0.0)
    V20     = st.number_input("V20", value=0.0)
    V21     = st.number_input("V21", value=0.0)
    V22     = st.number_input("V22", value=0.0)
    V23     = st.number_input("V23", value=0.0)
    V24     = st.number_input("V24", value=0.0)
    V25     = st.number_input("V25", value=0.0)
    V26     = st.number_input("V26", value=0.0)
    V27     = st.number_input("V27", value=0.0)
    V28     = st.number_input("V28", value=0.0)

st.divider()


if st.button("🔎 Check Transaction", use_container_width=True):
     
    hour=(time%86400)/3600

    input_data = pd.DataFrame([[
        time, V1, V2, V3, V4, V5, V6, V7, V8, V9,
        V10, V11, V12, V13, V14, V15, V16, V17, V18, V19,
        V20, V21, V22, V23, V24, V25, V26, V27, V28,
        amount, hour
    ]], columns=[
        "time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
        "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
        "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
        "amount", "hour"
    ])

    prediction=model.predict(input_data)[0]
    probability=model.predict_proba(input_data)[0][1]

    st.divider()

    st.subheader("Prediction Result")

    col_result,col_prob=st.columns(2)

    with col_result:
        if prediction==1:
            st.error("🚨 FRAUDULENT TRANSACTION DETECTED")
        else:
            st.success("✅ LEGITIMATE TRANSACTION")  


    with col_prob:
        st.metric(
            label="Fraud probability",
            value=f"{probability*100:.2f}%"
        )          
        st.progress(float(probability))


    st.divider()
    st.subheader("Risk Level")

    if probability < 0.3:
        st.info("🟢 LOW RISK")
    elif probability < 0.6:
        st.warning("🟡 MEDIUM RISK")
    else:
        st.error("🔴 HIGH RISK")

    st.divider()
    st.subheader("Key Feature Indicators")
    key_features = pd.DataFrame({
        "Feature": ["V14", "V12", "V10", "V17", "Amount", "Hour"],
        "Value":   [V14,   V12,   V10,   V17,   amount,   round(hour, 2)]
    })
    st.dataframe(key_features, use_container_width=True)


with st.sidebar:
    st.header("ℹ️ About")
    st.write("This app detects fraudulent credit card transactions using a machine learning model trained on real transaction data.")
    st.divider()
    st.header("📊 Model Info")
    st.write("**Algorithm:** XGBoost (Tuned)")
    st.write("**Technique:** SMOTE + RandomizedSearchCV")
    st.write("**Key Metrics:**")
    st.write("- ROC-AUC: ~0.98")
    st.write("- Handles class imbalance")
    st.divider()
    st.header("⚠️ Note")
    st.write("V1–V28 are PCA-transformed anonymized features from the original dataset.")     


   



