import streamlit as st
import pandas as pd
import joblib

# Load the saved pipeline
model = joblib.load("epic5_model_pipeline.pkl")

# Load CSVs
insurance_df = pd.read_csv("insurance_data.csv")

# Clean columns
insurance_df.columns = insurance_df.columns.str.strip().str.upper()
insurance_df["POLICY_NUMBER"] = insurance_df["POLICY_NUMBER"].astype(str).str.strip().str.upper()
insurance_df["AGENT_ID"] = insurance_df["AGENT_ID"].astype(str).str.strip().str.upper()
insurance_df["VENDOR_ID"] = insurance_df["VENDOR_ID"].astype(str).str.strip().str.upper()

# Streamlit UI
st.title("Insurance Fraud Detection System")
policy_number = st.selectbox("Policy Number", insurance_df["POLICY_NUMBER"].unique())
agent_id = st.selectbox("Agent ID", insurance_df["AGENT_ID"].unique())
vendor_id = st.selectbox("Vendor ID", insurance_df["VENDOR_ID"].unique())

if st.button("Predict"):
    try:
        row = insurance_df[
            (insurance_df["POLICY_NUMBER"] == policy_number) &
            (insurance_df["AGENT_ID"] == agent_id) &
            (insurance_df["VENDOR_ID"] == vendor_id)
        ].iloc[0]

        X_input = pd.DataFrame([{
            "AGE": float(row["AGE"]),
            "PREMIUM_AMOUNT": float(row["PREMIUM_AMOUNT"]),
            "CLAIM_AMOUNT": float(row["CLAIM_AMOUNT"]),
            "INCIDENT_HOUR_OF_THE_DAY": float(row["INCIDENT_HOUR_OF_THE_DAY"]),
            "AGENT_ID": row["AGENT_ID"],
            "VENDOR_ID": row["VENDOR_ID"],
            "INSURANCE_TYPE": row["INSURANCE_TYPE"]
        }])

        st.write("Features Sent to Model:")
        st.write(X_input)

        prediction = model.predict(X_input)[0]
        fraud_proba = model.predict_proba(X_input)[0][1]

        st.subheader("Prediction Result")
        st.success(f"Fraud Prediction: {prediction}")
        st.info(f"Fraud Probability: {fraud_proba:.2%}")

    except IndexError:
        st.error("Entered combination of Policy, Agent, and Vendor not found in dataset.")