"""Streamlit webapp for PaySim fraud detection."""

import os
import json
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "admin-key-123")
HEADERS = {"X-API-Key": API_KEY}

st.set_page_config(page_title="PaySim Fraud Detection", page_icon="💸", layout="wide")

# ---- Sidebar ----
st.sidebar.title("💸 PaySim Fraud Detection")
st.sidebar.markdown("Mobile Money Transaction Fraud Detection")
st.sidebar.markdown("---")

try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    if health["model_loaded"]:
        st.sidebar.success(f"API connected ({health['environment']})")
    else:
        st.sidebar.warning("API up but model not loaded")
except Exception:
    st.sidebar.error("API not reachable")
    health = None

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Transaction Check", "Batch Analysis", "Feedback", "Dashboard"],
)

TX_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

if page == "Transaction Check":
    st.title("🔎 Check a Transaction")
    st.markdown("Enter transaction details to check for fraud.")

    col1, col2 = st.columns(2)
    with col1:
        tx_type = st.selectbox("Transaction Type", TX_TYPES, index=4)
        amount = st.number_input("Amount", min_value=0.0, value=181000.0, step=100.0)
        step = st.number_input("Hour (step)", min_value=1, max_value=744, value=1)

    with col2:
        old_bal_orig = st.number_input("Sender Balance Before", min_value=0.0, value=181000.0)
        new_bal_orig = st.number_input("Sender Balance After", min_value=0.0, value=0.0)
        old_bal_dest = st.number_input("Receiver Balance Before", min_value=0.0, value=0.0)
        new_bal_dest = st.number_input("Receiver Balance After", min_value=0.0, value=0.0)

    if st.button("Check Transaction", type="primary"):
        payload = {
            "step": step,
            "type": tx_type,
            "amount": amount,
            "oldbalanceOrg": old_bal_orig,
            "newbalanceOrig": new_bal_orig,
            "oldbalanceDest": old_bal_dest,
            "newbalanceDest": new_bal_dest,
        }
        try:
            resp = requests.post(
                f"{API_URL}/predict/raw", json=payload, headers=HEADERS, timeout=10,
            )
            result = resp.json()

            if result["is_fraud"]:
                st.error(
                    f"**FRAUD DETECTED** — Confidence: {result['confidence']:.2%} "
                    f"— Risk: {result['risk_level']}"
                )
            else:
                st.success(
                    f"**Legitimate** — Confidence: {1 - result['confidence']:.2%} "
                    f"— Risk: {result['risk_level']}"
                )
            c1, c2, c3 = st.columns(3)
            c1.metric("Confidence", f"{result['confidence']:.2%}")
            c2.metric("Risk Level", result["risk_level"])
            c3.metric("Latency", f"{result['prediction_time_ms']:.1f} ms")
        except Exception as e:
            st.error(f"API error: {e}")

elif page == "Batch Analysis":
    st.title("📊 Batch Analysis")
    st.markdown(
        "Upload a CSV with columns: step, type, amount, "
        "oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest"
    )

    csv_file = st.file_uploader("Upload CSV", type=["csv"])
    if csv_file and st.button("Analyze", type="primary"):
        df = pd.read_csv(csv_file)
        st.info(f"Loaded {len(df)} transactions")

        required = ["type", "amount", "oldbalanceOrg", "newbalanceOrig",
                     "oldbalanceDest", "newbalanceDest"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            progress = st.progress(0)
            results = []
            for i, (_, row) in enumerate(df.iterrows()):
                payload = {
                    "step": int(row.get("step", 1)),
                    "type": row["type"],
                    "amount": float(row["amount"]),
                    "oldbalanceOrg": float(row["oldbalanceOrg"]),
                    "newbalanceOrig": float(row["newbalanceOrig"]),
                    "oldbalanceDest": float(row["oldbalanceDest"]),
                    "newbalanceDest": float(row["newbalanceDest"]),
                }
                try:
                    resp = requests.post(
                        f"{API_URL}/predict/raw", json=payload,
                        headers=HEADERS, timeout=10,
                    )
                    results.append(resp.json())
                except Exception:
                    results.append({"is_fraud": None, "confidence": None,
                                    "risk_level": "ERROR", "prediction_time_ms": None})
                progress.progress((i + 1) / len(df))

            df["is_fraud"] = [r.get("is_fraud") for r in results]
            df["confidence"] = [r.get("confidence") for r in results]
            df["risk_level"] = [r.get("risk_level") for r in results]

            c1, c2, c3, c4 = st.columns(4)
            n_fraud = sum(1 for r in results if r.get("is_fraud"))
            c1.metric("Total", len(results))
            c2.metric("Fraudulent", n_fraud)
            c3.metric("Legitimate", len(results) - n_fraud)
            c4.metric("Fraud Rate", f"{n_fraud / len(results):.2%}" if results else "N/A")

            st.dataframe(df[["type", "amount", "is_fraud", "confidence", "risk_level"]])

            fig = px.histogram(df, x="confidence", color="type", nbins=50,
                               title="Confidence by Transaction Type")
            st.plotly_chart(fig, use_container_width=True)

elif page == "Feedback":
    st.title("✏️ Correct a Prediction")

    col1, col2 = st.columns(2)
    with col1:
        fb_type = st.selectbox("Type", TX_TYPES, index=4, key="fb_type")
        fb_amount = st.number_input("Amount", min_value=0.0, value=10000.0, key="fb_amt")
        fb_step = st.number_input("Step", min_value=1, value=1, key="fb_step")
    with col2:
        fb_old_orig = st.number_input("Old Balance Orig", min_value=0.0, value=10000.0, key="fb_oo")
        fb_new_orig = st.number_input("New Balance Orig", min_value=0.0, value=0.0, key="fb_no")
        fb_old_dest = st.number_input("Old Balance Dest", min_value=0.0, value=0.0, key="fb_od")
        fb_new_dest = st.number_input("New Balance Dest", min_value=0.0, value=0.0, key="fb_nd")

    correct_label = st.selectbox("Correct Label", ["Legitimate (0)", "Fraud (1)"])
    comment = st.text_input("Comment (optional)")

    if st.button("Submit Feedback", type="primary"):
        payload = {
            "transaction": {
                "step": fb_step, "type": fb_type, "amount": fb_amount,
                "oldbalanceOrg": fb_old_orig, "newbalanceOrig": fb_new_orig,
                "oldbalanceDest": fb_old_dest, "newbalanceDest": fb_new_dest,
            },
            "correct_label": 1 if "Fraud" in correct_label else 0,
            "comment": comment,
        }
        resp = requests.post(f"{API_URL}/feedback", json=payload, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            st.success("Feedback submitted!")
        else:
            st.error(f"Error: {resp.text}")

elif page == "Dashboard":
    st.title("📈 Monitoring Dashboard")

    try:
        resp = requests.get(f"{API_URL}/metrics", timeout=5)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("API Status")
            if health:
                st.json(health)
        with col2:
            st.subheader("Prometheus Metrics")
            st.code(resp.text[:2000], language="text")
    except Exception:
        st.warning("Could not fetch metrics.")

    st.markdown("---")
    st.markdown(
        "[Grafana](http://localhost:3000) | "
        "[Prometheus](http://localhost:9090) | "
        "[MLflow](http://localhost:5000) | "
        "[Airflow](http://localhost:8080)"
    )
