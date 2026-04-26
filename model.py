import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap

#  load model
model = joblib.load('xgboost_model.pkl')

#  page config
st.set_page_config(page_title="ChurnIQ", layout="wide", page_icon="📊")

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0f1117; }
  [data-testid="stSidebar"] { background: #1a1d27; }
  h1, h2, h3, label, p, div { color: #fafafa !important; }
  .stButton > button {
    background: #1d9e75; color: #04342c; font-weight: 600;
    border: none; width: 100%; border-radius: 8px; padding: 10px;
  }
  .stButton > button:hover { background: #5dcaa5; }
  }
</style>
""", unsafe_allow_html=True)


# ── model metrics (top bar)
st.title("Customer Churn Prediction")
st.caption("Telco churn model · XGBoost tuned with RandomizedSearchCV")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy",  "80%")
m2.metric("F1 score",  "0.66")
m3.metric("Precision", "0.60")
m4.metric("Recall",    "0.73")

st.divider()

# ── CUSTOMER INFO 
st.subheader("📦 Customer Info")

col1, col2 = st.columns(2)

with col1:
    gender     = st.selectbox("Gender", ["Male", "Female"])
    senior     = st.selectbox("Senior citizen", ["No", "Yes"])
    partner    = st.selectbox("Partner", ["No", "Yes"])

with col2:
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure     = st.slider("Tenure (months)", 0, 72, 12)

st.divider()

# ── SERVICES
st.subheader("📶 Services")

col3, col4 = st.columns(2)

with col3:
    phone_service    = st.selectbox("Phone service", ["No", "Yes"])
    multiple_lines   = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    online_security  = st.selectbox("Online security", ["No", "Yes", "No internet service"])

with col4:
    online_backup    = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
    device_prot      = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
    tech_support     = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
    streaming_tv     = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])

st.divider()

# ── BILLING ──────────────────────────────
st.subheader("💳 Billing")

col5, col6 = st.columns(2)

with col5:
    contract  = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["No", "Yes"])
    payment   = st.selectbox("Payment method", [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ])

with col6:
    monthly_charges = st.slider("Monthly charges (₹)", 20.0, 120.0, 65.0, step=0.5)
    total_charges   = st.slider("Total charges (₹)", 0.0, 9000.0, monthly_charges * tenure, step=10.0)

st.divider()
predict_btn = st.button("Predict churn", type="primary")

with st.sidebar:
    st.title("📊 Customer Churn Prediction Dashboard")
    st.caption("XGBoost · RandomizedSearchCV · F1 = 0.66")
    st.divider()    

    st.markdown("### 🔍 Key Insights")
    st.write("- Month-to-month users churn more")
    st.write("- Fiber optic users have higher churn")
    st.write("- Long tenure customers are stable")

    st.divider()

    st.info("Fill customer details and click 'Predict Churn'")

# ── preprocessing helper 
def build_input():
    row = {
        "SeniorCitizen":    1 if senior == "Yes" else 0,
        "Partner":          1 if partner == "Yes" else 0,
        "Dependents":       1 if dependents == "Yes" else 0,
        "Tenure":           tenure,
        "MultipleLines":    1 if multiple_lines == "Yes" else 0,
        "OnlineSecurity":   1 if online_security == "Yes" else 0,
        "OnlineBackup":     1 if online_backup == "Yes" else 0,
        "DeviceProtection": 1 if device_prot == "Yes" else 0,
        "TechSupport":      1 if tech_support == "Yes" else 0,
        "StreamingTV":      1 if streaming_tv == "Yes" else 0,
        "StreamingMovies":  1 if streaming_movies == "Yes" else 0,
        "PaperlessBilling": 1 if paperless == "Yes" else 0,
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges,

        # dummies — internet service
        "InternetService_DSL":          1 if internet_service == "DSL" else 0,
        "InternetService_Fiber optic":  1 if internet_service == "Fiber optic" else 0,

        # dummies — contract
        "Contract_Monthly":             1 if contract == "Month-to-month" else 0,
        "Contract_One year":            1 if contract == "One year" else 0,

        # dummies — gender
        "Gender_Female":                1 if gender == "Female" else 0,

        # dummies — payment
        "PaymentMethod_Bank transfer (automatic)":  1 if payment == "Bank transfer (automatic)" else 0,
        "PaymentMethod_Credit card (automatic)":    1 if payment == "Credit card (automatic)" else 0,
    }
    # column order must exactly match training
    cols = [
        "SeniorCitizen", "Partner", "Dependents", "Tenure", "MultipleLines",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "PaperlessBilling", "MonthlyCharges",
        "TotalCharges", "InternetService_DSL", "InternetService_Fiber optic",
        "Contract_Monthly", "Contract_One year", "Gender_Female",
        "PaymentMethod_Bank transfer (automatic)",
        "PaymentMethod_Credit card (automatic)"
    ]
    return pd.DataFrame([row])[cols]

# ── prediction 
if predict_btn:
    input_df = build_input()
    prob     = model.predict_proba(input_df)[0][1]
    prob_pct = round(prob * 100, 2)

    risk  = "High risk"   if prob > 0.65 else "Medium risk" if prob > 0.40 else "Low risk"
    color = "red"         if prob > 0.65 else "orange"       if prob > 0.40 else "green"

    col1, col2 = st.columns([1, 1])
    # ── SHAP EXPLAINABILITY 
    st.divider()
    st.subheader("🔍 Why this customer may churn")
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)

    # Waterfall plot
    shap.plots.waterfall(shap_values[0])
    st.pyplot(plt.gcf())
    plt.clf()

    st.info("Positive values increase churn risk, negative values decrease it.")      

    with col1:
        st.subheader("Prediction result")
        st.metric("Churn probability", f"{prob_pct: .2f}%")
        st.progress(float(prob))
        st.markdown(f"**Risk level:** :{color}[{risk}]")
        st.markdown(f"**Decision:** {'Will churn' if prob >= 0.5 else 'Will not churn'}")

        st.divider()
        st.subheader("Recommended action")
        if prob > 0.65:
            color = "#ef4444" 
            st.error("Offer a 3-month loyalty discount or switch to an annual contract immediately.")
        elif prob > 0.40:
            color = "#f59e0b"
            st.warning("Send a satisfaction survey and consider a targeted retention offer.")
        else:
            color = "#22c55e" 
            st.success("Customer appears stable — continue regular engagement.")  
        st.markdown(f"""
             <div style="background-color: #2d2d2d; border-radius: 8px; height: 12px; width: 100%;">
            <div style="background-color: {color}; width: {prob_pct}%; height: 12px; border-radius: 8px;">
            </div>
        </div>
""",         unsafe_allow_html=True)    

    with col2:
        st.subheader("Input summary")
        summary = {
            "Tenure":           f"{tenure} months",
            "Contract":         contract,
            "Internet service": internet_service,
            "Monthly charges":  f"{monthly_charges}",
            "Total charges":    f"{total_charges}",
            "Payment method":   payment,
            "Senior citizen":   senior,
            "Partner":          partner,
        }
        df = pd.DataFrame(summary.items(), columns=["Feature", "Value"])
        st.table(df.set_index("Feature"))
        

else:
    st.info("Fill in the customer details in the sidebar and click Predict churn.")







