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

    # explaination
       # Top reasons
    shap_df = pd.DataFrame({
        "Feature": input_df.columns,
        "SHAP Value": shap_values.values[0]
    })

    # positive -> churn increase
    increase_df = shap_df[shap_df["SHAP Value"] > 0] .sort_values(by="SHAP Value", ascending=False)

    # negative -> churn decrease
    decrease_df = shap_df[shap_df["SHAP Value"] < 0] .sort_values(by="SHAP Value")  
    st.subheader("📌 Prediction Explanation")

    c1, c2 = st.columns(2)

# ── Human-readable feature name + business explanation mapping
    FEATURE_EXPLAIN = {
        # Risk-increasing (positive SHAP)
        "Contract_Monthly": {
            "name": "Month-to-Month Contract",
            "risk": "Customer is on a month-to-month contract — no long-term commitment makes it very easy to leave.",
        },
        "InternetService_Fiber optic": {
            "name": "Fiber Optic Internet",
            "risk": "Fiber optic users have higher churn — likely due to higher cost and more competitive alternatives.",
        },
        "Tenure": {
            "name": "Short Tenure",
            "risk": "Customer has been with the company for only {val} months — new customers have significantly higher churn risk.",
            "safe": "Customer has been with the company for {val} months — long-tenure customers are highly unlikely to leave.",
        },
        "MonthlyCharges": {
            "name": "High Monthly Charges",
            "risk": "Monthly charges are ₹{val} — high billing is a top driver of customer dissatisfaction and churn.",
            "safe": "Monthly charges are ₹{val} — affordable pricing strongly reduces churn intent.",
        },
        "TotalCharges": {
            "name": "Total Charges",
            "risk": "Total charges of ₹{val} — customer may feel the accumulated cost is no longer worth it.",
            "safe": "Total charges of ₹{val} reflect a long billing history — high-spend customers rarely churn.",
        },
        "PaperlessBilling": {
            "name": "Paperless Billing",
            "risk": "Paperless billing users tend to compare providers more actively, slightly increasing churn risk.",
            "safe": "Paper billing customers tend to have lower churn rates.",
        },
        "SeniorCitizen": {
            "name": "Senior Citizen",
            "risk": "Senior citizens can be more sensitive to pricing changes, increasing churn risk.",
            "safe": "Senior citizen loyalty patterns suggest this customer is well-retained.",
        },
        "TechSupport": {
            "name": "No Tech Support",
            "risk": "No tech support — unresolved technical issues are a very common churn trigger.",
            "safe": "Customer has Tech Support — reduces frustration and significantly lowers churn.",
        },
        "OnlineSecurity": {
            "name": "No Online Security",
            "risk": "No Online Security — missing value-added services reduces the perceived benefit of staying.",
            "safe": "Customer has Online Security — increases perceived value and reduces churn intent.",
        },
        "OnlineBackup": {
            "name": "No Online Backup",
            "risk": "No Online Backup — fewer bundled services means lower switching cost.",
            "safe": "Customer has Online Backup — bundled services increase switching cost.",
        },
        "DeviceProtection": {
            "name": "No Device Protection",
            "risk": "No Device Protection — fewer services means easier to switch providers.",
            "safe": "Customer has Device Protection — additional services strengthen loyalty.",
        },
        "MultipleLines": {
            "name": "Multiple Lines",
            "risk": "Single line customer — fewer services means lower switching cost.",
            "safe": "Customer has Multiple Lines — deeply engaged with the service.",
        },
        "StreamingTV": {
            "name": "Streaming TV",
            "risk": "No Streaming TV — fewer services means lower switching cost.",
            "safe": "Customer subscribes to Streaming TV — bundled services make switching harder.",
        },
        "StreamingMovies": {
            "name": "Streaming Movies",
            "risk": "No Streaming Movies — fewer value-added services reduces switching cost.",
            "safe": "Customer subscribes to Streaming Movies — bundled services increase loyalty.",
        },
        "Contract_One year": {
            "name": "One-Year Contract",
            "risk": "No annual contract — absence of commitment increases churn risk.",
            "safe": "Customer is on a one-year contract — annual commitment significantly reduces churn.",
        },
        "InternetService_DSL": {
            "name": "DSL Internet",
            "risk": "Non-DSL customer — billing or service factors are contributing to churn risk.",
            "safe": "Customer uses DSL internet — DSL users show lower churn rates than fiber optic customers.",
        },
        "PaymentMethod_Bank transfer (automatic)": {
            "name": "Auto Bank Transfer",
            "risk": "No automatic bank transfer — manual payments are associated with higher billing friction.",
            "safe": "Customer pays via automatic bank transfer — seamless billing strongly reduces churn.",
        },
        "PaymentMethod_Credit card (automatic)": {
            "name": "Auto Credit Card",
            "risk": "No automatic credit card — non-auto payments increase billing friction.",
            "safe": "Customer pays via automatic credit card — auto-payment reduces involuntary churn.",
        },
        "Partner": {
            "name": "No Partner",
            "risk": "Single customers can be slightly more likely to switch providers.",
            "safe": "Customer has a partner — partner accounts tend to be more stable.",
        },
        "Dependents": {
            "name": "No Dependents",
            "risk": "No dependents — accounts without families tend to be slightly more mobile.",
            "safe": "Customer has dependents — family accounts are significantly more stable.",
        },
    }

    def get_explanation(feature, shap_val, raw_val):
        """Returns (display_name, sentence) for a given feature + shap direction."""
        entry = FEATURE_EXPLAIN.get(feature, {})
        display_name = entry.get("name", feature.replace("_", " "))
        direction = "risk" if shap_val > 0 else "safe"
        sentence = entry.get(direction, f"{display_name} is contributing to this prediction.")
        # Fill numeric placeholders
        sentence = sentence.replace("{val}", f"{int(raw_val)}" if feature == "Tenure" else f"{raw_val:.0f}")
        return display_name, sentence

    with c1:
        st.subheader("🚨 High-Risk Customer Indicators")

        shown = 0
        for i in range(len(increase_df)):
            if shown >= 5:
                break
            feature  = increase_df.iloc[i]["Feature"]
            shap_val = round(increase_df.iloc[i]["SHAP Value"], 3)
            raw_val  = input_df.iloc[0].get(feature, 0) if hasattr(input_df, 'iloc') else 0

            display_name, sentence = get_explanation(feature, shap_val, raw_val)

            if shap_val >= 0.10:
                st.markdown(
                    f"<div style='background:#2a1515;border-left:3px solid #ef4444;"
                    f"border-radius:0 8px 8px 0;padding:10px 14px;margin-bottom:8px;'>"
                    f"<span style='font-size:11px;font-weight:600;color:#ef4444;'>🔴 High impact — {display_name}</span><br>"
                    f"<span style='font-size:13px;color:#fafafa;'>{sentence}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                shown += 1
            elif shap_val >= 0.04:
                st.markdown(
                    f"<div style='background:#2a2010;border-left:3px solid #f59e0b;"
                    f"border-radius:0 8px 8px 0;padding:10px 14px;margin-bottom:8px;'>"
                    f"<span style='font-size:11px;font-weight:600;color:#f59e0b;'>🟡 Medium impact — {display_name}</span><br>"
                    f"<span style='font-size:13px;color:#fafafa;'>{sentence}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                shown += 1

        if shown == 0:
            st.success("No significant risk-increasing factors detected.")

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Reasons NOT to churn
    with c2:
        st.subheader("🔐 Reasons Customer May Stay")

        shown = 0
        for i in range(len(decrease_df)):
            if shown >= 5:
                break
            feature  = decrease_df.iloc[i]["Feature"]
            shap_val = round(decrease_df.iloc[i]["SHAP Value"], 3)
            raw_val  = input_df.iloc[0].get(feature, 0) if hasattr(input_df, 'iloc') else 0

            display_name, sentence = get_explanation(feature, shap_val, raw_val)

            if shap_val <= -0.10:
                st.markdown(
                    f"<div style='background:#0d2a1e;border-left:3px solid #1d9e75;"
                    f"border-radius:0 8px 8px 0;padding:10px 14px;margin-bottom:8px;'>"
                    f"<span style='font-size:11px;font-weight:600;color:#1d9e75;'>🟢 Strong protector — {display_name}</span><br>"
                    f"<span style='font-size:13px;color:#fafafa;'>{sentence}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                shown += 1
            elif shap_val <= -0.02:   # threshold kam kiya — ab chhote factors bhi dikhenge
                st.markdown(
                    f"<div style='background:#0d1e2a;border-left:3px solid #378ADD;"
                    f"border-radius:0 8px 8px 0;padding:10px 14px;margin-bottom:8px;'>"
                    f"<span style='font-size:11px;font-weight:600;color:#378ADD;'>🔵 Moderate protector — {display_name}</span><br>"
                    f"<span style='font-size:13px;color:#fafafa;'>{sentence}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                shown += 1

        if shown == 0:
            st.warning("No strong protective factors — customer is at elevated risk.")

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Verdict banner
    n_risk = min(5, len(increase_df[increase_df["SHAP Value"] >= 0.04]))
    n_safe = min(5, len(decrease_df[decrease_df["SHAP Value"] <= -0.02]))

    if n_safe > n_risk:
        v_color, v_icon = "#1d9e75", "🟢"
        v_text = f"<b>{n_safe} protective factors</b> outweigh <b>{n_risk} risk factor(s)</b>. Customer is likely to stay — maintain their current experience."
    elif n_risk > n_safe:
        v_color, v_icon = "#ef4444", "🔴"
        v_text = f"<b>{n_risk} risk factors</b> outweigh <b>{n_safe} protective factor(s)</b>. Immediate retention action is recommended."
    else:
        v_color, v_icon = "#f59e0b", "🟡"
        v_text = f"Risk and protective factors are <b>balanced ({n_risk} each)</b>. Monitor this customer and consider a light-touch retention offer."

    st.markdown(
        f"<div style='background:#1a1d27;border:1px solid {v_color};border-radius:10px;"
        f"padding:14px 18px;margin-top:8px;'>"
        f"<span style='font-size:15px;font-weight:700;color:{v_color};'>{v_icon} Overall Verdict</span><br>"
        f"<span style='font-size:13px;color:#fafafa;line-height:1.7;'>{v_text}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )



        # st.info("Positive values increase churn risk, negative values decrease it.")      

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







