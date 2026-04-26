# 📊 Customer Churn Prediction Dashboard (XGBoost + SHAP)

An end-to-end machine learning project that predicts telecom customer churn, estimates churn probability, recommends retention actions, and explains individual predictions using SHAP.

## 🚀 Live Features
- Predict customer churn probability
- Classify customer risk level (Low / Medium / High)
- Recommend retention actions
- Explain predictions using SHAP (Explainable AI)
- Interactive Streamlit dashboard

---

## 📌 Problem Statement
Customer churn leads to major revenue loss for telecom companies.

This project aims to identify customers likely to leave so businesses can take proactive retention actions.

---

## 🛠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- SHAP
- Seaborn, Matplotlib
- Streamlit

---

## ⚙ Project Workflow

### 1. Data Preprocessing
- Handled missing values in `TotalCharges`
- Encoded categorical features
- Applied one-hot encoding
- Prepared features for modeling

---

### 2. Exploratory Data Analysis
Performed:
- Churn distribution analysis
- Crosstab analysis
- Correlation heatmap
- Chi-square significance testing

### Key findings:
- Month-to-month customers churn more
- Low tenure customers have higher churn risk
- Manual payment methods show higher churn
- Automatic payment users are more stable

---

### 3. Modeling
Models evaluated:
- Logistic Regression
- Random Forest
- XGBoost (Final Model)

---

### 4. Model Optimization
Implemented:
- RandomizedSearchCV for hyperparameter tuning
- Class imbalance handling using `scale_pos_weight`
- Threshold optimization using Precision-Recall curve

---

### 5. Explainable AI
Used SHAP to explain:

- Why a customer may churn
- Which features increase risk
- Which features reduce risk

---

## 📈 Final Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 80% |
| Precision | 0.60 |
| Recall | 0.73 |
| F1 Score | 0.66 |

### Confusion Matrix

```text
[[1317, 380]
 [140, 488]]
```

- Correctly detects 73% of churn customers  
- Balances churn detection and false alarms

---

## 🖥 Dashboard Features

### Input Customer Details
- Demographics
- Services
- Billing details

### Prediction Output
- Churn probability
- Risk level
- Recommended action

### Explainability
- SHAP waterfall explanation for each prediction

---

## 📸 Screenshots

Add project screenshots here:

### Dashboard
![Dashboard](screenshots/dashboard.png)

### Prediction Result
![Prediction](screenshots/prediction.png)

### SHAP Explanation
![SHAP](screenshots/shap.png)

---

## 📁 Project Structure

```bash
customer-churn-prediction/
│
├── app.py
├── churn_analysis.ipynb
├── telecom_churn_data.csv
├── xgboost_model.pkl
├── requirements.txt
└── README.md
```

---

## ⚙ Installation

Clone repository:

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run dashboard:

```bash
streamlit run app.py
```

---

## 💡 Example Use Case

Input:

- Contract: Month-to-month
- Tenure: 12 months
- Monthly Charges: 65
- Payment Method: Credit Card

Prediction:

- Churn Probability: 43%
- Risk Level: Medium Risk
- Action: Send retention offer

---

## 🎯 Business Impact
This model identifies **73% of potential churn customers**, helping businesses:

- Target at-risk users
- Improve retention campaigns
- Reduce revenue loss

---

## 🔮 Future Improvements
- Add ROC-AUC monitoring
- Deploy on Streamlit Cloud
- Add global SHAP summary plot
- Integrate real-time customer database

---

## 👤 Author
**Rishabh Gupta**  
Machine Learning | Data Science Enthusiast

GitHub: https://github.com/yourusername  
LinkedIn: https://linkedin.com/in/yourprofile