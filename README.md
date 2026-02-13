#🚀 Telco Customer Churn Prediction

##📌 Project Overview
This project builds a production-ready Machine Learning model to predict customer churn for a telecom company.
The primary objective is to identify high-risk customers in advance so the business can take proactive retention actions and reduce revenue loss.
Unlike traditional models optimized for accuracy, this solution is optimized for higher Recall to minimize missed churn customers.

##🧠 Model Architecture

-Algorithm: XGBoost Classifier

-Preprocessing: ColumnTransformer

-OneHotEncoding (Categorical Features)

-StandardScaler (Numerical Features)

-Pipeline: End-to-end Scikit-learn Pipeline

-Model Persistence: joblib

-Deployment: Streamlit Web App

##🏗 System Architecture Diagram
Raw Telco Data
       ↓
Data Cleaning & Preprocessing
       ↓
ColumnTransformer
 (OHE + Scaling)
       ↓
XGBoost Classifier
       ↓
Probability Output
       ↓
Custom Threshold (0.3)
       ↓
Final Churn Prediction
       ↓
Streamlit UI Deployment


This architecture ensures:

-Reproducibility

-Scalable preprocessing

-Business-aligned decision threshold

##🎯 Model Optimization Strategy

The model is optimized for higher Recall (0.78) for churned customers.
Instead of using the default probability threshold (0.5),
a custom threshold of 0.3 is applied to better capture churn customers.

Why Threshold = 0.3?
Missing a churn customer is more costly than flagging a non-churn customer.
Lower threshold increases Recall.
Business-focused optimization instead of metric-focused optimization.

##📊 Model Performance (Test Set)
Metric	Value
-Recall (Churn = 1)	0.78
-Precision (Churn = 1)	0.49
-Accuracy	0.72
-ROC-AUC Score	Strong discriminatory performance

##🛠 Tech Stack
-Python
-Scikit-learn
-XGBoost
-Pandas & NumPy
-Streamlit

##💡 Features

Real-time churn prediction
Churn probability score
High / Low risk classification
Business-driven threshold tuning
Interactive Streamlit dashboard

