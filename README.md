# 📉 Telco Customer Churn Prediction (Production-Ready ML Pipeline)

An end-to-end Machine Learning project designed to predict telecom customer churn using a business-focused modeling approach.

This project is built with a modular production-style architecture and includes threshold optimization to maximize business impact.

---

# 🧠 Business Context

Customer churn directly impacts company revenue.

In churn prediction:

- False Negative → Customer leaves unnoticed → Revenue Loss ❌
- False Positive → Extra retention effort → Acceptable cost ✅

Therefore, this project prioritizes **Recall over Accuracy**.

---

# 🎯 Project Objective

- Identify high-risk customers
- Reduce missed churn cases
- Support data-driven retention strategies
- Deploy a production-ready ML pipeline

---

# 📂 Dataset

- Source: Kaggle – Telco Customer Churn Dataset
- Records: 7,000+ customers
- Target Variable: `Churn (Yes / No)`

Features include:

- Demographics
- Account information
- Services subscribed
- Contract details
- Payment method
- Monthly & Total charges

---

# 🏗 Project Architecture

telco-customer-churn/
│
├── data/
├── models/
│ └── churn_pipeline.pkl
├── src/
│ ├── train.py
│ ├── predict.py
│ └── utils.py
├── app.py
├── requirements.txt
└── README.md


✔ Modular code  
✔ Separated training & inference logic  
✔ Production-ready structure  

---

# 🔄 Machine Learning Workflow

1. Data Cleaning
2. Feature Engineering
3. Categorical Encoding (One-Hot Encoding)
4. Feature Scaling
5. Train-Test Split
6. XGBoost Training
7. Cross-Validation
8. Threshold Optimization
9. Evaluation & Model Saving

---

# 🤖 Model Selection

### Final Model: XGBoost Classifier

Why XGBoost?

- Handles non-linear relationships
- Works well with mixed feature types
- Strong performance in tabular data

---

# 📊 Model Evaluation

## 🔹 Default Threshold (0.5)

| Metric      | Score |
|------------|--------|
| Accuracy   | ~77%   |
| Recall     | Lower  |
| False Negatives | Higher |

Problem: Many churn customers were missed.

---

## 🔹 Optimized Threshold (0.3)

| Metric      | Score |
|------------|--------|
| Accuracy   | ~73%   |
| Recall     | Significantly Improved |
| False Negatives | Reduced |

✔ Improved churn detection  
✔ Better business decision support  

---

# 📈 Cross Validation

- 5-Fold Cross Validation applied
- Stable Recall performance across folds
- ROC-AUC evaluated

This ensures the model is not overfitting.

---

# 📊 Confusion Matrix Comparison

### At 0.5 Threshold
- More False Negatives
- Lower churn detection

### At 0.3 Threshold
- Reduced False Negatives
- Improved Recall
- Slight drop in Precision (acceptable business tradeoff)

---

# 🎯 Why Threshold Optimization Matters

In real-world business scenarios:

- Missing a churn customer = Revenue Loss
- Flagging a safe customer = Retention campaign cost

Therefore, recall optimization improves business impact.

---

# 💾 Model Saving

Entire preprocessing + model pipeline saved as:

```python
joblib.dump(pipeline, "models/churn_pipeline.pkl")

The saved model includes:
Encoding
Scaling
Model
Threshold logic

#🚀 Deployment (Streamlit Application)

The project includes a fully interactive Streamlit app.

Run Locally
git clone <your-repo-link>
cd telco-customer-churn


conda create -n churn_env python=3.10
conda activate churn_env

pip install -r requirements.txt

streamlit run application.py
##![Dashboard](screenshots/dashboard.png)
