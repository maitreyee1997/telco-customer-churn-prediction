# 📉 Telco Customer Churn Prediction (Production-Ready ML Pipeline)

An end-to-end Machine Learning project designed to predict telecom customer churn using a business-focused modeling approach.

This project follows a modular production-style architecture and includes threshold optimization to maximize business impact.

---

# 🧠 Business Context

Customer churn directly impacts company revenue.

In churn prediction:

* False Negative → Customer leaves unnoticed → Revenue Loss ❌
* False Positive → Extra retention effort → Acceptable operational cost ✅

Therefore, this project prioritizes **Recall over Accuracy** to improve customer retention effectiveness.

---

# 🎯 Project Objective

* Identify high-risk customers
* Reduce missed churn cases
* Support data-driven retention strategies
* Deploy a production-ready ML pipeline

---

# 📂 Dataset Information

* Source: Kaggle – Telco Customer Churn Dataset
* Records: 7,000+ customers
* Target Variable: `Churn (Yes / No)`

### Features Include:

* Demographics
* Account information
* Services subscribed
* Contract details
* Payment method
* Monthly & Total charges

---

# 🏗 Project Architecture

```text id="jyn0hv"
telco-customer-churn/
│
├── data/
├── models/
│   └── churn_pipeline.pkl
├── src/
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── app.py
├── requirements.txt
└── README.md
```

✔ Modular code structure
✔ Separated training & inference logic
✔ Production-ready workflow

---

# 🔄 Machine Learning Workflow

1. Data Cleaning
2. Feature Engineering
3. One-Hot Encoding
4. Feature Scaling
5. Train-Test Split
6. XGBoost Training
7. Cross Validation
8. Threshold Optimization
9. Evaluation & Model Saving

---

# 🤖 Model Selection

## Final Model: XGBoost Classifier

### Why XGBoost?

* Handles non-linear relationships effectively
* Strong performance on tabular datasets
* Works well with mixed feature types
* Robust and scalable for production use

---

# 📊 Model Evaluation

## 🔹 Default Threshold (0.5)

| Metric          | Performance |
| --------------- | ----------- |
| Accuracy        | ~77%        |
| Recall          | Lower       |
| False Negatives | Higher      |

❌ Problem: Many actual churn customers were missed.

---

## 🔹 Optimized Threshold (0.3)

| Metric          | Performance            |
| --------------- | ---------------------- |
| Accuracy        | ~73%                   |
| Recall          | Significantly Improved |
| False Negatives | Reduced                |

✔ Improved churn detection
✔ Better customer retention targeting
✔ Stronger business decision support

---

# 📈 Cross Validation

* Applied 5-Fold Cross Validation
* Stable Recall performance across folds
* ROC-AUC evaluation performed

This helps ensure the model generalizes well and avoids overfitting.

---

# 📊 Confusion Matrix Analysis

## At Threshold = 0.5

* More False Negatives
* Lower churn detection rate

## At Threshold = 0.3

* Reduced False Negatives
* Improved Recall
* Slight Precision drop (acceptable business tradeoff)

---

# 🎯 Why Threshold Optimization Matters

In real-world business scenarios:

* Missing a churn customer = Revenue Loss
* Incorrectly flagging a safe customer = Retention campaign cost

Therefore, optimizing Recall improves overall business impact.

---

# 📌 Key Business Insight

Instead of maximizing only accuracy, this project focuses on reducing False Negatives because missing actual churn customers causes direct revenue loss.

By lowering the prediction threshold from `0.5 → 0.3`:

* More high-risk customers are identified
* Recall improves significantly
* Retention campaigns become more effective
* Business impact improves despite a small drop in precision

---

# 💾 Model Saving

Entire preprocessing + model pipeline saved using:

```python id="90sxb7"
joblib.dump(pipeline, "models/churn_pipeline.pkl")
```

The saved pipeline includes:

* Encoding
* Scaling
* Model
* Threshold logic

---

# 🌐 Live Dashboard

Interactive AI-assisted HTML dashboard:

👉 [Live Demo](YOUR_LIVE_DEMO_LINK)

The dashboard visualizes:

* Churn trends
* Feature importance
* Threshold comparison
* Customer risk prediction
* Business impact analysis

---

# 🚀 Deployment (Streamlit Application)

The project includes a fully interactive Streamlit application.

## Run Locally

```bash id="g2gq6w"
git clone <your-repo-link>
cd telco-customer-churn

pip install -r requirements.txt

streamlit run app.py
```

---

# 🧠 Key ML Concepts Applied

* Threshold Optimization
* Recall Optimization
* Cross Validation
* XGBoost Modeling
* Feature Engineering
* One-Hot Encoding
* Pipeline Serialization
* Business-Focused Evaluation
* Production-Style ML Workflow

---

# 🔮 Future Improvements

* SHAP Explainability
* Hyperparameter Tuning
* Docker Deployment
* Cloud Deployment (AWS/GCP)
* Real-time Prediction API
* Automated Retraining Pipeline

---

# 👩‍💻 Author

**Maitreyee**
Data Analyst | SQL | Power BI | Machine Learning | Customer Analytics

---

# ⭐ If You Found This Project Useful

Please consider giving this repository a ⭐ on GitHub!
