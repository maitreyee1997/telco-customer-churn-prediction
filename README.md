# 📉 Telco Customer Churn Prediction App

An end-to-end Machine Learning project that predicts whether a telecom customer is likely to churn.  
The model is trained on the Telco Customer Churn dataset and optimized to improve Recall for better churn detection.

---

## 📌 Project Overview

This project predicts whether a telecom customer will leave the company based on historical customer data.

It follows a complete Data Science workflow:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Encoding
- Model Training
- Threshold Optimization
- Model Evaluation
- Model Saving
- Deployment Ready Structure

---

## 🧠 Problem Statement

Customer churn leads to significant revenue loss in the telecom industry.

The objective of this project is to:

- Identify high-risk customers
- Minimize missed churn cases
- Improve customer retention strategies

Unlike traditional models optimized only for accuracy, this solution focuses on improving **Recall** to reduce false negatives (missed churn customers).

---

## 📂 Dataset Information

- Source: Kaggle – Telco Customer Churn Dataset
- Records: 7000+ customers
- Features Include:
  - Customer demographics
  - Account information
  - Services subscribed
  - Monthly & Total charges
  - Contract type
  - Payment method

Target Variable:
```
Churn (Yes / No)
```

---

## ⚙️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- Joblib

---

## 🤖 Model Used

### XGBoost Classifier (Final Model)

Pipeline Steps:

1️⃣ Data Cleaning  
2️⃣ Label Encoding / One-Hot Encoding  
3️⃣ Train-Test Split  
4️⃣ Model Training  
5️⃣ Threshold Optimization (0.3 for higher Recall)  
6️⃣ Model Evaluation  

---

## 📊 Model Performance

- Accuracy: 82%+
- Recall: Improved using custom threshold (0.3)
- Confusion Matrix Evaluated
- Better detection of churn customers compared to default threshold

Why threshold tuning?

Instead of default 0.5 threshold, using 0.3 improves Recall — meaning fewer churn customers are missed.

---

## 💾 Model Saving

The trained model is saved using:

```python
joblib.dump(model, "models/churn_model.pkl")
```

---

## 🚀 How to Run the Project Locally

### 1️⃣ Clone the Repository

```
git clone <your-repo-link>
cd telco_churn_prediction
```

### 2️⃣ Create and Activate Virtual Environment

```
conda create -n churn_env python=3.10
conda activate churn_env
```

### 3️⃣ Install Requirements

```
pip install -r requirements.txt
```

### 4️⃣ Run the Application (If Streamlit App Exists)

```
streamlit run application.py
```

---

## 📂 Project Structure

```
telco_customer_churn/
│
├── data/
├── models/
│   └── churn_model.pkl
├── notebook/
│   └── churn_model_training.ipynb
├── application.py
├── requirements.txt
└── README.md
```

---

## 📈 Future Improvements

- Hyperparameter tuning
- Cross-validation
- SHAP feature importance
- Model deployment on Streamlit Cloud
- API deployment using FastAPI

---

## 👩‍💻 Author

Maitreyee  
Data Analyst | Aspiring Data Scientist  

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
