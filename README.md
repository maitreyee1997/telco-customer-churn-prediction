# Telco Customer Churn Prediction using Machine Learning

## 📌 Project Overview

This project builds a production-ready Machine Learning model to predict customer churn for a telecom company.

The objective is to identify high-risk customers in advance so the business can take proactive retention actions and reduce revenue loss.

Unlike traditional models optimized only for accuracy, this solution is optimized for **higher Recall** to minimize missed churn customers.

---

## 🧠 Problem Statement

Telecom companies lose significant revenue due to customer churn.  
The goal is to predict whether a customer will leave the service based on historical customer data.

---

## 📂 Dataset

- Source: Kaggle (Telco Customer Churn Dataset)
- Records: 7000+ customers
- Features: Demographics, Account Info, Services, Charges

---

## ⚙️ Model Architecture

Algorithm Used: **XGBoost Classifier**

Pipeline Steps:
1. Data Cleaning
2. Feature Encoding
3. Train-Test Split
4. Model Training
5. Threshold Optimization (0.3 for higher Recall)
6. Model Evaluation

---

## 📊 Model Performance

- Accuracy: 82%+
- Recall: Improved using threshold = 0.3
- Confusion Matrix included

---

## 🏗️ Project Structure

