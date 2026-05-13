# 🕵️‍♂️ AML Fraud Detection using Random Forest

## 📌 Overview
This project builds a machine learning model to detect potential **money laundering (AML) transactions** using a **Random Forest classifier**.  

The model analyzes transaction-level data and engineered risk features to identify suspicious financial activity.

---

## 📊 Dataset
Synthetic Transaction Monitoring Dataset (AML)  
👉 https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml

Contains transaction details such as:
- Payment & receiving currencies
- Bank locations
- Transaction amounts
- Account identifiers
- Label: `Is_laundering`

---

## ⚙️ Features Engineering

Key engineered features:

- 💱 **Currency_Mismatch** → 1 if payment and receiving currencies differ
- 🌍 **Cross_Border** → 1 if sender and receiver are in different countries
- 💰 Numeric conversion of transaction amounts
- 🧾 One-hot encoding for categorical variables

---

## 🤖 Model

- 🌲 Algorithm: Random Forest Classifier
- 📦 n_estimators: 200
- 🌳 max_depth: 12
- ⚖️ class_weight: balanced (handles class imbalance)
- ⚡ n_jobs: -1 (uses all CPU cores)

---

## 🧪 Train/Test Split

- 80% training
- 20% testing
- Stratified split to preserve class distribution

---

## 📈 Evaluation Metrics

The model is evaluated using:

- 📊 Precision
- 📊 Recall
- 📊 F1-score
- 📉 ROC-AUC Score

---

## 🚀 Results

After training, the model outputs:
- Classification report
- ROC-AUC score for fraud detection performance

---

## 🧠 Goal

To build a baseline **Anti-Money Laundering (AML) detection system** that can:
- Identify suspicious financial behavior
- Handle imbalanced fraud datasets
- Serve as a foundation for more advanced fraud detection systems

---

## 🛠️ Tech Stack

- Python 🐍
- Pandas & NumPy
- Scikit-learn 🤖
- Random Forest 🌲

---

## 📁 Project Structure

aml-fraud-detection/
│
├── main.py / notebook
├── SAML-D dataset (not included)
├── README.md
└── requirements.txt


---

## ⚠️ Disclaimer

This project uses **synthetic data** and is intended for **educational and research purposes only**.

---

## ⭐ If you like this project
Give it a star and feel free to fork it!
