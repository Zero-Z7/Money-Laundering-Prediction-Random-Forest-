##
# AML Analysis with Random Forest
# @author: Zero_Z7 - Ignacio Velasco Delgado
# see {@link https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml} to get access to te data
#
##

#-#-#-#-#-#-#
# Libraries #
#-#-#-#-#-#-#

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

#-#-#-#-#-#-#
# Load Data #
#-#-#-#-#-#-#

# Default path when uncompressing the zip file, make sure to edit the CSV into samples if you just want to test it 
# that include cases where "Is_laundering" is 1 to properly train it

file_path = "SAML-D.csv/SAML-D - Sample 2.csv"

print("Loading Dataset...")
df = pd.read_csv(file_path)

print(f"="*148)
print(f"# DATASET INFORMATION #")
print(f"="*148)

print(f"✅ DATASET SHAPE: {df.shape}")
print("📌 COLUMNS:", " | ".join(map(str, df.columns)))

# We set our target column, set another target if changing the dataset, would be good to ask the user to enter the name
# manually

target = 'Is_laundering'

print("\n🎯 Target Distribution:")
print(df[target].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))
print("\n")

#-#-#-#-#-#-#
# Features  #
#-#-#-#-#-#-#

# Not all columns are useful, in this case we are dropping columns we are not interested in if changing the dataset we would
# have to choose which ones are not needed or have it cleaned from before. If we wanted a deepdive we would still hold to the
# sender_account and reciever_account, since they might pose a cluster of slightly suspicious transactions

drop_cols = ['Time', 'Date', 'Sender_account', 'Receiver_account']
df = df.drop(columns=drop_cols, errors='ignore')

# We check wether the amount is indeed in numeric
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

# We create a series of flags that will help identify possible transactions: currency mismatch and cross-border. More flags
# can be created with more complex mechanics, but for the sake of simplicity we are creating some basic ones.

# Currency mismatch flag
df['Currency_Mismatch'] = (df['Payment_currency'] != df['Received_currency']).astype(int)

# Cross-border flag
df['Cross_Border'] = (df['Sender_bank_location'] != df['Receiver_bank_location']).astype(int)

#-#-#-#-#-#-#-#-#-#-#
# Data Preparation  #
#-#-#-#-#-#-#-#-#-#-#

# We make some preparations for X and Y to further progress into prediction

X = df.drop(columns=[target, 'Laundering_type'], errors='ignore')
y = df[target]

# Categorical into numeric for simplicity
X = pd.get_dummies(X, drop_first=True)

#-#-#-#-#-#-#-#
# Prediction  #
#-#-#-#-#-#-#-#

print(f"="*148)
print(f"# PREDICTION #")
print(f"="*148)

print(f"\nFinal features: {X.shape[1]}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

print("\nTraining model...")
model.fit(X_train, y_train)

#-#-#-#-#-#
# RESULTS #
#-#-#-#-#-#

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n=== Model Performance ===")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")