import pandas as pd
import joblib
import os 
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
#from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import numpy as np

import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

st.title("Credit Card Fraud Prediction")

uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

model_choice = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    X = data.drop(columns=["Class"], axis = 1)
    y = data["Class"]
   
    pipeline = joblib.load(f"model/{model_files[model_choice]}")
    # Predict (NO scaling / encoding needed)
    probs = pipeline.predict_proba(X)[:, 1]
    y_pred = (probs >= 0.2).astype(int) 

    # Metrics
    st.subheader("📊 Classification Report")
    st.text(classification_report(y, y_pred))

    st.subheader("📉 Confusion Matrix")
    st.write(confusion_matrix(y, y_pred))