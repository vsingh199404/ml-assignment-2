import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report

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
    
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y )
    # Load FULL pipeline
    model = joblib.load(f"model/{model_files[model_choice]}")

    # Predict (NO scaling / encoding needed)
    y_pred = model.predict(X_test)

    # Metrics
    st.subheader("📊 Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("📉 Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))