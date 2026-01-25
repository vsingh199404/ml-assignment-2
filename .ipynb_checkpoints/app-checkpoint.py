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

    # Drop same columns as training
    data.drop(columns=["sl_no", "salary"], inplace=True, errors="ignore")

    X = data.drop("status", axis=1)
    y = data["status"].map({"Placed": 1, "Not Placed": 0})

    # Load FULL pipeline
    model = joblib.load(f"saved_models/{model_files[model_choice]}")

    # Predict (NO scaling / encoding needed)
    y_pred = model.predict(X)

    # Metrics
    st.subheader("📊 Classification Report")
    st.text(classification_report(y, y_pred))

    st.subheader("📉 Confusion Matrix")
    st.write(confusion_matrix(y, y_pred))