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
    
    #X_train, X_test, y_train, y_test = train_test_split(
    #X, y, test_size=0.8, random_state=42, stratify=y )
    # Load FULL pipeline

    #numeric_features = X.columns.tolist()

    

    #classes = np.array([0, 1])
    #class_weights = compute_class_weight(
        #class_weight="balanced",
        #classes=classes,
        #y=y_train
    #)

    #class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    

    #numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    #categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    #num_pipeline = Pipeline(steps=[
        #("imputer", SimpleImputer(strategy="mean")),
        #("scaler", StandardScaler())
    #])
    #cat_pipeline = Pipeline(steps=[
        #("imputer", SimpleImputer(strategy="most_frequent")),
        #("onehot", OneHotEncoder(drop="first", sparse_output=False,handle_unknown="ignore"))
    #])
    #preprocessor = ColumnTransformer(
        #transformers=[
            #("num", num_pipeline, numeric_cols),
            #("cat", cat_pipeline, categorical_cols)
       # ]
    #)
    

    pipeline = joblib.load(f"model/{model_files[model_choice]}")
    #pipeline = Pipeline(steps=[
        #("preprocessing", preprocessor),
        #("smote", SMOTE(random_state=42)),
        #("classifier", model)
    #])
    
    #pipeline.fit(X_train, y_train)
    # Predict (NO scaling / encoding needed)
    y_pred = pipeline.predict(X)

    # Metrics
    st.subheader("📊 Classification Report")
    st.text(classification_report(y, y_pred))

    st.subheader("📉 Confusion Matrix")
    st.write(confusion_matrix(y, y_pred))