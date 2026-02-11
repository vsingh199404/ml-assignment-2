# Machine Learning Model Comparison

## a. Problem Statement
The objective of this project is to detect fraudulent credit card transactions using machine learning techniques. Given the highly imbalanced nature of the Credit Card Fraud Detection dataset, the task involves classifying transactions as fraudulent or legitimate while ensuring effective detection of minority class instances. Multiple classification models are evaluated using metrics such as AUC, Precision, Recall, F1-score, and MCC to identify the most reliable model for fraud detection.

---

## b. Dataset Description
The Credit Card Fraud Detection dataset contains credit card transactions made by European cardholders, where each transaction is labeled as either fraudulent or legitimate. The dataset consists of numerical features obtained through Principal Component Analysis (PCA), along with the transaction amount and time. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for only 0.172% of all transactions.

---

## c. Models Used and Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.9975 | 0.9178 | 0.3974 | 0.8378 | 0.5391 | 0.5761 |
| Decision Tree | 0.9964 | 0.9173 | 0.3069 | 0.8378 | 0.4493 | 0.5059 |
| kNN | 0.9996 | 0.9188 | 0.8986 | 0.8378 | 0.8671 | 0.8674 |
| Naive Bayes | 0.9776 | 0.8944 | 0.0599 | 0.8108 | 0.1116 | 0.2168 |
| Random Forest (Ensemble) | 0.9983  | 0.9182 | 0.5000 | 0.8378 | 0.6263  | 0.6465 |
| XGBoost (Ensemble) | 0.9908 | 0.9212 | 0.1422 | 0.8514 | 0.2437 | 0.3458 |

---

## d. Observations on Model Performance

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | High recall and AUC indicate good fraud detection capability, but moderate precision shows a higher false positive rate. |
| Decision Tree | Detects fraud effectively with high recall but suffers from low precision, suggesting overfitting and weak generalization.  |
| kNN | Shows the best overall balance with very high precision, F1-score, and MCC, indicating excellent fraud detection performance. |
| Naive Bayes | Achieves high recall but extremely low precision, resulting in poor overall performance and many false positives. |
| Random Forest (Ensemble) | Provides a good balance between precision and recall with strong F1 and MCC, making it a robust and reliable model. |
| XGBoost (Ensemble) | Has the highest AUC, indicating strong ranking ability, but low precision and F1 suggest threshold tuning is required. |

---

