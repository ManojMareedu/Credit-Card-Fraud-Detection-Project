import pandas as pd
from typing import Tuple
from zenml import pipeline, step
from EDA import eda_step
from data_ingestion import load_data
from models import select_best_model 
from models import (
    train_test_split_data,
    CreditFraud_logreg_model,
    CreditFraud_knn_model,
    CreditFraud_decision_tree_model,
    CreditFraud_random_forest_model,
)

@step
def train_test_split_step(df: pd.DataFrame) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
]:
    X_train, X_test, y_train, y_test = train_test_split_data(df)
    return X_train, X_test, y_train, y_test

@pipeline
def fraud_detection_pipeline(file_path: str):
    df = load_data(file_path = file_path)
    df = eda_step(df=df)
    X_train, X_test, y_train, y_test = train_test_split_step(df=df)
    logreg_model, logreg_score = CreditFraud_logreg_model(X_train, y_train, X_test, y_test)
    knn_model, knn_score = CreditFraud_knn_model(X_train, y_train, X_test, y_test)
    dt_model, dt_score = CreditFraud_decision_tree_model(X_train, y_train, X_test, y_test)
    rf_model, rf_score = CreditFraud_random_forest_model(X_train, y_train, X_test, y_test)

    select_best_model(
        logreg_model, logreg_score,
        knn_model, knn_score,
        dt_model, dt_score,
        rf_model, rf_score
    )

if __name__ == "__main__":
    fraud_detection_pipeline(file_path="card_transdata.csv")