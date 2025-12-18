# Theme: Project feature build
# Scope: Dataset ingestion → preprocessing → pipeline wiring
# Dataset: loan_data.csv

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.preprocessing import build_preprocessor
from src.pipeline import build_pipeline
from sklearn.model_selection import train_test_split
import mlflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("loan_default_experiments")
import mlflow.sklearn

df = pd.read_csv("./data/loan_data.csv")

X = df.drop(columns=["loan_status"])
y = df["loan_status"]
num_cols = X.select_dtypes(include=["float64", "int64"]).columns.to_list()
cat_cols = X.select_dtypes(include=["object"]).columns.to_list()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

model = LogisticRegression(max_iter=1000)

preprocessor = build_preprocessor(cat_cols, num_cols)
pipeline = build_pipeline(preprocessor, model)

with mlflow.start_run():
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)

    pipeline.fit(X_train, y_train)

    train_acc = pipeline.score(X_train, y_train)
    val_acc = pipeline.score(X_test, y_test)

    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("val_accuracy", val_acc)

    try:
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
    except Exception as e:
        print("Model logging failed:", e)


import joblib
joblib.dump(pipeline, "artifacts/model.joblib")