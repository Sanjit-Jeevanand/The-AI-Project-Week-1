# Theme: Project feature build
# Scope: Dataset ingestion → preprocessing → pipeline wiring
# Dataset: loan_data.csv

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/loan_data.csv")

X = df.drop(columns=["loan_status"])
y = df["loan_status"]
num_cols = X.select_dtypes(include=["float64", "int64"]).columns.to_list()
cat_cols = X.select_dtypes(include=["object"]).columns.to_list()

model = LogisticRegression(max_iter=1000)
