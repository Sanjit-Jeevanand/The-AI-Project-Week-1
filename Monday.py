import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("data/loan_data.csv")

X = df.drop(columns=["loan_status"])
y = df["loan_status"]

# Numeric-only subset for bias–variance illustration
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
X = X[numeric_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

depths = [1, 3, 5, 10, 15]
train_acc, val_acc = [], []

for d in depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(X_train, y_train)
    train_acc.append(model.score(X_train, y_train))
    val_acc.append(model.score(X_test, y_test))

plt.plot(depths, train_acc, label="Train Accuracy")
plt.plot(depths, val_acc, label="Validation Accuracy")
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Bias–Variance Tradeoff")
plt.show()