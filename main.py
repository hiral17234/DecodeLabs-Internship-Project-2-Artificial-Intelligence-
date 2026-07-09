
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load Dataset

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["Species"] = iris.target
df["Species"] = df["Species"].map({
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
})

print("\n========== DATASET ==========\n")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nMissing Values\n")
print(df.isnull().sum())

print("\nDataset Information\n")
print(df.info())

print("\nStatistical Summary\n")
print(df.describe())

# Save Dataset

df.to_csv("datasets/iris.csv", index=False)

print("\nDataset saved successfully!")

# Data Visualization

sns.set_style("whitegrid")

# Pair Plot

pair_plot = sns.pairplot(
    df,
    hue="Species"
)

pair_plot.savefig(
    "screenshots/pairplot.png"
)

plt.close()

# Heatmap

plt.figure(figsize=(8,6))

sns.heatmap(
    df.drop("Species", axis=1).corr(),
    annot=True,
    cmap="Blues"
)

plt.title("Feature Correlation")

plt.savefig("screenshots/heatmap.png")

plt.close()

# Box Plot

plt.figure(figsize=(10,6))

sns.boxplot(data=df.drop("Species", axis=1))

plt.title("Feature Distribution")

plt.savefig("screenshots/boxplot.png")

plt.close()

# Preparing Data

X = iris.data

y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# Models

models = {

    "Logistic Regression": LogisticRegression(),

    "Decision Tree": DecisionTreeClassifier(),

    "KNN": KNeighborsClassifier(),

    "Support Vector Machine": SVC(),

    "Random Forest": RandomForestClassifier()

}

accuracy_scores = {}

best_model = None

best_accuracy = 0

print("\n==============================")

print("MODEL COMPARISON")

print("==============================")

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    accuracy_scores[name] = accuracy

    print(f"{name} Accuracy : {accuracy:.4f}")

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model

# Best Model

print("\nBest Model Selected")

print(best_model)

# Save Best Model

joblib.dump(
    best_model,
    "models/iris_model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("\nModel Saved Successfully!")

# Final Prediction

prediction = best_model.predict(X_test)

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        prediction,
        target_names=iris.target_names
    )
)

# Confusion Matrix


cm = confusion_matrix(
    y_test,
    prediction
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    cmap="Greens",
    fmt="d",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.savefig(
    "screenshots/confusion_matrix.png"
)

plt.close()

# Accuracy Comparison Graph

plt.figure(figsize=(8,5))

plt.bar(
    accuracy_scores.keys(),
    accuracy_scores.values()
)

plt.xticks(rotation=20)

plt.ylabel("Accuracy")

plt.title("Machine Learning Model Comparison")

plt.savefig(
    "screenshots/model_accuracy.png"
)

plt.close()

print("\nProject Completed Successfully!")
