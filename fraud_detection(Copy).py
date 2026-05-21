import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("dataset/creditcard.csv").sample(10000)

print(df.head())

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Handle imbalance using SMOTE
smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(X, y)

print("After SMOTE:")
print(y_resampled.value_counts())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier()

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Classification Report
print(classification_report(y_test, y_pred))

# Confusion Matrix
print(confusion_matrix(y_test, y_pred))

# Plot Fraud Distribution
fraud_counts = y.value_counts()

plt.bar(["Normal", "Fraud"], fraud_counts)

plt.title("Fraud vs Normal Transactions")

plt.xlabel("Transaction Type")

plt.ylabel("Count")

# Save graph image
plt.savefig("fraud_graph.png")

print("Project Executed Successfully")

plt.show()