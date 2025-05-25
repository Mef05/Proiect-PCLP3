import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Ensure output directory for plots
os.makedirs('plots', exist_ok=True)

# Load processed dataset
data = pd.read_csv('train_processed.csv')

# --- Encode categorical features so all columns are numeric ---
# Map 'Sex' to 0/1
data['Sex'] = data['Sex'].map({'female': 0, 'male': 1})

# One-hot encode 'Embarked' if still present
if 'Embarked' in data.columns:
    data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)

# --- end encoding ---

# Separate features and target
y_train = data['Survived']
X_train = data.drop(columns=['Survived'])

data = pd.read_csv('test.csv')

y_test = data['Survived']
X_test = data.drop(columns=['Survived'])

# Train a baseline logistic regression model
model = LogisticRegression(max_iter=1000000)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Baseline Logistic Regression Performance:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}\n")

# Detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Died','Survived']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', ax=ax,
    xticklabels=['Died','Survived'], yticklabels=['Died','Survived']
)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix - Baseline Logistic Regression')
plt.tight_layout()
plt.savefig('plots/confusion_matrix_baseline.png')
plt.show()