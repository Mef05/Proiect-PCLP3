import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Ensure output directory for plots
os.makedirs('plots', exist_ok=True)

# Load and encode training set
data = pd.read_csv('train_processed.csv')
data['Sex'] = data['Sex'].map({'female': 0, 'male': 1})
if 'Embarked' in data.columns:
    data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)

y_train = data.pop('Survived')
X_train = data

# Load and encode test set
data_test = pd.read_csv('test_processed.csv')
data_test['Sex'] = data_test['Sex'].map({'female': 0, 'male': 1})
if 'Embarked' in data_test.columns:
    data_test = pd.get_dummies(data_test, columns=['Embarked'], drop_first=True)

y_test = data_test.pop('Survived')
X_test = data_test

# align test to train columns
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Build pipeline with imputer
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train & predict
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Metrics
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)

print("Baseline Logistic Regression Performance:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Died','Survived']))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Died','Survived'],
            yticklabels=['Died','Survived'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix - Baseline Logistic Regression')
plt.tight_layout()
plt.savefig('plots/confusion_matrix_baseline.png')
plt.show()

# Error graph
errors = np.abs(y_test - y_pred)
plt.figure(figsize=(10, 6))
plt.bar(range(len(errors)), errors, color='red', alpha=0.7)
plt.title('Prediction Errors')
plt.xlabel('Sample Index')
plt.ylabel('Error (1 for incorrect prediction, 0 for correct)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('plots/prediction_errors.png')
plt.show()