import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Titanic datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Resize train to 500 rows and test to 200 rows
train_df = train_df.sample(n=500, random_state=42).reset_index(drop=True)
test_df = test_df.sample(n=200, random_state=42).reset_index(drop=True)

# Display basic info
print("Training set shape:", train_df.shape)
print("Test set shape:", test_df.shape)
print("\nTraining set columns:", train_df.columns.tolist())

# Missing Value Analysis
def analyze_missing(df, dataset_name):
    missing = df.isnull().sum()
    percent = (missing / len(df)) * 100
    result = pd.DataFrame({'Missing': missing, 'Percent (%)': percent})
    print(f"\nMissing values in {dataset_name} set:")
    print(result[result['Missing'] > 0])

analyze_missing(train_df, "training")
analyze_missing(test_df, "test")

# Visualize missing values
plt.figure(figsize=(12, 6))
sns.heatmap(train_df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values in Training Set')
plt.show()

plt.figure(figsize=(12, 6))
sns.heatmap(test_df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values in Test Set')
plt.show()

# Handle Missing Values
# Age – fill with median (avoid chained inplace)
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
test_df['Age']  = test_df['Age'].fillna(test_df['Age'].median())

# Embarked – fill with mode
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

# Cabin – drop column by reassignment
train_df = train_df.drop('Cabin', axis=1)
test_df  = test_df.drop('Cabin', axis=1)

# Fare – fill with median
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

# Descriptive Statistics
print("\nTraining set statistics:")
print(train_df.describe())

# Categorical features
cat_cols = ['Pclass', 'Sex', 'Embarked', 'Survived']
for col in cat_cols:
    if col in train_df.columns:
        print(f"\n{col} distribution:")
        print(train_df[col].value_counts(normalize=True))

# Data Visualization
# Numerical features distributions
num_cols = ['Age', 'Fare', 'SibSp', 'Parch']
train_df[num_cols].hist(figsize=(12, 8), bins=20)
plt.suptitle('Distributions of Numerical Features')
plt.show()

# Categorical features
for col in ['Pclass', 'Sex', 'Embarked']:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=train_df, x=col, hue='Survived')
    plt.title(f'Distribution of {col} by Survival')
    plt.show()

# Outlier detection
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=train_df, y=col)
    plt.title(f'Boxplot for {col}')
    plt.show()

# Correlation Analysis
# Encode categorical variables
train_encoded = pd.get_dummies(train_df, columns=['Sex', 'Embarked'], drop_first=True)

# Select numerical columns + encoded columns + target
corr_cols = num_cols + ['Sex_male', 'Embarked_Q', 'Embarked_S', 'Survived']
plt.figure(figsize=(10, 6))
sns.heatmap(train_encoded[corr_cols].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

# Target Analysis
# Age vs Survival
plt.figure(figsize=(8, 5))
sns.violinplot(data=train_df, x='Survived', y='Age')
plt.title('Age Distribution by Survival')
plt.xticks([0, 1], ['Died', 'Survived'])
plt.show()

# Fare vs Survival
plt.figure(figsize=(8, 5))
sns.boxplot(data=train_df, x='Survived', y='Fare')
plt.title('Fare Distribution by Survival')
plt.xticks([0, 1], ['Died', 'Survived'])
plt.show()

# Pclass vs Survival
plt.figure(figsize=(6, 4))
sns.countplot(data=train_df, x='Pclass', hue='Survived')
plt.title('Survival Count by Passenger Class')
plt.show()