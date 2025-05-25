import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cols = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

# Load Titanic datasets
test_df = pd.read_csv('test.csv')

# Display basic info
print("testing set shape:", test_df.shape)
print("\ntesting set columns:", test_df.columns.tolist())

# Missing Value Analysis
def analyze_missing(df, dataset_name):
    missing = df.isnull().sum()
    percent = (missing / len(df)) * 100
    result = pd.DataFrame({'Missing': missing, 'Percent (%)': percent})
    print(f"\nMissing values in {dataset_name} set:")
    print(result[result['Missing'] > 0])

analyze_missing(test_df, "testing")

# Visualize missing values
plt.figure(figsize=(12, 6))
sns.heatmap(test_df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values in testing Set')
plt.savefig('plots/test_missing_values.png')
# plt.show()


# Handle Missing Values
# Age – fill with median (avoid chained inplace)
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())

# Cabin – drop column by reassignment
test_df = test_df.drop('Cabin', axis=1)

# Embarked – fill with mode instead of median
test_df['Embarked'] = test_df['Embarked'].fillna(test_df['Embarked'].mode()[0])

# Descriptive Statistics
print("\ntesting set statistics:")
print(test_df.describe())

# Categorical features
cat_cols = ['Pclass', 'Sex', 'Embarked', 'Survived']
for col in cat_cols:
    if col in test_df.columns:
        print(f"\n{col} distribution:")
        print(test_df[col].value_counts(normalize=True))

# Data Visualization
# Numerical features distributions
num_cols = ['Age', 'Fare', 'SibSp', 'Parch']
test_df[num_cols].hist(figsize=(12, 8), bins=20)
plt.suptitle('Distributions of Numerical Features')
plt.savefig('plots/test_numerical_distributions.png')
# plt.show()

# Categorical features by Survival
for col in ['Pclass', 'Sex', 'Embarked']:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=test_df, x=col, hue='Survived')
    plt.title(f'Distribution of {col} by Survival')
    plt.savefig(f'plots/test_{col}_by_survival.png')
    # plt.show()

# Outlier detection
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=test_df, y=col)
    plt.title(f'Boxplot for {col}')
    plt.savefig(f'plots/test_{col}_boxplot.png')
    # plt.show()

# Correlation Analysis
# Encode categorical variables
test_encoded = pd.get_dummies(test_df, columns=['Sex', 'Embarked'], drop_first=True)

# Select numerical columns + encoded columns + target
corr_cols = num_cols + ['Sex_male', 'Embarked_Q', 'Embarked_S', 'Survived']
plt.figure(figsize=(10, 6))
sns.heatmap(test_encoded[corr_cols].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.savefig('plots/test_correlation_heatmap.png')
# plt.show()



# --- Feature vs Target Relationship Analysis ---
# Violin plots for numeric features vs binary target
for col in num_cols:
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=test_df, x='Survived', y=col, palette='Set2', hue='Survived')
    plt.title(f'{col} Distribution by Survival')
    plt.xticks([0, 1], ['Died', 'Survived'])
    plt.savefig(f'plots/test_{col}_violin_by_survival.png')
    # plt.show()

# Countplots for categorical features vs target
for col in cat_cols:
    if col != 'Survived':
        plt.figure(figsize=(8, 5))
        sns.countplot(data=test_df, x=col, hue='Survived', palette='Set1')
        plt.title(f'{col} vs Survival Count')
        plt.legend(title='Survived', labels=['Died', 'Survived'])
        plt.savefig(f'plots/test_{col}_count_by_survival.png')
        # plt.show()

# Scatter plot for two numeric features colored by target
plt.figure(figsize=(8, 6))
sns.scatterplot(data=test_df, x='Age', y='Fare', hue='Survived', palette='coolwarm', alpha=0.7)
plt.title('Age vs Fare by Survival')
plt.legend(title='Survived', labels=['Died', 'Survived'])
plt.savefig('plots/test_age_fare_scatter_by_survival.png')
# plt.show()


test_df = test_df.drop(columns=['Name'])  # Drop 'Name' column as it is not useful for analysis
test_df = test_df.drop(columns=['PassengerId'])  # Drop 'PassengerId' as it is not useful for analysis
test_df = test_df.drop(columns=['Ticket'])  # Drop 'Ticket' as it is not useful for analysis

# Save processed testing data  
test_df.to_csv('test_processed.csv', index=False)

