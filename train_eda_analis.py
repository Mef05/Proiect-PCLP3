import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cols = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

# Load Titanic datasets
train_df = pd.read_csv('train.csv')

# Display basic info
print("Training set shape:", train_df.shape)
print("\nTraining set columns:", train_df.columns.tolist())

# Missing Value Analysis
def analyze_missing(df, dataset_name):
    missing = df.isnull().sum()
    percent = (missing / len(df)) * 100
    result = pd.DataFrame({'Missing': missing, 'Percent (%)': percent})
    print(f"\nMissing values in {dataset_name} set:")
    print(result[result['Missing'] > 0])

analyze_missing(train_df, "training")

# Visualize missing values
plt.figure(figsize=(12, 6))
sns.heatmap(train_df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values in TRAINING Set')
plt.savefig('plots/train_missing_values.png')
# plt.show()


# Handle Missing Values
# Age – fill with median (avoid chained inplace)
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())

# Cabin – drop column by reassignment
train_df = train_df.drop('Cabin', axis=1)

# Embarked – fill with mode instead of median
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

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
plt.suptitle('Distributions of Numerical Features (TRAIN)')
plt.savefig('plots/train_numerical_distributions.png')
# plt.show()

# Categorical features by Survival
for col in ['Pclass', 'Sex', 'Embarked']:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=train_df, x=col, hue='Survived')
    plt.title(f'Distribution of {col} by Survival (TRAIN)')
    plt.savefig(f'plots/train_{col}_by_survival.png')
    # plt.show()

# Outlier detection
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=train_df, y=col)
    plt.title(f'Boxplot for {col} (TRAIN)')
    plt.savefig(f'plots/train_{col}_boxplot.png')
    # plt.show()

# Correlation Analysis
# Encode categorical variables
train_encoded = pd.get_dummies(train_df, columns=['Sex', 'Embarked'], drop_first=True)

# Select numerical columns + encoded columns + target
corr_cols = num_cols + ['Sex_male', 'Embarked_Q', 'Embarked_S', 'Survived']
plt.figure(figsize=(10, 6))
sns.heatmap(train_encoded[corr_cols].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap (TRAIN)')
plt.savefig('plots/train_correlation_heatmap.png')
# plt.show()



# --- Feature vs Target Relationship Analysis ---
# Violin plots for numeric features vs binary target
for col in num_cols:
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=train_df, x='Survived', y=col, palette='Set2', hue='Survived')
    plt.title(f'{col} Distribution by Survival (TRAIN)')
    plt.xticks([0, 1], ['Died', 'Survived'])
    plt.savefig(f'plots/train_{col}_violin_by_survival.png')
    # plt.show()

# Countplots for categorical features vs target
for col in cat_cols:
    if col != 'Survived':
        plt.figure(figsize=(8, 5))
        sns.countplot(data=train_df, x=col, hue='Survived', palette='Set1')
        plt.title(f'{col} vs Survival Count (TRAIN)')
        plt.legend(title='Survived', labels=['Died', 'Survived'])
        plt.savefig(f'plots/train_{col}_count_by_survival.png')
        # plt.show()

# Scatter plot for two numeric features colored by target
plt.figure(figsize=(8, 6))
sns.scatterplot(data=train_df, x='Age', y='Fare', hue='Survived', palette='coolwarm', alpha=0.7)
plt.title('Age vs Fare by Survival (TRAIN)')
plt.legend(title='Survived', labels=['Died', 'Survived'])
plt.savefig('plots/train_age_fare_scatter_by_survival.png')
# plt.show()


train_df = train_df.drop(columns=['Name'])  # Drop 'Name' column as it is not useful for analysis
train_df = train_df.drop(columns=['PassengerId'])  # Drop 'PassengerId' as it is not useful for analysis
train_df = train_df.drop(columns=['Ticket'])  # Drop 'Ticket' as it is not useful for analysis

# Save processed training data  
train_df.to_csv('train_processed.csv', index=False)

