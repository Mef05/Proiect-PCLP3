import pandas as pd

# Load train and test datasets
train_df = pd.read_csv('./titanic/train.csv')
test_df = pd.read_csv('./titanic/test.csv')

# Resize train to 500 rows and test to 200 rows
train_df = train_df.sample(n=500, random_state=42).reset_index(drop=True)
test_df = test_df.sample(n=200, random_state=42).reset_index(drop=True)

# Save resized datasets back to CSV
train_df.to_csv('./train.csv', index=False)
test_df.to_csv('./test.csv', index=False)

print("Resized datasets saved as train_resized.csv and test_resized.csv.")