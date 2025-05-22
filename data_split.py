from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=200, random_state=42)
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)
