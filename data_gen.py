import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


np.random.seed(42)
n_samples = 700  # 500 train + 200 test

data = {
    'Vârstă': np.random.randint(18, 91, n_samples),
    'Sex': np.random.choice(['Masculin', 'Feminin'], n_samples, p=[0.5, 0.5]),
    'Greutate': np.round(np.random.normal(70, 15, n_samples), 1),
    'Glucoză': np.random.uniform(70, 200, n_samples),
    'Tensiune': np.random.randint(90, 181, n_samples),
    'Fumător': np.random.choice(['Da', 'Nu'], n_samples, p=[0.3, 0.7]),
    'Activitate': np.random.choice(['Scăzută', 'Medie', 'Ridicată'], n_samples, p=[0.4, 0.4, 0.2]),
    'Diagnostic': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # 30% bolnavi
}

df = pd.DataFrame(data)

train_df, test_df = train_test_split(df, test_size=200, random_state=42)
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

