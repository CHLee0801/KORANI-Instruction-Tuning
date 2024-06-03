import pandas as pd 
import os 

for i in os.listdir():
    if '.py' in i:
        continue 
    train_df = pd.read_csv(f'{i}/train.csv')
    dev_df = pd.read_csv(f'{i}/dev.csv')
    train_df = train_df.sample(n=1000, random_state=42)
    dev_df = dev_df.sample(n=200, random_state=42)
    train_df.to_csv(f'{i}/train.csv', index=False)
    dev_df.to_csv(f'{i}/dev.csv', index=False)