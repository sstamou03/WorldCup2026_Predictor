import kagglehub
import os
import shutil
import pandas as pd

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Download and copy to data folder to show in Explorer
path = kagglehub.dataset_download("martj42/international-football-results-from-1872-to-2017")
if not os.path.exists("data/results.csv"):
    shutil.copy(os.path.join(path, "results.csv"), "data/results.csv")

df = pd.read_csv("data/results.csv")

#EDA 
#print(df.shape)
#print(df.dtypes)
#print(df.head())
#print(df.isnull().sum())

#casting
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

#handle null values
df = df.dropna(subset=['home_score', 'away_score'])
df['home_score'] = df['home_score'].astype(int)
df['away_score'] = df['away_score'].astype(int)

#print(df.isnull().sum())

#add result column
df['result'] = df.apply(
    lambda x: 'Win' if x['home_score'] > x['away_score'] else ( 'Draw' if x['home_score'] == x['away_score'] else 'Loss')
    , axis=1
)
#print(df['result'].value_counts())

for t in sorted(df['tournament'].unique()):
    print(t)
    



