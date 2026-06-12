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

#Features 

#1.Form Rolling Stat

home_df = df[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'result']].copy()
home_df.columns = ['date', 'team', 'opponent', 'score', 'opp_score', 'result']

away_df = df[['date', 'away_team', 'home_team', 'away_score', 'home_score', 'result']].copy()
away_df.columns = ['date', 'team', 'opponent', 'score', 'opp_score', 'result']

away_df["result"] = away_df["result"].map({"Win": "Loss", "Loss": "Win", "Draw": "Draw"})

df1 = pd.concat([home_df, away_df]).sort_values(["team", "date"]).reset_index(drop=True)


#Team's form 5 last games

df1["points"] = df1["result"].map({"Win" : 3, "Draw" : 1, "Loss" : 0})
grouped = df1.groupby("team")

#points of five latst matches
df1["form_5"] = grouped["points"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

#win rate
df1["win_rate"] = grouped["result"].transform(lambda x: (x=="Win").shift(1).rolling(20, min_periods=1).mean())

#avg goals of last 5 matches 
df1["avg_goals_5"] = grouped["score"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

#avg goals of last 5 matches pou tou kathisan
df1["avg_opp_goals_5"] = grouped["opp_score"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

def get_streak_fixed(series):
    streak_list = []
    current_streak = 0
    
    for val in series:
        if val == "Win":
            current_streak += 1
        else:
            current_streak = 0
        streak_list.append(current_streak)
        
    # to series for shifting and fillna for the first value
    return pd.Series(streak_list, index=series.index).shift(1).fillna(0)


#streak
df1["streak"] = grouped["result"].transform(get_streak_fixed)

#TODO
#double merge for the features for away and home team in the df

#TODO
#|feature 2| match context 
# -> weight for tournament
# -> neutral or not (1, 0)

#TODO
#|feature 3| Fifa rankings
# -> load dataset
# -> right casting in data
# -> merge it with df
# -> features from ranks (discusion)
# -> drop matches before 1992

#TODO
#|feature 4| H2H

            
#print(df1.head(10))

path1 = kagglehub.dataset_download("lucasyukioimafuko/fifa-mens-world-ranking")
if not os.path.exists("data/fifa_mens_rank.csv"):
    shutil.copy(os.path.join(path1, "fifa_mens_rank.csv"), "data/fifa_mens_rank.csv")

df_rank = pd.read_csv("data/fifa_mens_rank.csv")


colsdrop = ['acronym', 'total.points', 'previous.points', 'diff.points']
df_rank = df_rank.drop(columns=colsdrop, errors='ignore')

print(df_rank.head(15))



