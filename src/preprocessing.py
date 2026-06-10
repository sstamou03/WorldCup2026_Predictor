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
df1["win_rate"] = grouped["result"].transform(lambda x: (x == "Win").cumsum().shift(1) / range(len(x)))

# Helper functions to calculate streaks (consecutive matches) before each game
def get_win_streak(series):
    streak = []
    current = 0
    for val in series:
        streak.append(current)
        if val == "Win":
            current += 1
        else:
            current = 0
    return pd.Series(streak, index=series.index)

def get_undefeated_streak(series):
    streak = []
    current = 0
    for val in series:
        streak.append(current)
        if val in ["Win", "Draw"]:
            current += 1
        else:
            current = 0
    return pd.Series(streak, index=series.index)

#avg goals scored in last 5 matches 
df1["avg_goals_scored_5"] = grouped["score"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

#avg goals conceded in last 5 matches
df1["avg_goals_conceded_5"] = grouped["opp_score"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

# streaks (consecutive wins and consecutive undefeated matches before the game)
df1["win_streak"] = grouped["result"].transform(get_win_streak)
df1["undefeated_streak"] = grouped["result"].transform(get_undefeated_streak)

print(df1.head(10))


