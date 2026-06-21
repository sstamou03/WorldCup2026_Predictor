import sys
import kagglehub
import os
import shutil
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

results_url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
shootouts_url = "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"

df = pd.read_csv(results_url)

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


# resolve Shootout
df_shootouts = pd.read_csv(shootouts_url)
df_shootouts['date'] = pd.to_datetime(df_shootouts['date'])
df_shootouts = df_shootouts[['date', 'home_team', 'away_team', 'winner']]

# concate df and df_shootouts
df = pd.merge(df, df_shootouts, on=['date', 'home_team', 'away_team'], how='left')

# update the result after penalties
def resolve_penalties(row):
    if row['result'] == 'Draw' and pd.notna(row['winner']):
        if row['winner'] == row['home_team']:
            return 'Win'
        elif row['winner'] == row['away_team']:
            return 'Loss'
    return row['result']

df['result'] = df.apply(resolve_penalties, axis=1)

df = df.drop(columns=['winner'])



#Features 
#================================================================================================================
#|feture 1| Form Rolling Stat

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

#print(df1.head(10))

#double merge for the features for away and home team in the df
features = ['date','team','form_5','win_rate', 'avg_goals_5', 'avg_opp_goals_5', 'streak']
df_features1 = df1[features].copy()

home_features = df_features1.copy()
home_features.columns = ['date', 'home_team', 'home_form_5', 'home_win_rate', 'home_avg_goals_5', 'home_avg_opp_goals_5', 'home_streak']

away_features = df_features1.copy()
away_features.columns = ['date', 'away_team', 'away_form_5', 'away_win_rate', 'away_avg_goals_5', 'away_avg_opp_goals_5', 'away_streak']

df = pd.merge(df, home_features, on=['date', 'home_team'], how='left')
df = pd.merge(df, away_features, on=['date', 'away_team'], how='left')

#print(df.head())
#================================================================================================================


#================================================================================================================
#|feature 2| match context 
# -> weight for tournament
# -> neutral or not (1, 0)


#delete tournaments which dont contain any of the teams.
target_teams = [
    # AFC
    'Australia', 'Iran', 'Iraq', 'Japan', 'Jordan', 'Qatar', 'Saudi Arabia', 'South Korea', 'Uzbekistan',
    # CAF
    'Algeria', 'Cape Verde', 'DR Congo', 'Egypt', 'Ghana', 'Ivory Coast', 'Morocco', 'Senegal', 'South Africa', 'Tunisia',
    # CONCACAF
    'Canada', 'Curaçao', 'Haiti', 'Mexico', 'Panama', 'United States',
    # CONMEBOL
    'Argentina', 'Brazil', 'Colombia', 'Ecuador', 'Paraguay', 'Uruguay',
    # OFC
    'New Zealand',
    # UEFA 
    'Austria', 'Belgium', 'Bosnia and Herzegovina', 'Croatia', 'Czech Republic', 'England', 
    'France', 'Germany', 'Netherlands', 'Norway', 'Portugal', 'Scotland', 'Spain', 
    'Sweden', 'Switzerland', 'Turkey'
]

mask_target = df['home_team'].isin(target_teams) | df['away_team'].isin(target_teams)

valid_tournaments = df[mask_target]['tournament'].unique()

df = df[df['tournament'].isin(valid_tournaments)].copy().reset_index(drop=True)


#weight for tournament

# for t in df['tournament'].unique():
#     print(t)

def weight_tournament(tournament):

    '''
        weight by tournament

        Scale 5: The Absolute (Only the final stage of the World Cup)
        Scale 4: Final Stages of Continental Cups(e.g. Euro, Copa America, Asian Cup, etc.)
        Scale 3: Official Qualifiers and Nations League/Confederations
        Scale 1: The Friendlies (the lowest possible weight)
        Scale 2: All other small tournaments that 'survived' (e.g. Kirin Cup)
    '''

    t = tournament.lower()

    if 'world cup' in t and 'qualification' not in t and 'qualifier' not in t:
        return 5
        
    elif any(cup in t for cup in ['euro', 'copa américa', 'african', 'asian cup', 'gold cup']):
        if 'qualification' not in t and 'qualifier' not in t:   
            return 4
        else:
            return 3
            
    elif 'qualification' in t or 'qualifier' in t or 'nations league' in t or 'confederations' in t:
        return 3
        
    elif 'friendly' in t:
        return 1
    else:
        return 2


df['tournament_weight'] = df['tournament'].apply(weight_tournament)

#make True to 1 or False to 0
df['is_neutral'] = df['neutral'].astype(int)
#================================================================================================================#================================================================================================================



#TODO
#|feature 3| Fifa rankings
# -> load dataset
# -> right casting in data
# -> merge it with df
# -> features from ranks (discusion)
# -> drop matches before 1992

            
path1 = kagglehub.dataset_download("lucasyukioimafuko/fifa-mens-world-ranking")
if not os.path.exists("data/fifa_mens_rank.csv"):
    shutil.copy(os.path.join(path1, "fifa_mens_rank.csv"), "data/fifa_mens_rank.csv")

df_rank = pd.read_csv("data/fifa_mens_rank.csv")

df_rank = df_rank.rename(columns={'rank_date': 'date', 'country_full': 'team', 'total_points': 'total.points'})

name_mapping = {
    'IR Iran': 'Iran', 'Korea Republic': 'South Korea', 'USA': 'United States',
    'Korea DPR': 'North Korea', "Côte d'Ivoire": 'Ivory Coast', 'Congo DR': 'DR Congo',
    'Cape Verde Islands': 'Cape Verde', 'China PR': 'China', 'Kyrgyz Republic': 'Kyrgyzstan',
    'North Macedonia': 'Macedonia', 'Republic of Ireland': 'Ireland', 'Türkiye': 'Turkey'
}
df_rank['team'] = df_rank['team'].replace(name_mapping)

df_rank = df_rank[['date', 'team', 'total.points']].copy()

df_rank = df_rank.groupby(['date', 'team'], as_index=False)['total.points'].mean()

df_rank = df_rank.rename(columns={'total.points': 'avg.points'})

df_rank = df_rank.sort_values(by=['date', 'avg.points'], ascending=[False, False])




#TODO
#================================================================================================================#================================================================================================================
#|feature 4| H2H

df["matchup"] = df.apply(lambda x: '-'.join(sorted([x['home_team'], x['away_team']])), axis=1)

#our lists 
h2h_matches, home_h2h_wins, away_h2h_wins = [], [], []
history ={}

for i, row in df.iterrows():
    
    matchup = row['matchup']
    home = row["home_team"]
    away = row["away_team"]
    result = row["result"]

    #init if new matchup
    if matchup not in history:
        history[matchup] = {home: 0, away: 0, 'Total': 0}

    #read past of the matchup
    stats = history[matchup]
    h2h_matches.append(stats.get('Total', 0))
    home_h2h_wins.append(stats.get(home, 0))
    away_h2h_wins.append(stats.get(away, 0))

    #update the history
    if result == "Win":
        history[matchup][home] = history[matchup].get(home, 0) + 1
    elif result == "Loss":
        history[matchup][away] = history[matchup].get(away, 0) + 1

    history[matchup]['Total'] = history[matchup].get('Total', 0) + 1

df['h2h_matches'] = h2h_matches
df['home_h2h_wins'] = home_h2h_wins
df['away_h2h_wins'] = away_h2h_wins

#h2h win rate 
df["home_h2h_win_rate"] = np.where(
    df['h2h_matches'] > 0, 
    df['home_h2h_wins'] / df['h2h_matches'],
    0.5
)

df['away_h2h_win_rate'] = np.where(
    df['h2h_matches'] > 0, 
    df['away_h2h_wins'] / df['h2h_matches'],
    0.5
)

df= df.drop(columns=['matchup'])  
    
#================================================================================================================#================================================================================================================
df['match_year'] = df['date'].dt.year

df_rank_home = df_rank.rename(columns={'date': 'match_year', 'team': 'home_team', 'avg.points': 'home_avg.points'})
df = pd.merge(df, df_rank_home, on=['match_year', 'home_team'], how='left')

df_rank_away = df_rank.rename(columns={'date': 'match_year', 'team': 'away_team', 'avg.points': 'away_avg.points'})
df = pd.merge(df, df_rank_away, on=['match_year', 'away_team'], how='left')

df = df[df['match_year'] >= 1993].reset_index(drop=True)

df = df.sort_values('date').reset_index(drop=True)

df['home_avg.points'] = df.groupby('home_team')['home_avg.points'].ffill()
df['away_avg.points'] = df.groupby('away_team')['away_avg.points'].ffill()

rankings_2025 = {
    'Argentina': 1865.00, 'France': 1855.20, 'Spain': 1830.10, 'England': 1810.50,
    'Brazil': 1785.40, 'Portugal': 1750.00, 'Netherlands': 1745.00, 'Belgium': 1735.00,
    'Morocco': 1730.00, 'Croatia': 1720.00, 'Colombia': 1705.00, 'Uruguay': 1690.00,
    'Senegal': 1680.00, 'Mexico': 1675.00, 'United States': 1665.00, 'Japan': 1650.00,
    'Switzerland': 1640.00, 'Iran': 1610.00, 'Turkey': 1595.00, 'Ecuador': 1585.00,
    'South Korea': 1580.00, 'Austria': 1575.00, 'Australia': 1565.00, 'Algeria': 1560.00,
    'Egypt': 1550.00, 'Canada': 1545.00, 'Norway': 1540.00, 'Ivory Coast': 1530.00,
    'Panama': 1525.00, 'Sweden': 1515.00, 'Czech Republic': 1510.00, 'Paraguay': 1500.00,
    'Scotland': 1495.00, 'Tunisia': 1485.00, 'DR Congo': 1465.00, 'Qatar': 1455.00,
    'Uzbekistan': 1450.00, 'Iraq': 1440.00, 'South Africa': 1435.00, 'Saudi Arabia': 1410.00,
    'Bosnia and Herzegovina': 1395.00, 'Jordan': 1380.00, 'Cape Verde': 1365.00,
    'Ghana': 1355.00, 'Curaçao': 1305.00, 'Haiti': 1285.00, 'New Zealand': 1270.00
}

live_rankings_2026 = {
    'Argentina': 1877.27, 'Spain': 1874.71, 'France': 1870.70, 'England': 1828.02,
    'Portugal': 1767.85, 'Brazil': 1765.86, 'Morocco': 1755.10, 'Netherlands': 1753.57,
    'Belgium': 1742.24, 'Germany': 1735.77, 'Croatia': 1714.87, 'Colombia': 1698.35,
    'Mexico': 1687.48, 'Senegal': 1684.07, 'Uruguay': 1673.07, 'United States': 1671.23,
    'Japan': 1661.58, 'Switzerland': 1650.06, 'Iran': 1619.58, 'Turkey': 1605.00,
    'Ecuador': 1598.00, 'Austria': 1597.00, 'South Korea': 1591.00, 'Australia': 1579.00,
    'Algeria': 1571.00, 'Egypt': 1562.00, 'Canada': 1559.00, 'Norway': 1557.00,
    'Ivory Coast': 1540.00, 'Panama': 1539.00, 'Sweden': 1509.00, 'Czech Republic': 1505.00, 
    'Paraguay': 1505.00, 'Scotland': 1503.00, 'Tunisia': 1476.00, 'DR Congo': 1474.00, 
    'Uzbekistan': 1458.00, 'Qatar': 1450.00, 'Iraq': 1446.00, 'South Africa': 1428.00, 
    'Saudi Arabia': 1423.00, 'Bosnia and Herzegovina': 1387.00, 'Jordan': 1387.00, 
    'Cape Verde': 1371.00, 'Ghana': 1346.00, 'Curaçao': 1294.00, 'Haiti': 1293.00, 
    'New Zealand': 1275.00
}

df.loc[df['match_year'] == 2025, 'home_avg.points'] = df['home_team'].map(rankings_2025).fillna(df['home_avg.points'])
df.loc[df['match_year'] == 2025, 'away_avg.points'] = df['away_team'].map(rankings_2025).fillna(df['away_avg.points'])

df.loc[df['match_year'] == 2026, 'home_avg.points'] = df['home_team'].map(live_rankings_2026).fillna(df['home_avg.points'])
df.loc[df['match_year'] == 2026, 'away_avg.points'] = df['away_team'].map(live_rankings_2026).fillna(df['away_avg.points'])

df['points_diff'] = df['home_avg.points'] - df['away_avg.points']
df['form_diff'] = df['home_form_5'] - df['away_form_5']
df['streak_diff'] = df['home_streak'] - df['away_streak']

df['target_result'] = df['result'].map({'Win': 2, 'Draw': 1, 'Loss': 0})

df = df.dropna(subset=['home_form_5', 'away_form_5', 'home_avg.points', 'away_avg.points']).reset_index(drop=True)

cols_to_drop = ['home_score', 'away_score', 'opp_score', 'match_year', 'result', 'neutral', 'city', 'country', 'tournament']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

os.makedirs("data", exist_ok=True)
df.to_csv("data/world_cup_ready.csv", index=False)

#print(df.head(50))

