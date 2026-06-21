import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 1. Load model and historical data
try:
    model = joblib.load("models/football_model_xgb_timebased.pkl")
    print("✅ XGBoost model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit()

df_data = pd.read_csv("data/world_cup_ready.csv")

# Get most recent stats per team
latest_stats = df_data.sort_values('date').groupby('home_team').last().reset_index()
teams_dict = latest_stats.set_index('home_team').to_dict('index')

# 2. Official FIFA World Cup 2026 Groups (48 teams)
groups = {
    'Group A': ['Mexico', 'South Africa', 'Korea Republic', 'Czech Republic'],
    'Group B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
    'Group C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'Group D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'Group E': ['Germany', 'Curacao', "Cote d'Ivoire", 'Ecuador'],
    'Group F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'Group G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'Group H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'Group I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'Group J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'Group K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
    'Group L': ['England', 'Croatia', 'Ghana', 'Panama']
}

# 3. Match prediction logic
def predict_match(team1, team2, is_knockout=False):
    default_stats = {'home_avg.points': 1000, 'home_form_rolling': 0.5, 'home_streak_rolling': 0}
    
    stats1 = teams_dict.get(team1, default_stats)
    stats2 = teams_dict.get(team2, default_stats)
    
    # Calculate feature differences
    points_diff = stats1.get('home_avg.points', 1000) - stats2.get('home_avg.points', 1000)
    form_diff = stats1.get('home_form_rolling', 0.5) - stats2.get('home_form_rolling', 0.5)
    streak_diff = stats1.get('home_streak_rolling', 0) - stats2.get('home_streak_rolling', 0)
    
    match_features = np.array([[points_diff, form_diff, streak_diff, 1.0, 1.0, 0.5, 0.5]])
    probs = model.predict_proba(match_features)[0]
    
    if not is_knockout:
        result = np.argmax(probs)
        if result == 2: return team1, 3, team2, 0
        elif result == 1: return team1, 1, team2, 1
        else: return team1, 0, team2, 3
    else:
        # Split draw probability for knockout stages
        team1_score = probs[2] + (probs[1] / 2)
        team2_score = probs[0] + (probs[1] / 2)
        return team1 if team1_score >= team2_score else team2

# 4. Knockout stage runner
def run_knockout_stage(teams_list, stage_name):
    print(f"\n--- {stage_name} ---")
    winners = []
    for i in range(0, len(teams_list), 2):
        t1, t2 = teams_list[i], teams_list[i+1]
        winner = predict_match(t1, t2, is_knockout=True)
        print(f"{t1} vs {t2} -> Winner: {winner}")
        winners.append(winner)
    return winners

# =====================================================================
# 5. Group Stage Simulation & Standings
# =====================================================================
print("\n=== STARTING WORLD CUP 2026 SIMULATION (FIFA BRACKET) ===")

group_winners = {}
group_runners_up = {}
thirds_stats = []  # Stores tuple: (team_name, points, group_letter)

for group_name, group_teams in groups.items():
    standings = {team: 0 for team in group_teams}
    
    for i in range(len(group_teams)):
        for j in range(i + 1, len(group_teams)):
            t1, t2 = group_teams[i], group_teams[j]
            _, s1, _, s2 = predict_match(t1, t2, is_knockout=False)
            standings[t1] += s1
            standings[t2] += s2
            
    sorted_standings = sorted(standings.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n[{group_name} Standings]")
    for team, pts in sorted_standings:
        print(f"  {team}: {pts} pts")
    
    g_letter = group_name[-1]
    group_winners[g_letter] = sorted_standings[0][0]
    group_runners_up[g_letter] = sorted_standings[1][0]
    
    # Store 3rd place team details for the overall ranking
    thirds_stats.append((sorted_standings[2][0], sorted_standings[2][1], g_letter))

# =====================================================================
# 6. Third-Place Teams Global Leaderboard
# =====================================================================
# Sort all 12 third-place teams by points
best_thirds_sorted = sorted(thirds_stats, key=lambda x: x[1], reverse=True)

print("\n=======================================")
print("  3RD PLACE TEAMS GLOBAL LEADERBOARD   ")
print("=======================================")
for idx, (team, pts, group) in enumerate(best_thirds_sorted, 1):
    status = "QUALIFIED ✅" if idx <= 8 else "ELIMINATED ❌"
    print(f"{idx}. {team} (Group {group}): {pts} pts -> {status}")
print("=======================================\n")

# Extract only the top 8 qualified third-place teams
top_8_thirds = [item[0] for item in best_thirds_sorted[:8]]
# =====================================================================
# =====================================================================
# 7. Official FIFA Round of 32 Bracket Allocation (Matches 73 - 88)
# =====================================================================
round_of_32_matches = [
    # Sunday, 28 June 2026
    (group_runners_up['A'], group_runners_up['B']),   # Match 73: Group A runners-up v Group B runners-up
    
    # Monday, 29 June 2026
    (group_winners['E'], top_8_thirds[0]),            # Match 74: Group E winners v Third place
    (group_winners['F'], group_runners_up['C']),      # Match 75: Group F winners v Group C runners-up
    (group_winners['C'], group_runners_up['F']),      # Match 76: Group C winners v Group F runners-up
    
    # Tuesday, 30 June 2026
    (group_winners['I'], top_8_thirds[1]),            # Match 77: Group I winners v Third place
    (group_runners_up['E'], group_runners_up['I']),   # Match 78: Group E runners-up v Group I runners-up
    (group_winners['A'], top_8_thirds[2]),            # Match 79: Mexico (Winner A) v Third place
    
    # Wednesday, 1 July 2026
    (group_winners['L'], top_8_thirds[3]),            # Match 80: Group L winners v Third place
    (group_winners['D'], top_8_thirds[4]),            # Match 81: USA (Winner D) v Third place
    (group_winners['G'], top_8_thirds[5]),            # Match 82: Group G winners v Third place
    
    # Thursday, 2 July 2026
    (group_runners_up['K'], group_runners_up['L']),   # Match 83: Group K runners-up v Group L runners-up
    (group_winners['H'], group_runners_up['J']),      # Match 84: Group H winners v Group J runners-up
    (group_winners['B'], top_8_thirds[6]),            # Match 85: Group B winners v Third place
    
    # Friday, 3 July 2026
    (group_winners['J'], group_runners_up['H']),      # Match 86: Group J winners v Group H runners-up
    (group_winners['K'], top_8_thirds[7]),            # Match 87: Group K winners v Third place
    (group_runners_up['D'], group_runners_up['G'])    # Match 88: Group D runners-up v Group G runners-up
]

# Run Knockout Phases
print("=== KNOCKOUT STAGES ===")
round_of_16_teams = []
print("\n--- ROUND OF 32 ---")
for idx, (t1, t2) in enumerate(round_of_32_matches, 73):
    winner = predict_match(t1, t2, is_knockout=True)
    print(f"Match {idx}: {t1} vs {t2} -> Winner: {winner}")
    round_of_16_teams.append(winner)

round_of_16 = run_knockout_stage(round_of_16_teams, "ROUND OF 16")
quarters = run_knockout_stage(round_of_16, "QUARTER-FINALS")
semis = run_knockout_stage(quarters, "SEMI-FINALS")

# Grand Final
print("\n=== WORLD CUP GRAND FINAL ===")
champion = predict_match(semis[0], semis[1], is_knockout=True)
print(f"Final Match: {semis[0]} vs {semis[1]}")
print(f"\n🏆 WORLD CUP CHAMPION 2026: {champion} 🏆\n")