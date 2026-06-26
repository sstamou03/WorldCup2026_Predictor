import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

#load model
model = joblib.load("models/football_model_xgb_timebased.pkl")

#load data
df_data = pd.read_csv("data/world_cup_ready.csv")
df_data['date'] = pd.to_datetime(df_data['date'])

latest_stats = df_data.sort_values('date').groupby('home_team').last().reset_index()
teams_dict = latest_stats.set_index('home_team').to_dict('index')

#get h2h stats
h2h_lookup = {}
for _, row in df_data.iterrows():
    key = frozenset({row['home_team'], row['away_team']})
    # keep all h2h records, not just the last one
    if key not in h2h_lookup:
        h2h_lookup[key] = {
            'home': row['home_team'],
            'away': row['away_team'],
            'home_h2h_win_rate': row.get('home_h2h_win_rate', 0.5),
            'away_h2h_win_rate': row.get('away_h2h_win_rate', 0.5),
        }


groups = {
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'],
    'B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'E': ['Germany', 'Curaçao', 'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}


real_results = [
    # GROUP A
    # Matchday 1
    ('Mexico', 'South Africa', 2, 0),
    ('South Korea', 'Czech Republic', 2, 1),
    # Matchday 2
    ('Czech Republic', 'South Africa', 1, 1),
    ('Mexico', 'South Korea', 1, 0),
    # Matchday 3
    ('Mexico', 'Czech Republic', 3, 0),
    ('South Africa', 'South Korea', 1, 0),

    # GROUP B
    # Matchday 1
    ('Canada', 'Bosnia and Herzegovina', 1, 1),
    ('Qatar', 'Switzerland', 1, 1),
    # Matchday 2
    ('Canada', 'Qatar', 6, 0),
    ('Switzerland', 'Bosnia and Herzegovina', 4, 1),
    # Matchday 3
    ('Switzerland', 'Canada', 3, 1),
    ('Bosnia and Herzegovina', 'Qatar', 3, 1),

    # GROUP C
    # Matchday 1
    ('Brazil', 'Morocco', 1, 1),
    ('Scotland', 'Haiti', 1, 0),
    # Matchday 2
    ('Brazil', 'Haiti', 3, 0),
    ('Morocco', 'Scotland', 1, 0),
    # Matchday 3
    ('Scotland', 'Brazil', 0, 3),
    ('Morocco', 'Haiti', 4, 2),

    # GROUP D
    # Matchday 1
    ('United States', 'Paraguay', 4, 1),
    ('Australia', 'Turkey', 2, 0),
    # Matchday 2
    ('United States', 'Australia', 2, 0),
    ('Paraguay', 'Turkey', 1, 0),
    # Matchday 3
    ('Turkey', 'United States', 3, 2),
    ('Paraguay', 'Australia', 0, 0),

    # GROUP E
    # Matchday 1
    ('Germany', 'Curaçao', 7, 1),
    ('Ivory Coast', 'Ecuador', 1, 0),
    # Matchday 2
    ('Germany', 'Ivory Coast', 2, 1),
    ('Ecuador', 'Curaçao', 0, 0),
    # Matchday 3
    ('Curaçao', 'Ivory Coast', 0, 2),
    ('Ecuador', 'Germany', 2, 1),

    # GROUP F
    # Matchday 1
    ('Netherlands', 'Japan', 2, 2),
    ('Sweden', 'Tunisia', 5, 1),
    # Matchday 2
    ('Netherlands', 'Sweden', 5, 1),
    ('Japan', 'Tunisia', 4, 0),
    # Matchday 3
    ('Japan', 'Sweden', 1, 1),
    ('Netherlands', 'Tunisia', 3, 1),

    # GROUP G
    # Matchday 1
    ('Belgium', 'Egypt', 1, 1),
    ('Iran', 'New Zealand', 2, 2),
    # Matchday 2
    ('Egypt', 'New Zealand', 3, 1),
    ('Belgium', 'Iran', 0, 0),
    # Matchday 3 → παίζονται 26/6, θα ενημερωθούν

    # GROUP H
    # Matchday 1
    ('Spain', 'Cape Verde', 0, 0),
    ('Saudi Arabia', 'Uruguay', 1, 1),
    # Matchday 2
    ('Spain', 'Saudi Arabia', 4, 0),
    ('Uruguay', 'Cape Verde', 2, 2),
    # Matchday 3 → παίζονται 26/6, θα ενημερωθούν

    # GROUP I
    # Matchday 1
    ('France', 'Senegal', 3, 1),
    ('Iraq', 'Norway', 1, 4),
    # Matchday 2
    ('France', 'Iraq', 3, 0),
    ('Norway', 'Senegal', 3, 2),
    # Matchday 3 → παίζονται 26/6, θα ενημερωθούν

    # GROUP J
    # Matchday 1
    ('Argentina', 'Algeria', 3, 0),
    ('Austria', 'Jordan', 3, 1),
    # Matchday 2
    ('Argentina', 'Austria', 2, 0),
    ('Jordan', 'Algeria', 1, 2),
    # Matchday 3 → παίζονται 27/6, θα ενημερωθούν

    # GROUP K
    # Matchday 1
    ('Portugal', 'DR Congo', 1, 1),
    ('Uzbekistan', 'Colombia', 1, 3),
    # Matchday 2
    ('Portugal', 'Uzbekistan', 5, 0),
    ('Colombia', 'DR Congo', 1, 0),
    # Matchday 3 → παίζονται 27/6, θα ενημερωθούν

    # GROUP L
    # Matchday 1
    ('England', 'Croatia', 4, 2),
    ('Ghana', 'Panama', 1, 0),
    # Matchday 2
    ('England', 'Ghana', 0, 0),
    ('Panama', 'Croatia', 0, 1),
    # Matchday 3 → παίζονται 27/6, θα ενημερωθούν
]

# Build a lookup dict: frozenset({team1, team2}) -> list of (home, away, hg, ag)
played_matches = {}
for home, away, hg, ag in real_results:
    key = frozenset({home, away})                         #frozenset: immutable set -> used as key
    if key not in played_matches:
        played_matches[key] = []
    played_matches[key].append((home, away, hg, ag))


#
def get_group_match_result(team1, team2):
    """
    Returns (team1_pts, team2_pts, tag) for a group stage match.
    First checks real results; if not found, uses the model to predict.
    """
    key = frozenset({team1, team2})
    if key in played_matches:

        # if already played use that result and update pts
        home, away, hg, ag = played_matches[key][0]

        tag = f"REAL: {home} {hg}-{ag} {away}"

        # assign points to team1 and team2 correctly
        if hg > ag:
            home_pts, away_pts = 3, 0
        elif hg < ag:
            home_pts, away_pts = 0, 3
        else:
            home_pts, away_pts = 1, 1

        if team1 == home:
            return home_pts, away_pts, tag
        else:
            return away_pts, home_pts, tag

    else:
        # if not played use the model to predict
        return predict_group_match(team1, team2)


def get_h2h_rates(team1, team2):

    """Get the h2h win rates for a pair of teams from historical data."""

    key = frozenset({team1, team2})
    if key in h2h_lookup:

        h2h = h2h_lookup[key]

        # if team1 was 'home' in the last recorded match
        if h2h['home'] == team1:

            return h2h['home_h2h_win_rate'], h2h['away_h2h_win_rate']

        else:

            return h2h['away_h2h_win_rate'], h2h['home_h2h_win_rate']

    return 0.5, 0.5  # no h2h history


def predict_group_match(team1, team2):

    """Use the XGBoost model to predict a group stage match."""

    default_stats = {'home_avg.points': 1000, 'home_form_5': 1.5, 'home_streak': 0}

    stats1 = teams_dict.get(team1, default_stats)
    stats2 = teams_dict.get(team2, default_stats)

    # warn if a team is missing from the dataset
    if team1 not in teams_dict:
        print(f"  [WARNING] '{team1}' not found in dataset, using default stats.")
    if team2 not in teams_dict:
        print(f"  [WARNING] '{team2}' not found in dataset, using default stats.")

    points_diff = stats1.get('home_avg.points', 1000) - stats2.get('home_avg.points', 1000)
    form_diff = stats1.get('home_form_5', 1.5) - stats2.get('home_form_5', 1.5)
    streak_diff = stats1.get('home_streak', 0) - stats2.get('home_streak', 0)

    h2h_home, h2h_away = get_h2h_rates(team1, team2)

    # features: points_diff, form_diff, streak_diff, tournament_weight, is_neutral, home_h2h_win_rate, away_h2h_win_rate
    match_features = np.array([[points_diff, form_diff, streak_diff, 5.0, 1.0, h2h_home, h2h_away]])

    probs = model.predict_proba(match_features)[0]

    result = np.argmax(probs)

    win_pct = f"{probs[2]*100:.0f}%"
    draw_pct = f"{probs[1]*100:.0f}%"
    loss_pct = f"{probs[0]*100:.0f}%"

    if result == 2:
        return 3, 0, f"PREDICTED: {team1} wins (W:{win_pct} D:{draw_pct} L:{loss_pct})"
    elif result == 1:
        return 1, 1, f"PREDICTED: Draw (W:{win_pct} D:{draw_pct} L:{loss_pct})"
    else:
        return 0, 3, f"PREDICTED: {team2} wins (W:{win_pct} D:{draw_pct} L:{loss_pct})"


def predict_knockout_match(team1, team2):

    """Use the model for knockout stage (no draws allowed)."""

    default_stats = {'home_avg.points': 1000, 'home_form_5': 1.5, 'home_streak': 0}
    stats1 = teams_dict.get(team1, default_stats)
    stats2 = teams_dict.get(team2, default_stats)

    points_diff = stats1.get('home_avg.points', 1000) - stats2.get('home_avg.points', 1000)
    form_diff = stats1.get('home_form_5', 1.5) - stats2.get('home_form_5', 1.5)
    streak_diff = stats1.get('home_streak', 0) - stats2.get('home_streak', 0)

    h2h_home, h2h_away = get_h2h_rates(team1, team2)

    match_features = np.array([[points_diff, form_diff, streak_diff, 5.0, 1.0, h2h_home, h2h_away]])

    probs = model.predict_proba(match_features)[0]

    # In knockout: split draw probability between the two teams
    team1_score = probs[2] + (probs[1] / 2)
    team2_score = probs[0] + (probs[1] / 2)

    return team1 if team1_score >= team2_score else team2


#=====================group stage simulation=====================
# =====================================================================
# 5. GROUP STAGE SIMULATION
# =====================================================================
print("\n╔" + "═"*70 + "╗")
print("║" + "FIFA WORLD CUP 2026 SIMULATION".center(70) + "║")
print("╚" + "═"*70 + "╝")

group_winners = {}
group_runners_up = {}
thirds_stats = []

for g_letter, group_teams in groups.items():
    # track pts, gf, ga for each team
    standings = {team: {'pts': 0, 'gf': 0, 'ga': 0} for team in group_teams}

    header_text = f" GROUP {g_letter}: {' | '.join(group_teams)} "
    print(f"\n┌" + "─"*70 + "┐")
    print(f"│{header_text.center(70)}│")
    print(f"└" + "─"*70 + "┘")
    print("  [Match Results]")

    # matches
    for i in range(len(group_teams)):
        for j in range(i + 1, len(group_teams)):
            t1, t2 = group_teams[i], group_teams[j]
            t1_pts, t2_pts, tag = get_group_match_result(t1, t2)

            standings[t1]['pts'] += t1_pts
            standings[t2]['pts'] += t2_pts

            # track goals for goal difference tiebreaker
            key = frozenset({t1, t2})
            if key in played_matches:
                home, away, hg, ag = played_matches[key][0]
                if t1 == home:
                    standings[t1]['gf'] += hg; standings[t1]['ga'] += ag
                    standings[t2]['gf'] += ag; standings[t2]['ga'] += hg
                else:
                    standings[t1]['gf'] += ag; standings[t1]['ga'] += hg
                    standings[t2]['gf'] += hg; standings[t2]['ga'] += ag

            match_str = f"{t1} vs {t2}"
            print(f"  * {match_str:<35} | {tag}")

    # sort standings: first by pts, then by goal difference as tiebreaker
    sorted_standings = sorted(
        standings.items(),
        key=lambda x: (x[1]['pts'], x[1]['gf'] - x[1]['ga'], x[1]['gf']),
        reverse=True
    )

    print("\n  [Standings]")
    print("  ┌" + "─"*32 + "┬" + "─"*7 + "┬" + "─"*7 + "┐")
    print("  │ " + "Team".ljust(30) + " │ " + "Pts".ljust(5) + " │ " + "GD".ljust(5) + " │")
    print("  ├" + "─"*32 + "┼" + "─"*7 + "┼" + "─"*7 + "┤")

    for idx, (team, stats) in enumerate(sorted_standings, 1):
        team_display = f"{idx}. {team}"
        gd = stats['gf'] - stats['ga']
        gd_str = f"+{gd}" if gd > 0 else str(gd)
        print(f"  │ {team_display:<30} │ {stats['pts']:>4}  │ {gd_str:>4}  │")

    print("  └" + "─"*32 + "┴" + "─"*7 + "┴" + "─"*7 + "┘")

    group_winners[g_letter] = sorted_standings[0][0]
    group_runners_up[g_letter] = sorted_standings[1][0]
    thirds_stats.append((sorted_standings[2][0], sorted_standings[2][1]['pts'], g_letter))


#=====================third-place teams=====================
best_thirds_sorted = sorted(thirds_stats, key=lambda x: x[1], reverse=True)

print("\n╔" + "═"*70 + "╗")
print("║" + "3RD PLACE TEAMS GLOBAL LEADERBOARD".center(70) + "║")
print("╚" + "═"*70 + "╝")

print("  ┌" + "─"*32 + "┬" + "─"*7 + "┬" + "─"*7 + "┬" + "─"*16 + "┐")
print("  │ " + "Team".ljust(30) + " │ " + "Grp".center(5) + " │ " + "Pts".center(5) + " │ " + "Status".center(14) + " │")
print("  ├" + "─"*32 + "┼" + "─"*7 + "┼" + "─"*7 + "┼" + "─"*16 + "┤")

for idx, (team, pts, group) in enumerate(best_thirds_sorted, 1):
    status = "QUALIFIED" if idx <= 8 else "ELIMINATED"
    team_display = f"{idx}. {team}"

    print(f"  │ {team_display:<30} │ {group:^5} │ {pts:^5} │ {status:^14} │")

    if idx == 8:
        print("  ├" + "─"*32 + "┼" + "─"*7 + "┼" + "─"*7 + "┼" + "─"*16 + "┤")

print("  └" + "─"*32 + "┴" + "─"*7 + "┴" + "─"*7 + "┴" + "─"*16 + "┘")

top_8_thirds = [item[0] for item in best_thirds_sorted[:8]]


#=====================knockout=====================
def run_knockout_stage(teams_list, stage_name):
    print("\n  ┌" + "─"*66 + "┐")
    print("  │" + stage_name.center(66) + "│")
    print("  └" + "─"*66 + "┘")
    winners = []
    losers = []
    for i in range(0, len(teams_list), 2):
        t1, t2 = teams_list[i], teams_list[i+1]
        winner = predict_knockout_match(t1, t2)
        loser = t2 if winner == t1 else t1
        match_str = f"{t1} vs {t2}"
        print(f"    {match_str:<45} | Winner: {winner}")
        winners.append(winner)
        losers.append(loser)
    return winners, losers


print("\n╔" + "═"*70 + "╗")
print("║" + "KNOCKOUT STAGES".center(70) + "║")
print("╚" + "═"*70 + "╝")

# Official FIFA Round of 32 Bracket (Matches 73-88)
round_of_32_matches = [
    (group_runners_up['A'], group_runners_up['B']),    # Match 73
    (group_winners['E'], top_8_thirds[0]),             # Match 74
    (group_winners['F'], group_runners_up['C']),       # Match 75
    (group_winners['C'], group_runners_up['F']),       # Match 76
    (group_winners['I'], top_8_thirds[1]),             # Match 77
    (group_runners_up['E'], group_runners_up['I']),    # Match 78
    (group_winners['A'], top_8_thirds[2]),             # Match 79
    (group_winners['L'], top_8_thirds[3]),             # Match 80
    (group_winners['D'], top_8_thirds[4]),             # Match 81
    (group_winners['G'], top_8_thirds[5]),             # Match 82
    (group_runners_up['K'], group_runners_up['L']),    # Match 83
    (group_winners['H'], group_runners_up['J']),       # Match 84
    (group_winners['B'], top_8_thirds[6]),             # Match 85
    (group_winners['J'], group_runners_up['H']),       # Match 86
    (group_winners['K'], top_8_thirds[7]),             # Match 87
    (group_runners_up['D'], group_runners_up['G']),    # Match 88
]

# Print Round of 32
print("\n  ┌" + "─"*66 + "┐")
print("  │" + "ROUND OF 32".center(66) + "│")
print("  └" + "─"*66 + "┘")

round_of_16_teams = []
for idx, (t1, t2) in enumerate(round_of_32_matches, 73):
    winner = predict_knockout_match(t1, t2)
    match_str = f"Match {idx}: {t1} vs {t2}"
    print(f"    {match_str:<45} | Winner: {winner}")
    round_of_16_teams.append(winner)

# Progress through the next stages
round_of_16, _ = run_knockout_stage(round_of_16_teams, "ROUND OF 16")
quarters, _ = run_knockout_stage(round_of_16, "QUARTER-FINALS")
semis, sf_losers = run_knockout_stage(quarters, "SEMI-FINALS")

# ---------------------------------------------------------------------
# 3rd Place Match
# (losers are tracked directly from run_knockout_stage, no guesswork)
# ---------------------------------------------------------------------
third_place = None

if len(sf_losers) == 2:
    print("\n  ┌" + "─"*66 + "┐")
    print("  │" + "3RD PLACE MATCH".center(66) + "│")
    print("  └" + "─"*66 + "┘")
    third_place = predict_knockout_match(sf_losers[0], sf_losers[1])
    match_str = f"{sf_losers[0]} vs {sf_losers[1]}"
    print(f"    {match_str:<45} | Winner: {third_place}")

# ---------------------------------------------------------------------
# Grand Final
# ---------------------------------------------------------------------
print("\n╔" + "═"*70 + "╗")
print("║" + "GRAND FINAL".center(70) + "║")
print("╚" + "═"*70 + "╝")

champion = predict_knockout_match(semis[0], semis[1])
runner_up = semis[1] if champion == semis[0] else semis[0]

print(f"\n  Final Match: {semis[0]} vs {semis[1]}\n")

# Final Results Board
print("  ┌" + "─"*66 + "┐")
print(f"  │ WINNER (CHAMPION): {champion:<45} │")
print(f"  │ RUNNER-UP:         {runner_up:<45} │")
if third_place:
    print(f"  │ THIRD PLACE:       {third_place:<45} │")
print("  └" + "─"*66 + "┘\n")