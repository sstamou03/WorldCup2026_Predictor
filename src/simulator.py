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
    ('Mexico', 'South Africa', 2, 0),
    ('South Korea', 'Czech Republic', 2, 1),
    ('Czech Republic', 'South Africa', 1, 1),
    ('Mexico', 'South Korea', 1, 0),
    ('Mexico', 'Czech Republic', 3, 0),
    ('South Africa', 'South Korea', 1, 0),

    # GROUP B
    ('Canada', 'Bosnia and Herzegovina', 1, 1),
    ('Qatar', 'Switzerland', 1, 1),
    ('Canada', 'Qatar', 6, 0),
    ('Switzerland', 'Bosnia and Herzegovina', 4, 1),
    ('Switzerland', 'Canada', 3, 1),
    ('Bosnia and Herzegovina', 'Qatar', 3, 1),

    # GROUP C
    ('Brazil', 'Morocco', 1, 1),
    ('Scotland', 'Haiti', 1, 0),
    ('Brazil', 'Haiti', 3, 0),
    ('Morocco', 'Scotland', 1, 0),
    ('Scotland', 'Brazil', 0, 3),
    ('Morocco', 'Haiti', 4, 2),

    # GROUP D
    ('United States', 'Paraguay', 4, 1),
    ('Australia', 'Turkey', 2, 0),
    ('United States', 'Australia', 2, 0),
    ('Paraguay', 'Turkey', 1, 0),
    ('Turkey', 'United States', 3, 2),
    ('Paraguay', 'Australia', 0, 0),

    # GROUP E
    ('Germany', 'Curaçao', 7, 1),
    ('Ivory Coast', 'Ecuador', 1, 0),
    ('Germany', 'Ivory Coast', 2, 1),
    ('Ecuador', 'Curaçao', 0, 0),
    ('Curaçao', 'Ivory Coast', 0, 2),
    ('Ecuador', 'Germany', 2, 1),

    # GROUP F
    ('Netherlands', 'Japan', 2, 2),
    ('Sweden', 'Tunisia', 5, 1),
    ('Netherlands', 'Sweden', 5, 1),
    ('Japan', 'Tunisia', 4, 0),
    ('Japan', 'Sweden', 1, 1),
    ('Netherlands', 'Tunisia', 3, 1),

    # GROUP G
    ('Belgium', 'Egypt', 1, 1),
    ('Iran', 'New Zealand', 2, 2),
    ('Egypt', 'New Zealand', 3, 1),
    ('Belgium', 'Iran', 0, 0),
    ('Belgium', 'New Zealand', 3, 0),
    ('Egypt', 'Iran', 1, 1),

    # GROUP H
    ('Spain', 'Cape Verde', 0, 0),
    ('Saudi Arabia', 'Uruguay', 1, 1),
    ('Spain', 'Saudi Arabia', 4, 0),
    ('Uruguay', 'Cape Verde', 2, 2),
    ('Spain', 'Uruguay', 2, 1),
    ('Cape Verde', 'Saudi Arabia', 1, 0),

    # GROUP I
    ('France', 'Senegal', 3, 1),
    ('Iraq', 'Norway', 1, 4),
    ('France', 'Iraq', 3, 0),
    ('Norway', 'Senegal', 3, 2),
    ('France', 'Norway', 2, 2),
    ('Senegal', 'Iraq', 2, 0),

    # GROUP J
    ('Argentina', 'Algeria', 3, 0),
    ('Austria', 'Jordan', 3, 1),
    ('Argentina', 'Austria', 2, 0),
    ('Jordan', 'Algeria', 1, 2),
    ('Argentina', 'Jordan', 4, 0),
    ('Algeria', 'Austria', 1, 2),

    # GROUP K
    ('Portugal', 'DR Congo', 1, 1),
    ('Uzbekistan', 'Colombia', 1, 3),
    ('Portugal', 'Uzbekistan', 5, 0),
    ('Colombia', 'DR Congo', 1, 0),
    ('Portugal', 'Colombia', 2, 1),
    ('DR Congo', 'Uzbekistan', 2, 0),

    # GROUP L
    ('England', 'Croatia', 4, 2),
    ('Ghana', 'Panama', 1, 0),
    ('England', 'Ghana', 0, 0),
    ('Panama', 'Croatia', 0, 1),
    ('England', 'Panama', 3, 0),
    ('Croatia', 'Ghana', 1, 1),
]

played_matches = {}
for home, away, hg, ag in real_results:
    key = frozenset({home, away})
    if key not in played_matches:
        played_matches[key] = []
    played_matches[key].append((home, away, hg, ag))


def get_group_match_result(team1, team2):
    key = frozenset({team1, team2})
    if key in played_matches:
        home, away, hg, ag = played_matches[key][0]
        tag = f"REAL: {home} {hg}-{ag} {away}"
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
        return predict_group_match(team1, team2)


def get_h2h_rates(team1, team2):
    key = frozenset({team1, team2})
    if key in h2h_lookup:
        h2h = h2h_lookup[key]
        if h2h['home'] == team1:
            return h2h['home_h2h_win_rate'], h2h['away_h2h_win_rate']
        else:
            return h2h['away_h2h_win_rate'], h2h['home_h2h_win_rate']
    return 0.5, 0.5


def predict_group_match(team1, team2):
    default_stats = {'home_avg.points': 1000, 'home_form_5': 1.5, 'home_streak': 0}
    stats1 = teams_dict.get(team1, default_stats)
    stats2 = teams_dict.get(team2, default_stats)
    if team1 not in teams_dict:
        print(f"  [WARNING] '{team1}' not found in dataset, using default stats.")
    if team2 not in teams_dict:
        print(f"  [WARNING] '{team2}' not found in dataset, using default stats.")
    points_diff = stats1.get('home_avg.points', 1000) - stats2.get('home_avg.points', 1000)
    form_diff   = stats1.get('home_form_5', 1.5)      - stats2.get('home_form_5', 1.5)
    streak_diff = stats1.get('home_streak', 0)         - stats2.get('home_streak', 0)
    h2h_home, h2h_away = get_h2h_rates(team1, team2)
    match_features = np.array([[points_diff, form_diff, streak_diff, 5.0, 1.0, h2h_home, h2h_away]])
    probs  = model.predict_proba(match_features)[0]
    result = np.argmax(probs)
    win_pct  = f"{probs[2]*100:.0f}%"
    draw_pct = f"{probs[1]*100:.0f}%"
    loss_pct = f"{probs[0]*100:.0f}%"
    if result == 2:
        return 3, 0, f"PREDICTED: {team1} wins (W:{win_pct} D:{draw_pct} L:{loss_pct})"
    elif result == 1:
        return 1, 1, f"PREDICTED: Draw (W:{win_pct} D:{draw_pct} L:{loss_pct})"
    else:
        return 0, 3, f"PREDICTED: {team2} wins (W:{win_pct} D:{draw_pct} L:{loss_pct})"


def predict_knockout_match(team1, team2):
    default_stats = {'home_avg.points': 1000, 'home_form_5': 1.5, 'home_streak': 0}
    stats1 = teams_dict.get(team1, default_stats)
    stats2 = teams_dict.get(team2, default_stats)
    points_diff = stats1.get('home_avg.points', 1000) - stats2.get('home_avg.points', 1000)
    form_diff   = stats1.get('home_form_5', 1.5)      - stats2.get('home_form_5', 1.5)
    streak_diff = stats1.get('home_streak', 0)         - stats2.get('home_streak', 0)
    h2h_home, h2h_away = get_h2h_rates(team1, team2)
    match_features = np.array([[points_diff, form_diff, streak_diff, 5.0, 1.0, h2h_home, h2h_away]])
    probs = model.predict_proba(match_features)[0]
    team1_score = probs[2] + (probs[1] / 2)
    team2_score = probs[0] + (probs[1] / 2)
    return team1 if team1_score >= team2_score else team2


# =====================================================================
# GROUP STAGE SIMULATION
# =====================================================================
print("\n╔" + "═"*70 + "╗")
print("║" + "FIFA WORLD CUP 2026 SIMULATION".center(70) + "║")
print("╚" + "═"*70 + "╝")

group_winners   = {}
group_runners_up = {}
thirds_stats    = []

for g_letter, group_teams in groups.items():
    standings = {team: {'pts': 0, 'gf': 0, 'ga': 0} for team in group_teams}

    header_text = f" GROUP {g_letter}: {' | '.join(group_teams)} "
    print(f"\n┌" + "─"*70 + "┐")
    print(f"│{header_text.center(70)}│")
    print(f"└" + "─"*70 + "┘")
    print("  [Match Results]")

    for i in range(len(group_teams)):
        for j in range(i + 1, len(group_teams)):
            t1, t2 = group_teams[i], group_teams[j]
            t1_pts, t2_pts, tag = get_group_match_result(t1, t2)
            standings[t1]['pts'] += t1_pts
            standings[t2]['pts'] += t2_pts
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
        gd     = stats['gf'] - stats['ga']
        gd_str = f"+{gd}" if gd > 0 else str(gd)
        print(f"  │ {team_display:<30} │ {stats['pts']:>4}  │ {gd_str:>4}  │")
    print("  └" + "─"*32 + "┴" + "─"*7 + "┴" + "─"*7 + "┘")

    group_winners[g_letter]    = sorted_standings[0][0]
    group_runners_up[g_letter] = sorted_standings[1][0]
    thirds_stats.append((sorted_standings[2][0], sorted_standings[2][1]['pts'], g_letter))


# =====================================================================
# BEST 8 THIRD-PLACE TEAMS
# =====================================================================
best_thirds_sorted = sorted(thirds_stats, key=lambda x: x[1], reverse=True)

print("\n╔" + "═"*70 + "╗")
print("║" + "3RD PLACE TEAMS GLOBAL LEADERBOARD".center(70) + "║")
print("╚" + "═"*70 + "╝")
print("  ┌" + "─"*32 + "┬" + "─"*7 + "┬" + "─"*7 + "┬" + "─"*16 + "┐")
print("  │ " + "Team".ljust(30) + " │ " + "Grp".center(5) + " │ " + "Pts".center(5) + " │ " + "Status".center(14) + " │")
print("  ├" + "─"*32 + "┼" + "─"*7 + "┼" + "─"*7 + "┼" + "─"*16 + "┤")
for idx, (team, pts, group) in enumerate(best_thirds_sorted, 1):
    status       = "QUALIFIED" if idx <= 8 else "ELIMINATED"
    team_display = f"{idx}. {team}"
    print(f"  │ {team_display:<30} │ {group:^5} │ {pts:^5} │ {status:^14} │")
    if idx == 8:
        print("  ├" + "─"*32 + "┼" + "─"*7 + "┼" + "─"*7 + "┼" + "─"*16 + "┤")
print("  └" + "─"*32 + "┴" + "─"*7 + "┴" + "─"*7 + "┴" + "─"*16 + "┘")

# ── Lookup: group letter -> third-place team ──────────────────────────
# Only the 8 qualified thirds are usable in the bracket
qualified_thirds = best_thirds_sorted[:8]
slots = ['D', 'F', 'E', 'K', 'B', 'I', 'J', 'L']
qualified_map = {group: team for team, pts, group in qualified_thirds}

thirds_by_group = {}
for g in slots:
    if g in qualified_map:
        thirds_by_group[g] = qualified_map[g]

# Map missing slots to any qualified third-place groups that aren't in the default slots
missing_slots = [g for g in slots if g not in qualified_map]
extra_groups = [g for g in qualified_map if g not in slots]
for i, g in enumerate(missing_slots):
    extra_g = extra_groups[i]
    thirds_by_group[g] = qualified_map[extra_g]


# =====================================================================
# KNOCKOUT STAGE HELPERS
# =====================================================================
def run_knockout_stage(teams_list, stage_name):
    print("\n  ┌" + "─"*66 + "┐")
    print("  │" + stage_name.center(66) + "│")
    print("  └" + "─"*66 + "┘")
    winners = []
    losers  = []
    for i in range(0, len(teams_list), 2):
        t1, t2   = teams_list[i], teams_list[i+1]
        winner   = predict_knockout_match(t1, t2)
        loser    = t2 if winner == t1 else t1
        match_str = f"{t1} vs {t2}"
        print(f"    {match_str:<45} | Winner: {winner}")
        winners.append(winner)
        losers.append(loser)
    return winners, losers


# =====================================================================
# ROUND OF 32  –  Official FIFA 2026 bracket
#
# Fixed slots (group winner / runner-up positions are predetermined).
# Third-place slots are assigned by the GROUP LETTER the 3rd team came
# from, NOT by their overall ranking among the 8 qualifiers.
#
# Match 73 : Runner-up A  vs Runner-up B
# Match 74 : Winner E     vs 3rd-place from D
# Match 75 : Winner F     vs Runner-up C
# Match 76 : Winner C     vs Runner-up F
# Match 77 : Winner I     vs 3rd-place from F
# Match 78 : Runner-up E  vs Runner-up I
# Match 79 : Winner A     vs 3rd-place from E
# Match 80 : Winner L     vs 3rd-place from K
# Match 81 : Winner D     vs 3rd-place from B
# Match 82 : Winner G     vs 3rd-place from I
# Match 83 : Winner K     vs Runner-up L
# Match 84 : Winner H     vs Runner-up J
# Match 85 : Winner B     vs 3rd-place from J
# Match 86 : Winner J     vs Runner-up H
# Match 87 : Winner K ... wait – Colombia is K winner, Portugal runner-up
#            Correct: Runner-up K  vs Runner-up L  → Match 83
#                     Winner K     vs 3rd-place L  → Match 87
# Match 88 : Runner-up D  vs Runner-up G
# =====================================================================
print("\n╔" + "═"*70 + "╗")
print("║" + "KNOCKOUT STAGES".center(70) + "║")
print("╚" + "═"*70 + "╝")

round_of_32_matches = [
    (group_runners_up['A'],  group_runners_up['B']),   # Match 73
    (group_winners['E'],     thirds_by_group['D']),    # Match 74
    (group_winners['F'],     group_runners_up['C']),   # Match 75
    (group_winners['C'],     group_runners_up['F']),   # Match 76
    (group_winners['I'],     thirds_by_group['F']),    # Match 77
    (group_runners_up['E'],  group_runners_up['I']),   # Match 78
    (group_winners['A'],     thirds_by_group['E']),    # Match 79
    (group_winners['L'],     thirds_by_group['K']),    # Match 80
    (group_winners['D'],     thirds_by_group['B']),    # Match 81
    (group_winners['G'],     thirds_by_group['I']),    # Match 82
    (group_runners_up['K'],  group_runners_up['L']),   # Match 83
    (group_winners['H'],     group_runners_up['J']),   # Match 84
    (group_winners['B'],     thirds_by_group['J']),    # Match 85
    (group_winners['J'],     group_runners_up['H']),   # Match 86
    (group_winners['K'],     thirds_by_group['L']),    # Match 87
    (group_runners_up['D'],  group_runners_up['G']),   # Match 88
]

print("\n  ┌" + "─"*66 + "┐")
print("  │" + "ROUND OF 32".center(66) + "│")
print("  └" + "─"*66 + "┘")

round_of_16_teams = []
for idx, (t1, t2) in enumerate(round_of_32_matches, 73):
    winner    = predict_knockout_match(t1, t2)
    match_str = f"Match {idx}: {t1} vs {t2}"
    print(f"    {match_str:<55} | Winner: {winner}")
    round_of_16_teams.append(winner)

# =====================================================================
# ROUND OF 16  (pairs: W73-W75, W74-W77, W76-W78, W79-W80,
#               W81-W82, W83-W84, W85-W87, W86-W88)
# =====================================================================
ro16_ordered = [
    round_of_16_teams[0],  round_of_16_teams[2],   # W73 vs W75
    round_of_16_teams[1],  round_of_16_teams[4],   # W74 vs W77
    round_of_16_teams[3],  round_of_16_teams[5],   # W76 vs W78
    round_of_16_teams[6],  round_of_16_teams[7],   # W79 vs W80
    round_of_16_teams[8],  round_of_16_teams[9],   # W81 vs W82
    round_of_16_teams[10], round_of_16_teams[11],  # W83 vs W84
    round_of_16_teams[12], round_of_16_teams[14],  # W85 vs W87
    round_of_16_teams[13], round_of_16_teams[15],  # W86 vs W88
]

round_of_16, _  = run_knockout_stage(ro16_ordered,       "ROUND OF 16")
quarters,    _  = run_knockout_stage(round_of_16,        "QUARTER-FINALS")
semis,  sf_losers = run_knockout_stage(quarters,         "SEMI-FINALS")

# =====================================================================
# 3RD PLACE MATCH
# =====================================================================
third_place = None
if len(sf_losers) == 2:
    print("\n  ┌" + "─"*66 + "┐")
    print("  │" + "3RD PLACE MATCH".center(66) + "│")
    print("  └" + "─"*66 + "┘")
    third_place = predict_knockout_match(sf_losers[0], sf_losers[1])
    match_str   = f"{sf_losers[0]} vs {sf_losers[1]}"
    print(f"    {match_str:<45} | Winner: {third_place}")

# =====================================================================
# GRAND FINAL
# =====================================================================
print("\n╔" + "═"*70 + "╗")
print("║" + "GRAND FINAL".center(70) + "║")
print("╚" + "═"*70 + "╝")

champion   = predict_knockout_match(semis[0], semis[1])
runner_up  = semis[1] if champion == semis[0] else semis[0]

print(f"\n  Final Match: {semis[0]} vs {semis[1]}\n")
print("  ┌" + "─"*66 + "┐")
print(f"  │ WINNER (CHAMPION): {champion:<45} │")
print(f"  │ RUNNER-UP:         {runner_up:<45} │")
if third_place:
    print(f"  │ THIRD PLACE:       {third_place:<45} │")
print("  └" + "─"*66 + "┘\n")