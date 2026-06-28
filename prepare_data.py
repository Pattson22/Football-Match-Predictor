import pandas as pd
import os
import numpy as np
from collections import defaultdict
import json

data_folder = "my_football_data"

if not os.path.exists(data_folder):
    print(f"Error: folder '{data_folder}' not found.")
    exit()

# --- 1. Load all CSVs ---
all_dataframes = []
for filename in os.listdir(data_folder):
    if filename.endswith(".csv"):
        try:
            df = pd.read_csv(os.path.join(data_folder, filename),
                             encoding='ISO-8859-1', on_bad_lines='skip')
            df['_season_file'] = filename
            all_dataframes.append(df)
        except Exception as e:
            print(f"  Skipping {filename}: {e}")

if not all_dataframes:
    print("No data loaded. Stopping.")
    exit()

master_df = pd.concat(all_dataframes, ignore_index=True)
print(f"Loaded {len(all_dataframes)} files — {len(master_df)} total rows.")

# --- 2. Essential columns ---
master_df.dropna(subset=['FTHG', 'FTAG', 'HomeTeam', 'AwayTeam'], inplace=True)
master_df['FTHG'] = pd.to_numeric(master_df['FTHG'], errors='coerce')
master_df['FTAG'] = pd.to_numeric(master_df['FTAG'], errors='coerce')
master_df.dropna(subset=['FTHG', 'FTAG'], inplace=True)

if 'Date' in master_df.columns:
    master_df['Date'] = pd.to_datetime(master_df['Date'], dayfirst=True, errors='coerce')
    master_df.sort_values('Date', inplace=True)
    master_df.reset_index(drop=True, inplace=True)
    print("Sorted chronologically.")

# --- 3. Outcome ---
master_df['Outcome'] = np.select(
    [master_df['FTHG'] > master_df['FTAG'],
     master_df['FTHG'] == master_df['FTAG'],
     master_df['FTHG'] < master_df['FTAG']],
    [0, 1, 2]
)

# --- 4. Average consensus odds (7 bookmakers) ---
H_COLS = [c for c in ['B365H','BWH','IWH','WHH','VCH','LBH','PSH'] if c in master_df.columns]
D_COLS = [c for c in ['B365D','BWD','IWD','WHD','VCD','LBD','PSD'] if c in master_df.columns]
A_COLS = [c for c in ['B365A','BWA','IWA','WHA','VCA','LBA','PSA'] if c in master_df.columns]
for c in H_COLS + D_COLS + A_COLS:
    master_df[c] = pd.to_numeric(master_df[c], errors='coerce')
master_df['avg_H'] = master_df[H_COLS].mean(axis=1)
master_df['avg_D'] = master_df[D_COLS].mean(axis=1)
master_df['avg_A'] = master_df[A_COLS].mean(axis=1)
print(f"Consensus odds from {len(H_COLS)} bookmakers.")

# --- 4b. Implied probabilities (strip bookmaker overround) ---
raw_h = 1.0 / master_df['avg_H']
raw_d = 1.0 / master_df['avg_D']
raw_a = 1.0 / master_df['avg_A']
total_raw = raw_h + raw_d + raw_a
master_df['imp_H'] = raw_h / total_raw
master_df['imp_D'] = raw_d / total_raw
master_df['imp_A'] = raw_a / total_raw
print("Implied probabilities computed.")

# --- 5. Elo ratings ---
print("Computing Elo ratings...")
ELO_K        = 32
ELO_HOME_ADV = 100
ELO_START    = 1500.0
team_elo = defaultdict(lambda: ELO_START)
home_elo_col, away_elo_col = [], []

for _, row in master_df.iterrows():
    ht, at = row['HomeTeam'], row['AwayTeam']
    home_elo_col.append(team_elo[ht])
    away_elo_col.append(team_elo[at])
    e_h = 1.0 / (1.0 + 10.0 ** ((team_elo[at] - team_elo[ht] - ELO_HOME_ADV) / 400.0))
    e_a = 1.0 - e_h
    s_h = 1.0 if row['FTHG'] > row['FTAG'] else (0.5 if row['FTHG'] == row['FTAG'] else 0.0)
    s_a = 1.0 - s_h
    team_elo[ht] += ELO_K * (s_h - e_h)
    team_elo[at] += ELO_K * (s_a - e_a)

master_df['home_elo'] = home_elo_col
master_df['away_elo'] = away_elo_col
master_df['elo_diff'] = master_df['home_elo'] - master_df['away_elo']
print("Elo ratings computed.")

# --- 6. Rolling form + momentum + rest days ---
print("Computing rolling form, momentum and rest-day features...")

WINDOW    = 5
WINDOW3   = 3
MIN_FORM  = 3
MAX_REST  = 14

team_history      = defaultdict(list)
team_home_history = defaultdict(list)
team_away_history = defaultdict(list)
team_last_date    = {}

(home_form_pts_col, home_gspg_col, home_gcpg_col,
 away_form_pts_col, away_gspg_col, away_gcpg_col,
 home_venue_pts_col, home_venue_gs_col, home_venue_gc_col,
 away_venue_pts_col, away_venue_gs_col, away_venue_gc_col,
 home_form3_col, away_form3_col,
 home_rest_col, away_rest_col) = ([] for _ in range(16))


def get_form(history, window=WINDOW, min_games=MIN_FORM):
    if len(history) < min_games:
        return np.nan, np.nan, np.nan
    recent = history[-window:]
    return (
        float(np.mean([h['pts'] for h in recent])),
        float(np.mean([h['gs']  for h in recent])),
        float(np.mean([h['gc']  for h in recent])),
    )


for _, row in master_df.iterrows():
    ht, at = row['HomeTeam'], row['AwayTeam']
    hg, ag = row['FTHG'], row['FTAG']
    date   = row['Date']

    # All-venue form (last 5)
    hp, hgs, hgc = get_form(team_history[ht])
    ap, ags, agc = get_form(team_history[at])
    home_form_pts_col.append(hp); home_gspg_col.append(hgs); home_gcpg_col.append(hgc)
    away_form_pts_col.append(ap); away_gspg_col.append(ags); away_gcpg_col.append(agc)

    # Venue-split form
    hvp, hvgs, hvgc = get_form(team_home_history[ht], min_games=2)
    avp, avgs, avgc = get_form(team_away_history[at], min_games=2)
    home_venue_pts_col.append(hvp); home_venue_gs_col.append(hvgs); home_venue_gc_col.append(hvgc)
    away_venue_pts_col.append(avp); away_venue_gs_col.append(avgs); away_venue_gc_col.append(avgc)

    # Short-term form (last 3) for momentum signal
    hp3, _, _ = get_form(team_history[ht], window=WINDOW3, min_games=2)
    ap3, _, _ = get_form(team_history[at], window=WINDOW3, min_games=2)
    home_form3_col.append(hp3)
    away_form3_col.append(ap3)

    # Rest days (days since last match, capped)
    h_rest = np.nan
    a_rest = np.nan
    if pd.notna(date):
        if ht in team_last_date and pd.notna(team_last_date[ht]):
            h_rest = float(min((date - team_last_date[ht]).days, MAX_REST))
        if at in team_last_date and pd.notna(team_last_date[at]):
            a_rest = float(min((date - team_last_date[at]).days, MAX_REST))
    home_rest_col.append(h_rest)
    away_rest_col.append(a_rest)

    # Points earned this match
    h_pts = 3.0 if hg > ag else (1.0 if hg == ag else 0.0)
    a_pts = 3.0 - h_pts if hg != ag else 1.0

    # Update histories AFTER recording pre-match stats
    team_history[ht].append({'pts': h_pts, 'gs': hg, 'gc': ag})
    team_history[at].append({'pts': a_pts, 'gs': ag, 'gc': hg})
    team_home_history[ht].append({'pts': h_pts, 'gs': hg, 'gc': ag})
    team_away_history[at].append({'pts': a_pts, 'gs': ag, 'gc': hg})
    if pd.notna(date):
        team_last_date[ht] = date
        team_last_date[at] = date

master_df['home_form_pts']       = home_form_pts_col
master_df['home_gspg']           = home_gspg_col
master_df['home_gcpg']           = home_gcpg_col
master_df['away_form_pts']       = away_form_pts_col
master_df['away_gspg']           = away_gspg_col
master_df['away_gcpg']           = away_gcpg_col
master_df['home_venue_form_pts'] = home_venue_pts_col
master_df['home_venue_gspg']     = home_venue_gs_col
master_df['home_venue_gcpg']     = home_venue_gc_col
master_df['away_venue_form_pts'] = away_venue_pts_col
master_df['away_venue_gspg']     = away_venue_gs_col
master_df['away_venue_gcpg']     = away_venue_gc_col
master_df['home_form3_pts']      = home_form3_col
master_df['away_form3_pts']      = away_form3_col
master_df['home_rest_days']      = home_rest_col
master_df['away_rest_days']      = away_rest_col

# Momentum = short-term trend vs medium-term baseline (positive = improving)
master_df['home_momentum'] = master_df['home_form3_pts'] - master_df['home_form_pts']
master_df['away_momentum'] = master_df['away_form3_pts'] - master_df['away_form_pts']
print("Form / momentum / rest days computed.")

# --- 7. Season-to-date standings ---
print("Computing season standings features...")
season_records = defaultdict(lambda: {'pts': 0, 'gd': 0, 'played': 0})
home_s_ppg_col, home_s_gd_col = [], []
away_s_ppg_col, away_s_gd_col = [], []

for _, row in master_df.iterrows():
    ht = row['HomeTeam'];  at = row['AwayTeam']
    sf = row['_season_file']
    hg = row['FTHG'];      ag = row['FTAG']
    hk = (sf, ht);         ak = (sf, at)

    h_played = season_records[hk]['played']
    a_played = season_records[ak]['played']
    home_s_ppg_col.append(season_records[hk]['pts'] / max(1, h_played))
    home_s_gd_col.append(float(season_records[hk]['gd']))
    away_s_ppg_col.append(season_records[ak]['pts'] / max(1, a_played))
    away_s_gd_col.append(float(season_records[ak]['gd']))

    h_pts, a_pts = (3, 0) if hg > ag else ((1, 1) if hg == ag else (0, 3))
    season_records[hk]['pts']    += h_pts
    season_records[hk]['gd']     += int(hg - ag)
    season_records[hk]['played'] += 1
    season_records[ak]['pts']    += a_pts
    season_records[ak]['gd']     += int(ag - hg)
    season_records[ak]['played'] += 1

master_df['home_season_ppg'] = home_s_ppg_col
master_df['home_season_gd']  = home_s_gd_col
master_df['away_season_ppg'] = away_s_ppg_col
master_df['away_season_gd']  = away_s_gd_col
print("Season standings computed.")

# --- 7b. H2H record (last 5 meetings between the same pair) ---
print("Computing head-to-head features...")
h2h_history = defaultdict(list)   # frozenset({ht, at}) -> list of {winner, draw}
H2H_WINDOW = 5
H2H_MIN    = 2

h2h_home_wr_col   = []
h2h_draw_rate_col = []

for _, row in master_df.iterrows():
    ht, at = row['HomeTeam'], row['AwayTeam']
    hg, ag = row['FTHG'], row['FTAG']
    key  = frozenset({ht, at})
    hist = h2h_history[key]

    if len(hist) >= H2H_MIN:
        recent = hist[-H2H_WINDOW:]
        h2h_home_wr_col.append(sum(1 for r in recent if r['winner'] == ht) / len(recent))
        h2h_draw_rate_col.append(sum(1 for r in recent if r['winner'] is None) / len(recent))
    else:
        h2h_home_wr_col.append(np.nan)
        h2h_draw_rate_col.append(np.nan)

    winner = ht if hg > ag else (None if hg == ag else at)
    h2h_history[key].append({'winner': winner})

master_df['h2h_home_win_rate'] = h2h_home_wr_col
master_df['h2h_draw_rate']     = h2h_draw_rate_col
print("H2H features computed.")

# --- 7c. Differentials and matchup features ---
master_df['form_pts_diff']           = master_df['home_form_pts']   - master_df['away_form_pts']
master_df['season_ppg_diff']         = master_df['home_season_ppg'] - master_df['away_season_ppg']
master_df['home_attack_vs_away_def'] = master_df['home_gspg'] - master_df['away_gcpg']
master_df['away_attack_vs_home_def'] = master_df['away_gspg'] - master_df['home_gcpg']

# --- 8. Fill soft-feature NaNs with sensible league-average defaults ---
master_df = master_df.assign(
    home_rest_days   = master_df['home_rest_days'].fillna(7.0),
    away_rest_days   = master_df['away_rest_days'].fillna(7.0),
    home_momentum    = master_df['home_momentum'].fillna(0.0),
    away_momentum    = master_df['away_momentum'].fillna(0.0),
    h2h_home_win_rate = master_df['h2h_home_win_rate'].fillna(0.46),
    h2h_draw_rate    = master_df['h2h_draw_rate'].fillna(0.26),
)

# --- 9. Save artifacts for live prediction ---

# Elo ratings
elo_snapshot = {str(t): round(r, 1) for t, r in team_elo.items()}
with open('team_elo.json', 'w', encoding='utf-8') as f:
    json.dump(elo_snapshot, f, indent=2)
print(f"Saved Elo ratings for {len(elo_snapshot)} teams.")

# Rolling form + venue-split + short-term (for momentum)
team_form = {}
for team, history in team_history.items():
    if len(history) < MIN_FORM:
        continue
    recent = history[-WINDOW:]
    entry = {
        'form_pts': round(float(np.mean([h['pts'] for h in recent])), 3),
        'gspg':     round(float(np.mean([h['gs']  for h in recent])), 3),
        'gcpg':     round(float(np.mean([h['gc']  for h in recent])), 3),
    }
    # Short-term (last 3) for momentum
    if len(history) >= 2:
        r3 = history[-WINDOW3:]
        entry['form3_pts'] = round(float(np.mean([h['pts'] for h in r3])), 3)
    # Venue splits
    hh = team_home_history[team]
    if len(hh) >= 2:
        hr = hh[-WINDOW:]
        entry['home_form_pts'] = round(float(np.mean([h['pts'] for h in hr])), 3)
        entry['home_gspg']     = round(float(np.mean([h['gs']  for h in hr])), 3)
        entry['home_gcpg']     = round(float(np.mean([h['gc']  for h in hr])), 3)
    ah = team_away_history[team]
    if len(ah) >= 2:
        ar = ah[-WINDOW:]
        entry['away_form_pts'] = round(float(np.mean([h['pts'] for h in ar])), 3)
        entry['away_gspg']     = round(float(np.mean([h['gs']  for h in ar])), 3)
        entry['away_gcpg']     = round(float(np.mean([h['gc']  for h in ar])), 3)
    team_form[str(team)] = entry

with open('team_form.json', 'w', encoding='utf-8') as f:
    json.dump(team_form, f, indent=2)
print(f"Saved form data for {len(team_form)} teams.")

# H2H cache keyed both ways (home||away and away||home)
h2h_cache = {}
for key, hist in h2h_history.items():
    if len(hist) < H2H_MIN:
        continue
    recent = hist[-H2H_WINDOW:]
    teams  = list(key)
    t1, t2 = teams[0], teams[1]
    dr    = round(sum(1 for r in recent if r['winner'] is None) / len(recent), 3)
    hw_t1 = round(sum(1 for r in recent if r['winner'] == t1) / len(recent), 3)
    hw_t2 = round(sum(1 for r in recent if r['winner'] == t2) / len(recent), 3)
    h2h_cache[f"{t1}||{t2}"] = {'hw': hw_t1, 'dr': dr}
    h2h_cache[f"{t2}||{t1}"] = {'hw': hw_t2, 'dr': dr}

with open('h2h_cache.json', 'w', encoding='utf-8') as f:
    json.dump(h2h_cache, f, indent=2)
print(f"Saved H2H data for {len(h2h_cache)//2} matchups.")

# --- 10. Final feature set ---
FEATURE_COLS = [
    # Odds (raw + normalised)
    'avg_H', 'avg_D', 'avg_A',
    'imp_H', 'imp_D', 'imp_A',
    # Elo
    'home_elo', 'away_elo', 'elo_diff',
    # All-venue rolling form (last 5)
    'home_form_pts', 'home_gspg', 'home_gcpg',
    'away_form_pts', 'away_gspg', 'away_gcpg',
    # Venue-split form
    'home_venue_form_pts', 'home_venue_gspg', 'home_venue_gcpg',
    'away_venue_form_pts', 'away_venue_gspg', 'away_venue_gcpg',
    # Season standings
    'home_season_ppg', 'home_season_gd',
    'away_season_ppg', 'away_season_gd',
    # Differentials and matchup
    'form_pts_diff', 'season_ppg_diff',
    'home_attack_vs_away_def', 'away_attack_vs_home_def',
    # Momentum (3-game trend vs 5-game baseline)
    'home_momentum', 'away_momentum',
    # Rest days
    'home_rest_days', 'away_rest_days',
    # Head-to-head
    'h2h_home_win_rate', 'h2h_draw_rate',
]
ALL_COLS = ['Outcome'] + FEATURE_COLS

master_df.dropna(subset=ALL_COLS, inplace=True)
print(f"After dropping missing: {len(master_df)} rows remain.")

if len(master_df) == 0:
    print("Error: No rows remaining.")
    exit()

master_df[ALL_COLS].to_csv("processed_football_data.csv", index=False)
print(f"Saved {len(master_df)} rows with {len(FEATURE_COLS)} features to processed_football_data.csv")
print(f"\nFeatures ({len(FEATURE_COLS)}): {FEATURE_COLS}")
