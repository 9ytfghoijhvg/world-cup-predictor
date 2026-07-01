import numpy as np 
import pandas as pd 
from penalty_feature import get_penalty_stats
from group_stage_feature import add_group_stage_features
from knockout_history_feature import get_knockout_history
from rolling_goals_feature import add_rolling_goals_features

# Load World Cup matches (for knockout match preparation)
df_wc = pd.read_csv("../data/matches_1930_2022.csv")
group_stage = pd.read_csv("../data/group_stage_2026.csv")

# Load ALL international matches (for rolling features)
df_all = pd.read_csv("../data/international_results.csv")

# Filter out future matches with NA scores
df_all = df_all[df_all['home_score'].notna()].copy()
df_all['date'] = pd.to_datetime(df_all['date'])

# Rename columns to match World Cup dataset format
df_all = df_all.rename(columns={'date': 'Date'})

print(f"Total international matches loaded: {len(df_all)}")
print(f"Date range: {df_all['Date'].min()} to {df_all['Date'].max()}")

knockout_rounds = ['Final', 'Third-place match', 'Semi-finals', 
                   'Quarter-finals', 'Round of 16', 'Second round',
                   'Final stage']

ko = df_wc[df_wc['Round'].isin(knockout_rounds)].copy()
print(f"\nTotal knockout matches: {len(ko)}")
print(ko['Year'].min(), "to", ko['Year'].max())

# Create winner column: 1 = home advanced, 0 = home didn't advance
condition = (ko['home_score'] > ko['away_score']) | ((ko['home_score'] == ko['away_score']) & (ko['home_penalty'] > ko['away_penalty'])) 
ko['home_advanced'] = np.where(condition, 1, 0)

# Flip dataset (create ko_expanded with both home and away perspectives)
ko_flipped = ko.copy()

# Now flip the team-perspective columns
ko_flipped[['home_team', 'away_team']] = ko[['away_team', 'home_team']].values
ko_flipped[['home_score', 'away_score']] = ko[['away_score', 'home_score']].values
ko_flipped[['home_xg', 'away_xg']] = ko[['away_xg', 'home_xg']].values
ko_flipped[['home_penalty', 'away_penalty']] = ko[['away_penalty', 'home_penalty']].values
ko_flipped['home_advanced'] = 1 - ko_flipped['home_advanced']

# Combine
ko_expanded = pd.concat([ko, ko_flipped], ignore_index=True)

# Convert Date column to datetime for proper merging
ko_expanded['Date'] = pd.to_datetime(ko_expanded['Date'])

# Load Elo ratings (use calculated 2022 ratings for all teams in training data)
elo = pd.read_csv("../data/elo_ratings_all_teams_2022.csv")
print(f"Loaded Elo ratings for {len(elo)} teams (as of end of 2022)")

# Merge for home team Elo (match on team name only)
ko_expanded = ko_expanded.merge(elo[['team', 'rating']], 
                                 left_on='home_team', 
                                 right_on='team', 
                                 how='left')
ko_expanded.rename(columns={'rating': 'home_elo'}, inplace=True)
ko_expanded = ko_expanded.drop(columns=['team'])

# Merge for away team Elo
ko_expanded = ko_expanded.merge(elo[['team', 'rating']], 
                                 left_on='away_team', 
                                 right_on='team', 
                                 how='left')
ko_expanded.rename(columns={'rating': 'away_elo'}, inplace=True)
ko_expanded = ko_expanded.drop(columns=['team'])

# Calculate Elo difference
ko_expanded['elo_diff'] = ko_expanded['home_elo'] - ko_expanded['away_elo']

# === ROLLING xG FROM WORLD CUP MATCHES ===
# Impute missing xG with actual goals (highly correlated)
print("\n=== Imputing missing xG data ===")
df_wc_imputed = df_wc.copy()
df_wc_imputed['home_xg'] = df_wc_imputed['home_xg'].fillna(df_wc_imputed['home_score'])
df_wc_imputed['away_xg'] = df_wc_imputed['away_xg'].fillna(df_wc_imputed['away_score'])

xg_imputed_count = df_wc['home_xg'].isna().sum()
print(f"Imputed {xg_imputed_count} missing xG values using actual goals scored")

home_view = df_wc_imputed[['Date', 'home_team', 'home_xg']].copy()
home_view.columns = ['Date', 'team', 'xg']

away_view = df_wc_imputed[['Date', 'away_team', 'away_xg']].copy()
away_view.columns = ['Date', 'team', 'xg']

team_matches = pd.concat([home_view, away_view], ignore_index=True)
team_matches = team_matches.sort_values(['team', 'Date']).reset_index(drop=True)

team_matches['rolling_xg_10'] = team_matches.groupby('team')['xg'].transform(
    lambda x: x.rolling(window=10, min_periods=1).mean()
)
rolling_averages = team_matches[['Date', 'team', 'rolling_xg_10']].copy()
rolling_averages['Date'] = pd.to_datetime(rolling_averages['Date'])

# Merge xG for home
ko_expanded = ko_expanded.merge(rolling_averages[['Date', 'team', 'rolling_xg_10']], 
                                 left_on=['Date', 'home_team'], 
                                 right_on=['Date', 'team'], 
                                 how='left')
ko_expanded.rename(columns={'rolling_xg_10': 'home_rolling_xg_10'}, inplace=True)

# Merge xG for away
ko_expanded = ko_expanded.merge(rolling_averages[['Date', 'team', 'rolling_xg_10']], 
                                 left_on=['Date', 'away_team'], 
                                 right_on=['Date', 'team'], 
                                 how='left')
ko_expanded.rename(columns={'rolling_xg_10': 'away_rolling_xg_10'}, inplace=True)

# Host advantage feature
ko_expanded['host_advantage'] = np.where(
    ko_expanded['home_team'] == ko_expanded['Host'], 2,
    np.where(ko_expanded['away_team'] == ko_expanded['Host'], -2, 0)
)

# Add and merge penalty stats
penalty_stats = get_penalty_stats(ko_expanded)
ko_expanded = ko_expanded.merge(penalty_stats[['team', 'win_rate']], 
                                 left_on=['home_team'], 
                                 right_on=['team'], 
                                 how='left')
ko_expanded.rename(columns={'win_rate': 'home_penalty_win_rate'}, inplace=True)
ko_expanded = ko_expanded.drop(columns=['team'])

ko_expanded = ko_expanded.merge(penalty_stats[['team', 'win_rate']], 
                                 left_on=['away_team'], 
                                 right_on=['team'], 
                                 how='left')
ko_expanded.rename(columns={'win_rate': 'away_penalty_win_rate'}, inplace=True)
ko_expanded = ko_expanded.drop(columns=['team'])

# Add knockout history features
knockout_history = get_knockout_history(ko)
ko_expanded = ko_expanded.merge(knockout_history, 
                                 left_on=['home_team'], 
                                 right_on=['team'], 
                                 how='left')
ko_expanded.rename(columns={'knockout_history_score': 'home_knockout_history'}, inplace=True)
ko_expanded = ko_expanded.drop(columns=['team'])

ko_expanded = ko_expanded.merge(knockout_history, 
                                 left_on=['away_team'], 
                                 right_on=['team'], 
                                 how='left')
ko_expanded.rename(columns={'knockout_history_score': 'away_knockout_history'}, inplace=True)
ko_expanded = ko_expanded.drop(columns=['team'])

# Add group stage features (2026 tournament specific)
ko_expanded = add_group_stage_features(ko_expanded, group_stage)

# === ADD ROLLING GOALS FROM ALL INTERNATIONAL MATCHES ===
print("\n=== Adding rolling goals features from all international matches ===")
ko_expanded = add_rolling_goals_features(ko_expanded, df_all, window=10)

# === IMPUTE REMAINING MISSING VALUES ===
print("\n=== Imputing remaining missing values ===")

# Define features for validation
feature_cols = ['elo_diff', 'home_rolling_xg_10', 'away_rolling_xg_10', 'host_advantage', 
                'home_penalty_win_rate', 'away_penalty_win_rate', 'home_group_points', 
                'home_group_gd', 'away_group_points', 'away_group_gd',
                'home_rolling_gf_10', 'home_rolling_ga_10', 'away_rolling_gf_10', 'away_rolling_ga_10']
target_col = 'home_advanced'

# Impute missing Elo with global average
if ko_expanded['elo_diff'].isna().any():
    global_avg_elo = elo['rating'].mean()
    elo_missing = ko_expanded['elo_diff'].isna().sum()
    ko_expanded['elo_diff'] = ko_expanded['elo_diff'].fillna(0)  # Missing both = assume equal
    print(f"Imputed {elo_missing} missing Elo differences (set to 0 = equal teams)")

# Impute missing penalty win rates with 0.5 (50% = neutral)
penalty_cols = ['home_penalty_win_rate', 'away_penalty_win_rate']
for col in penalty_cols:
    missing = ko_expanded[col].isna().sum()
    if missing > 0:
        ko_expanded[col] = ko_expanded[col].fillna(0.5)
        print(f"Imputed {missing} missing {col} (set to 0.5 = neutral)")

# Impute missing group stage stats with tournament averages
group_cols = ['home_group_points', 'home_group_gd', 'away_group_points', 'away_group_gd']
for col in group_cols:
    missing = ko_expanded[col].isna().sum()
    if missing > 0:
        # Use 0 as default (team didn't have group data)
        ko_expanded[col] = ko_expanded[col].fillna(0)
        print(f"Imputed {missing} missing {col} (set to 0)")

# Impute missing rolling goals with team's average from available data
for col in ['home_rolling_gf_10', 'home_rolling_ga_10', 'away_rolling_gf_10', 'away_rolling_ga_10']:
    missing = ko_expanded[col].isna().sum()
    if missing > 0:
        # Use 1.0 as default (neutral value)
        ko_expanded[col] = ko_expanded[col].fillna(1.0)
        print(f"Imputed {missing} missing {col} (set to 1.0)")

# Final check
total_rows = len(ko_expanded)
clean_rows = ko_expanded[feature_cols + [target_col]].dropna().shape[0]
print(f"\n✅ Clean rows after imputation: {clean_rows}/{total_rows} ({clean_rows/total_rows*100:.1f}%)")

# Save prepared data for modeling
ko_expanded.to_csv('../data/knockout_matches_prepared.csv', index=False)
print(f"\n✅ Saved prepared data: {len(ko_expanded)} rows with features ready for modeling")

# Show sample of new features
print("\n=== Sample of rolling goals features ===")
sample = ko_expanded[['Date', 'home_team', 'away_team', 'home_rolling_gf_10', 
                      'home_rolling_ga_10', 'away_rolling_gf_10', 'away_rolling_ga_10']].tail(10)
print(sample)
