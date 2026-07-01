import numpy as np 
import pandas as pd 
from penalty_feature import get_penalty_stats
from group_stage_feature import add_group_stage_features
from knockout_history_feature import get_knockout_history

df = pd.read_csv("data/matches_1930_2022.csv")
group_stage = pd.read_csv("data/group_stage_2026.csv")

knockout_rounds = ['Final', 'Third-place match', 'Semi-finals', 
                   'Quarter-finals', 'Round of 16', 'Second round',
                   'Final stage']

ko = df[df['Round'].isin(knockout_rounds)].copy()
print(f"Total knockout matches: {len(ko)}")
print(ko['Year'].min(), "to", ko['Year'].max())

# Create winner column: 1 = home advanced, 0 = home didn't advance
condition = (ko['home_score'] > ko['away_score']) | ((ko['home_score'] == ko['away_score']) & (ko['home_penalty'] > ko['away_penalty'])) 
ko['home_advanced'] = np.where(condition, 1, 0)

# TODO: FEATURES TO ADD (Phase 1 - Simple):
# 1. Elo rating difference (merge from elo_ratings_wc2026.csv)
# 2. Rolling xG average (last 5-10 matches, for and against)
# 3. Host advantage (use Host column, scale: 2=home country, 1=co-host/nearby, 0=neutral, -1/-2=opponent perspective)

# TODO: FEATURES FOR PHASE 2 (Advanced - add later):
# - Group stage goal difference
# - Squad market value difference (Transfermarkt)
# - Tournament form (group stage points, goals)
# - Penalty shootout history per team

# Flip dataset (create ko_expanded with both home and away perspectives)

# Start with ALL columns from ko
ko_flipped = ko.copy()

# Now flip the team-perspective columns
ko_flipped[['home_team', 'away_team']] = ko[['away_team', 'home_team']].values
ko_flipped[['home_score', 'away_score']] = ko[['away_score', 'home_score']].values
ko_flipped[['home_xg', 'away_xg']] = ko[['away_xg', 'home_xg']].values
ko_flipped[['home_penalty', 'away_penalty']] = ko[['away_penalty', 'home_penalty']].values
ko_flipped['home_advanced'] = 1 - ko_flipped['home_advanced']

# Combine
ko_expanded = pd.concat([ko, ko_flipped], ignore_index=True)


# Load Elo ratings
elo = pd.read_csv("data/elo_ratings_wc2026.csv")

# Merge for home team Elo
ko_expanded = ko_expanded.merge(elo[['year', 'country', 'rating']], 
                                 left_on=['Year', 'home_team'], 
                                 right_on=['year', 'country'], 
                                 how='left')
ko_expanded.rename(columns={'rating': 'home_elo'}, inplace=True)

# Merge for away team Elo
ko_expanded = ko_expanded.merge(elo[['year', 'country', 'rating']], 
                                 left_on=['Year', 'away_team'], 
                                 right_on=['year', 'country'], 
                                 how='left')
ko_expanded.rename(columns={'rating': 'away_elo'}, inplace=True)

# Calculate Elo difference
ko_expanded['elo_diff'] = ko_expanded['home_elo'] - ko_expanded['away_elo']

# Dataset 1: Home team perspective
home_view = ko[['Date', 'home_team', 'home_xg']].copy()
home_view.columns = ['Date', 'team', 'xg']

# Dataset 2: Away team perspective (flip it)
away_view = ko[['Date', 'away_team', 'away_xg']].copy()
away_view.columns = ['Date', 'team', 'xg']

# Combine them
team_matches = pd.concat([home_view, away_view], ignore_index=True)
team_matches = team_matches.sort_values(['team', 'Date']).reset_index(drop=True)

# Calculate rolling average per team, keeping all columns
team_matches['rolling_xg_10'] = team_matches.groupby('team')['xg'].transform(
    lambda x: x.rolling(window=10, min_periods=1).mean()
)
rolling_averages = team_matches[['Date', 'team', 'rolling_xg_10']].copy()

# Merge average for home
ko_expanded = ko_expanded.merge(rolling_averages[['Date', 'team', 'rolling_xg_10']], 
                                 left_on=['Date', 'home_team'], 
                                 right_on=['Date', 'team'], 
                                 how='left')
ko_expanded.rename(columns={'rolling_xg_10': 'home_rolling_xg_10'}, inplace=True)

# Merge average for away
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

# Add group stage features
ko_expanded = add_group_stage_features(ko_expanded, group_stage)

# Save prepared data for modeling
ko_expanded.to_csv('data/knockout_matches_prepared.csv', index=False)
print(f"\nSaved prepared data: {len(ko_expanded)} rows with features ready for modeling")

