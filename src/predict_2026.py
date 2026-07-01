import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from features.penalty_feature import get_penalty_stats
from features.group_stage_feature import add_group_stage_features
from features.knockout_history_feature import get_knockout_history

# Load trained model data
df_train = pd.read_csv('data/knockout_matches_prepared.csv')

feature_cols = ['elo_diff', 
'home_rolling_xg_10', 
'away_rolling_xg_10', 
'host_advantage', 
'home_penalty_win_rate', 
'away_penalty_win_rate', 
'home_group_points', 
'home_group_gd', 
'away_group_points', 
'away_group_gd']

target_col = 'home_advanced'

# Train model on full historical data
df_clean = df_train[feature_cols + [target_col]].dropna()
X_train = df_clean[feature_cols]
y_train = df_clean[target_col]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Load 2026 data sources
elo_2026 = pd.read_csv('data/elo_ratings_wc2026.csv')
group_stage_2026 = pd.read_csv('data/group_stage_2026.csv')

def get_2026_team_features(team_name):
    features = {}
    
    # Get Elo rating (use most recent)
    elo_row = elo_2026[elo_2026['country'] == team_name].sort_values('snapshot_date', ascending=False).iloc[0]
    features['elo'] = elo_row['rating']
    
    # Get group stage stats
    group_row = group_stage_2026[group_stage_2026['team'] == team_name]
    if len(group_row) > 0:
        features['group_points'] = group_row.iloc[0]['points']
        features['group_gd'] = group_row.iloc[0]['goal_difference']
    else:
        features['group_points'] = 0
        features['group_gd'] = 0
    
    # Get rolling xG (use historical average from training data - simplified)
    # In a real scenario, you'd have 2026 match data to calculate this
    team_xg_data = df_train[
        ((df_train['home_team'] == team_name) | (df_train['away_team'] == team_name))
    ]
    if len(team_xg_data) > 0:
        home_xg = team_xg_data[team_xg_data['home_team'] == team_name]['home_rolling_xg_10'].mean()
        away_xg = team_xg_data[team_xg_data['away_team'] == team_name]['away_rolling_xg_10'].mean()
        features['rolling_xg'] = np.nanmean([home_xg, away_xg])
    else:
        features['rolling_xg'] = 1.0  # Default
    
    # Get penalty win rate (from historical data)
    penalty_stats = get_penalty_stats(df_train)
    penalty_row = penalty_stats[penalty_stats['team'] == team_name]
    if len(penalty_row) > 0:
        features['penalty_win_rate'] = penalty_row.iloc[0]['win_rate']
    else:
        features['penalty_win_rate'] = 0.5
    
    # Knockout history
    knockout_hist = get_knockout_history(df_train)
    ko_row = knockout_hist[knockout_hist['team'] == team_name]
    if len(ko_row) > 0:
        features['knockout_history'] = ko_row.iloc[0]['knockout_history_score']
    else:
        features['knockout_history'] = 2.5  # Average
    
    return features

def predict_match(home_team, away_team, host_team=None):
    """Predict outcome of a 2026 World Cup match"""
    
    home_features = get_2026_team_features(home_team)
    away_features = get_2026_team_features(away_team)
    
    # Calculate host advantage
    host_advantage = 0
    if host_team == home_team:
        host_advantage = 2
    elif host_team == away_team:
        host_advantage = -2
    
    # Build feature vector for prediction
    X_pred = pd.DataFrame([{
        'elo_diff': home_features['elo'] - away_features['elo'],
        'home_rolling_xg_10': home_features['rolling_xg'],
        'away_rolling_xg_10': away_features['rolling_xg'],
        'host_advantage': host_advantage,
        'home_penalty_win_rate': home_features['penalty_win_rate'],
        'away_penalty_win_rate': away_features['penalty_win_rate'],
        'home_group_points': home_features['group_points'],
        'home_group_gd': home_features['group_gd'],
        'away_group_points': away_features['group_points'],
        'away_group_gd': away_features['group_gd']
    }])
    
    # Predict probability
    prob = model.predict_proba(X_pred)[0]
    home_win_prob = prob[1]  # Probability of home team advancing
    away_win_prob = prob[0]
    
    return {
        'home_team': home_team,
        'away_team': away_team,
        'home_win_prob': home_win_prob,
        'away_win_prob': away_win_prob,
        'prediction': home_team if home_win_prob > 0.5 else away_team
    }


