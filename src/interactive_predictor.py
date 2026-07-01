import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.penalty_feature import get_penalty_stats
from features.group_stage_feature import add_group_stage_features
from features.knockout_history_feature import get_knockout_history
from features.betting_odds import get_match_betting_odds, display_betting_context

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

def get_all_teams():
    """Get list of all 2026 teams"""
    return sorted(elo_2026['country'].unique())

def get_2026_team_features(team_name):
    """Get all features for a 2026 team for prediction"""
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
    
    # Get rolling xG (use historical average from training data)
    team_xg_data = df_train[
        ((df_train['home_team'] == team_name) | (df_train['away_team'] == team_name))
    ]
    if len(team_xg_data) > 0:
        home_xg = team_xg_data[team_xg_data['home_team'] == team_name]['home_rolling_xg_10'].mean()
        away_xg = team_xg_data[team_xg_data['away_team'] == team_name]['away_rolling_xg_10'].mean()
        features['rolling_xg'] = np.nanmean([home_xg, away_xg])
    else:
        features['rolling_xg'] = 1.0
    
    # Get penalty win rate
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
        features['knockout_history'] = 2.5
    
    return features

def predict_match(home_team, away_team, host_team=None):
    """Predict outcome of a 2026 World Cup match"""
    
    home_features = get_2026_team_features(home_team)
    away_features = get_2026_team_features(away_team)
    
    # Calculate host advantage
    host_advantage = 0
    if host_team and host_team == home_team:
        host_advantage = 2
    elif host_team and host_team == away_team:
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
    home_win_prob = prob[1]
    away_win_prob = prob[0]
    
    return {
        'home_team': home_team,
        'away_team': away_team,
        'home_win_prob': home_win_prob,
        'away_win_prob': away_win_prob,
        'prediction': home_team if home_win_prob > 0.5 else away_team,
        'home_elo': home_features['elo'],
        'away_elo': away_features['elo']
    }

def display_result(result):
    """Display prediction result nicely"""
    print("\n" + "=" * 70)
    print(f"{result['home_team']} vs {result['away_team']}")
    print("=" * 70)
    print(f"Elo Ratings: {result['home_team']} ({result['home_elo']:.0f}) vs {result['away_team']} ({result['away_elo']:.0f})")
    print()
    print(f"{result['home_team']} advance: {result['home_win_prob']:.1%}")
    print(f"{result['away_team']} advance: {result['away_win_prob']:.1%}")
    print()
    print(f"🏆 Prediction: {result['prediction']} advance")
    print("=" * 70 + "\n")

def main():
    """Interactive prediction loop"""
    print("\n" + "=" * 70)
    print("2026 FIFA World Cup Knockout Stage Predictor")
    print("=" * 70)
    
    teams = get_all_teams()
    print(f"\nAvailable teams ({len(teams)}):")
    for i, team in enumerate(teams, 1):
        print(f"  {i:2d}. {team}")
    
    while True:
        print("\n" + "-" * 70)
        home_input = input("Enter home team name (or 'quit' to exit): ").strip()
        
        if home_input.lower() == 'quit':
            print("\nThanks for using the World Cup Predictor!")
            break
        
        # Find matching team (case-insensitive)
        home_team = None
        for team in teams:
            if team.lower() == home_input.lower():
                home_team = team
                break
        
        if not home_team:
            print(f"❌ Team '{home_input}' not found. Try again.")
            continue
        
        away_input = input("Enter away team name: ").strip()
        
        away_team = None
        for team in teams:
            if team.lower() == away_input.lower():
                away_team = team
                break
        
        if not away_team:
            print(f"❌ Team '{away_input}' not found. Try again.")
            continue
        
        if home_team == away_team:
            print("❌ Home and away teams must be different!")
            continue
        
        # Optional: specify host team
        host_input = input("Host country (press Enter if neutral): ").strip()
        host_team = None
        if host_input:
            for team in teams:
                if team.lower() == host_input.lower():
                    host_team = team
                    break
            if not host_team and host_input:
                print(f"⚠️  Host country '{host_input}' not found, treating as neutral venue.")
        
        # Make prediction
        result = predict_match(home_team, away_team, host_team)
        display_result(result)
        
        # Show betting context if available
        betting_context = display_betting_context(home_team, away_team, result['home_win_prob'])
        if betting_context:
            print(betting_context)

if __name__ == "__main__":
    main()
