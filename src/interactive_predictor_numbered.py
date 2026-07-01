import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys
import os
import pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.penalty_feature import get_penalty_stats
from features.group_stage_feature import add_group_stage_features
from features.knockout_history_feature import get_knockout_history
from features.betting_odds import get_match_betting_odds, remaining_r32_matches

# Load trained model from disk
MODEL_PATH = 'models/random_forest_model.pkl'
SCORE_MODEL_HOME_PATH = 'models/score_predictor_home.pkl'
SCORE_MODEL_AWAY_PATH = 'models/score_predictor_away.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    # Load score prediction models
    with open(SCORE_MODEL_HOME_PATH, 'rb') as f:
        score_model_home = pickle.load(f)
    with open(SCORE_MODEL_AWAY_PATH, 'rb') as f:
        score_model_away = pickle.load(f)
    
except FileNotFoundError as e:
    print(f"⚠️  Model file not found: {e}")
    print(f"   Training model on-the-fly (slower)...")
    
    # Fallback: train on-the-fly
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
    'away_group_gd',
    'home_rolling_gf_10',
    'home_rolling_ga_10',
    'away_rolling_gf_10',
    'away_rolling_ga_10']
    
    target_col = 'home_advanced'
    df_clean = df_train[feature_cols + [target_col]].dropna()
    X_train = df_clean[feature_cols]
    y_train = df_clean[target_col]
    
    model = RandomForestClassifier(n_estimators=1000, random_state=42)
    model.fit(X_train, y_train)
    
    score_model_home = None
    score_model_away = None

# Load training data for feature lookup
df_train = pd.read_csv('data/knockout_matches_prepared.csv')

# Load 2026 data sources
elo_2026 = pd.read_csv('data/elo_ratings_wc2026.csv')  # Official 2026 ratings
elo_2026_calculated = pd.read_csv('data/elo_ratings_all_teams_2026.csv')  # Our calculated ratings

# Merge both - prefer official where available, use calculated as fallback
elo_2026_combined = elo_2026[['country', 'rating']].rename(columns={'country': 'team'})
elo_2026_calculated_renamed = elo_2026_calculated.rename(columns={'team': 'team_calc', 'rating': 'rating_calc'})

# Use calculated ratings for all teams, then override with official where available
elo_2026_final = elo_2026_calculated.copy()
for idx, row in elo_2026_combined.iterrows():
    team = row['team']
    rating = row['rating']
    # Update if team exists in our calculated set
    elo_2026_final.loc[elo_2026_final['team'] == team, 'rating'] = rating

elo_2026 = elo_2026_final

group_stage_2026 = pd.read_csv('data/group_stage_2026.csv')

# Load ALL international matches for rolling goals calculation
df_all_international = pd.read_csv('data/international_results.csv')
df_all_international = df_all_international[df_all_international['home_score'].notna()].copy()
df_all_international['date'] = pd.to_datetime(df_all_international['date'])

def get_all_teams():
    """Get list of all 2026 teams"""
    return sorted(elo_2026['country'].unique())

def get_rolling_goals_for_team(team_name, window=10):
    """Calculate rolling goals for/against for a team from all international matches"""
    # Get all matches for this team (home and away)
    home_matches = df_all_international[df_all_international['home_team'] == team_name][['date', 'home_score', 'away_score']].copy()
    home_matches.columns = ['date', 'goals_for', 'goals_against']
    
    away_matches = df_all_international[df_all_international['away_team'] == team_name][['date', 'away_score', 'home_score']].copy()
    away_matches.columns = ['date', 'goals_for', 'goals_against']
    
    # Combine and sort by date
    all_matches = pd.concat([home_matches, away_matches], ignore_index=True)
    all_matches = all_matches.sort_values('date')
    
    if len(all_matches) == 0:
        return 1.0, 1.0  # Default values
    
    # Take last N matches and calculate average
    recent_matches = all_matches.tail(window)
    rolling_gf = recent_matches['goals_for'].mean()
    rolling_ga = recent_matches['goals_against'].mean()
    
    return rolling_gf, rolling_ga

def get_2026_team_features(team_name):
    """Get all features for a 2026 team for prediction"""
    features = {}
    
    # Get Elo rating (use most recent)
    elo_row = elo_2026[elo_2026['team'] == team_name]
    if len(elo_row) > 0:
        features['elo'] = elo_row.iloc[0]['rating']
    else:
        # Fallback: use average Elo if team not found
        features['elo'] = 1500
    
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
    
    # Get rolling goals from ALL international matches (last 10 matches)
    rolling_gf, rolling_ga = get_rolling_goals_for_team(team_name, window=10)
    features['rolling_gf'] = rolling_gf
    features['rolling_ga'] = rolling_ga
    
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
        'away_group_gd': away_features['group_gd'],
        'home_rolling_gf_10': home_features['rolling_gf'],
        'home_rolling_ga_10': home_features['rolling_ga'],
        'away_rolling_gf_10': away_features['rolling_gf'],
        'away_rolling_ga_10': away_features['rolling_ga']
    }])
    
    # Predict probability
    prob = model.predict_proba(X_pred)[0]
    home_win_prob = prob[1]
    away_win_prob = prob[0]
    
    # Predict score if models available
    predicted_score = None
    if score_model_home is not None and score_model_away is not None:
        score_features = ['elo_diff', 'home_rolling_xg_10', 'away_rolling_xg_10',
                          'home_rolling_gf_10', 'home_rolling_ga_10',
                          'away_rolling_gf_10', 'away_rolling_ga_10', 'host_advantage']
        X_score = X_pred[score_features]
        
        pred_home_goals = score_model_home.predict(X_score)[0]
        pred_away_goals = score_model_away.predict(X_score)[0]
        
        predicted_score = {
            'home_goals': round(pred_home_goals, 1),
            'away_goals': round(pred_away_goals, 1),
            'home_goals_rounded': round(pred_home_goals),
            'away_goals_rounded': round(pred_away_goals)
        }
    
    return {
        'home_team': home_team,
        'away_team': away_team,
        'home_win_prob': home_win_prob,
        'away_win_prob': away_win_prob,
        'prediction': home_team if home_win_prob > 0.5 else away_team,
        'home_elo': home_features['elo'],
        'away_elo': away_features['elo'],
        'predicted_score': predicted_score
    }

def display_result(result, odds_data=None):
    """Display prediction result nicely"""
    print("\n" + "=" * 70)
    print(f"{result['home_team']} vs {result['away_team']}")
    print("=" * 70)
    
    # Display betting odds instead of Elo ratings
    if odds_data:
        home_line = f"{odds_data['home_moneyline']:+d}" if odds_data['home_moneyline'] > 0 else f"{odds_data['home_moneyline']}"
        away_line = f"{odds_data['away_moneyline']:+d}" if odds_data['away_moneyline'] > 0 else f"{odds_data['away_moneyline']}"
        print(f"Betting Odds: {result['home_team']} {home_line} | Draw {odds_data['draw_moneyline']:+d} | {result['away_team']} {away_line}")
    else:
        print(f"Betting Odds: Not available for this matchup")
    
    print()
    
    # Display predicted score if available
    if result.get('predicted_score'):
        score = result['predicted_score']
        print(f"📊 Predicted Score: {result['home_team']} {score['home_goals_rounded']}-{score['away_goals_rounded']} {result['away_team']}")
        print(f"   (Expected: {score['home_goals']:.1f} - {score['away_goals']:.1f})")
        print()
    
    print(f"{result['home_team']} advance: {result['home_win_prob']:.1%}")
    print(f"{result['away_team']} advance: {result['away_win_prob']:.1%}")
    print()
    print(f"🏆 Prediction: {result['prediction']} advance")
    print("=" * 70 + "\n")

def display_match_list():
    """Display all remaining Round of 32 matches"""
    print("\n" + "=" * 70)
    print("2026 FIFA World Cup - Round of 32 (Remaining Matches)")
    print("=" * 70)
    print()
    
    for i, (home, away) in enumerate(remaining_r32_matches, 1):
        odds_data = get_match_betting_odds(home, away)
        if odds_data and not odds_data['is_played']:
            # Format odds nicely
            home_line = f"{odds_data['home_moneyline']:+d}" if odds_data['home_moneyline'] > 0 else f"{odds_data['home_moneyline']}"
            print(f"{i:2d}. {home:<25} vs {away:<25} ({home_line})")
        elif odds_data and odds_data['is_played']:
            print(f"   [{home:<25} vs {away:<25} - PLAYED]")
    
    print()
    print("=" * 70 + "\n")

def main():
    """Interactive prediction loop with numbered match selection"""
    print("\n" + "=" * 70)
    print("2026 FIFA World Cup Knockout Stage Predictor")
    print("=" * 70)
    
    while True:
        display_match_list()
        
        user_input = input("Enter match number (1-9) or 'quit' to exit: ").strip()
        
        if user_input.lower() == 'quit':
            print("\nThanks for using the World Cup Predictor!")
            break
        
        # Validate input
        try:
            match_num = int(user_input)
            if match_num < 1 or match_num > len(remaining_r32_matches):
                print(f"❌ Please enter a number between 1 and {len(remaining_r32_matches)}")
                continue
        except ValueError:
            print("❌ Invalid input. Please enter a number or 'quit'")
            continue
        
        # Get the selected match
        home_team, away_team = remaining_r32_matches[match_num - 1]
        
        # Check if match was already played
        odds_data = get_match_betting_odds(home_team, away_team)
        if odds_data and odds_data['is_played']:
            print(f"\n⚠️  This match has already been played!")
            continue
        
        # Make prediction
        result = predict_match(home_team, away_team)
        
        # Get betting odds for display
        odds_data = get_match_betting_odds(home_team, away_team)
        
        display_result(result, odds_data)

if __name__ == "__main__":
    main()
