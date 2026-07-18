import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys
import os
import pickle

# Resolve all paths relative to the project root so the script works from any cwd
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from features.penalty_feature import get_penalty_stats
from features.group_stage_feature import add_group_stage_features
from features.knockout_history_feature import get_knockout_history
from features.betting_odds import get_match_betting_odds, remaining_r32_matches, knockout_odds_2026, knockout_odds_2026_played
from features.update_rolling_features import prepare_knockout_with_updated_rolling

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
    'away_rolling_ga_10',
    'home_h2h_win_rate',
    'away_h2h_win_rate',
    'home_h2h_goal_diff',
    'away_h2h_goal_diff',
    'h2h_meeting_count',
    'home_odds_prob',
    'away_odds_prob']
    
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

def get_head_to_head_2026(home_team, away_team):
    """Calculate head-to-head stats for 2026 predictions (1-2 year lookback from July 2026)"""
    from datetime import datetime, timedelta
    
    # Lookback window: 1-2 years before July 2, 2026
    match_date = datetime(2026, 7, 2)
    lookback_start = match_date - timedelta(days=730)  # 2 years
    lookback_end = match_date - timedelta(days=1)
    
    # Filter matches between these two teams in the window
    h2h_matches = df_all_international[
        (df_all_international['date'] >= lookback_start) &
        (df_all_international['date'] <= lookback_end) &
        (
            ((df_all_international['home_team'] == home_team) & (df_all_international['away_team'] == away_team)) |
            ((df_all_international['home_team'] == away_team) & (df_all_international['away_team'] == home_team))
        )
    ]
    
    if len(h2h_matches) == 0:
        return {
            'home_h2h_win_rate': 0.5,
            'away_h2h_win_rate': 0.5,
            'home_h2h_goal_diff': 0.0,
            'away_h2h_goal_diff': 0.0,
            'h2h_meeting_count': 0
        }
    
    home_wins = 0
    home_goal_diff_sum = 0
    
    for _, match in h2h_matches.iterrows():
        if match['home_team'] == home_team:
            goal_diff = match['home_score'] - match['away_score']
            if goal_diff > 0:
                home_wins += 1
        else:
            goal_diff = match['away_score'] - match['home_score']
            if goal_diff > 0:
                home_wins += 1
        
        home_goal_diff_sum += goal_diff
    
    return {
        'home_h2h_win_rate': home_wins / len(h2h_matches),
        'away_h2h_win_rate': (len(h2h_matches) - home_wins) / len(h2h_matches),
        'home_h2h_goal_diff': home_goal_diff_sum / len(h2h_matches),
        'away_h2h_goal_diff': -home_goal_diff_sum / len(h2h_matches),
        'h2h_meeting_count': len(h2h_matches)
    }

def moneyline_to_probability(moneyline):
    """Convert American moneyline odds to implied probability"""
    if moneyline < 0:
        return abs(moneyline) / (abs(moneyline) + 100)
    else:
        return 100 / (moneyline + 100)

def get_betting_odds_probs(home_team, away_team):
    """Get betting odds probabilities for a match"""
    # Combine all odds
    all_odds = {**knockout_odds_2026, **knockout_odds_2026_played}
    
    match_key = (home_team, away_team)
    
    if match_key in all_odds:
        odds_tuple = all_odds[match_key]
        home_moneyline, draw_moneyline, away_moneyline = odds_tuple
        return {
            'home_odds_prob': moneyline_to_probability(home_moneyline),
            'away_odds_prob': moneyline_to_probability(away_moneyline)
        }
    
    return {
        'home_odds_prob': 0.5,
        'away_odds_prob': 0.5
    }

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

# Score model feature order (must match training)
SCORE_FEATURES = ['elo_diff', 'home_rolling_xg_10', 'away_rolling_xg_10',
                  'home_rolling_gf_10', 'home_rolling_ga_10',
                  'away_rolling_gf_10', 'away_rolling_ga_10', 'host_advantage']

# Cache for the updated-rolling pipeline so it runs ONCE per process instead of
# on every prediction (which previously re-appended matches and wrote a backup
# file each call).
_updated_cache = None

def get_updated_model_and_data():
    """Run the rolling-feature update + retrain once, then cache the result."""
    global _updated_cache
    if _updated_cache is not None:
        return _updated_cache

    print("🔄 Updating rolling features from tournament matches...")
    # backup=False: no per-run backup CSVs; dedup keeps this idempotent
    ko_updated = prepare_knockout_with_updated_rolling(backup=False)

    # Reload the in-memory international results so current-form lookups
    # (get_rolling_goals_for_team) reflect any tournament matches just appended.
    global df_all_international
    df_all_international = pd.read_csv('data/international_results.csv')
    df_all_international = df_all_international[df_all_international['home_score'].notna()].copy()
    df_all_international['date'] = pd.to_datetime(df_all_international['date'])

    feature_cols = ['elo_diff', 'home_rolling_xg_10', 'away_rolling_xg_10',
                    'host_advantage', 'home_penalty_win_rate', 'away_penalty_win_rate',
                    'home_group_points', 'home_group_gd', 'away_group_points', 'away_group_gd']
    target_col = 'home_advanced'

    df_clean = ko_updated[feature_cols + [target_col]].dropna()
    X_train = df_clean[feature_cols]
    y_train = df_clean[target_col]

    model_updated = RandomForestClassifier(n_estimators=100, random_state=42)
    model_updated.fit(X_train, y_train)
    print(f"✓ Retrained model on {len(df_clean)} matches with updated rolling features")

    _updated_cache = (model_updated, ko_updated, feature_cols)
    return _updated_cache

def reconcile_score_with_prediction(predicted_score, advancing_team, home_team, away_team):
    """Ensure the predicted scoreline doesn't contradict who is predicted to advance.

    The win-probability classifier and the score models are independent, so they
    can disagree (e.g. classifier says away advances but score says home wins).
    If they contradict, assign the winning scoreline to the advancing team so the
    displayed result is internally consistent.
    """
    if predicted_score is None:
        return predicted_score

    home_r = predicted_score['home_goals_rounded']
    away_r = predicted_score['away_goals_rounded']

    # If it's a draw, the advancing team wins on penalties — already consistent.
    if home_r == away_r:
        return predicted_score

    score_winner = home_team if home_r > away_r else away_team
    if score_winner == advancing_team:
        return predicted_score

    # Contradiction: swap goals so the advancing team holds the winning scoreline.
    predicted_score['home_goals'], predicted_score['away_goals'] = \
        predicted_score['away_goals'], predicted_score['home_goals']
    predicted_score['home_goals_rounded'], predicted_score['away_goals_rounded'] = \
        predicted_score['away_goals_rounded'], predicted_score['home_goals_rounded']
    return predicted_score

def predict_match(home_team, away_team, host_team=None, use_updated_rolling=True):
    """
    Predict outcome of a 2026 World Cup match with UPDATED rolling features.
    
    Parameters:
    - home_team: Home team name
    - away_team: Away team name
    - host_team: Which team is hosting (for host advantage)
    - use_updated_rolling: Use updated rolling features from tournament matches (default: True)
    """
    
    if use_updated_rolling:
        try:
            # Run/reuse the cached rolling-feature update + retrained model
            model_updated, _ko_updated, _feature_cols = get_updated_model_and_data()

            # Build features from each team's CURRENT form (last-10 international
            # matches, including the tournament games just appended) rather than a
            # stale historical knockout row. get_2026_team_features() reads the
            # tournament-updated international data via get_rolling_goals_for_team().
            home_features = get_2026_team_features(home_team)
            away_features = get_2026_team_features(away_team)

            elo_diff = home_features['elo'] - away_features['elo']

            home_gf = home_features['rolling_gf']
            home_ga = home_features['rolling_ga']
            away_gf = away_features['rolling_gf']
            away_ga = away_features['rolling_ga']

            # xG proxy = current rolling goals-for (rolling_xg_feature uses goals as
            # the xG proxy, so this keeps prediction inputs consistent with training).
            home_xg = home_gf
            away_xg = away_gf

            home_penalty = home_features['penalty_win_rate']
            away_penalty = away_features['penalty_win_rate']

            host_advantage = 0
            if host_team == home_team:
                host_advantage = 2
            elif host_team == away_team:
                host_advantage = -2

            # Predict with updated model
            X_pred = pd.DataFrame([{
                'elo_diff': elo_diff,
                'home_rolling_xg_10': home_xg,
                'away_rolling_xg_10': away_xg,
                'host_advantage': host_advantage,
                'home_penalty_win_rate': home_penalty,
                'away_penalty_win_rate': away_penalty,
                'home_group_points': home_features['group_points'],
                'home_group_gd': home_features['group_gd'],
                'away_group_points': away_features['group_points'],
                'away_group_gd': away_features['group_gd']
            }])
            
            prob = model_updated.predict_proba(X_pred)[0]
            home_win_prob = prob[1]
            away_win_prob = prob[0]
            
            # Predict score with updated model if available
            predicted_score = None
            if score_model_home is not None and score_model_away is not None:
                # Pass a named DataFrame (matching training) to avoid sklearn
                # "X does not have valid feature names" warnings.
                X_score = pd.DataFrame([[elo_diff, home_xg, away_xg, home_gf, home_ga,
                                         away_gf, away_ga, host_advantage]],
                                       columns=SCORE_FEATURES)
                X_score = X_score.fillna(1.5)

                pred_home_goals = max(0, score_model_home.predict(X_score)[0])
                pred_away_goals = max(0, score_model_away.predict(X_score)[0])

                predicted_score = {
                    'home_goals': round(pred_home_goals, 1),
                    'away_goals': round(pred_away_goals, 1),
                    'home_goals_rounded': round(pred_home_goals),
                    'away_goals_rounded': round(pred_away_goals)
                }

            advancing_team = home_team if home_win_prob > 0.5 else away_team
            predicted_score = reconcile_score_with_prediction(
                predicted_score, advancing_team, home_team, away_team)

            return {
                'home_team': home_team,
                'away_team': away_team,
                'home_win_prob': home_win_prob,
                'away_win_prob': away_win_prob,
                'prediction': advancing_team,
                'home_rolling_xg': home_xg,
                'away_rolling_xg': away_xg,
                'predicted_score': predicted_score,
                'note': '✓ Using UPDATED rolling features from tournament matches'
            }
        
        except Exception as e:
            print(f"⚠️ Error with updated rolling: {e}")
            use_updated_rolling = False
    
    # Fallback to original method if updated fails
    if not use_updated_rolling:
        home_features = get_2026_team_features(home_team)
        away_features = get_2026_team_features(away_team)
        
        # Calculate host advantage
        host_advantage = 0
        if host_team and host_team == home_team:
            host_advantage = 2
        elif host_team and host_team == away_team:
            host_advantage = -2
        
        # Get head-to-head features
        h2h_features = get_head_to_head_2026(home_team, away_team)
        
        # Get betting odds features
        odds_features = get_betting_odds_probs(home_team, away_team)
        
        # Build feature vector for prediction (21 features)
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
            'away_rolling_ga_10': away_features['rolling_ga'],
            'home_h2h_win_rate': h2h_features['home_h2h_win_rate'],
            'away_h2h_win_rate': h2h_features['away_h2h_win_rate'],
            'home_h2h_goal_diff': h2h_features['home_h2h_goal_diff'],
            'away_h2h_goal_diff': h2h_features['away_h2h_goal_diff'],
            'h2h_meeting_count': h2h_features['h2h_meeting_count'],
            'home_odds_prob': odds_features['home_odds_prob'],
            'away_odds_prob': odds_features['away_odds_prob']
        }])
        
        # Predict probability
        prob = model.predict_proba(X_pred)[0]
        home_win_prob = prob[1]
        away_win_prob = prob[0]
        
        # Predict score if models available
        predicted_score = None
        if score_model_home is not None and score_model_away is not None:
            X_score = X_pred[SCORE_FEATURES]

            pred_home_goals = score_model_home.predict(X_score)[0]
            pred_away_goals = score_model_away.predict(X_score)[0]
            
            prob_diff = home_win_prob - 0.5
            adjustment_factor = prob_diff * 1.0
            home_boost = 1 + adjustment_factor
            away_boost = 1 - adjustment_factor
            
            pred_home_goals_adj = pred_home_goals * home_boost
            pred_away_goals_adj = pred_away_goals * away_boost
            
            pred_home_goals_adj = max(0, pred_home_goals_adj)
            pred_away_goals_adj = max(0, pred_away_goals_adj)
            
            predicted_score = {
                'home_goals': round(pred_home_goals_adj, 1),
                'away_goals': round(pred_away_goals_adj, 1),
                'home_goals_rounded': round(pred_home_goals_adj),
                'away_goals_rounded': round(pred_away_goals_adj)
            }

        advancing_team = home_team if home_win_prob > 0.5 else away_team
        predicted_score = reconcile_score_with_prediction(
            predicted_score, advancing_team, home_team, away_team)

        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_win_prob': home_win_prob,
            'away_win_prob': away_win_prob,
            'prediction': advancing_team,
            'home_elo': home_features['elo'],
            'away_elo': away_features['elo'],
            'predicted_score': predicted_score,
            'note': '⚠️ Using stagnant rolling features (fallback mode)'
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
    """Display Finals matches"""
    print("\n" + "=" * 70)
    print("2026 FIFA World Cup - Finals")
    print("=" * 70)
    print("\n1. France vs England (3rd Place Match - July 16)")
    print("2. Spain vs Argentina (Final - July 19)")
    print("\n" + "=" * 70 + "\n")

def main():
    """Interactive prediction loop for finals"""
    print("\n" + "=" * 70)
    print("2026 FIFA World Cup Finals Predictor")
    print("=" * 70)
    print("\nSemifinals Complete!")
    print("  Spain beat France 2-0")
    print("  Argentina beat England 2-1")
    print("\nNow predicting the Finals and 3rd Place Match...\n")
    
    finals_matches = [
        ("France", "England"),  # 3rd Place
        ("Spain", "Argentina")  # Final
    ]
    
    while True:
        display_match_list()
        
        user_input = input("Enter match number (1-2) or 'quit' to exit: ").strip()
        
        if user_input.lower() == 'quit':
            print("\nThanks for using the World Cup Predictor!")
            break
        
        # Validate input
        try:
            match_num = int(user_input)
            if match_num < 1 or match_num > len(finals_matches):
                print(f"❌ Please enter a number between 1 and {len(finals_matches)}")
                continue
        except ValueError:
            print("❌ Invalid input. Please enter a number or 'quit'")
            continue
        
        # Get the selected match
        home_team, away_team = finals_matches[match_num - 1]
        
        # Make prediction
        try:
            result = predict_match(home_team, away_team)
            display_result(result, odds_data=None)
        except Exception as e:
            print(f"\n❌ Error making prediction: {e}")
            print("Please try again.\n")
            continue

if __name__ == "__main__":
    main()
