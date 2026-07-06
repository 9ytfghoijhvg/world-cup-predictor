import pandas as pd
import os
from datetime import datetime
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


PREDICTIONS_CSV = 'data/prediction_log.csv'

def log_prediction(home_team, away_team, predicted_winner, home_win_prob, predicted_home_score, predicted_away_score, actual_winner=None, actual_home_score=None, actual_away_score=None):
    """
    Log a prediction for a match.
    """
    
    # Create or load existing log
    if os.path.exists(PREDICTIONS_CSV):
        df = pd.read_csv(PREDICTIONS_CSV)
    else:
        df = pd.DataFrame(columns=[
            'match_date', 'home_team', 'away_team', 'predicted_winner', 'home_win_prob',
            'predicted_home_score', 'predicted_away_score', 'actual_winner',
            'actual_home_score', 'actual_away_score', 'prediction_correct',
            'score_home_error', 'score_away_error', 'created_at', 'updated_at'
        ])
    
    # Check if prediction already exists (avoid duplicates)
    if len(df) > 0:
        existing_mask = (df['home_team'] == home_team) & (df['away_team'] == away_team)
        existing = df[existing_mask]
    else:
        existing = pd.DataFrame()
    
    if len(existing) > 0:
        # Update existing prediction with actual result
        idx = existing.index[0]
        if actual_winner is not None:
            df.at[idx, 'actual_winner'] = actual_winner
            df.at[idx, 'actual_home_score'] = actual_home_score
            df.at[idx, 'actual_away_score'] = actual_away_score
            df.at[idx, 'updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Calculate metrics
            df.at[idx, 'prediction_correct'] = 1 if predicted_winner == actual_winner else 0
            df.at[idx, 'score_home_error'] = abs(df.at[idx, 'predicted_home_score'] - actual_home_score)
            df.at[idx, 'score_away_error'] = abs(df.at[idx, 'predicted_away_score'] - actual_away_score)
    else:
        # Add new prediction
        new_row = pd.DataFrame([{
            'match_date': datetime.now().strftime('%Y-%m-%d'),
            'home_team': home_team,
            'away_team': away_team,
            'predicted_winner': predicted_winner,
            'home_win_prob': round(home_win_prob, 4),
            'predicted_home_score': round(predicted_home_score, 1),
            'predicted_away_score': round(predicted_away_score, 1),
            'actual_winner': actual_winner,
            'actual_home_score': actual_home_score,
            'actual_away_score': actual_away_score,
            'prediction_correct': 1 if (actual_winner and predicted_winner == actual_winner) else (0 if actual_winner else None),
            'score_home_error': abs(round(predicted_home_score, 1) - actual_home_score) if actual_home_score is not None else None,
            'score_away_error': abs(round(predicted_away_score, 1) - actual_away_score) if actual_away_score is not None else None,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': None
        }])
        df = pd.concat([df, new_row], ignore_index=True)
    
    # Save updated log
    df.to_csv(PREDICTIONS_CSV, index=False)
    
    return df[(df['home_team'] == home_team) & (df['away_team'] == away_team)].iloc[-1]


def get_accuracy_stats():
    """
    Get overall accuracy statistics from all predictions.
    
    Returns:
    - Dict with winner prediction accuracy, score prediction MAE, etc.
    """
    
    if not os.path.exists(PREDICTIONS_CSV):
        return None
    
    df = pd.read_csv(PREDICTIONS_CSV)
    
    # Filter to matches with actual results
    completed = df[df['actual_winner'].notna()]
    
    if len(completed) == 0:
        return None
    
    winner_accuracy = completed['prediction_correct'].mean()
    home_score_mae = completed['score_home_error'].mean()
    away_score_mae = completed['score_away_error'].mean()
    
    stats = {
        'total_matches_predicted': len(df),
        'matches_completed': len(completed),
        'matches_remaining': len(df) - len(completed),
        'winner_prediction_accuracy': round(winner_accuracy, 3),
        'home_score_mae': round(home_score_mae, 2),
        'away_score_mae': round(away_score_mae, 2),
        'avg_score_mae': round((home_score_mae + away_score_mae) / 2, 2),
        'correct_winners': int(completed['prediction_correct'].sum()),
        'incorrect_winners': len(completed) - int(completed['prediction_correct'].sum()),
    }
    
    return stats


def display_accuracy():
    """Display accuracy statistics in a formatted way."""
    
    stats = get_accuracy_stats()
    
    if stats is None:
        print("No predictions yet.")
        return
    
    print("\n" + "="*70)
    print("PREDICTION TRACKING RESULTS")
    print("="*70)
    print(f"\nMatches predicted: {stats['total_matches_predicted']}")
    print(f"Matches completed: {stats['matches_completed']}")
    print(f"Matches remaining: {stats['matches_remaining']}")
    
    if stats['matches_completed'] > 0:
        print(f"\n📊 WINNER PREDICTION:")
        print(f"  Accuracy: {stats['winner_prediction_accuracy']:.1%}")
        print(f"  Correct: {stats['correct_winners']}")
        print(f"  Incorrect: {stats['incorrect_winners']}")
        
        print(f"\n⚽ SCORE PREDICTION (MAE - Mean Absolute Error):")
        print(f"  Home goals MAE: {stats['home_score_mae']:.2f}")
        print(f"  Away goals MAE: {stats['away_score_mae']:.2f}")
        print(f"  Average MAE: {stats['avg_score_mae']:.2f}")
    
    print("\n" + "="*70 + "\n")


def show_predictions_table():
    """Display all predictions and results in a table."""
    
    if not os.path.exists(PREDICTIONS_CSV):
        print("No predictions logged yet.")
        return
    
    df = pd.read_csv(PREDICTIONS_CSV)
    
    # Select columns to display
    display_cols = ['match_date', 'home_team', 'away_team', 'predicted_winner', 
                    'predicted_home_score', 'predicted_away_score', 
                    'actual_winner', 'actual_home_score', 'actual_away_score', 'prediction_correct']
    
    print("\n" + "="*150)
    print("ALL PREDICTIONS AND RESULTS")
    print("="*150)
    print(df[display_cols].to_string(index=False))
    print("="*150 + "\n")


if __name__ == "__main__":
    # Example: Log a prediction
    # log_prediction('England', 'DR Congo', 'England', 0.936, 2, 1, actual_winner='England', actual_home_score=2, actual_away_score=1)
    
    show_predictions_table()
    display_accuracy()
