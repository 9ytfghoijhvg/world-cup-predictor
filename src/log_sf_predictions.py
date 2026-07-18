"""
Log semifinal predictions and actual results, then retrain model.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.interactive_predictor_numbered import predict_match


def log_semifinal_predictions():
    """Log the semifinal predictions before they happened"""
    
    print("\n" + "=" * 70)
    print("LOGGING SEMIFINAL PREDICTIONS")
    print("=" * 70)
    
    # Load prediction log
    pred_log = pd.read_csv('data/prediction_log.csv')
    
    # Check if semifinals are already logged
    sf_count = len(pred_log[pred_log['match_date'].isin(['2026-07-14', '2026-07-15'])])
    
    if sf_count >= 2:
        print("\n✓ Semifinals already logged in prediction_log.csv")
        return
    
    # Make predictions for semifinals
    print("\nPredicting Semifinals...")
    
    semifinals = [
        ("Spain", "France", "2026-07-14"),
        ("Argentina", "England", "2026-07-15")
    ]
    
    sf_results = []
    for home, away, date in semifinals:
        print(f"\nPredicting: {home} vs {away} ({date})")
        result = predict_match(home, away)
        
        # Store prediction
        sf_results.append({
            'match_date': date,
            'home_team': home,
            'away_team': away,
            'predicted_winner': result['prediction'],
            'home_win_prob': result['home_win_prob'],
            'predicted_home_score': result['predicted_score']['home_goals_rounded'] if result['predicted_score'] else 1,
            'predicted_away_score': result['predicted_score']['away_goals_rounded'] if result['predicted_score'] else 0,
            'actual_winner': '',
            'actual_home_score': '',
            'actual_away_score': '',
            'prediction_correct': '',
            'score_home_error': '',
            'score_away_error': '',
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': ''
        })
        
        print(f"  Prediction: {result['prediction']} ({result['home_win_prob']:.0%})")
        print(f"  Predicted Score: {result['predicted_score']['home_goals_rounded']}-{result['predicted_score']['away_goals_rounded']}")
    
    # Append to prediction log
    sf_df = pd.DataFrame(sf_results)
    pred_log_updated = pd.concat([pred_log, sf_df], ignore_index=True)
    pred_log_updated.to_csv('data/prediction_log.csv', index=False)
    
    print("\n✓ Semifinal predictions logged to prediction_log.csv")
    return sf_results


def log_semifinal_results():
    """Update prediction log with actual semifinal results"""
    
    print("\n" + "=" * 70)
    print("LOGGING SEMIFINAL RESULTS")
    print("=" * 70)
    
    # Actual semifinal results
    actual_results = {
        ("Spain", "France", "2026-07-14"): (2, 0, "Spain"),
        ("Argentina", "England", "2026-07-15"): (2, 1, "Argentina")
    }
    
    # Load prediction log
    pred_log = pd.read_csv('data/prediction_log.csv')
    
    # Convert columns to proper types
    pred_log['actual_home_score'] = pred_log['actual_home_score'].astype('object')
    pred_log['actual_away_score'] = pred_log['actual_away_score'].astype('object')
    pred_log['actual_winner'] = pred_log['actual_winner'].astype('object')
    pred_log['prediction_correct'] = pred_log['prediction_correct'].astype('object')
    pred_log['score_home_error'] = pred_log['score_home_error'].astype('object')
    pred_log['score_away_error'] = pred_log['score_away_error'].astype('object')
    pred_log['updated_at'] = pred_log['updated_at'].astype('object')
    
    # Update with actual results
    for (home, away, date), (home_score, away_score, winner) in actual_results.items():
        mask = (pred_log['match_date'] == date) & (pred_log['home_team'] == home)
        
        if mask.any():
            idx = pred_log[mask].index[0]
            
            # Update actual results
            pred_log.at[idx, 'actual_home_score'] = home_score
            pred_log.at[idx, 'actual_away_score'] = away_score
            pred_log.at[idx, 'actual_winner'] = winner
            
            # Calculate prediction accuracy
            predicted_winner = pred_log.at[idx, 'predicted_winner']
            pred_log.at[idx, 'prediction_correct'] = 1 if predicted_winner == winner else 0
            
            # Calculate score errors
            pred_log.at[idx, 'score_home_error'] = abs(int(pred_log.at[idx, 'predicted_home_score']) - home_score)
            pred_log.at[idx, 'score_away_error'] = abs(int(pred_log.at[idx, 'predicted_away_score']) - away_score)
            
            pred_log.at[idx, 'updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"\n✓ {home} {home_score}-{away_score} {away}")
            print(f"  Predicted: {predicted_winner} | Actual: {winner}")
            print(f"  Correct: {'Yes' if predicted_winner == winner else 'No'}")
    
    # Save updated log
    pred_log.to_csv('data/prediction_log.csv', index=False)
    print("\n✓ Semifinal results logged to prediction_log.csv")


def retrain_model_with_sf_results():
    """Retrain the model including semifinal results"""
    
    print("\n" + "=" * 70)
    print("RETRAINING MODEL WITH SEMIFINAL DATA")
    print("=" * 70)
    
    from features.update_rolling_features import prepare_knockout_with_updated_rolling
    from sklearn.ensemble import RandomForestClassifier
    import pickle
    
    # Prepare updated knockout data (includes semifinals)
    print("\nUpdating rolling features with semifinal data...")
    ko_updated = prepare_knockout_with_updated_rolling()
    
    # Retrain model
    print("\nRetraining Random Forest model...")
    
    feature_cols = ['elo_diff', 'home_rolling_xg_10', 'away_rolling_xg_10', 
                   'host_advantage', 'home_penalty_win_rate', 'away_penalty_win_rate', 
                   'home_group_points', 'home_group_gd', 'away_group_points', 'away_group_gd']
    
    target_col = 'home_advanced'
    
    df_clean = ko_updated[feature_cols + [target_col]].dropna()
    X_train = df_clean[feature_cols]
    y_train = df_clean[target_col]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save updated model
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✓ Model retrained on {len(df_clean)} matches")
    print(f"✓ Model saved to models/random_forest_model.pkl")
    
    # Show model performance
    train_accuracy = model.score(X_train, y_train)
    print(f"✓ Training Accuracy: {train_accuracy:.1%}")


if __name__ == "__main__":
    import time
    
    print("\n" + "=" * 70)
    print("SEMIFINAL PREDICTION LOG & MODEL RETRAIN")
    print("=" * 70)
    
    try:
        # Step 1: Log predictions (if not already done)
        log_semifinal_predictions()
        
        # Step 2: Log actual results
        log_semifinal_results()
        
        # Step 3: Retrain model with new data
        retrain_model_with_sf_results()
        
        print("\n" + "=" * 70)
        print("✓ COMPLETE!")
        print("=" * 70)
        print("\nSummary:")
        print("  ✓ Semifinal predictions logged")
        print("  ✓ Actual results logged")
        print("  ✓ Model retrained with semifinal data")
        print("\nThe interactive predictor now uses the updated model for Finals!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
