"""
Example: Making predictions with automatically updated rolling features.

This example shows how to use the new automatic rolling feature update system
to make predictions that reflect current tournament form.

Run this from the project root:
    python3 examples/predict_with_updated_rolling.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict_2026_with_updated_rolling import predict_match_with_updated_rolling
from features.update_rolling_features import update_international_results
import pandas as pd


def example_1_simple_prediction():
    """Example 1: Simple one-line prediction with auto-updates"""
    print("=" * 60)
    print("EXAMPLE 1: Simple Prediction with Auto-Updated Rolling Features")
    print("=" * 60)
    
    result = predict_match_with_updated_rolling("Argentina", "France")
    
    print(f"\nMatch: {result['home_team']} vs {result['away_team']}")
    print(f"Prediction: {result['prediction']} to advance")
    print(f"Home Win Probability: {result['home_win_prob']:.1%}")
    print(f"Away Win Probability: {result['away_win_prob']:.1%}")
    print(f"\nRolling xG (Current Tournament Form):")
    print(f"  {result['home_team']}: {result['home_rolling_xg']:.2f}")
    print(f"  {result['away_team']}: {result['away_rolling_xg']:.2f}")
    print(f"\n✓ {result['note']}")
    

def example_2_multiple_predictions():
    """Example 2: Batch predictions with updated features"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Batch Predictions (Semifinal Scenario)")
    print("=" * 60)
    
    # Predict both semifinals
    matches = [
        ("Argentina", "Germany"),
        ("Spain", "France"),
    ]
    
    print(f"\nUpdating rolling features once (for all predictions)...\n")
    
    results = []
    for home, away in matches:
        # First call updates, subsequent calls use cached data
        result = predict_match_with_updated_rolling(
            home, away, 
            auto_update=(home == matches[0][0])  # Only update on first call
        )
        results.append(result)
    
    # Display results
    print("\nSemifinal Predictions:")
    print("-" * 60)
    for result in results:
        print(f"{result['home_team']:15} {result['home_win_prob']:.1%}  |  {result['away_win_prob']:.1%}  {result['away_team']:15}")
        print(f"  Winner: {result['prediction']}")
        print(f"  Rolling xG: {result['home_rolling_xg']:.2f} vs {result['away_rolling_xg']:.2f}")
        print()


def example_3_with_manual_update():
    """Example 3: Manual control - update first, then predict"""
    print("=" * 60)
    print("EXAMPLE 3: Manual Update Then Predict")
    print("=" * 60)
    
    # Step 1: Manually update rolling features
    print("\nStep 1: Updating rolling features from tournament matches...")
    new_matches, count = update_international_results()
    
    if count > 0:
        print(f"  ✓ Added {count} new tournament matches")
    else:
        print(f"  ✓ All matches already included")
    
    # Step 2: Make prediction (no auto-update needed)
    print("\nStep 2: Making prediction with updated rolling features...")
    result = predict_match_with_updated_rolling(
        "Brazil", "Uruguay",
        auto_update=False  # Don't update again, we just did it
    )
    
    print(f"\nMatch: {result['home_team']} vs {result['away_team']}")
    print(f"Prediction: {result['prediction']} to advance")
    print(f"Probability: {result['home_win_prob']:.1%} ({result['away_win_prob']:.1%})")


def example_4_compare_features():
    """Example 4: Compare rolling features before and after updates"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Compare Rolling Features (Before vs After)")
    print("=" * 60)
    
    import pandas as pd
    
    # Load original and updated data
    try:
        original = pd.read_csv('data/knockout_matches_prepared.csv')
        updated = pd.read_csv('data/knockout_matches_prepared_updated.csv')
        
        print("\nComparing rolling xG features for Spain:")
        print("-" * 60)
        
        # Find Spain matches
        spain_orig = original[original['home_team'] == 'Spain']['home_rolling_xg_10'].values
        spain_upd = updated[updated['home_team'] == 'Spain']['home_rolling_xg_10'].values
        
        if len(spain_orig) > 0 and len(spain_upd) > 0:
            avg_orig = spain_orig.mean()
            avg_upd = spain_upd.mean()
            print(f"Average Rolling xG (Original): {avg_orig:.3f}")
            print(f"Average Rolling xG (Updated):  {avg_upd:.3f}")
            print(f"Change: {avg_upd - avg_orig:+.3f}")
            
            if abs(avg_upd - avg_orig) > 0.001:
                print(f"\n✓ Features updated! Tournament matches affect rolling stats.")
            else:
                print(f"\nℹ No significant change (normal if same data set)")
    except FileNotFoundError:
        print("Data files not found - run example 1 first")


def example_5_show_tournament_form():
    """Example 5: Display tournament form of teams"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Current Tournament Form")
    print("=" * 60)
    
    teams = ["Argentina", "Brazil", "France", "Germany", "Spain"]
    
    print(f"\nGetting rolling xG for top teams (current tournament form):\n")
    
    results = []
    for team in teams:
        # Get features for team
        result = predict_match_with_updated_rolling(
            team, "Uruguay",  # Dummy opponent to get features
            auto_update=False
        )
        results.append({
            'team': team,
            'rolling_xg': result['home_rolling_xg']
        })
    
    # Sort by rolling xG
    results_sorted = sorted(results, key=lambda x: x['rolling_xg'], reverse=True)
    
    print("Rolling xG by Team (Higher = Better Current Form):")
    print("-" * 40)
    for i, r in enumerate(results_sorted, 1):
        print(f"{i}. {r['team']:15} {r['rolling_xg']:.2f} xG/match")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Rolling Features Update System - Examples")
    print("=" * 60)
    
    try:
        # Run all examples
        example_1_simple_prediction()
        example_2_multiple_predictions()
        example_3_with_manual_update()
        example_4_compare_features()
        example_5_show_tournament_form()
        
        print("\n" + "=" * 60)
        print("✓ All examples completed successfully!")
        print("=" * 60)
        print("\nFor more details, see:")
        print("  - ROLLING_FEATURES_UPDATE.md (full documentation)")
        print("  - ROLLING_FEATURES_QUICKSTART.md (quick reference)")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure you:")
        print("  1. Are in the project root directory")
        print("  2. Have the virtual environment activated")
        print("  3. Have prediction log data (prediction_log.csv)")
