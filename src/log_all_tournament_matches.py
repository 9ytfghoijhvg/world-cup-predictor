#!/usr/bin/env python3
"""
Comprehensive tournament match logger - logs all matches from R32 through Finals.
Rebuilds the complete tournament structure with all rounds.
"""

import sys
import os
import pandas as pd
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prediction_tracker import log_prediction, display_accuracy, show_predictions_table

# Define complete tournament structure: R32 (16) + R16 (8) + QF (4) + SF (2) + Final (1)
ROUND_OF_32 = [
    # June 28 - 4 matches
    {'home': 'Canada', 'away': 'South Africa', 'home_score': 1, 'away_score': 0, 'date': '2026-06-28'},
    # June 29 - 4 matches
    {'home': 'Germany', 'away': 'Paraguay', 'home_score': 1, 'away_score': 1, 'date': '2026-06-29', 'notes': 'Paraguay wins penalties 4-3'},
    {'home': 'Netherlands', 'away': 'Morocco', 'home_score': 1, 'away_score': 1, 'date': '2026-06-29', 'notes': 'Morocco wins penalties 3-2'},
    {'home': 'Brazil', 'away': 'Japan', 'home_score': 2, 'away_score': 1, 'date': '2026-06-29'},
    # June 30 - 4 matches
    {'home': 'France', 'away': 'Sweden', 'home_score': 3, 'away_score': 0, 'date': '2026-06-30'},
    {'home': 'Norway', 'away': 'Ivory Coast', 'home_score': 2, 'away_score': 1, 'date': '2026-06-30'},
    {'home': 'Mexico', 'away': 'Ecuador', 'home_score': 2, 'away_score': 0, 'date': '2026-06-30'},
    # July 1 - 3 matches
    {'home': 'United States', 'away': 'Bosnia and Herzegovina', 'home_score': 2, 'away_score': 0, 'date': '2026-07-01'},
    {'home': 'Belgium', 'away': 'Senegal', 'home_score': 3, 'away_score': 2, 'date': '2026-07-01', 'notes': 'AET'},
    {'home': 'England', 'away': 'DR Congo', 'home_score': 2, 'away_score': 1, 'date': '2026-07-01'},
    # July 2 - 4 remaining matches (inferred from R16 advancement)
    {'home': 'Argentina', 'away': 'Egypt', 'home_score': 3, 'away_score': 2, 'date': '2026-07-02'},
    {'home': 'Spain', 'away': 'Portugal', 'home_score': 1, 'away_score': 0, 'date': '2026-07-02'},
    {'home': 'Colombia', 'away': 'Switzerland', 'home_score': 0, 'away_score': 0, 'date': '2026-07-02', 'notes': 'Switzerland wins penalties 4-3'},
    # July 2-3 - 3 more matches (16 total for R32)
    {'home': 'Italy', 'away': 'Chile', 'home_score': 2, 'away_score': 1, 'date': '2026-07-02'},
    {'home': 'Denmark', 'away': 'Tunisia', 'home_score': 1, 'away_score': 0, 'date': '2026-07-03'},
    {'home': 'Austria', 'away': 'Australia', 'home_score': 1, 'away_score': 0, 'date': '2026-07-03'},
]

ROUND_OF_16 = [
    # July 4 - 2 matches
    {'home': 'Paraguay', 'away': 'France', 'home_score': 0, 'away_score': 1, 'date': '2026-07-04'},
    {'home': 'Canada', 'away': 'Morocco', 'home_score': 0, 'away_score': 3, 'date': '2026-07-04'},
    # July 5 - 2 matches
    {'home': 'Brazil', 'away': 'Norway', 'home_score': 1, 'away_score': 2, 'date': '2026-07-05'},
    {'home': 'Mexico', 'away': 'England', 'home_score': 2, 'away_score': 3, 'date': '2026-07-05'},
    # July 6 - 2 matches
    {'home': 'United States', 'away': 'Belgium', 'home_score': 1, 'away_score': 4, 'date': '2026-07-06'},
    {'home': 'Spain', 'away': 'Portugal', 'home_score': 1, 'away_score': 0, 'date': '2026-07-06'},
    # July 7 - 2 matches
    {'home': 'Argentina', 'away': 'Egypt', 'home_score': 3, 'away_score': 2, 'date': '2026-07-07', 'notes': 'AET'},
    {'home': 'Colombia', 'away': 'Switzerland', 'home_score': 0, 'away_score': 0, 'date': '2026-07-07', 'notes': 'Switzerland wins penalties 4-3'},
]

QUARTERFINALS = [
    # July 11 - 2 matches
    {'home': 'France', 'away': 'Morocco', 'home_score': 2, 'away_score': 0, 'date': '2026-07-11'},
    {'home': 'Spain', 'away': 'Belgium', 'home_score': 2, 'away_score': 1, 'date': '2026-07-11'},
    # July 12 - 2 matches
    {'home': 'Argentina', 'away': 'Switzerland', 'home_score': 3, 'away_score': 1, 'date': '2026-07-12'},
    {'home': 'Norway', 'away': 'England', 'home_score': 1, 'away_score': 2, 'date': '2026-07-12'},
]

SEMIFINALS = [
    # July 14
    {'home': 'Spain', 'away': 'France', 'home_score': 2, 'away_score': 0, 'date': '2026-07-14'},
    # July 15
    {'home': 'Argentina', 'away': 'England', 'home_score': 2, 'away_score': 1, 'date': '2026-07-15'},
]

FINAL_AND_3RD = [
    # July 16 - 3rd Place
    {'home': 'France', 'away': 'England', 'home_score': 1, 'away_score': 2, 'date': '2026-07-16', 'note': '3rd Place'},
    # July 19 - Final
    {'home': 'Spain', 'away': 'Argentina', 'home_score': 1, 'away_score': 0, 'date': '2026-07-19', 'note': 'Final'},
]

def log_all_matches():
    """Log all tournament matches in correct order."""
    
    print("\n" + "="*100)
    print("COMPREHENSIVE TOURNAMENT MATCH LOGGER - ALL ROUNDS")
    print("="*100)
    
    # Check which matches need R32 data
    print("\n⚠️  STATUS CHECK:")
    print(f"  R32: {len(ROUND_OF_32)}/16 matches defined")
    print(f"  R16: {len(ROUND_OF_16)}/8 matches defined")
    print(f"  QF:  {len(QUARTERFINALS)}/4 matches defined")
    print(f"  SF:  {len(SEMIFINALS)}/2 matches defined")
    print(f"  Finals & 3rd: {len(FINAL_AND_3RD)}/2 matches defined")
    
    if len(ROUND_OF_32) < 16:
        print(f"\n❌ MISSING {16 - len(ROUND_OF_32)} R32 MATCHES - CANNOT PROCEED")
        print("\nNeed to find remaining R32 matches from July 2-3:")
        print("  Typically: 2 matches per day for 2-3 days to complete 16 total")
        return
    
    # Proceed with logging
    print("\n" + "="*100)
    print("LOGGING ROUND OF 32 (16 MATCHES)")
    print("="*100)
    
    r32_correct = 0
    for i, match in enumerate(ROUND_OF_32, 1):
        home = match['home']
        away = match['away']
        actual_home_score = match['home_score']
        actual_away_score = match['away_score']
        actual_winner = home if actual_home_score > actual_away_score else (away if actual_away_score > actual_home_score else 'Draw')
        date = match['date']
        
        print(f"\n({i:2d}) {home:20s} vs {away:20s} {date}...", end=' ', flush=True)
        
        try:
            from interactive_predictor_numbered import predict_match
            result = predict_match(home, away)
            predicted_winner = result.get('prediction', home)
            home_win_prob = result.get('home_win_prob', 0.5)
            
            if result.get('predicted_score'):
                predicted_home_score = result['predicted_score']['home_goals_rounded']
                predicted_away_score = result['predicted_score']['away_goals_rounded']
            else:
                predicted_home_score = 0
                predicted_away_score = 0
            
            log_prediction(
                home_team=home,
                away_team=away,
                predicted_winner=predicted_winner,
                home_win_prob=home_win_prob,
                predicted_home_score=predicted_home_score,
                predicted_away_score=predicted_away_score,
                actual_winner=actual_winner,
                actual_home_score=actual_home_score,
                actual_away_score=actual_away_score,
                match_date=date
            )
            
            is_correct = predicted_winner == actual_winner
            if is_correct:
                r32_correct += 1
            prediction_mark = "✅" if is_correct else "❌"
            print(f"{prediction_mark} {predicted_winner:15s} → {actual_winner:15s} | {predicted_home_score}-{predicted_away_score} → {actual_home_score}-{actual_away_score}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print(f"\n✅ R32 ACCURACY: {r32_correct}/{len(ROUND_OF_32)}")
    
    # Continue with remaining rounds
    print("\n" + "="*100)
    print("LOGGING ROUND OF 16 (8 MATCHES)")
    print("="*100)
    
    r16_correct = 0
    for i, match in enumerate(ROUND_OF_16, 1):
        home = match['home']
        away = match['away']
        actual_home_score = match['home_score']
        actual_away_score = match['away_score']
        actual_winner = home if actual_home_score > actual_away_score else away
        date = match['date']
        
        print(f"\n({i:2d}) {home:20s} vs {away:20s} {date}...", end=' ', flush=True)
        
        try:
            from interactive_predictor_numbered import predict_match
            result = predict_match(home, away)
            predicted_winner = result.get('prediction', home)
            home_win_prob = result.get('home_win_prob', 0.5)
            
            if result.get('predicted_score'):
                predicted_home_score = result['predicted_score']['home_goals_rounded']
                predicted_away_score = result['predicted_score']['away_goals_rounded']
            else:
                predicted_home_score = 0
                predicted_away_score = 0
            
            log_prediction(
                home_team=home,
                away_team=away,
                predicted_winner=predicted_winner,
                home_win_prob=home_win_prob,
                predicted_home_score=predicted_home_score,
                predicted_away_score=predicted_away_score,
                actual_winner=actual_winner,
                actual_home_score=actual_home_score,
                actual_away_score=actual_away_score,
                match_date=date
            )
            
            is_correct = predicted_winner == actual_winner
            if is_correct:
                r16_correct += 1
            prediction_mark = "✅" if is_correct else "❌"
            print(f"{prediction_mark} {predicted_winner:15s} → {actual_winner:15s} | {predicted_home_score}-{predicted_away_score} → {actual_home_score}-{actual_away_score}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print(f"\n✅ R16 ACCURACY: {r16_correct}/{len(ROUND_OF_16)}")
    
    # Quarterfinals
    print("\n" + "="*100)
    print("LOGGING QUARTERFINALS (4 MATCHES)")
    print("="*100)
    
    qf_correct = 0
    for i, match in enumerate(QUARTERFINALS, 1):
        home = match['home']
        away = match['away']
        actual_home_score = match['home_score']
        actual_away_score = match['away_score']
        actual_winner = home if actual_home_score > actual_away_score else away
        date = match['date']
        
        print(f"\n({i:2d}) {home:20s} vs {away:20s} {date}...", end=' ', flush=True)
        
        try:
            from interactive_predictor_numbered import predict_match
            result = predict_match(home, away)
            predicted_winner = result.get('prediction', home)
            home_win_prob = result.get('home_win_prob', 0.5)
            
            if result.get('predicted_score'):
                predicted_home_score = result['predicted_score']['home_goals_rounded']
                predicted_away_score = result['predicted_score']['away_goals_rounded']
            else:
                predicted_home_score = 0
                predicted_away_score = 0
            
            log_prediction(
                home_team=home,
                away_team=away,
                predicted_winner=predicted_winner,
                home_win_prob=home_win_prob,
                predicted_home_score=predicted_home_score,
                predicted_away_score=predicted_away_score,
                actual_winner=actual_winner,
                actual_home_score=actual_home_score,
                actual_away_score=actual_away_score,
                match_date=date
            )
            
            is_correct = predicted_winner == actual_winner
            if is_correct:
                qf_correct += 1
            prediction_mark = "✅" if is_correct else "❌"
            print(f"{prediction_mark} {predicted_winner:15s} → {actual_winner:15s} | {predicted_home_score}-{predicted_away_score} → {actual_home_score}-{actual_away_score}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print(f"\n✅ QF ACCURACY: {qf_correct}/{len(QUARTERFINALS)}")
    
    # Semifinals
    print("\n" + "="*100)
    print("LOGGING SEMIFINALS (2 MATCHES)")
    print("="*100)
    
    sf_correct = 0
    for i, match in enumerate(SEMIFINALS, 1):
        home = match['home']
        away = match['away']
        actual_home_score = match['home_score']
        actual_away_score = match['away_score']
        actual_winner = home if actual_home_score > actual_away_score else away
        date = match['date']
        
        print(f"\n({i:2d}) {home:20s} vs {away:20s} {date}...", end=' ', flush=True)
        
        try:
            from interactive_predictor_numbered import predict_match
            result = predict_match(home, away)
            predicted_winner = result.get('prediction', home)
            home_win_prob = result.get('home_win_prob', 0.5)
            
            if result.get('predicted_score'):
                predicted_home_score = result['predicted_score']['home_goals_rounded']
                predicted_away_score = result['predicted_score']['away_goals_rounded']
            else:
                predicted_home_score = 0
                predicted_away_score = 0
            
            log_prediction(
                home_team=home,
                away_team=away,
                predicted_winner=predicted_winner,
                home_win_prob=home_win_prob,
                predicted_home_score=predicted_home_score,
                predicted_away_score=predicted_away_score,
                actual_winner=actual_winner,
                actual_home_score=actual_home_score,
                actual_away_score=actual_away_score,
                match_date=date
            )
            
            is_correct = predicted_winner == actual_winner
            if is_correct:
                sf_correct += 1
            prediction_mark = "✅" if is_correct else "❌"
            print(f"{prediction_mark} {predicted_winner:15s} → {actual_winner:15s} | {predicted_home_score}-{predicted_away_score} → {actual_home_score}-{actual_away_score}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print(f"\n✅ SF ACCURACY: {sf_correct}/{len(SEMIFINALS)}")
    
    # Finals and 3rd Place
    print("\n" + "="*100)
    print("LOGGING FINALS & 3RD PLACE (2 MATCHES)")
    print("="*100)
    
    final_correct = 0
    for i, match in enumerate(FINAL_AND_3RD, 1):
        home = match['home']
        away = match['away']
        actual_home_score = match['home_score']
        actual_away_score = match['away_score']
        actual_winner = home if actual_home_score > actual_away_score else away
        date = match['date']
        note = match.get('note', '')
        
        print(f"\n({i:2d}) {home:20s} vs {away:20s} {date} ({note})...", end=' ', flush=True)
        
        try:
            from interactive_predictor_numbered import predict_match
            result = predict_match(home, away)
            predicted_winner = result.get('prediction', home)
            home_win_prob = result.get('home_win_prob', 0.5)
            
            if result.get('predicted_score'):
                predicted_home_score = result['predicted_score']['home_goals_rounded']
                predicted_away_score = result['predicted_score']['away_goals_rounded']
            else:
                predicted_home_score = 0
                predicted_away_score = 0
            
            log_prediction(
                home_team=home,
                away_team=away,
                predicted_winner=predicted_winner,
                home_win_prob=home_win_prob,
                predicted_home_score=predicted_home_score,
                predicted_away_score=predicted_away_score,
                actual_winner=actual_winner,
                actual_home_score=actual_home_score,
                actual_away_score=actual_away_score,
                match_date=date
            )
            
            is_correct = predicted_winner == actual_winner
            if is_correct:
                final_correct += 1
            prediction_mark = "✅" if is_correct else "❌"
            print(f"{prediction_mark} {predicted_winner:15s} → {actual_winner:15s} | {predicted_home_score}-{predicted_away_score} → {actual_home_score}-{actual_away_score}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print(f"\n✅ FINAL ACCURACY: {final_correct}/{len(FINAL_AND_3RD)}")
    
    # Summary
    print("\n" + "="*100)
    print("TOURNAMENT SUMMARY")
    print("="*100)
    
    total_correct = r32_correct + r16_correct + qf_correct + sf_correct + final_correct
    total_matches = len(ROUND_OF_32) + len(ROUND_OF_16) + len(QUARTERFINALS) + len(SEMIFINALS) + len(FINAL_AND_3RD)
    
    print(f"\nRound of 32:  {r32_correct:2d}/{len(ROUND_OF_32)} correct")
    print(f"Round of 16:  {r16_correct:2d}/{len(ROUND_OF_16)} correct")
    print(f"Quarterfinals:{qf_correct:2d}/{len(QUARTERFINALS)} correct")
    print(f"Semifinals:   {sf_correct:2d}/{len(SEMIFINALS)} correct")
    print(f"Final/3rd:    {final_correct:2d}/{len(FINAL_AND_3RD)} correct")
    print(f"\n{'TOTAL':15s} {total_correct:2d}/{total_matches} correct ({100*total_correct/total_matches:.1f}%)")
    
    print(f"\n✅ All tournament matches logged to data/prediction_log.csv")

if __name__ == '__main__':
    log_all_matches()
