"""
Point-in-time backtest of the 2026 World Cup knockout stage.

For every knockout game (chronologically), features are computed using ONLY
data available before that game's kickoff — a team's rolling goals/xG reflect
its last 10 matches prior to the game, so France's numbers before its Round-of-32
match differ from its numbers before the final. The trained classifier predicts
who advances (+ probability) and the score models predict a scoreline. Results
are written to data/prediction_log.csv alongside the actual outcomes.

Static features (Elo, group-stage stats, historical penalty win-rate) are known
before the knockouts begin and are used as-is for all rounds.
"""
import os
import sys
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from features.penalty_feature import get_penalty_stats
from src.knockout_2026_results import as_dicts

CLASSIFIER_FEATURES = ['elo_diff', 'home_rolling_xg_10', 'away_rolling_xg_10', 'host_advantage',
                       'home_penalty_win_rate', 'away_penalty_win_rate',
                       'home_group_points', 'home_group_gd', 'away_group_points', 'away_group_gd']
SCORE_FEATURES = ['elo_diff', 'home_rolling_xg_10', 'away_rolling_xg_10',
                  'home_rolling_gf_10', 'home_rolling_ga_10',
                  'away_rolling_gf_10', 'away_rolling_ga_10', 'host_advantage']

# ---------------------------------------------------------------- data loading
model = pickle.load(open('models/random_forest_model.pkl', 'rb'))
score_model_home = pickle.load(open('models/score_predictor_home.pkl', 'rb'))
score_model_away = pickle.load(open('models/score_predictor_away.pkl', 'rb'))

intl = pd.read_csv('data/international_results.csv')
intl = intl[intl['home_score'].notna()].copy()
intl['date'] = pd.to_datetime(intl['date'])

group_2026 = pd.read_csv('data/group_stage_2026.csv').set_index('team')
penalty_stats = get_penalty_stats(pd.read_csv('data/knockout_matches_prepared.csv')).set_index('team')

# Elo: calculated ratings for all teams, overridden by official WC2026 ratings
elo = dict(zip(*[pd.read_csv('data/elo_ratings_all_teams_2026.csv')[c] for c in ('team', 'rating')]))
official = pd.read_csv('data/elo_ratings_wc2026.csv')
elo.update(dict(zip(official['country'], official['rating'])))


def rolling_form(team, as_of_date, window=10):
    """Mean goals-for / goals-against over a team's last `window` matches
    strictly before `as_of_date`. Returns (gf, ga)."""
    m = intl[intl['date'] < as_of_date]
    home = m[m['home_team'] == team][['date', 'home_score', 'away_score']]
    home.columns = ['date', 'gf', 'ga']
    away = m[m['away_team'] == team][['date', 'away_score', 'home_score']]
    away.columns = ['date', 'gf', 'ga']
    both = pd.concat([home, away]).sort_values('date').tail(window)
    if both.empty:
        return 1.0, 1.0
    return float(both['gf'].mean()), float(both['ga'].mean())


def group_feat(team):
    if team in group_2026.index:
        r = group_2026.loc[team]
        return float(r['points']), float(r['goal_difference'])
    return 0.0, 0.0


def penalty_rate(team):
    return float(penalty_stats.loc[team, 'win_rate']) if team in penalty_stats.index else 0.5


def h2h_lookback(home, away, as_of_date, days=730):
    """Head-to-head win rate / goal diff for the home team over the `days`
    before the game (kept for reference; not used by the 10-feature model)."""
    start = as_of_date - timedelta(days=days)
    m = intl[(intl['date'] >= start) & (intl['date'] < as_of_date)]
    m = m[(((m['home_team'] == home) & (m['away_team'] == away)) |
           ((m['home_team'] == away) & (m['away_team'] == home)))]
    if m.empty:
        return {'meetings': 0, 'home_win_rate': 0.5, 'home_goal_diff': 0.0}
    wins, gd = 0, 0
    for _, x in m.iterrows():
        diff = (x['home_score'] - x['away_score']) if x['home_team'] == home else (x['away_score'] - x['home_score'])
        wins += diff > 0
        gd += diff
    return {'meetings': len(m), 'home_win_rate': wins / len(m), 'home_goal_diff': gd / len(m)}


def predict_game(g):
    date = pd.Timestamp(g['date'])
    home, away = g['home_team'], g['away_team']

    home_gf, home_ga = rolling_form(home, date)
    away_gf, away_ga = rolling_form(away, date)
    home_pts, home_gd = group_feat(home)
    away_pts, away_gd = group_feat(away)
    elo_diff = elo.get(home, 1500) - elo.get(away, 1500)
    host_adv = g['host_advantage']

    feats = {
        'elo_diff': elo_diff,
        'home_rolling_xg_10': home_gf, 'away_rolling_xg_10': away_gf,
        'host_advantage': host_adv,
        'home_penalty_win_rate': penalty_rate(home), 'away_penalty_win_rate': penalty_rate(away),
        'home_group_points': home_pts, 'home_group_gd': home_gd,
        'away_group_points': away_pts, 'away_group_gd': away_gd,
        'home_rolling_gf_10': home_gf, 'home_rolling_ga_10': home_ga,
        'away_rolling_gf_10': away_gf, 'away_rolling_ga_10': away_ga,
    }

    # Advance probability
    X_cls = pd.DataFrame([feats])[CLASSIFIER_FEATURES]
    home_prob = float(model.predict_proba(X_cls)[0][1])
    predicted_winner = home if home_prob > 0.5 else away

    # Scoreline, nudged toward the predicted winner, then reconciled
    X_sc = pd.DataFrame([feats])[SCORE_FEATURES]
    ph = float(score_model_home.predict(X_sc)[0])
    pa = float(score_model_away.predict(X_sc)[0])
    boost = home_prob - 0.5
    ph, pa = max(0, ph * (1 + boost)), max(0, pa * (1 - boost))
    hr, ar = round(ph), round(pa)
    # Keep the scoreline consistent with who is predicted to advance
    if hr != ar and (home if hr > ar else away) != predicted_winner:
        hr, ar = ar, hr

    return {
        'match_date': g['date'],
        'round': g['round'],
        'home_team': home,
        'away_team': away,
        'predicted_winner': predicted_winner,
        'home_win_prob': round(home_prob, 4),
        'predicted_home_score': hr,
        'predicted_away_score': ar,
        'actual_winner': g['winner'],
        'actual_home_score': g['home_score'],
        'actual_away_score': g['away_score'],
        'prediction_correct': predicted_winner == g['winner'],
        'score_home_error': abs(hr - g['home_score']),
        'score_away_error': abs(ar - g['away_score']),
        'verified_result': g['verified'],
    }


def main():
    games = sorted(as_dicts(), key=lambda x: x['date'])
    now = datetime.now().isoformat(timespec='seconds')
    rows = [predict_game(g) for g in games]
    log = pd.DataFrame(rows)
    log['created_at'] = now
    log['updated_at'] = now
    log.to_csv('data/prediction_log.csv', index=False)

    correct = log['prediction_correct'].sum()
    print(f"Logged {len(log)} knockout games -> data/prediction_log.csv")
    print(f"Winner accuracy: {correct}/{len(log)} = {correct/len(log):.1%}")
    print(f"Mean score error (home/away goals): "
          f"{log['score_home_error'].mean():.2f} / {log['score_away_error'].mean():.2f}")
    print()
    cols = ['round', 'home_team', 'away_team', 'predicted_winner', 'home_win_prob',
            'predicted_home_score', 'predicted_away_score', 'actual_home_score',
            'actual_away_score', 'actual_winner', 'prediction_correct']
    with pd.option_context('display.max_rows', None, 'display.width', 200):
        print(log[cols].to_string(index=False))


if __name__ == '__main__':
    main()
