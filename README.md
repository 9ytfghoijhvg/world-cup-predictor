# World Cup 2026 Knockout Predictor

I built a Random Forest model to predict FIFA World Cup 2026 knockout stage matches. It uses historical match data from 1930–2022, plus every international match since 1872 (~49,000 games) for form features. The model now also predicts actual scores using Poisson regression. Cross-validation shows 64.8% accuracy on winner prediction and ~0.9 goal error on scores.

## How to run it

```bash
python src/interactive_predictor_numbered.py
```

You get a numbered list of remaining Round of 32 matches. Pick one and the model gives you:
- Predicted score (e.g., 2-1)
- Win probabilities for each team
- Betting odds for context

```
======================================================================
England vs DR Congo
======================================================================
Betting Odds: England -380 | Draw +440 | DR Congo +1300

📊 Predicted Score: England 2-1 DR Congo
   (Expected: 1.6 - 0.7)

England advance: 93.6%
DR Congo advance: 6.4%

🏆 Prediction: England advance
======================================================================
```

## What the model actually does

**Data pipeline** (`features/prepare_data_with_international.py`) — Loads every World Cup knockout match since 1930 and 49,484 international matches for rolling form features. Builds 21 features total (was 14 in v1). The rolling goals against features are the most important at 12.6% each. Elo difference is critical at 12.1%. Added head-to-head records (1-2 year lookback before each match) and betting odds as features. Rolling xG, goals for, penalty rates, group stats, and host advantage round out the rest.

**Elo calculation** (`features/elo_calculator.py`) — I built a proper Elo calculator using official K-factors (World Cup = 60, continental tournaments = 50, qualifiers = 40, friendlies = 20) because the pre-made Elo dataset only had 2026 qualifiers. Now we have ratings for 333 teams across history.

**Training** (`training/train_and_save_model.py`) — Random Forest with 1000 trees trained on 552 knockout matches (was 48 before I added imputation). The model intelligently fills missing xG data with actual goals scored (highly correlated) and uses sensible defaults for other missing values. Cross-validation: 64.8% accuracy.

**Score prediction** (`training/train_score_predictor.py`) — Two Poisson regression models (one for home goals, one for away goals). Uses rolling goals, xG, and Elo features. Predicts within ~0.9 goals per team on average.

**Betting odds** (`features/betting_odds.py`) — Compiled odds from Fox Sports, FanDuel, OddsPortal, and Betfair for 2018, 2022, and 2026 knockout matches.

**Interactive CLI** (`src/interactive_predictor_numbered.py`) — Lists the remaining Round of 16 matches with betting odds, lets you pick which matchup to analyze.

**Prediction tracking** (`src/backtest_knockout.py`, `src/knockout_2026_results.py`) — Backtests all 32 knockout games with point-in-time features (each game uses only pre-kickoff data), logging predicted vs actual winner and score. Tracks winner accuracy and score MAE per round. Data stored in `data/prediction_log.csv`.

## Tournament performance (point-in-time backtest)

The full knockout stage is backtested by [src/backtest_knockout.py](src/backtest_knockout.py):
every game is predicted using only data available before kickoff (rolling form
evolves round by round), then compared to the official result. All 32 games are
logged to `data/prediction_log.csv` with a `round` column.

| Round | Winner accuracy | Score MAE |
|---|---|---|
| Round of 32 | 11/16 (69%) | 0.81 |
| Round of 16 | 6/8 (75%) | 1.12 |
| Quarter-finals | 3/4 (75%) | 0.50 |
| Semi-finals | 2/2 (100%) | 0.75 |
| Third place | 0/1 | 1.00 |
| Final | 0/1 | 1.00 |
| **Overall** | **22/32 (68.8%)** | **0.86** |

Misses: Germany→Paraguay, Netherlands→Morocco, Ecuador→Mexico, Croatia→Portugal,
Australia→Egypt (R32); Brazil→Norway, Colombia→Switzerland (R16); Belgium→Spain
(QF); plus the third-place and final upsets. The third-place match and final were
unplayed as of the results snapshot (flagged `verified_result=False` in the log).

Regenerate anytime with `python src/backtest_knockout.py`.

## Project structure

```
world-cup-predictor/
├── src/
│   ├── interactive_predictor_numbered.py   # run this
│   ├── log_r32_predictions.py              # new: tracks R32 predictions
│   ├── log_r16_predictions.py              # new: tracks R16 predictions
│   ├── prediction_tracker.py               # new: logs predictions to CSV
│   └── predict_2026.py
├── features/
│   ├── elo_calculator.py                   # new: calculate Elo for all teams
│   ├── rolling_goals_feature.py            # new: form from 49k matches
│   ├── rolling_xg_feature.py               # new: xG from all international matches
│   ├── head_to_head_feature.py             # new: opponent-specific stats
│   ├── betting_odds_feature.py             # new: odds as features
│   ├── betting_odds.py
│   ├── group_stage_feature.py
│   ├── knockout_history_feature.py
│   ├── penalty_feature.py
│   ├── prepare_data_with_international.py  # updated: 21 features, imputation
│   └── prepare_data.py
├── training/
│   ├── train_and_save_model.py             # new: saves model to disk
│   ├── train_score_predictor.py            # new: Poisson for scores
│   ├── retrain_with_live_data.py           # new: online learning from tournament
│   ├── train_random_forest.py
│   ├── cross_validate_models.py            # updated: compares v1 vs v2 vs v3
│   └── compare_models.py
├── models/
│   ├── random_forest_model.pkl             # pre-trained, loads instantly
│   ├── score_predictor_home.pkl
│   └── score_predictor_away.pkl
├── data/
│   ├── international_results.csv           # new: 49k matches 1872-2026
│   ├── elo_ratings_all_teams_2022.csv      # new: 333 teams
│   ├── elo_ratings_all_teams_2026.csv      # new: for predictions
│   ├── prediction_log.csv                  # new: live tournament tracking
│   ├── matches_1930_2022.csv
│   ├── knockout_matches_prepared.csv       # 552 samples, 21 features
│   ├── elo_ratings_wc2026.csv
│   └── group_stage_2026.csv
└── README.md
```

## What changed from v1

**v1 issues:**
- Only 48 training samples (88% of data had missing xG)
- Elo ratings only for 2026 qualifiers (Russia, Denmark, etc. missing)
- No score predictions
- Rolling features only from World Cup matches (too sparse)
- No head-to-head or betting odds features

**v2 fixes:**
- **552 training samples** (11.5x more) — Added intelligent imputation for missing xG (uses actual goals as proxy) and other features
- **Complete Elo ratings** — Calculated for all 333 teams using official K-factors from international match history
- **Score prediction** — Poisson regression models predict actual scorelines (~0.9 goal error)
- **Rolling goals from all internationals** — Uses 49,484 matches to calculate form (last 10 games) instead of just World Cups. These are the most important features at 12.4-12.6%.
- **Head-to-head features** — Win rate and goal difference between teams in 1-2 years before the World Cup (captures opponent-specific form)
- **Betting odds as features** — Converts moneyline to implied probability ("wisdom of crowds")
- **Prediction tracking** — Live tournament logging with accuracy and error metrics
- **Better accuracy** — 65% vs 58.9% on cross-validation, **80% on actual R32 matches**

## Feature importance (what the model relies on)

1. **Away team goals against** (12.5%) — How many goals they concede recently
2. **Home team goals against** (12.5%) — Defensive form is critical
3. **Elo difference** (12.1%) — Still the foundation
4. **Home rolling xG** (7.5%) — Expected goals from recent matches
5. **Away rolling xG** (7.5%) — Attacking threat
6. **Rolling goals for** (8.2% home, 7.6% away) — Offensive form
7-14. Penalty rates, group stage stats, host advantage, head-to-head stats (~5% each)

Defense matters most in knockouts. Teams that don't concede advance. Head-to-head records capture opponent-specific dynamics.

## Things I know are limited

- Only World Cup matches in training data (no friendlies/qualifiers for modeling, just for rolling features)
- No player-level data (injuries, suspensions, form)
- Score predictions round to integers (can't predict 2.3 goals)
- Cross-validation variance on 552 samples
- Head-to-head data sparse for early tournaments (only 1-2 matches for some historic pairings)

## Data sources

- Historical World Cup matches: Kaggle dataset (964 matches, 1930–2022)
- International matches: martj42/international_results GitHub (49,484 matches, 1872-2026)
- Elo ratings: Calculated using official K-factors from match history
- 2026 group stage results: Manually entered from FIFA
- Betting odds: Fox Sports, FanDuel, OddsPortal, Betfair archives

## Setup

```bash
git clone https://github.com/yourusername/world-cup-predictor.git
cd world-cup-predictor
pip install -r requirements.txt
python src/interactive_predictor_numbered.py
```

Needs Python 3.9+, scikit-learn, pandas, numpy.

Or test sample predictions:
```bash
python test_multiple_predictions.py
```

## What I'd add next

- Ensemble with gradient boosting (XGBoost/LightGBM likely +2-5% accuracy)
- Player-level data (top scorers, injuries via Transfermarkt)
- Tournament progression features (days rest, extra time in previous round)
- Confidence intervals on predictions
- Web interface (Flask/Streamlit)
- Live model retraining as tournament progresses (online learning from prediction log)

Built during the 2026 World Cup as a learning project. The model tracks live performance and improves through the tournament. MIT License.
