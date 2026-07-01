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

**Data pipeline** (`features/prepare_data_with_international.py`) — Loads every World Cup knockout match since 1930 and 49,484 international matches for rolling form features. Builds 14 features total. The new rolling goals against features (how many goals teams concede in their last 10 internationals) are the most important at 12-13% each. Elo difference is still critical at 12%. Then rolling xG, goals for, penalty win rates, group stats, and host advantage.

**Elo calculation** (`features/elo_calculator.py`) — I built a proper Elo calculator using official K-factors (World Cup = 60, continental tournaments = 50, qualifiers = 40, friendlies = 20) because the pre-made Elo dataset only had 2026 qualifiers. Now we have ratings for 333 teams across history.

**Training** (`training/train_and_save_model.py`) — Random Forest with 1000 trees trained on 552 knockout matches (was 48 before I added imputation). The model intelligently fills missing xG data with actual goals scored (highly correlated) and uses sensible defaults for other missing values. Cross-validation: 64.8% accuracy.

**Score prediction** (`training/train_score_predictor.py`) — Two Poisson regression models (one for home goals, one for away goals). Uses rolling goals, xG, and Elo features. Predicts within ~0.9 goals per team on average.

**Betting odds** (`features/betting_odds.py`) — Compiled odds from Fox Sports, FanDuel, OddsPortal, and Betfair for 2018, 2022, and 2026 knockout matches.

**Interactive CLI** (`src/interactive_predictor_numbered.py`) — Lists the 9 remaining R32 matches with betting odds, lets you pick which matchup to analyze.

## Current predictions (Round of 32)

| Match | Model | Predicted Score | Betting favorite |
|-------|-------|-----------------|------------------|
| England vs DR Congo | England 94% | 2-1 | England (-380) |
| Belgium vs Senegal | Belgium 65% | 2-1 | Belgium slight |
| USA vs Bosnia-Herzegovina | USA 76% | 2-1 | USA (-280) |
| Spain vs Austria | Spain 78% | 2-1 | Spain (-340) |
| Croatia vs Portugal | Portugal 58% | 1-2 | Portugal slight |
| Switzerland vs Algeria | Switzerland 72% | 2-1 | Switzerland |
| Australia vs Egypt | Australia 56% | 2-1 | Australia |
| Argentina vs Cape Verde | Argentina 98% | 3-0 | Argentina (-700) |
| Ghana vs Colombia | Colombia 73% | 1-2 | Colombia |

## Sample predictions (big matchups)

```
Brazil vs Argentina: Argentina 58% (1-2)
England vs Germany: Germany 52% (1-2)
Spain vs France: Spain 54% (1-1)
Argentina vs France (2022 rematch): Argentina 54% (1-1)
```

## Project structure

```
world-cup-predictor/
├── src/
│   ├── interactive_predictor_numbered.py   # run this
│   └── predict_2026.py
├── features/
│   ├── elo_calculator.py                   # new: calculate Elo for all teams
│   ├── rolling_goals_feature.py            # new: form from 49k matches
│   ├── betting_odds.py
│   ├── group_stage_feature.py
│   ├── knockout_history_feature.py
│   ├── penalty_feature.py
│   ├── prepare_data_with_international.py  # new: with imputation
│   └── prepare_data.py
├── training/
│   ├── train_and_save_model.py             # new: saves model to disk
│   ├── train_score_predictor.py            # new: Poisson for scores
│   ├── train_random_forest.py
│   └── compare_models.py
├── models/
│   ├── random_forest_model.pkl             # pre-trained, loads instantly
│   ├── score_predictor_home.pkl
│   └── score_predictor_away.pkl
├── data/
│   ├── international_results.csv           # new: 49k matches 1872-2026
│   ├── elo_ratings_all_teams_2022.csv      # new: 333 teams
│   ├── elo_ratings_all_teams_2026.csv      # new: for predictions
│   ├── matches_1930_2022.csv
│   ├── knockout_matches_prepared.csv       # 552 samples now
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

**v2 fixes:**
- **552 training samples** (11.5x more) — Added intelligent imputation for missing xG (uses actual goals as proxy) and other features
- **Complete Elo ratings** — Calculated for all 333 teams using official K-factors from international match history
- **Score prediction** — Poisson regression models predict actual scorelines (~0.9 goal error)
- **Rolling goals from all internationals** — Uses 49,484 matches to calculate form (last 10 games) instead of just World Cups. These are now the most important features.
- **Better accuracy** — 64.8% vs 58.9% on cross-validation

## Feature importance (what the model relies on)

1. **Away team goals against** (12.6%) — How many goals they concede recently
2. **Home team goals against** (12.4%) — Defensive form is critical
3. **Elo difference** (12.1%) — Still the foundation
4. **Rolling xG** (11.3% home, 10.9% away) — Expected goals from recent matches
5. **Rolling goals for** (8.2% home, 8.2% away) — Offensive form
6-10. Penalty rates, group stage stats, host advantage (~5% each)

Defense matters more than offense in knockouts. Teams that don't concede advance.

## Things I know are limited

- Only World Cup matches in training data (no friendlies/qualifiers for modeling, just for rolling features)
- No player-level data (injuries, suspensions, form)
- Score predictions round to integers (can't predict 2.3 goals)
- Cross-validation variance on 552 samples
- No live form tracking during tournament

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
- Head-to-head historical records between specific teams
- Confidence intervals on predictions
- Web interface (Flask/Streamlit)

Built during the 2026 World Cup as a learning project. I used Amazon Kiro to help with feature engineering, Elo calculation, data imputation, and model training. The rolling goals insight (most important features) came from exploring what actual matters in knockout football. MIT License.
