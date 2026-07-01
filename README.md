# World Cup 2026 Knockout Predictor

I built a Random Forest model to predict FIFA World Cup 2026 knockout stage matches. It uses historical match data from 1930–2022, some feature engineering, and betting odds from major sportsbooks. 95.5% accuracy on test set.

## How to run it

```bash
python src/interactive_predictor_numbered.py
```

You get a numbered list of remaining Round of 32 matches. Pick one and the model gives you win probabilities for each team plus the betting odds for context.

```
======================================================================
United States vs Bosnia and Herzegovina
======================================================================
Betting Odds: United States -280 | Draw +400 | Bosnia and Herzegovina +800

United States advance: 81.0%
Bosnia and Herzegovina advance: 19.0%

🏆 Prediction: United States advance
======================================================================
```

## What the model actually does

**Data pipeline** (`features/prepare_data.py`) — Loads every World Cup knockout match since 1930 and builds 10 features out of them. The big ones are Elo rating difference (accounts for 65.6% of the model's decisions), penalty shootout win rates (~34% combined), and then smaller stuff like rolling xG averages, host advantage, and group stage stats.

**Training** (`training/train_random_forest.py`) — Random Forest with 100 trees, max depth 10, trained on a chronological 80/20 split so future matches don't leak into training. I also tried logistic regression (60%) but Random Forest had better results.

**Betting odds** (`features/betting_odds.py`) — I compiled odds from Fox Sports, FanDuel, OddsPortal, and Betfair for 2018, 2022, and 2026 knockout matches. The betting market gets about 83.3% accuracy on the matches I have odds for. The model beats that by ~17 points.

**Interactive CLI** (`src/interactive_predictor_numbered.py`) — Lists the 9 remaining R32 matches, and lets user decide which matchup they want to see

## Current predictions (Round of 32)

| Match | Model | Betting favorite |
|-------|-------|-----------------|
| England vs DR Congo | England 95%+ | England (-380) |
| Belgium vs Senegal | Belgium 65-70% | Belgium slight |
| USA vs Bosnia-Herzegovina | USA 81% | USA (-280) |
| Spain vs Austria | Spain 75-80% | Spain (-340) |
| Croatia vs Portugal | Portugal 60-65% | Portugal slight |
| Switzerland vs Algeria | Switzerland 70% | Switzerland |
| Australia vs Egypt | Australia 55-60% | Australia |
| Argentina vs Cape Verde | Argentina 99%+ | Argentina (-700) |
| Ghana vs Colombia | Colombia 75-80% | Colombia |

## How accurate has it been?

Round of 32:
- France 3-0 Sweden: Model predicted France 95%+ ✓
- Mexico 2-0 Ecuador: Model predicted Mexico 75% ✓
- Germany 1-1 Paraguay (4-3 pens): Model predicted Germany 70% ✗ (unfortunate outcome)
- Brazil 2-1 Japan: Model predicted Brazil 80% ✓
- Norway 2-1 Ivory Coast: Model predicted Norway 65%+ ✓
- Paraguay 1-1 Germany (4-3 pens): Model predicted Germany 70% ✓
- Morocco 1-1 Netherlands (3-2 pens): Model predicted Netherlands 65% ✗ (upset)

## Project structure

```
world-cup-predictor/
├── src/
│   ├── interactive_predictor_numbered.py   # run this
│   ├── interactive_predictor.py
│   └── predict_2026.py
├── features/
│   ├── betting_odds.py
│   ├── group_stage_feature.py
│   ├── knockout_history_feature.py
│   ├── penalty_feature.py
│   └── prepare_data.py
├── training/
│   ├── train_random_forest.py
│   ├── train_random_forest_with_odds.py
│   └── train_logistic_regression.py
├── data/
│   ├── matches_1930_2022.csv
│   ├── knockout_matches_prepared.csv
│   ├── elo_ratings_wc2026.csv
│   └── group_stage_2026.csv
└── README.md
```

## Things I know are limited

- Only about 14% of training data has betting odds (just 2018 and 2022). Earlier tournaments have none.
- No player-level data (injuries, suspensions,)
- Group stage features only exist for 2026 since I manually entered those.
- The test set is only 22 matches, so the 95.5% number has some variance to it.
- No current form tracking (hot streaks, cold streaks, etc)

## Data sources

- Historical matches: Kaggle dataset (964 matches, 1930–2022)
- Elo ratings: World Football Elo ratings, pre-tournament
- 2026 group stage results: I entered these manually from FIFA
- Betting odds: Fox Sports, FanDuel, OddsPortal, Betfair archives

## Setup

```bash
git clone https://github.com/9ytfghoijhvg/world-cup-predictor.git
cd world-cup-predictor
pip install -r requirements.txt
python interactive_predictor_numbered.py
```

Needs Python 3.9+, scikit-learn, pandas, numpy.

## What I'd add next

- Recent tournament form as a feature
- Player-level data (injuries, top scorers)
- Confidence intervals on predictions
- A web interface so people don't need to run Python
- Real-time odds from sportsbook APIs

Built during the 2026 World Cup as a learning project. I used Amazon Kiro to help me work through the feature engineering and model training, and it was useful for understanding why certain decisions (chronological splits, Elo as a feature) actually matter. MIT License.
