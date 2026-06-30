# World Cup Match Predictor — Project Brief

## Goal
Build a binary classifier that predicts which team advances in FIFA World Cup knockout matches. Starting with the USA vs Bosnia Round of 32 match (July 1, 2026), then expanding to other sports (MLB, NBA, NFL, fantasy leagues).

## Key Design Decisions
- **Binary classification** (team advances vs eliminated), NOT three-way (win/draw/loss) — draws don't exist in knockout rounds
- Accuracy ceiling for soccer prediction is ~50-60% three-way; binary should be higher
- Pre-match features ONLY — no in-match stats like possession or shots (we don't have those before kickoff)
- Start simple (logistic regression), then upgrade to random forest / XGBoost

## Feature List (ranked by predictive power)
1. **Elo rating difference** — single most predictive feature per Groll et al. research
2. **Rolling xG average** (last 5-10 matches) — xG for and xG against; more predictive than actual goals
3. **Group stage goal difference**
4. **Squad market value difference** (Transfermarkt)
5. **Tournament form** — group stage points, goals scored/conceded
6. **Host advantage score** — scale: 2 (home country), 1 (co-host/nearby), 0 (neutral), -1/-2 (opponent perspective). USA gets a 2 for this tournament.
7. **Penalty shootout history** — relevant for knockout rounds specifically

## Research Findings
- Elo ratings beat FIFA rankings for prediction (Hvattum & Arntzen, 2010)
- xG difference outperforms raw goal difference and total-shots ratio
- Feature quality matters more than model choice (XGBoost vs RF vs logistic regression converge to similar ceilings)
- Gradient-boosted trees (XGBoost/CatBoost) are current best-in-class for tabular forecasting
- Bookmaker odds are nearly impossible to beat as a baseline
- Always use **time-based (chronological) train/test splits**, never random splits
- Host nations historically overperform at World Cups

## Datasets to Use

### Training Data (historical knockout matches)
- **Kaggle: "Football - FIFA World Cup, 1930-2026"** — `matches_1930_2022.csv`, all match results with stages
  - URL: https://www.kaggle.com/datasets/piterfm/fifa-football-world-cup
  - Filter to knockout matches from 2002 onward

### Elo Ratings
- **Kaggle: "2026 FIFA World Cup — Historical Elo Ratings"** — 4,683 rows, 1901-2026, 22 columns
  - URL: https://www.kaggle.com/datasets/afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings
  - Merge by team + year into match data

### 2026 Live Tournament Data
- **GitHub: FIFA-World-Cup-2026-Dataset** — updated daily, includes xG, match events, Elo, market values, 1,248 players
  - URL: https://github.com/mominullptr/FIFA-World-Cup-2026-Dataset
  - Use for current USA and Bosnia stats

### xG Data
- FBref.com, understat.com for historical xG
- RealGM xG Tracker for 2026: https://soccer.realgm.com/analysis/559/

## USA vs Bosnia Context (July 1, 2026 — Round of 32)

### USA (Group D winners, 2W-0D-1L, 8 GF / 4 GA)
- 4-1 vs Paraguay (1.35 xG)
- 2-0 vs Australia (1.08 xG)
- 2-3 vs Türkiye (2.06 xG, rotated squad)
- Cumulative xG differential: +0.31
- Playing at Levi's Stadium, Santa Clara (home soil)

### Bosnia (Group B 3rd place, 1W-1D-1L, 5 GF / 6 GA)
- 1-1 vs Canada (0.98 xG)
- 1-4 vs Switzerland (0.23 xG, red card collapse)
- 3-1 vs Qatar
- First ever knockout stage appearance

### Current Odds
- USA -185 moneyline, Bosnia +800, Draw +320
- USA -1.5 spread +110

## Pipeline Steps
1. ✅ Define features (done)
2. ⬜ Download historical knockout match data from Kaggle
3. ⬜ Merge in Elo ratings by team + year
4. ⬜ Add binary "advanced" column (1 = advanced, 0 = eliminated)
5. ⬜ Feature engineering (rolling xG, market value diff, host advantage)
6. ⬜ Train baseline model (logistic regression)
7. ⬜ Evaluate with chronological train/test split
8. ⬜ Upgrade to random forest / XGBoost
9. ⬜ Predict USA vs Bosnia
10. ⬜ Expand to other matches / sports

## Tech Stack
- Python 3, pandas, scikit-learn, matplotlib, numpy
- Jupyter notebook for exploration, .py for final model
- Virtual environment: `python3 -m venv venv && source venv/bin/activate`

## Academic References
- Groll, Ley, Schauberger & Van Eetvelde (2019) — hybrid random forest for World Cup prediction
- Hvattum & Arntzen (2010) — Elo rating effectiveness in ordered logit models
- arXiv:2403.07669 — ML for Soccer Match Result Prediction review
- arXiv:1906.01131 — Hybrid ML forecasts for FIFA Women's World Cup 2019
