# World Cup 2026 Predictor

A Machine Learning powered simulator for the FIFA World Cup 2026. This project uses historical match data, team form, and FIFA rankings to predict match outcomes and simulate the entire tournament from the Group Stage to the Grand Final.

## Simulator Logic & Methodology

The core idea behind this simulator is a **hybrid prediction approach** combining machine learning with real-world progression:

1. **Phase-by-Phase Prediction:** Initially, the simulator uses an trained model (XGBoost) to predict the outcome of every single match in the tournament. Based on the team stats (points difference, form, streak, head-to-head records), it calculates the win/draw/loss probabilities for the Group Stage and the advancing team for the Knockout Stages.
2. **Real Results Injection:** As the actual World Cup progresses, we feed the **real match results** back into the simulator. By updating the played matches (in both the Group Stage and Knockouts), the simulator dynamically adjusts the tournament bracket and standings.
3. **Adaptive Forecasting:** By constantly replacing past predictions with factual data, the simulator recalculates the remaining future bracket. This allows us to find the most accurate and realistic path to the final, adapting to upsets and unexpected early eliminations.

**Initial Pure Prediction:** Before introducing any real-world results into the dataset, the model's pure initial prediction for the World Cup 2026 Champion was **Argentina**.

## Machine Learning Model & Features

The simulation is powered by an **XGBoost Classifier** (`XGBClassifier`), which was trained using a Time-Series Cross-Validation split to ensure realistic forecasting without data leakage. 

For every match, the model evaluates the following engineered features:

| Feature Name | Description |
| :--- | :--- |
| `points_diff` | Difference in FIFA ranking points between the two teams |
| `form_diff` | Difference in the teams' recent form (based on their last 5 matches) |
| `streak_diff` | Difference in their current winning/losing streak |
| `tournament_weight` | Importance multiplier based on the tournament type (e.g., Friendlies vs World Cup) |
| `is_neutral` | Boolean (0 or 1) indicating if the match is played on neutral ground |
| `home_h2h_win_rate` | Historical Head-to-Head win rate for the "Home" team against this opponent |
| `away_h2h_win_rate` | Historical Head-to-Head win rate for the "Away" team against this opponent |

## How to Update with Real Results

The main simulation logic is located in `src/simulator.py`. 

To update the simulator with real-world results as the tournament happens, edit the following lists inside the file:

- **Group Stage:** Add the match scores to the `real_results` list.
  ```python
  ('France', 'Senegal', 3, 1) # Home, Away, Home Goals, Away Goals
  ```
- **Knockout Stage:** Add the advancing teams to the `real_knockout_results` list.
  ```python
  ('Mexico', 'Canada', 'Mexico') # Team 1, Team 2, Winner
  ```

When running the simulator, the console output will clearly tag matches as `[REAL]` (based on your inputted data) or `[PREDICTED]` (computed by the ML model).

## Project Pipeline & Structure

The repository follows a clean, three-step data science pipeline:

1. **Preprocessing (`src/preprocessing.py`)**: Fetches, cleans, and structures historical football data, merging it with FIFA rankings to compute the necessary features.
2. **Training (`src/train_xgb_timebased.py`)**: Trains the main XGBoost model using Time-Series Cross-Validation and saves the model artifact to the `models/` directory. *(Alternative models like Random Forest and Baseline algorithms are also available in the `src/` folder for comparison).*
3. **Simulation (`src/simulator.py`)**: The main execution script that loads the trained model, simulates the tournament logic, applies real results, and predicts the Knockout brackets.

## Installation & Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the simulator:**
   ```bash
   python src/simulator.py
   ```
