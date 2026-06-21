import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 1. Load dataset
df = pd.read_csv("data/world_cup_ready.csv")

# Sort by date
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# 2. Define features and target
features = [
    'points_diff', 
    'form_diff', 
    'streak_diff', 
    'tournament_weight', 
    'is_neutral',
    'home_h2h_win_rate',
    'away_h2h_win_rate'
]

X = df[features]
y = df['target_result']

# 3. Time-Based split (80% Train - 20% Test)
split_index = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print(f"Total matches: {len(df)}")
print(f"Train set (Past): {len(X_train)} matches (Until {df['date'].iloc[split_index].strftime('%Y-%m-%d')})")
print(f"Test set (Future): {len(X_test)} matches (From {df['date'].iloc[split_index].strftime('%Y-%m-%d')} onwards)")

# 4. Initialize XGBoost model
model_xgb = XGBClassifier(
    n_estimators=100, 
    max_depth=4, 
    learning_rate=0.05, 
    random_state=42, 
    eval_metric='mlogloss'
)

# 5. Time-Series Cross-Validation (5-Fold)
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(model_xgb, X_train, y_train, cv=tscv, scoring='accuracy')

print("\n--- Cross-Validation Results ---")
print(f"Scores per fold: {cv_scores * 100}")
print(f"Mean CV Accuracy: {cv_scores.mean() * 100:.2f}%")
print(f"CV Standard Deviation: {cv_scores.std() * 100:.2f}%")
print("-" * 32)

# 6. Final training
model_xgb.fit(X_train, y_train)

# 7. Evaluate on Future Test Set
y_pred = model_xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n======================================")
print(f"Future Test Set Accuracy: {accuracy * 100:.2f}%")
print("======================================")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Loss (0)', 'Draw (1)', 'Win (2)']))

# 8. Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model_xgb, "models/football_model_xgb_timebased.pkl")