import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 1. Load dataset
df = pd.read_csv("data/world_cup_ready.csv")

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

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. XGBoost model
model_xgb = XGBClassifier(
    n_estimators=100, 
    max_depth=5, 
    learning_rate=0.1, 
    random_state=42, 
    eval_metric='mlogloss'
)

# 5. 5-Fold Cross-Validation
cv_scores = cross_val_score(model_xgb, X_train, y_train, cv=5, scoring='accuracy')

print("\n--- Cross-Validation Results ---")
print(f"Scores per fold: {cv_scores * 100}")
print(f"Mean CV Accuracy: {cv_scores.mean() * 100:.2f}%")
print(f"CV Standard Deviation: {cv_scores.std() * 100:.2f}%")
print("-" * 32)

# 6. Final training
model_xgb.fit(X_train, y_train)

# 7. Evaluate on Test Set
y_pred = model_xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n======================================")
print(f"Test Set Accuracy: {accuracy * 100:.2f}%")
print("======================================")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Loss (0)', 'Draw (1)', 'Win (2)']))

# 8. Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model_xgb, "models/football_model_xgb.pkl")
print("\n✅ Model saved to models/football_model_xgb.pkl")
