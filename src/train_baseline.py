import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
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

# 4. Initialize Pipeline (Imputer + Scaler + Logistic Regression)
baseline_model = make_pipeline(
    SimpleImputer(strategy='constant', fill_value=0),
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

# 5. 5-Fold Cross-Validation
print("\nRunning 5-Fold Cross-Validation...")
cv_scores = cross_val_score(baseline_model, X_train, y_train, cv=5, scoring='accuracy')

print("\n--- Cross-Validation Results ---")
print(f"Scores per fold: {cv_scores * 100}")
print(f"Mean CV Accuracy: {cv_scores.mean() * 100:.2f}%")
print(f"CV Standard Deviation: {cv_scores.std() * 100:.2f}%")
print("-" * 32)

# 6. Final training
baseline_model.fit(X_train, y_train)

# 7. Evaluate on Test Set
y_pred = baseline_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n======================================")
print(f"Test Set Accuracy: {accuracy * 100:.2f}%")
print("======================================")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Loss (0)', 'Draw (1)', 'Win (2)']))

# 8. Save model
os.makedirs("models", exist_ok=True)
joblib.dump(baseline_model, "models/football_model_baseline.pkl")