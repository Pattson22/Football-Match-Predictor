import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import joblib

# --- 1. Load Data ---
try:
    df = pd.read_csv("processed_football_data.csv")
except FileNotFoundError:
    print("Error: processed_football_data.csv not found. Run prepare_data.py first.")
    exit()

print(f"Loaded {len(df)} rows.")
y = df['Outcome']
X = df.drop(columns=['Outcome'])
print(f"Features ({len(X.columns)}): {list(X.columns)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}  Test: {len(X_test)}\n")

# Balanced class weights — boosts draw recall without removing any data
sample_weights = compute_sample_weight('balanced', y_train)

# --- 2. Baseline: GradientBoosting ---
print("=" * 50)
print("Baseline: GradientBoosting")
print("=" * 50)
gb = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.05,
    max_depth=4, min_samples_leaf=20,
    subsample=0.8, random_state=42,
)
gb.fit(X_train, y_train, sample_weight=sample_weights)
gb_acc = accuracy_score(y_test, gb.predict(X_test)) * 100
print(f"Test accuracy: {gb_acc:.2f}%\n")

# --- 3. XGBoost with RandomizedSearch ---
print("=" * 50)
print("XGBoost — RandomizedSearchCV (50 iterations, 5-fold CV)")
print("=" * 50)

param_dist = {
    'n_estimators':     [200, 300, 500, 700, 1000],
    'learning_rate':    [0.01, 0.03, 0.05, 0.08, 0.1, 0.15],
    'max_depth':        [3, 4, 5, 6],
    'min_child_weight': [1, 3, 5, 10, 20],
    'subsample':        [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 1.0],
    'gamma':            [0, 0.05, 0.1, 0.3, 0.5, 1.0],
    'reg_alpha':        [0, 0.01, 0.1, 0.5, 1.0],
    'reg_lambda':       [0.5, 1.0, 1.5, 2.0, 5.0],
}

xgb_base = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

search = RandomizedSearchCV(
    xgb_base,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1,
    refit=True,
)

search.fit(X_train, y_train, sample_weight=sample_weights)

print(f"\nBest CV score: {search.best_score_ * 100:.2f}%")
print("Best params:")
for k, v in sorted(search.best_params_.items()):
    print(f"  {k:<22} {v}")

best_xgb = search.best_estimator_
xgb_acc  = accuracy_score(y_test, best_xgb.predict(X_test)) * 100
print(f"\nTest accuracy: {xgb_acc:.2f}%")

print("\nPer-class breakdown:")
print(classification_report(y_test, best_xgb.predict(X_test),
      target_names=['Home Win', 'Draw', 'Away Win'], digits=3))

# --- 4. Pick the winner ---
print("=" * 50)
print(f"GradientBoosting: {gb_acc:.2f}%")
print(f"XGBoost (tuned):  {xgb_acc:.2f}%")

if xgb_acc >= gb_acc:
    winner       = best_xgb
    winner_acc   = xgb_acc
    winner_name  = "XGBoost"
else:
    winner       = gb
    winner_acc   = gb_acc
    winner_name  = "GradientBoosting"

print(f"Winner: {winner_name}  ({winner_acc:.2f}%)")
print("=" * 50)

# --- 5. Feature importances ---
print("\nFeature importances:")
importances = (best_xgb.feature_importances_
               if winner_name == "XGBoost"
               else gb.feature_importances_)
for feat, imp in sorted(zip(X.columns, importances), key=lambda x: -x[1]):
    bar = '#' * int(imp * 50)
    print(f"  {feat:<22} {imp:.3f}  {bar}")

# --- 6. Cross-validation on winner ---
print("\nRunning 5-fold CV on winner...")
cv = cross_val_score(winner, X, y, cv=5, scoring='accuracy', n_jobs=-1)
print(f"CV Accuracy: {cv.mean()*100:.2f}% (+/- {cv.std()*100:.2f}%)")

# --- 7. Save ---
with open("accuracy.txt", "w") as f:
    f.write(f"{cv.mean()*100:.2f}")

joblib.dump({'model': winner, 'features': list(X.columns)}, 'football_model.joblib')
print(f"\nSaved {winner_name} model ({winner_acc:.2f}%) to football_model.joblib")
