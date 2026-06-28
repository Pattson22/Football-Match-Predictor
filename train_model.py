import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
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

# --- 2. XGBoost with fixed good params (fast training path) ---
print("=" * 50)
print("XGBoost — fixed hyperparams (fast training)")
print("=" * 50)

best_xgb = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.7,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

best_xgb.fit(X_train, y_train, sample_weight=sample_weights)
xgb_acc = accuracy_score(y_test, best_xgb.predict(X_test)) * 100
print(f"Test accuracy: {xgb_acc:.2f}%")

print("\nPer-class breakdown:")
print(classification_report(y_test, best_xgb.predict(X_test),
      target_names=['Home Win', 'Draw', 'Away Win'], digits=3))

winner      = best_xgb
winner_acc  = xgb_acc
winner_name = "XGBoost"
print("=" * 50)

# --- 5. Feature importances ---
print("\nFeature importances:")
importances = best_xgb.feature_importances_
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
