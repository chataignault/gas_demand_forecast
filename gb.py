# %%

import os 
import numpy as np
import polars as pl
import datetime as dt
from enum import Enum
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV 

IMG_FOLDER = Path("img")
if not os.path.exists(IMG_FOLDER):
    os.mkdir(IMG_FOLDER)

IMG_FOLDER = Path("img")
if not os.path.exists(IMG_FOLDER):
    os.mkdir(IMG_FOLDER)

class Month(int, Enum):
    Jan = 1
    Feb = 2
    Mar = 3
    Apr = 4
    May = 5
    Jun = 6
    Jul = 7
    Aug = 8
    Sep = 9
    Oct = 10
    Nov = 11
    Dec = 12

class Quarter(Enum):
    Q1 = set([Month.Jan, Month.Feb, Month.Mar])
    Q2 = set([Month.Apr, Month.May, Month.Jun])
    Q3 = set([Month.Jul, Month.Aug, Month.Sep])
    Q4 = set([Month.Oct, Month.Nov, Month.Dec])

TARGET_COL = 'demand'
DATE_COL = 'date'

# %%

train = pl.read_csv('train.csv').drop('id').with_columns(
    pl.col('date').str.to_date()
)

# add categorical level to include partial date information
for q in Quarter:
    train = train.with_columns(
        pl.col('date').dt.month().is_in(q.value).cast(pl.Int8).alias(q.name)
    )

features = [c for c in train.columns if c not in ['demand', 'date']]

y = train.select(pl.col(TARGET_COL)).to_numpy().T[0]
X = train.select(features).to_numpy()

# %%

# Set random seed for reproducibility
np.random.seed(42)

# Split data into 90% train and 10% holdout validation
n_samples = len(X)
indices = np.random.permutation(n_samples)
holdout_size = int(0.1 * n_samples)
train_idx = indices[holdout_size:]
holdout_idx = indices[:holdout_size]

X_train, y_train = X[train_idx], y[train_idx]
X_holdout, y_holdout = X[holdout_idx], y[holdout_idx]

# Define hyperparameter search space
param_distributions = {
    'n_estimators': [100, 200, 300, 500, 800],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'max_depth': [3, 4, 5, 6, 7],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'max_features': ['sqrt', 'log2', None],
}

# Base estimator
base_model = GradientBoostingRegressor(
    loss='squared_error',
    criterion='friedman_mse',
    random_state=42
)

# Set up 5-fold CV with shuffle
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# RandomizedSearchCV
print("Starting RandomizedSearchCV with 50 iterations...")
random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=50,
    cv=cv,
    scoring='neg_mean_squared_error',
    random_state=42,
    verbose=2,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("\n" + "="*60)
print("RandomizedSearchCV Results")
print("="*60)
print(f"Best CV Score (MSE): {-random_search.best_score_:.4f}")
print(f"Best CV Score (RMSE): {np.sqrt(-random_search.best_score_):.4f}")
print("\nBest Parameters:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")

# Evaluate on holdout set
best_model = random_search.best_estimator_
y_holdout_pred = best_model.predict(X_holdout)
holdout_mse = mean_squared_error(y_holdout, y_holdout_pred)
holdout_rmse = np.sqrt(holdout_mse)

print("\n" + "="*60)
print("Holdout Validation Results")
print("="*60)
print(f"Holdout MSE: {holdout_mse:.4f}")
print(f"Holdout RMSE: {holdout_rmse:.4f}")
print(f"CV vs Holdout RMSE difference: {abs(np.sqrt(-random_search.best_score_) - holdout_rmse):.4f}")

# Train final model on ALL training data with best parameters
print("\n" + "="*60)
print("Training final model on all training data...")
print("="*60)
model = GradientBoostingRegressor(**random_search.best_params_, random_state=42)
model.fit(X, y)

y_hat = model.predict(X)

dates = [d for d in train.select('date').to_numpy().T[0]]

# %%

def norm2(x:np.ndarray, y:np.ndarray):
    """Mean Squared Error"""
    assert len(x) == len(y)
    assert len(x) > 0
    return np.sqrt(np.mean((x - y)**2))

mean_squared_error(y_hat, y), norm2(y_hat, y)

# %%

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(dates, y, label='target', alpha=.8)
ax.plot(dates, y_hat, label='predicted', alpha=.8)

ax.grid()
ax.set_title("Gradient boost in-sample prediction vs target")

fig.savefig(IMG_FOLDER / 'gb_in_sample.png')

# %%

test = pl.read_csv('test.csv').drop('id').with_columns(
    pl.col('date').str.to_date()
)

# add categorical level to include partial date information
for q in Quarter:
    test = test.with_columns(
        pl.col('date').dt.month().is_in(q.value).cast(pl.Int8).alias(q.name)
    )


dates_test = [d for d in test.select('date').to_numpy().T[0]]

X_test = test.select(features).to_numpy()

y_hat_test = model.predict(X_test)

# %%

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(dates, y, label='in-sample')
ax.plot(dates_test, y_hat_test, label='OOS', alpha=.8)

ax.legend()
ax.grid()

ax.set_title("Gradient-boost in sample target and OOS prediction")

fig.savefig(IMG_FOLDER / 'gb_oos_prediction.png')

# %%

# Feature importance analysis
feature_importance = model.feature_importances_
feature_names = features

# Sort features by importance
sorted_idx = np.argsort(feature_importance)[::-1]
top_n = 20  # Show top 20 features

fig, ax = plt.subplots(figsize=(10, 8))

ax.barh(range(top_n), feature_importance[sorted_idx][:top_n])
ax.set_yticks(range(top_n))
ax.set_yticklabels([feature_names[i] for i in sorted_idx[:top_n]])
ax.invert_yaxis()
ax.set_xlabel('Feature Importance')
ax.set_title(f'Top {top_n} Most Important Features - Gradient Boosting')
ax.grid(axis='x', alpha=0.3)

fig.tight_layout()
fig.savefig(IMG_FOLDER / 'gb_feature_importance.png')

print("\n" + "="*60)
print(f"Top {top_n} Feature Importances:")
print("="*60)
for i in range(top_n):
    idx = sorted_idx[i]
    print(f"{i+1:2d}. {feature_names[idx]:30s}: {feature_importance[idx]:.4f}")
