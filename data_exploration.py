

# %%
import os
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import datetime as dt
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from enum import Enum
from IPython.display import display

TARGET_COL = 'demand'
DATE_COL = 'date'

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

TEMP_C_TO_K = 273.15


# %% Dataframe analysis

train = pd.read_csv("train.csv")

# data checks
assert train["date"].is_unique 
assert train["id"].is_unique 

print("Non floating-point columns :")
print(train.dtypes.pipe(lambda x: x.loc[x != "float64"]).index.to_list())

start_date, end_date = train["date"].agg(['min', 'max'])

print("\nTrain dates :")
print(start_date, end_date)

print("\nNumber of days missing days in the date range :")
print((
    len(train) - len(pd.date_range(start=start_date, end=end_date)
)))
print("One and only one data point per day (weekends included)")

print(f"\nNumber of columns : {len(train.columns)}")
print("Types of features :")
print(set(['_'.join((x.split('_')[:-1])) for x in train.columns if x not in ['date', 'id', 'demand']]))

train['date'] = train['date'].apply(lambda d: dt.date.fromisoformat(d))

train = train.set_index("date")

features = [c for c in train.columns if c not in ['id', 'demand']]

pos = (train[features].min() > 0.).reset_index().rename(columns={'index': 'name', 0: 'is_positive'})
pos['base_name'] = pos['name'].str.split('_').apply(lambda l : "_".join(l[:-1]))
print("\nFeatures that are positive :")
display(pos.groupby('base_name')['is_positive'].all().reset_index())

temp_features = [f for f in features if f.startswith("temp")]

train[temp_features] += TEMP_C_TO_K

train['demand'].plot(
    figsize=(10, 5),
    grid=True,
    title="Target value : daily demand"
)

plt.tight_layout()

plt.savefig(IMG_FOLDER / 'train_demand.png')

del train

# %% Define first objective function

def norm2(x:np.ndarray, y:np.ndarray):
    """Mean Squared Error"""
    assert len(x) == len(y)
    assert len(x) > 0
    return np.sqrt(np.mean((x - y)**2))

# %% Linear regression as benchmark : feature set definition

train = (
    pl.read_csv("train.csv").drop("id")
).with_columns(
    pl.col('date').str.to_date()
)
# feature engineer : log K temp
log_k_temp_features = []
for c in temp_features:
    train = train.with_columns((np.log((pl.col(c) + TEMP_C_TO_K) / TEMP_C_TO_K)).alias(c+"_logk"))
    log_k_temp_features.append(c+"_logk")

# feature engineer : add level per quarter :
for q in Quarter:
    train = train.with_columns(
        pl.col('date').dt.month().is_in(q.value).cast(pl.Int8).alias(q.name)
    )

y = train.select(TARGET_COL).to_numpy().T[0]
X = train.select(pl.exclude([TARGET_COL, DATE_COL])).to_numpy()
x_mean = np.mean(X, axis=0).copy()

print("Shape of the final feature set :", X.shape)

# %% Feature Importance Analysis with Standardized Features

print("\n" + "="*60)
print("Feature Importance Analysis (with standardized features)")
print("="*60)

# Copy X to avoid interfering with training data
X_analysis = X.copy()

# Get feature names from the train dataframe
feature_names = [col for col in train.columns if col not in [TARGET_COL, DATE_COL]]

# Standardize features (mean=0, std=1)
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X_analysis)

# Fit Lasso with cross-validation on standardized data
param_grid_analysis = {
    'alpha': np.logspace(-4, 2, 50)
}

kfold_analysis = KFold(n_splits=5, shuffle=True, random_state=42)
lasso_analysis = Lasso(fit_intercept=True, max_iter=10000)

grid_search_analysis = GridSearchCV(
    estimator=lasso_analysis,
    param_grid=param_grid_analysis,
    cv=kfold_analysis,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=0
)

grid_search_analysis.fit(X_standardized, y)

# Get the best model and coefficients
best_lasso = grid_search_analysis.best_estimator_
coefficients = best_lasso.coef_

# Create feature importance table
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coefficients,
    'abs_coefficient': np.abs(coefficients)
})

# Add indicator for retained features (non-zero coefficients)
feature_importance['retained'] = feature_importance['abs_coefficient'] > 0

# Sort by absolute coefficient value
feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)

print(f"\nBest alpha for standardized features: {grid_search_analysis.best_params_['alpha']:.6f}")
print(f"Best CV RMSE: {-grid_search_analysis.best_score_:.4f}")
print(f"\nNumber of retained features: {feature_importance['retained'].sum()} / {len(feature_names)}")
print(f"Number of zeroed-out features: {(~feature_importance['retained']).sum()}")

print("\n--- Top 20 Most Important Features (by absolute coefficient) ---")
display(feature_importance.head(20))

print("\n--- All Retained Features ---")
retained_features = feature_importance[feature_importance['retained']]
display(retained_features)

print("="*60 + "\n")

retained_features["feature category"] = retained_features["feature"].str.split("_").apply(lambda x: x[0])
retained_features.groupby("feature category")["abs_coefficient"].mean().sort_values(ascending=False).to_frame()

# %% Main Model Training (on original scale)

# Define parameter grid for Lasso regularization
param_grid = {
    'alpha': np.logspace(-4, 2, 50)  # Test alpha values from 0.0001 to 100
}

# Set up 5-fold cross-validation with shuffling
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Create Lasso model
lasso = Lasso(fit_intercept=True, copy_X=True, max_iter=10000)

# Perform grid search with cross-validation
grid_search = GridSearchCV(
    estimator=lasso,
    param_grid=param_grid,
    cv=kfold,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

# Fit the grid search
grid_search.fit(X, y)

# Get the best model
lin = grid_search.best_estimator_
print(f"\nBest alpha: {grid_search.best_params_['alpha']:.6f}")
print(f"Best CV RMSE: {-grid_search.best_score_:.4f}")

y_hat = lin.predict(X)

print(norm2(y, y_hat))
dates = [d for d in train.select('date').to_numpy().T[0]]

del train

# %% Plot benchmark in-sample prediction

fig, ax = plt.subplots(
    figsize=(10, 5),
)

ax.plot(dates, y, label='target', alpha=.7)
ax.plot(dates, y_hat, label='predicted (lasso)', alpha=.7)

ax.grid()
ax.legend()
ax.set_title("In-sample gas demand")
fig.tight_layout()

fig.savefig(IMG_FOLDER / 'linear_prediction.png')

print(f"Root mean squared error : {norm2(y, y_hat)}")

# %% Test dataframe prediction

test = pd.read_csv('test.csv')
test_start, test_end = test[DATE_COL].agg(['min', 'max'])

print(f"\nOut-of-sample data from {test_start} to {test_end}")
print("Which is exactly two years.")
# dates_test = [dt.date.fromisoformat(d) for d in test[DATE_COL].to_numpy()]
dates_test = [d for d in test[DATE_COL].to_numpy()]

# %%
test = pl.from_dataframe(test).with_columns(
    pl.col('date').str.to_date()
)
for c in temp_features:
    test = test.with_columns((np.log((pl.col(c) + TEMP_C_TO_K) / TEMP_C_TO_K)).alias(c+"_logk"))

# feature engineer : add level per quarter :
for q in Quarter:
    test = test.with_columns(
        pl.col('date').dt.month().is_in(q.value).cast(pl.Int8).alias(q.name)
    )

X_test = test.to_pandas()[features+log_k_temp_features+[q.name for q in Quarter]].to_numpy()

# %% 

predicted_demand = lin.predict(X_test)

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(dates_test, predicted_demand)

plt.tight_layout()

#%% Plot in-sample with out-of-sample

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(dates, y, label='IS', alpha=1.)
# ax.plot(dates, y_hat, label='IS-pred', alpha=.4)
ax.plot(dates_test, predicted_demand, c='orange', label='OOS-pred', alpha=.8)

ax.grid()
ax.legend()

ax.set_title("Linear regression OOS prediction after training data")

fig.tight_layout()

fig.savefig(IMG_FOLDER / 'lin_oos.png')

