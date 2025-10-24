

# %%
import os
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import datetime as dt
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
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

# %% Linear regression as benchmark

train = (
    pl.read_csv("train.csv").drop("id")
)

lin = LinearRegression(
    fit_intercept=True,
    copy_X=True
    )


# # feature engineer : log K temp
# log_k_temp_features = []
# for c in temp_features:
#     train = train.with_columns(np.log((pl.col(c) + TEMP_C_TO_K) / TEMP_C_TO_K))
#     log_k_temp_features.append(c+"_logk")

y = train.select(TARGET_COL).to_numpy().T[0]
X = train.select(pl.exclude([TARGET_COL, DATE_COL])).to_numpy()
x_mean = np.mean(X, axis=0).copy()

print(x_mean.shape)
print(X.shape)

lin.fit(X, y)

y_hat = lin.predict(X)

print(norm2(y, y_hat))
dates = [dt.date.fromisoformat(d) for d in train.select('date').to_numpy().T[0]]

del train

# %% Plot benchmark in-sample prediction

fig, ax = plt.subplots(
    figsize=(10, 5),
)

ax.plot(dates, y, label='target', alpha=.7)
ax.plot(dates, y_hat, label='predicted', alpha=.7)

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
print("Which is exactly one year.")
dates_test = [dt.date.fromisoformat(d) for d in test[DATE_COL].to_numpy()]

# %%
X_test = test[features].to_numpy()

# %% Relative difference of features

var_diff = (np.var(X, axis=0) - np.var(X_test, axis=0)) / np.var(X, axis=0)
var_diff = pd.Series(var_diff, index=features)
var_diff = var_diff.loc[var_diff.abs() > .05]
colours = var_diff.apply(lambda v : 'green' if v > 0. else 'red')
var_diff.sort_values().plot(
    kind='bar',
    title='OOS features, variance, relative difference',
    color=var_diff.sort_values().index.to_series().map(colours).to_list(),
    alpha=.5
    )

plt.tight_layout()

plt.savefig(IMG_FOLDER / 'oos_var_diff.png')

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

# %%
