

# %%

import numpy as np
import pandas as pd
import polars as pl
import datetime as dt
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


TARGET_COL = 'demand'
DATE_COL = 'date'

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

# %%

train['demand'].plot(
    figsize=(10, 5),
    grid=True,
    title="Target value : daily demand"
)

plt.tight_layout()

del train

# %% Define first objective function

def squared_error(x:np.ndarray, y:np.ndarray):
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
    copy_X=False
    )

y = train.select(TARGET_COL).to_numpy().T[0]
X = train.select(pl.exclude([TARGET_COL, DATE_COL])).to_numpy()
x_mean = np.mean(X, axis=0)

print(x_mean.shape)
print(X.shape)

lin.fit(X, y)

y_hat = lin.predict(X+x_mean)

print(squared_error(y, y_hat))
dates = [dt.date.fromisoformat(d) for d in train.select('date').to_numpy().T[0]]



# del train
# %% Plot benchmark in-sample prediction


fig, ax = plt.subplots(
    figsize=(10, 5),
)

ax.plot(dates, y, label='target')
ax.plot(dates, y_hat, label='predicted')

ax.legend()

fig.tight_layout()

print(f"Squared error : {squared_error(y, y_hat)}")

# %% Test dataframe prediction

test = pd.read_csv('test.csv')
test_start, test_end = test[DATE_COL].agg(['min', 'max'])

dates_test = [dt.date.fromisoformat(d) for d in test[DATE_COL].to_numpy()]



# %%
X_test = test[[c for c in test.columns if c not in ['date', 'id']]].to_numpy()

# %% Relative difference of features

var_diff = (np.var(X, axis=0) - np.var(X_test, axis=0)) / np.var(X, axis=0)
pd.Series(var_diff).sort_values().plot(kind='bar')

# %%

predicted_demand = lin.predict(X_test - x_mean)

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(dates_test, predicted_demand)

plt.tight_layout()

#%% Plot in-sample with out-of-sample

fig, ax = plt.subplots(figsize=(10, 5))


ax.plot(dates, y, label='IS')
ax.plot(dates_test, predicted_demand, c='orange', label='OOS')

fig.tight_layout()

# %%
