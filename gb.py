# %%

import os 
from pathlib import Path
import pandas as pd
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor
from IPython.display import display
from matplotlib import pyplot as plt
import datetime as dt
from sklearn.metrics import mean_squared_error
import numpy as np

IMG_FOLDER = Path("img")
if not os.path.exists(IMG_FOLDER):
    os.mkdir(IMG_FOLDER)

TARGET_COL = 'demand'

# %%

train = pl.read_csv('train.csv').drop('id')
features = [c for c in train.columns if c not in ['demand', 'date']]

# %%

y = train.select(pl.col(TARGET_COL)).to_numpy().T[0]
X = train.select(features).to_numpy()

# %%

model = GradientBoostingRegressor(
    loss='squared_error',
    learning_rate=0.1,
    n_estimators=100,
    subsample=1.,
    criterion='friedman_mse',
    min_samples_split=2,
    max_depth=3,
    max_features=None,
)


model.fit(X, y)

y_hat = model.predict(X)

dates = [dt.date.fromisoformat(d) for d in train.select('date').to_numpy().T[0]]


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

test = pl.read_csv('test.csv').drop('id')

dates_test = [dt.date.fromisoformat(d) for d in test.select('date').to_numpy().T[0]]

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
