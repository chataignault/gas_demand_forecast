# %%

import os 
from pathlib import Path
import pandas as pd
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor
from IPython.display import display
from matplotlib import pyplot as plt

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

# %%
from sklearn.metrics import mean_squared_error
import numpy as np

def norm2(x:np.ndarray, y:np.ndarray):
    """Mean Squared Error"""
    assert len(x) == len(y)
    assert len(x) > 0
    return np.sqrt(np.mean((x - y)**2))

mean_squared_error(y_hat, y), norm2(y_hat, y)
