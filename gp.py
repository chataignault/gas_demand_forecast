# %%

import os
from pathlib import Path
import pandas as pd
import polars as pl
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C, ExpSineSquared
from sklearn.preprocessing import StandardScaler
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

print(f"Number of training samples: {len(train)}")
print(f"Number of features: {len(features)}")

# %%

y = train.select(pl.col(TARGET_COL)).to_numpy().T[0]
X = train.select(features).to_numpy()

# Normalize features for GP (critical for performance)
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

print(f"X shape: {X_scaled.shape}")
print(f"y shape: {y_scaled.shape}")

# %% Define kernel for seasonal gas demand

# Composite kernel structure:
# 1. RBF and Periodic: captures smooth seasonal variations (yearly cycle)
# 2. RBF: captures longer-term trends and non-periodic variations
# 3. WhiteKernel: measurement noise

# Periodic kernel for yearly seasonality
# Period should be around 365 days (converted to feature space)
# We use a large length_scale_bounds to allow optimization
periodic_kernel = ExpSineSquared(
    length_scale=1.0,
    periodicity=1.0,
    length_scale_bounds=(1e-2, 1e3),
    periodicity_bounds=(0.5, 2.0)
)

# RBF kernel for smooth variations
rbf_seasonal = RBF(
    length_scale=1.0,
    length_scale_bounds=(1e-2, 1e3)
)

# Product kernel for smooth seasonal pattern
seasonal_component = C(1.0, (1e-3, 1e3)) * rbf_seasonal * periodic_kernel

# Additional RBF for longer-term non-periodic trends
trend_component = C(1.0, (1e-3, 1e3)) * RBF(
    length_scale=10.0,
    length_scale_bounds=(1e-1, 1e4)
)

# White noise kernel
noise_kernel = WhiteKernel(
    noise_level=0.1,
    noise_level_bounds=(1e-5, 1e1)
)

# Complete composite kernel
kernel = seasonal_component + trend_component + noise_kernel

print("Kernel structure:")
print(kernel)

# %% Initialize and fit Gaussian Process

model = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=10,  # Multiple random starts for hyperparameter optimization
    normalize_y=False,  # Already normalized manually
    alpha=1e-10,  # Additional noise for numerical stability
    random_state=42
)

print("\nFitting Gaussian Process...")
print("This may take a few minutes due to kernel hyperparameter optimization...")

model.fit(X_scaled, y_scaled)

print("\nOptimized kernel:")
print(model.kernel_)
print(f"\nLog-marginal-likelihood: {model.log_marginal_likelihood(model.kernel_.theta):.3f}")

# %% In-sample prediction

y_hat_scaled, y_std_scaled = model.predict(X_scaled, return_std=True)

# Transform back to original scale
y_hat = scaler_y.inverse_transform(y_hat_scaled.reshape(-1, 1)).ravel()
y_std = y_std_scaled * scaler_y.scale_[0]  # Scale the standard deviation

dates = [dt.date.fromisoformat(d) for d in train.select('date').to_numpy().T[0]]

# %% Evaluation metrics

def norm2(x: np.ndarray, y: np.ndarray):
    """Root Mean Squared Error"""
    assert len(x) == len(y)
    assert len(x) > 0
    return np.sqrt(np.mean((x - y)**2))

rmse = norm2(y_hat, y)
mse = mean_squared_error(y, y_hat)

print(f"\nIn-sample metrics:")
print(f"RMSE: {rmse:.4f}")
print(f"MSE: {mse:.4f}")

# Calculate coverage of 95% confidence interval
lower_bound = y_hat - 1.96 * y_std
upper_bound = y_hat + 1.96 * y_std
coverage = np.mean((y >= lower_bound) & (y <= upper_bound))
print(f"95% CI coverage: {coverage:.2%}")

# %% Plot in-sample prediction with uncertainty

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(dates, y, label='True demand', alpha=0.8, linewidth=1.5)
ax.plot(dates, y_hat, label='GP prediction', alpha=0.8, linewidth=1.5)

# Plot confidence interval
ax.fill_between(
    dates,
    lower_bound,
    upper_bound,
    alpha=0.2,
    label='95% confidence interval'
)

ax.grid(alpha=0.3)
ax.legend()
ax.set_title("Gaussian Process in-sample prediction vs target")
ax.set_xlabel("Date")
ax.set_ylabel("Demand")

fig.tight_layout()
fig.savefig(IMG_FOLDER / 'gp_in_sample.png')

print(f"\nSaved plot: {IMG_FOLDER / 'gp_in_sample.png'}")

# %% Out-of-sample prediction

test = pl.read_csv('test.csv').drop('id')

dates_test = [dt.date.fromisoformat(d) for d in test.select('date').to_numpy().T[0]]

X_test = test.select(features).to_numpy()
X_test_scaled = scaler_X.transform(X_test)

print(f"\nOut-of-sample data: {len(X_test)} samples")

# Predict with uncertainty
y_hat_test_scaled, y_std_test_scaled = model.predict(X_test_scaled, return_std=True)

# Transform back to original scale
y_hat_test = scaler_y.inverse_transform(y_hat_test_scaled.reshape(-1, 1)).ravel()
y_std_test = y_std_test_scaled * scaler_y.scale_[0]

lower_bound_test = y_hat_test - 1.96 * y_std_test
upper_bound_test = y_hat_test + 1.96 * y_std_test

print(f"OOS predictions - min: {y_hat_test.min():.2f}, max: {y_hat_test.max():.2f}")
print(f"Average uncertainty (std): {y_std_test.mean():.2f}")

# %% Plot out-of-sample prediction

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(dates, y, label='In-sample demand', alpha=0.7)
ax.plot(dates_test, y_hat_test, label='OOS prediction', alpha=0.8, linewidth=1.5, color='orange')

# Plot OOS confidence interval
ax.fill_between(
    dates_test,
    lower_bound_test,
    upper_bound_test,
    alpha=0.2,
    color='orange',
    label='95% confidence interval (OOS)'
)

ax.legend()
ax.grid(alpha=0.3)
ax.set_title("Gaussian Process: in-sample target and OOS prediction")
ax.set_xlabel("Date")
ax.set_ylabel("Demand")

fig.tight_layout()
fig.savefig(IMG_FOLDER / 'gp_oos_prediction.png')

print(f"Saved plot: {IMG_FOLDER / 'gp_oos_prediction.png'}")

# %% Detailed OOS prediction plot

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(dates_test, y_hat_test, label='GP OOS prediction', linewidth=2)
ax.fill_between(
    dates_test,
    lower_bound_test,
    upper_bound_test,
    alpha=0.3,
    label='95% confidence interval'
)

ax.grid(alpha=0.3)
ax.legend()
ax.set_title("Gaussian Process OOS prediction with uncertainty")
ax.set_xlabel("Date")
ax.set_ylabel("Demand")

fig.tight_layout()
fig.savefig(IMG_FOLDER / 'gp_oos_detail.png')

print(f"Saved plot: {IMG_FOLDER / 'gp_oos_detail.png'}")

# %% Check for negative predictions

negative_count = np.sum(y_hat_test < 0)
print(f"\nNegative predictions in OOS: {negative_count} out of {len(y_hat_test)}")

if negative_count > 0:
    print(f"Minimum OOS prediction: {y_hat_test.min():.4f}")
else:
    print("No negative predictions - good!")

# %%
