# Quantitative modelling task to predict domestic gas demand

The task is to perform a regression on a time-series
with a daily frequency.

In-sample target `demand` :

<img src="img/train_demand.png" width="500px" />

The out-of-sample distribution 
which contains features for exactly a whole year,
seemingly has a different distribution than the in-sample 
empirical distribution, for instance for order 2 moments,
many have large discrepancies between in/out sample. 

<img src="img/oos_var_diff.png" width="500px" />


## Metrics 

## Linear Regression benchmark

- 5-fold cross-validation 


<img src="img/linear_prediction.png" width="500px" />


## Gradient Boost Regressor prediction

- Non-linear feature engineering
- Feature importance analysis


<img src="img/gb_in_sample.png" width="500px" />


## Gaussian Process model

- Kernel choice

