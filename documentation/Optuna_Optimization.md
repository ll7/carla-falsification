# Optuna

## What is Optuna? 

Optuna is framework for hyperparameter optimization. You can use it with any machine learning or deep learning framework. 


## Setup a multicomputer setup 

First you need to install optuna for python
- pip install optuna


After install it, verify if optuna works and understand the basics with this small example:
'''
import optuna

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

study = optuna.create_study()
study.optimize(objective, n_trials=100)

study.best_params  # E.g. {'x': 2.002108042}
'''
