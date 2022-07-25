# import lightgbm as lgb
import joblib
import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split

import optuna
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

SEED = 42

np.random.seed(SEED)

storage = optuna.storages.RDBStorage(
     url='mysql://test:123@137.250.121.31/optuna',
     engine_kwargs={
         'pool_size': 20,
         'max_overflow': 0
     }
 )

study = optuna.load_study(
        study_name="rl-learning", storage=storage
    )


plot = optuna.visualization.plot_param_importances(study)
# plot = plot_optimization_history(study)
plot.show()