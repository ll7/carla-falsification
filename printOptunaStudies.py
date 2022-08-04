import numpy as np
import optuna
from optuna.visualization import plot_optimization_history

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
        # study_name="learning2", storage=storage
        # study_name="learning4", storage=storage
        study_name="learning5", storage=storage
    )




# === Suggested Values after 150 Trials ===

# args_p = {
#         "learnrate": 0.001
#         "epochs": 1200
#         "batch_size": 1024,
#         "gamma": 0.735
#         "gae_lambda": 0.95,
#         "clip_range": 0.1,
#         "ent_coef": 0.005,
#         "vf_coef": 0.57,
#         "policy_kwargs": policy_kwargs
#     }
# first_layer = 128
# secound_layer = 128
# third_layer = 64
# layers = 3
# activation_function = "Tanh"

plot = optuna.visualization.plot_param_importances(study)
plot.show()

plot = plot_optimization_history(study)
plot.show()

plot = optuna.visualization.plot_contour(study)
plot = optuna.visualization.plot_contour(study, params=["first_layer", "secound_layer", "third_layer"])
plot.show()

fig = optuna.visualization.plot_slice(study, params=["first_layer", "secound_layer","layers"])
fig.show()
fig = optuna.visualization.plot_slice(study, params=["first_layer", "secound_layer", "third_layer", "layers"])
fig.show()
fig = optuna.visualization.plot_slice(study, params=["learnrate", "epochs","gamma"])
fig.show()
fig = optuna.visualization.plot_slice(study, params=["learnrate", "first_layer", "secound_layer"])
fig.show()
fig = optuna.visualization.plot_slice(study, params=["batch_size", "clip_range","gae_lambda"])
fig.show()
fig = optuna.visualization.plot_slice(study, params=["ent_coef", "vf_coef", "activation_function"])
fig.show()

fig = optuna.visualization.plot_parallel_coordinate(study)
fig.show()
