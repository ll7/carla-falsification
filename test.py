import optuna

def objective(trial):
    x = trial.suggest_float("x", 0, 10)
    return x ** 2


storage = optuna.storages.RDBStorage(
     url='mysql://test:123@137.250.121.31/optuna',
     engine_kwargs={
         'pool_size': 20,
         'max_overflow': 0
     }
 )


study = optuna.create_study(study_name="learning7", storage=storage, direction='maximize')
# study.optimize(objective)

# study = optuna.load_study(
#         study_name="test1", storage=storage
#     )
# study.optimize(objective, n_trials=10)

