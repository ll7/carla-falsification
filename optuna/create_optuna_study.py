import optuna

storage = optuna.storages.RDBStorage(
     url='mysql://test:123@137.250.121.31/optuna',
     engine_kwargs={
         'pool_size': 20,
         'max_overflow': 0
     }
 )

study = optuna.create_study(study_name="learning8", storage=storage, direction='maximize')
