# Optuna

## What is Optuna? 

Optuna is an automatic hyperparameter optimization software framework, particularly designed for machine learning. It features an imperative, define-by-run style user API. Thanks to our define-by-run API, the code written with Optuna enjoys high modularity, and the user of Optuna can dynamically construct the search spaces for the hyperparameters.


#### Key Features are: 

- Lightweight, versatile, and platform agnostic architecture
- Pythonic search spaces
- Efficient optimization algorithms
- Easy parallelization
- Quick visualization





## Setup a multicomputer setup 

First you need to install optuna for python
    
    pip install optuna


After install it, verify if optuna works and understand the basics with this small example:

    import optuna

    def objective(trial):
        x = trial.suggest_float('x', -10, 10)
        return (x - 2) ** 2

    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

    study.best_params  # E.g. {'x': 2.002108042}

For a multicomputer setup it's needed to create a database for the communication. 

Open mysql with root user  
In my case I insall MySQL-DB and create a user with a password and with pivileges to create/ write/ read and update db's. 
    
    sudo mysql
    CREATE USER 'myuser'@'%' IDENTIFIED BY 'mypass';
    GRANT ALL ON *.* TO 'myuser'@'%';
    FLUSH PRIVILEGES;
    
Create a New Datebase:

    CREATE DATABASE IF NOT EXISTS example; 
    
Next connect with the database and create a new study once

    storage = optuna.storages.RDBStorage(
    url='mysql://USER:PASSWORD@IP-ADRESS/optuna')
    study = optuna.create_study(study_name="NAME_OF_STUDY", storage=storage, direction='maximize')
    
Note: direction says in which direction to optimize more params can be found: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html?highlight=direction#optuna.study.Study.directions

After creating the Database and Tables you are ready to start the training. For this you just need to connect to the database and start the training on one or multiple computers.For the storage parm use the connection what was set priviosly. 

    study = optuna.load_study(study_name="NAME_OF_STUDY", storage=storage)
    study.optimize(objective, n_trials=1000)

For Evaluation you can load the trained study und visualise it. For example you can plot the history. Different Visualizations can be found here: https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/005_visualization.html#sphx-glr-tutorial-10-key-features-005-visualization-py

    study = optuna.load_study(study_name="NAME_OF_STUDY", storage=storage)
    plot = plot_optimization_history(study)
    plot.show()



Source: 

https://optuna.readthedocs.io/en/stable/index.html

https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html
