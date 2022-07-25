# carla-falsification
This Project is about ceating an Gym Env for Carla and Train an agent via reinforcement learning to find mistakes in autonomous driving vehicles. 

## Learn Reinforcement Learing: 
- https://github.com/ll7/carla-falsification/edit/master/documentation/Learn-RL.md

## Aufgaben Stellung: 
- https://github.com/ll7/carla-falsification/blob/master/documentation/Aufgabe.md


## Zeitplan: 
- https://github.com/ll7/carla-falsification/blob/master/documentation/Zeitplan.md


## Installation 

### CUDA and Drives 
You need to Install at least Cuda 11.3 and Nvidea Driver 
First install Cuda then driver 

### CARLA: 

https://carla.readthedocs.io/en/latest/start_quickstart/

pip3 install --upgrade pip

pip3 install --user pygame numpy

sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 1AF1527DE64CB8D9

sudo add-apt-repository "deb [arch=amd64] http://dist.carla.org/carla $(lsb_release -sc) main"

sudo apt-get update # Update the Debian package index

sudo apt-get install carla-simulator=0.9.13 # Install Carla 0.9.13

cd /opt/carla-simulator # Open the folder where CARLA is installed

apt-get install libomp5

### Envirionment 
pip3 install gym

pip3 install tensorflow

pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu113 # https://pytorch.org/get-started/locally/****

pip3 install stable_baselines3

### Extra for visulalisation and param optisation

pip3 install optuna 

pip3 install tensorboard

### Visualization

pip install plotly

pip install sklearn

sudo apt-get install libmysqlclient-dev
sudo -H pip3 install mysqlclient

#### Used Verions: 
- pygame                            2.1.2
- numpy                             1.22.2
- carla                             0.9.13
- gym                               0.21.0
- tensorflow                        2.9.1
- tensorboard                       2.9.1
- stable-baselines3                 1.5.0
- optuna                            2.10.1

## Start Training: 
python3 training.py

