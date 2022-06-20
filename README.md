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


