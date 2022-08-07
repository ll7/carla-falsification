# carla-falsification
This Project is about ceating an Gym Env for Carla and Train an agent via reinforcement learning to find mistakes in autonomous driving vehicles. 

## Learn Reinforcement Learing and Carla: 
<details>
  <summary>Deitails</summary>
  
Start with RL:
- https://github.com/ll7/carla-falsification/edit/master/documentation/Learn-RL.md

Mistakes that can be avoided:
- https://github.com/ll7/carla-falsification/blob/master/documentation/Mistakes%20that%20can%20be%20avoided.md

Carla Determinism Problem:
- https://github.com/ll7/carla-falsification/blob/master/documentation/determinism-problem.md

</details>

## Aufgaben Stellung and Zeitplan: 
- https://github.com/ll7/carla-falsification/blob/master/documentation/Aufgabe.md
- https://github.com/ll7/carla-falsification/blob/master/documentation/Zeitplan.md




## Installation 

<details>
  <summary>Deitails Installation</summary>
  
### CUDA and Drives 
You need to Install at least Cuda 11.3 and Nvidea Driver 
https://developer.nvidia.com/cuda-11.3.0-download-archive

    wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
    sudo sh cuda_11.3.0_465.19.01_linux.run


First install Cuda then driver 

### CARLA: 

https://carla.readthedocs.io/en/latest/start_quickstart/

Update Pip and install pygame and numpy:

    pip3 install --upgrade pip
    pip3 install --user pygame numpy

Add Carla Repo to apt and install Simulator version 0.9.13

    sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 1AF1527DE64CB8D9
    sudo add-apt-repository "deb [arch=amd64] http://dist.carla.org/carla $(lsb_release -sc) main"

    sudo apt-get update # Update the Debian package index

    sudo apt-get install carla-simulator=0.9.13 # Install Carla 0.9.13

Go to the instalation folder: 

    cd /opt/carla-simulator # Open the folder where CARLA is installed

Install dependanies for starting Carla: 
    
    apt-get install libomp5

### Envirionment 

For the training and environment you need to install several packages. 

    pip3 install gym

    pip3 install tensorflow
    
    pip3 install stable_baselines3
    
    pip3 install optuna 
    
For Nvidia RTX 3080 only the nightly build seems to work ... 

    pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu113 # https://pytorch.org/get-started/locally/****

    

### Extra for visulalisation and param optisation

For the visualizion and for param optimization you need to install the following tools/ packages. For more information abaut Optuna zou can read here: https://github.com/ll7/carla-falsification/blob/master/documentation/Optuna_Optimization.md

    pip3 install optuna 

    pip3 install tensorboard

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

</details>

## Start Training: 

<details>
  <summary>Deitails Training</summary>
  
  
For starting a training you need to clone Repo first, install all dependancies (From above) and then you can start carla and afterwards the Training. For performance reasons rendering is deactivated by default. If you want to see what the agent is doing uncomment the line "# env.render("human")" in the main of training.py

Start Carla: 
    
    cd /opt/carla-simulator
    ./CarlaUE4.sh
    
Start Training: 

    cd /home
    git@github.com:ll7/carla-falsification.git
    cd ./carla-falsification/
    python3 training.py >> log.txt

</details>

## Visualize Training 

<details>
  <summary>Deitails Visualization</summary>
  
While training you watch traing Progress in Tensaboard. For do so go into the save folder of tensorboad, open terminal and open models. 

    cd ./tmp/optuna_tb_big_net_2
    tensorboard --logdir=./

As mentioned before, if you want to see what the agent is doing uncomment the line "# env.render("human")" in the main of training.py

For visualize results of Optuna, modify storage and study name if needed in printOptunaStudies.py and execute it. 

    cd home/carla-falsification/
    python3 printOptunaStudies.py

</details>
