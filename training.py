import sys
import time
import gym
from time import sleep
from typing import Callable, Optional, Dict, List, Tuple, Type, Union
import optuna
from stable_baselines3.common.logger import configure

from rl_environment import CustomEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn
import torch as th

sys.path.append(".")

env = None


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, time_steps_per_training, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.log_interval = time_steps_per_training
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.best_result = -9999
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        # if self.n_calls % self.log_interval == 299:
        #     print(self.locals['infos'][0]['actions'])
        if self.n_calls % self.log_interval == 0:

            try:
                log_reward = self.locals['infos'][0]['episode']['r']
                # self.logger.record('Log_Reward', log_reward)
                try:
                    if log_reward > self.best_result:
                        print(self.n_calls / self.log_interval, ':', log_reward)
                        self.best_result = log_reward
                        # Just save models with score higher -5 #TODO
                        if (log_reward>-15):
                            self.model.save("./tmp/Callback" + str(log_reward))
                            actions = self.locals['infos'][0]['actions']
                            try:
                                f = open("./tmp/Actions" + str(log_reward), "w")
                                for action in actions:
                                    f.write("%s\n" % action)
                                f.close()
                            except:
                                print("Save actions failed!")
                        # print(actions)
                except:
                    print("save faild")
            except:
                ...

            # value = np.random.random()
            # self.logger.record('random_value', value)

        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def training_test(training_steps, time_steps_per_training,
                  save_name, log_interall, learn_rate=0.003, policy_kwargs=None, p_kwargs=None):
    print("args_p", p_kwargs)
    # env = CustomEnv(time_steps_per_training)
    tmp_path = "./tmp/optuna_tb_big_net_2/" + str(training_steps) + "/" + str(save_name)
    new_logger = configure(tmp_path, ["tensorboard", "stdout"])

    # required before you can step the environment
    env.reset()

    cb = CustomCallback(time_steps_per_training)
    if policy_kwargs is None:
        policy_kwargs = dict(activation_fn=th.nn.ReLU,
                             net_arch=[dict(pi=[64, 64, 2048], vf=[64, 64, 2048])])
    model = None
    if p_kwargs is None:
        model = PPO("MlpPolicy", env, verbose=2, learning_rate=linear_schedule(learn_rate), policy_kwargs=policy_kwargs)
    else:
        print("Creating Model")
        batch_size = p_kwargs["batch_size"]
        # n_epochs = p_kwargs["n_epochs"]
        gamma = p_kwargs["gamma"]
        gae_lambda = p_kwargs["gae_lambda"]
        clip_range = p_kwargs["clip_range"]
        ent_coef = p_kwargs["ent_coef"]
        vf_coef = p_kwargs["vf_coef"]

        model = PPO("MlpPolicy",
                    env=env,
                    verbose=2,
                    learning_rate=linear_schedule(learn_rate),
                    policy_kwargs=policy_kwargs,
                    batch_size=batch_size,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    clip_range=clip_range,
                    ent_coef=ent_coef,
                    vf_coef=vf_coef,
                    use_sde=True
                    )
    print(model.policy)
    model.set_logger(new_logger)
    model.learn(total_timesteps=int(training_steps * time_steps_per_training),
                log_interval=log_interall,
                callback=cb
                #,eval_freq= time_steps_per_training * 10
                )
    model.save("./tmp/test_Model" + str(save_name))
    # env.close()
    print("End Learning")
    return cb.best_result

def render_model(model, env, time_sleep=0.01):
    obs = env.reset()
    rewards = 0
    for _ in range(env.max_tick_count):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        rewards += reward
        # env.render()
        sleep(time_sleep)
        if done:
            break
    return rewards


def create_policy_kwargs(layer, layersize, activation_fn):
    if activation_fn == "ReLU":
        activation_fn = th.nn.ReLU
    elif activation_fn == "Sigmoid":
        activation_fn = th.nn.Sigmoid
    elif activation_fn == "Tanh":
        activation_fn = th.nn.Tanh
    else:
        activation_fn = th.nn.Tanh

    pi2 = []
    vf = []
    for i in range(layer):
        pi2.append(layersize[i])
        vf.append(layersize[i])

    # pi2 = [512, 256, 128]
    # vf = [512, 256, 128]
    policy_kwargs = dict(activation_fn=activation_fn,
                             net_arch=[dict(pi=pi2, vf=vf)])
    # print("policy_kwargs:", policy_kwargs)
    return policy_kwargs


def optuna_trial(trial):
    """Trial for Optuna with all variables"""
    print("Start Trail")
    layers = trial.suggest_categorical('layers', [2, 3, 4])

    first_layer = trial.suggest_categorical('first_layer', [64, 128, 256, 512, 1024, 2048])
    secound_layer = trial.suggest_categorical('secound_layer', [64, 128, 256, 512, 1024, 2048])
    third_layer = 0
    fourth_layer = 0
    if layers > 2:
        third_layer = trial.suggest_categorical('third_layer', [64, 128, 256, 512, 1024, 2048])
    if layers > 3:
        fourth_layer = trial.suggest_categorical('fourth_layer', [64, 128, 256, 512, 1024, 2048])

    activation_function = trial.suggest_categorical('activation_function', ["ReLU", "Sigmoid", "Tanh"])
    policy_kwargs = create_policy_kwargs(
        layers,
        (first_layer, secound_layer, third_layer, fourth_layer),
        activation_function
    )
    epochs = trial.suggest_int('epochs', 700, 1500)
    learnrate = trial.suggest_float('learnrate', 5e-6, 0.007)
    # policy = None
    # algorithm = None
    batch_size = trial.suggest_categorical('batch_size', [512, 1024, 2048])
    gamma = trial.suggest_float('gamma', 0.5, 0.999)
    gae_lambda = trial.suggest_float('gae_lambda', 0.8, 1.0)
    clip_range = trial.suggest_discrete_uniform('clip_range', 0.1, 0.5, 0.1)
    # clip_range_vf=None,
    # normalize_advantage=True,
    ent_coef = trial.suggest_float('ent_coef', 0.0, 0.05)
    vf_coef = trial.suggest_float('vf_coef', 0.3, 1.0)
    # max_grad_norm=0.5,
    # use_sde=False,
    # sde_sample_freq=- 1,
    # target_kl=None,

    args_p = {
        "learnrate": learnrate,
        "epochs": epochs,
        "batch_size": batch_size,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "policy_kwargs": policy_kwargs
    }
    print("args_p", args_p)
    return validate_trys(args_p)


def validate_trys(p_kwargs):
    """Cause one trial is not significant more runs for one evaluation is needed"""
    scores = []
    learnrate = p_kwargs["learnrate"]
    policy_kwargs = p_kwargs["policy_kwargs"]
    training_steps = p_kwargs["epochs"]

    ### Mean of more runs because huge variaty of results but what we really want is a high reward...
    for i in range(5):
        save_name = str(learnrate) + "_" + str(training_steps) + "_" + str(i)

        scores.append(training_test(training_steps, time_steps_per_training, save_name,
                                    log_interall, learnrate, policy_kwargs, p_kwargs))
    # Mean + Max / 2
    return (sum(scores) / len(scores) + max(scores)) / 2


def opt_training(n_trials, hostname, user, password, db_name):
    """ Connect to bd and optimize it"""
    url = 'mysql://' + user + ':' + password + '@' + hostname + '/' + db_name
    # Connect to db
    storage = optuna.storages.RDBStorage(
        url=url,
        engine_kwargs={
            'pool_size': 20,
            'max_overflow': 0
        }
    )

    # Load Study from db
    study = optuna.load_study(
        study_name="learning8", storage=storage
    )

    # start to optimize study
    study.optimize(optuna_trial, n_trials=n_trials)


def manual_training():
    """Train with fixed values"""
    args_p = {
        "learnrate": 0.001,
        "epochs": 10,
        "batch_size": 10,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.1,
        "clip_range": 0,
        "ent_coef": 0,
        "vf_coef": 0,
        "policy_kwargs": 0
    }
    learnrate = args_p["learnrate"]
    policy_kwargs = args_p["policy_kwargs"]
    training_steps = args_p["epochs"]

    init_learnrates = [0.00015, 0.0003, 0.0006]
    for i2 in range(2):
        for i in range(len(init_learnrates)):
            save_name = str(init_learnrates[i]) + "_" + str(training_steps) + "_" + str(i2)
            training_test(training_steps, time_steps_per_training, save_name,
                          log_interall, learnrate, policy_kwargs, None)


if __name__ == '__main__':
    time_steps_per_training = 512
    log_interall = 1

    env = CustomEnv(time_steps_per_training)

    # For Visual training
    # env.render("human")

    #DB-Data:
    hostname = '137.250.121.31'
    user = 'test'
    password = '123'
    db_name = 'optuna'
    # Start Optuna Parameter Optimization
    opt_training(n_trials=500, hostname=hostname, user=user, password=password, db_name=db_name)
