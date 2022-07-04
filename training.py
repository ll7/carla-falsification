import sys
import time
import gym
from time import sleep
import tensorflow as tf
import datetime
from typing import Callable
import optuna
from stable_baselines3.common.logger import configure

from my_rl import CustomEnv
import os
import numpy as np
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback

sys.path.append(".")


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
                self.logger.record('Log_Reward', log_reward)
                try:
                    if log_reward > self.best_result:
                        print(self.n_calls / self.log_interval, ':', log_reward)

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
                        self.best_result = log_reward
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


def carla_training(training_steps, time_steps_per_training, log_interall, learning_rate, save_name):
    env = CustomEnv(time_steps_per_training)

    tmp_path = "./tmp/CartPole_DQN/" + str(save_name)
    new_logger = configure(tmp_path, ["tensorboard", "stdout"])

    model = PPO('MlpPolicy', env,
                learning_rate=learning_rate,
                # n_steps=time_steps_per_training,
                # batch_size=time_steps_per_training,
                # n_epochs=300,
                # gamma=0.99,
                # gae_lambda=0.95,
                # clip_range=0.2,
                # clip_range_vf=None,
                # normalize_advantage=True,
                # ent_coef=0.0,
                # vf_coef=0.5,
                # max_grad_norm=0.5,
                # use_sde=False,
                # sde_sample_freq=- 1,
                # target_kl=None,
                tensorboard_log="./ppo_tensorlog/1/",
                # create_eval_env=False,
                # policy_kwargs=None,
                verbose=1,
                # seed=123,
                # device='auto',
                # _init_setup_model=True
                )

    model.set_logger(new_logger)
    obs = env.reset()
    model.learn(total_timesteps=int(time_steps_per_training * training_steps),
                callback=CustomCallback(time_steps_per_training),
                tb_log_name='PPO_Log', log_interval=log_interall)

    model.save("./tmp/myModel" + str(save_name))
    print('Reward:', render_model(model, env))
    env.close()


def training_test(training_steps, time_steps_per_training, save_name, log_interall, learn_rate=0.0003):
    env = CustomEnv(time_steps_per_training)
    tmp_path = "./tmp/optuna_tb/" + str(save_name)
    new_logger = configure(tmp_path, ["tensorboard", "stdout"])

    # required before you can step the environment
    env.reset()

    cb = CustomCallback(time_steps_per_training)
    model = PPO('MlpPolicy', env, verbose=2, learning_rate=learn_rate)
    model.set_logger(new_logger)
    model.learn(total_timesteps=int(training_steps * time_steps_per_training),
                log_interval=log_interall,
                callback=cb)
    model.save("./tmp/test_Model" + str(save_name))
    env.close()

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


def optuna_trial(trial):
    learnrate = trial.suggest_float('learnrate', 5e-6, 0.01)
    # policy = None
    # algorithm = None
    # batch_size = trial.suggest_int('batch_size', 20, 3000)
    # n_epochs = 300,
    # gamma = trial.suggest_float('gamma', 0.7, 0.999)
    # gae_lambda = trial.suggest_float('gae_lambda', 0.9, 1.0)
    # clip_range = trial.suggest_discrete_uniform('clip_range', 0.1, 0.5, 0.1)
    # clip_range_vf=None,
    # normalize_advantage=True,
    # ent_coef = trial.suggest_float('ent_coef', 0.0, 0.02)
    # vf_coef = trial.suggest_float('vf_coef', 0.5, 1.0)
    # max_grad_norm=0.5,
    # use_sde=False,
    # sde_sample_freq=- 1,
    # target_kl=None,


    scores = []

    ### Mean of 3 runs because huge variaty of results
    for i in range(3):
        save_name = str(learnrate) + "_" + str(training_steps) + "_" + str(i)
        scores.append(training_test(training_steps, time_steps_per_training, save_name, log_interall, learnrate))

    return sum(scores) / len(scores)

def opt_training():
    study = optuna.create_study(direction='maximize')
    study.optimize(optuna_trial, n_trials=50)

def manual_training():

    init_learnrates = [0.00015, 0.0003, 0.0006]
    for i in range(len(init_learnrates)):
        save_name = str(init_learnrates[i])+"_"+str(training_steps)
        training_test(training_steps, time_steps_per_training, save_name, log_interall, init_learnrates[i])


if __name__ == '__main__':
    training_steps = 1500
    time_steps_per_training = 512
    log_interall = 1
    opt_training()
    # manual_training()

