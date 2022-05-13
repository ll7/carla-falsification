import sys
import time
import gym
from time import sleep
import tensorflow as tf
import datetime

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

    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
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

        if self.n_calls % 100 == 0:
            try:
                log_reward = self.locals['infos'][0]['episode']['r']
                if (self.best_result < log_reward):
                    self.best_result = log_reward
                print(self.n_calls, ':', log_reward)
                self.logger.record('Log_Reward', log_reward)
            except:
                ...

            value = np.random.random()
            self.logger.record('random_value', value)

        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


def eazy_env():
    env = CustomEnv()

    tmp_path = "./tmp/CartPole_DQN"
    new_logger = configure(tmp_path, ["tensorboard", "stdout"])

    model = PPO('MlpPolicy', env, verbose=2, n_steps=2000, batch_size=100)
    model.set_logger(new_logger)
    obs = env.reset()
    rewards = 0
    for _ in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        rewards += reward
        # env.render()
        if done:
            print("first:", rewards)
            rewards = 0
            obs = env.reset()

    model.learn(total_timesteps=int(100), log_interval=50, callback=CustomCallback(),
                reset_num_timesteps=True)
    # for i in range(1):
    #     print("Run ", i)
    #     model.learn(total_timesteps=int(100), log_interval=50, callback=CustomCallback(),
    #                 reset_num_timesteps=True)

    model.save("./tmp/myModel")

    obs = env.reset()
    rewards = 0
    for _ in range(env.max_tick_count):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        rewards += reward
        # env.render()
        sleep(0.2)
        if done:
            print("last:", rewards)
            rewards = 0
            obs = env.close()
            break


def sec_env():
    env = CustomEnv()
    LOG_DIR = './tmp/train/logs/'
    model = PPO('MlpPolicy', env, tensorboard_log=LOG_DIR, verbose=1)
    new_logger = configure(LOG_DIR, ["stdout", "tensorboard"])

    obs = env.reset()
    rewards = 0
    model.learn(total_timesteps=10, callback=CustomCallback())

    model.set_logger(new_logger)
    # model.save("./tmp/CartPole_DQN_model")

    obs = env.reset()
    rewards = 0
    for _ in range(env.max_tick_count):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        rewards += reward
        sleep(0.1)
        if done:
            print("last:", rewards)
            rewards = 0
            obs = env.close()
            break


eazy_env()
