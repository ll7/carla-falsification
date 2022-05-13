import sys
import time
import gym
from time import sleep
import tensorflow as tf
import datetime
from typing import Callable

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

        if self.n_calls % self.log_interval == 0:
            try:
                log_reward = self.locals['infos'][0]['episode']['r']
                if (self.best_result > log_reward):
                    self.best_result = log_reward
                print(self.n_calls/self.log_interval, ':', log_reward)
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

def carla_training(training_steps, time_steps_per_training):
    env = CustomEnv(time_steps_per_training)

    tmp_path = "./tmp/CartPole_DQN"
    new_logger = configure(tmp_path, ["tensorboard", "stdout"])

    model = PPO('MlpPolicy', env,
                learning_rate=0.0003,
                n_steps=training_steps*time_steps_per_training,
                batch_size=time_steps_per_training,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                clip_range_vf=None,
                normalize_advantage=True,
                ent_coef=0.0,
                vf_coef=0.5,
                max_grad_norm=0.5,
                use_sde=False,
                sde_sample_freq=- 1,
                target_kl=None,
                tensorboard_log=None,
                create_eval_env=False,
                policy_kwargs=None,
                verbose=2,
                seed=0,
                device='auto',
                _init_setup_model=True)

    model.set_logger(new_logger)
    obs = env.reset()

    # rewards = 0
    # for _ in range(time_steps_per_training):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     rewards += reward
    #     # env.render()
    #     if done:
    #         print("first:", rewards)
    #         rewards = 0
    #         obs = env.reset()

    model.learn(total_timesteps=int(time_steps_per_training),
                callback=CustomCallback(time_steps_per_training))

    # model.learn(10, callback=None, log_interval=1, eval_env=None, eval_freq=- 1,
    #       n_eval_episodes=5, tb_log_name='PPO', eval_log_path=None, reset_num_timesteps=True)

    model.save("./tmp/myModel")

    obs = env.reset()
    rewards = 0
    for _ in range(time_steps_per_training):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        rewards += reward
        # env.render()
        sleep(0.1)
        if done:
            print("last:", rewards)
            rewards = 0
            obs = env.close()
            break


if __name__ == '__main__':
    training_steps = 3
    time_steps_per_training = 250

    carla_training(training_steps, time_steps_per_training)
