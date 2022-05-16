import sys
import time
import gym
from time import sleep
import tensorflow as tf
import datetime

from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from my_rl import CustomEnv
import os
import numpy as np
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback

sys.path.append(".")


def eazy_env():
    env = CustomEnv(300)

    tmp_path = "./tmp/CartPole_DQN"
    new_logger = configure(tmp_path, ["tensorboard", "stdout"])
    # model = PPO.load("./tmp/myModel", env=env)
    time.sleep(0.3)
    model = PPO.load("./tmp/Callback-25.837323", env=env)
    # model.set_random_seed(123)
    time.sleep(0.3)
    obs = env.reset()
    rewards = 0
    for i in range(10):
        for _ in range(env.max_tick_count):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            rewards += reward
            # env.render()
            if i > 0:
                sleep(0.01)
            if done:
                print("Reward:", rewards)
                rewards = 0
                env.reset()
                break

    obs = env.close()
    #
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    # print(mean_reward, std_reward)


eazy_env()
