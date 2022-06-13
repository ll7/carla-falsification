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


def test_determinisWalker(ticks, save_path):
    env = CustomEnv(ticks)
    model = PPO.load("../tmp/"+save_path, env=env)
    time.sleep(0.3)
    obs = env.reset()
    actions = []
    positions = []
    for index in range(env.max_tick_count):
        positions.append(env.walker.get_transform().location)
        action, _states = model.predict(obs, deterministic=True)
        actions.append(action.tolist())
        env.step(action=action)
    for epoch in range(100):
        sprint = True
        env.reset()
        for index, action in enumerate(actions):
            if positions[index].x != env.walker.get_transform().location.x:
                if sprint:
                    print("Oh no:", index, env.walker.get_transform().location.x, "soll:", positions[index].x)
                    sprint = False
            env.step(action=action)
        if epoch % 10 == 0:
            print(epoch)
    env.close()

if __name__ == '__main__':
    ticks = 300
    save_name = "myModel10k.zip"
    test_determinisWalker(ticks, save_name)