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
    model = PPO.load(save_path, env=env)
    obs = env.reset()
    env.render()
    actions = []
    positions = []
    for index in range(env.max_tick_count):
        positions.append(env.walker.get_transform().location)
        action, _states = model.predict(obs, deterministic=True)
        actions.append(action.tolist())
        env.step(action=action)
    print(actions)
    for epoch in range(20):
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

def test_Results(ticks, save_path):
    env = CustomEnv(ticks)
    model = PPO.load(save_path, env=env)
    obs = env.reset()
    env.render()
    action1 = []
    auswertung = []
    rewards = 0
    for i in range(10):
        for i2 in range(env.max_tick_count):
            action, _states = model.predict(obs, deterministic=True)
            if i == 0:
                action1.append(action)
            else:
                action = action1[i2]
            obs, reward, done, info = env.step(action)
            rewards += reward
            if done:
                print(rewards, env.walker.get_transform().location)
                auswertung.append(rewards)
                rewards = 0
                env.reset()
                time.sleep(0.5)
                break
    # print(action1)
    env.close()

    import statistics
    mean = statistics.mean(auswertung)
    max_v = max(auswertung)
    min_v = min(auswertung)
    print('max:', max_v, 'min:', min_v, 'mean', mean, 'max-min:',  max_v-min_v)

    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    # print(mean_reward, std_reward)

def load_action(action_file):
    import ast
    file1 = open(action_file, 'r')
    Lines = file1.readlines()
    array = []
    for line in Lines:
        array.append(ast.literal_eval(line.strip()))
    return array


def test_actions(ticks, action_file, mode="human"):
    actions = load_action(action_file)
    print(actions)
    env = CustomEnv(ticks)
    obs = env.reset()
    env.render(mode=mode)
    action1 = []
    auswertung = []
    rewards = 0

    while len(actions) < ticks:
        actions.append(actions[-1])

    for i in range(10):
        for i2 in range(env.max_tick_count):
            action = actions[i2]
            obs, reward, done, info = env.step(action)
            rewards += reward
            # time.sleep(0.01)
            if done:
                print(rewards)
                auswertung.append(rewards)
                rewards = 0
                env.reset()
                time.sleep(0.5)
                break
    # print(action1)
    env.close()


if __name__ == '__main__':
    # Modes: humanSync, human

    ticks = 512
    # folder = "../tmp/"
    folder = "../tmp/GoodModels/"
    save_name = folder + "myModel1e-05_3000.zip"
    action_file = folder + "Actions-3.73375"
    # test_determinisWalker(ticks, save_name)
    # test_Results(ticks, save_name)

    test_actions(ticks, action_file, "humanSync")
    test_actions(ticks, action_file, "human")
