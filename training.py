import sys
import time

import gym
from time import sleep

from stable_baselines3 import DQN, A2C, PPO

sys.path.append(".")
from my_rl import CustomEnv

tmp_path = "./tmp/CartPole_DQN"

from stable_baselines3.common.logger import configure

new_logger = configure(tmp_path, ["tensorboard", "stdout"])


env = CustomEnv()


batch = 64 #trainig batch size
eph = 0.9 #ephsilon
eph_min = 0.01
decay = 0.995 #decay for ephsilon
gamma = 0.99 # discount factor


# model = DQN("MlpPolicy", env, verbose=2, batch_size=batch, gamma=gamma,
#             exploration_initial_eps=decay, exploration_final_eps=eph_min)

model = PPO('MlpPolicy', env, verbose=2)

model.set_logger(new_logger)


obs = env.reset()
rewards = 0
for _ in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    rewards+=reward
    # env.render()
    if done:
        print("first:", rewards)
        rewards = 0
        obs = env.reset()

model.learn(int(200), log_interval=50)

# model.save("./tmp/CartPole_DQN_model")

obs = env.reset()
rewards = 0
for _ in range(env.max_tick_count):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    rewards+=reward
    # env.render()
    sleep(0.2)
    if done:
        print("last:", rewards)
        rewards = 0
        obs = env.reset()
        continue









