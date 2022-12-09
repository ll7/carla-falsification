"""
The `validate_model.py` script is a utility script for evaluating the performance 
of a trained reinforcement learning model. It loads the trained model from a specified 
file path, creates an instance of the `CustomEnv` class, and then uses the model to 
make predictions on the environment. The script can also render the environment in 
either "humanSync" or "human" mode, allowing the user to see the model's performance 
in real-time. Additionally, the script computes and prints some advanced statistics 
on the model's performance.
"""

import sys
import time
from rl_environment import CustomEnv
from stable_baselines3 import PPO

sys.path.append(".")

""" Load Model from path and render it"""
def test_Results(ticks, save_path, rendMode="humanSync"):
    env = CustomEnv(ticks)
    model = PPO.load(save_path, env=env)
    obs = env.reset()
    env.render(mode=rendMode)
    action1 = []
    auswertung = []
    rewards = 0
    for i in range(10):
        for i2 in range(env.max_tick_count):
            action, _states = model.predict(obs, deterministic=True)
            # action, _states = model.predict(obs)
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

    # For some advanced statistics
    import statistics
    mean = statistics.mean(auswertung)
    max_v = max(auswertung)
    min_v = min(auswertung)
    print('max:', max_v, 'min:', min_v, 'mean', mean, 'max-min:',  max_v-min_v)

    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    # print(mean_reward, std_reward)


if __name__ == '__main__':

    ticks_max = 512
    folder = "../tmp/"
    save_name = folder + "test_Model0.001_50_0"
    test_Results(ticks_max, save_name)
