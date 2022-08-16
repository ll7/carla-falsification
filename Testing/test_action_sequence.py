import sys
import time
sys.path.append(".")
sys.path.append("/home/imech031/Desktop/carla-simulator/PythonAPI/carla-falsification")
from rl_environment import CustomEnv
import matplotlib.pyplot as plt



def load_action(action_file):
    import ast
    file1 = open(action_file, 'r')
    Lines = file1.readlines()
    array = []
    for line in Lines:
        array.append(ast.literal_eval(line.strip()))
    return array


def test_actions(ticks, action_file, mode="human", plot=True):
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

    for i in range(5):
        for i2 in range(env.max_tick_count):
            action = actions[i2]
            if len(action) == 2:
                action.append(5)
            if i == 0:
                action[2] = action[2]/5
            obs, reward, done, info = env.step(action)
            if plot:
                rewards_list = env.rewards
                plt.cla()
                plt.plot(range(len(rewards_list)), rewards_list, label='whole')
                plt.plot(range(len(rewards_list)), env.open_your_eyes_track, label='angle')
                plt.plot(range(len(rewards_list)), env.reward_distance_track, label='dist')
                plt.legend()
                plt.pause(0.1)
            rewards += reward
            # time.sleep(0.045)
            if done:
                print(rewards)
                auswertung.append(rewards)
                rewards = 0
                env.reset()
                time.sleep(3)
                break
    # print(action1)
    env.close()


if __name__ == '__main__':
    # Modes: humanSync, human
    ticks_max = 512
    action_file = "../tmp/Actions-2.3253"
    action_file = "../tmp/Actions-8.075186"

    plt.show()

    test_actions(ticks_max, action_file, "humanSync", plot=True)

    # test_actions(ticks_max, action_file, "human", plot=False)
