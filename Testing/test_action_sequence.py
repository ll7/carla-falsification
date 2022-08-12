import sys
import time
from rl_environment import CustomEnv

sys.path.append(".")


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

    for i in range(5):
        for i2 in range(env.max_tick_count):
            action = actions[i2]
            if len(action) == 2:
                action.append(5)
            if i == 0:
                action[2] = action[2]/5
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
    ticks_max = 512
    action_file = "../tmp/Actions-7.357062"
    # test_actions(ticks, action_file, "humanSync")
    test_actions(ticks_max, action_file, "human")
