#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import datetime
import glob
import math
import os
import sys

from setuptools.command.dist_info import dist_info

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
import random
import time
from gym.spaces import Box

import gym
from gym import spaces
import numpy as np
from stable_baselines3.common import env_checker

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""


    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:

        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        obs_size = 4
        high = np.array([-999] * obs_size)  # 360 degree scan to a max of 4.5 meters
        low = np.array([999] * obs_size)

        self.observation_space = spaces.Box(low=low, high=high,
                                            shape=(4,), dtype=np.float32)
        print(self.observation_space, self.observation_space.sample())
        # Example for using image as input (channel-first; channel-last also works):

        self.host = 'localhost'
        self.town = 'Town01'
        self.done = False
        self.reward = -100
        self.max_tick_count = 100
        # Steps per Tick
        self.fixed_time_step = 1
        self.max_walking_speed = 15.0 / 3.6  # m/s
        self.actor_list = []
        self.pos_car_default = [335, 260]
        self.pos_walker_default = [335.489990234375, 273]


        self.pos_car = self.pos_car_default
        self.pos_walker = self.pos_walker_default
        self.observation = [self.pos_car, self.pos_walker]

        self.client = carla.Client(self.host, 2000)
        self.client.set_timeout(60.0)
        self.world = self.client.get_world()

        self.spawn_point = self.world.get_map().get_spawn_points()[0]
        # set correct map
        self.map = self.world.get_map()
        if not self.map.name.endswith(self.town):
            self.world = self.client.load_world(self.town)
            while not self.world.get_map().name.endswith(self.town):
                time.sleep(0.5)
            self.world = self.client.get_world()
            self.map = self.world.get_map()

        time.sleep(2)

        # set fixed time-step
        # https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/#fixed-time-step
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.fixed_time_step
        self.world.apply_settings(self.settings)

        # set synchronous mode
        # https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/#client-server-synchrony
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True  # Enables synchronous mode
        self.world.apply_settings(self.settings)

        self.blueprint_library = self.world.get_blueprint_library()

        # === walker ===
        self.__spawn_walker()
        carla_point = carla.Location(x=self.pos_walker[0], y=self.pos_walker[1], z=1)
        self.draw_waypoint(carla_point, 'o')
        carla_point = carla.Location(x=self.pos_car[0], y=self.pos_car[1], z=1)
        self.draw_waypoint(carla_point, 'x')
        self.world.tick()
        print("Init success")


    def __spawn_walker(self):
        self.walker_bp = self.blueprint_library.filter('0012')[0]
        self.walker_spawn_transform = self.spawn_point
        print(self.walker_spawn_transform)
        self.walker = self.world.spawn_actor(
            self.walker_bp, self.walker_spawn_transform)
        self.actor_list.append(self.walker)

    def draw_waypoint(self, location, index, life_time=120.0):

        self.world.debug.draw_string(location, str(index), draw_shadow=False,
                                color=carla.Color(r=255, g=0, b=0), life_time=life_time,
                                persistent_lines=True)

    def __reward_calculation(self):
        return -math.dist(self.pos_walker, self.pos_car)


    def step(self, action):
        action_length = np.linalg.norm(action)
        if action_length == 0.0:
            # the chances are slim, but theoretically both actions could be 0.0
            unit_action = np.array([0.0, 0.0], dtype=np.float32)
        elif action_length > 1.0:
            # create a vector for the action with the length of zero
            unit_action = action / action_length
        else:
            unit_action = action
        self.pos_walker = [self.walker.get_transform().location.x,
                           self.walker.get_transform().location.y]

        direction = carla.Vector3D(x=float(unit_action[0]), y=float(unit_action[1]), z=0.0)

        walker_control = carla.WalkerControl(
            direction, speed=self.max_walking_speed)

        self.walker.apply_control(walker_control)
        # print("start tick")
        #### TICK ####
        self.world.tick()
        self.tick_count += 1
        ##############

        self.reward = self.__reward_calculation()
        if self.reward > -5:
            self.reward = 5
            self.done = True
        if self.tick_count >= self.max_tick_count:
            self.done = True

        # slow down simulation in verbose mode
        # print("calc obs")
        # TODO return
        observation = self.pos_car + self.pos_walker
        observation = np.array(observation)
        # print("observation:", observation)
        return observation, self.reward, self.done, self.info

    def reset(self):
        #TODO
        self.observation = self.spawn_point

        self.tick_count = 0
        self.reward = 0
        self.done = False
        self.info = {}
        self.pos_car = self.pos_car_default
        self.pos_walker = self.pos_walker_default
        carla_point = carla.Location(x=self.pos_walker[0], y=self.pos_walker[1], z=1)
        self.walker.set_location(carla_point)
        self.world.tick()  # this has to be the last line in reset
        observation = self.pos_car + self.pos_walker
        observation = np.array(observation)
        return observation

    # def render(self, mode='human'):
    #     ...
    def close (self):
        self.client = carla.Client(self.host, 2000)
        self.client.set_timeout(60.0)
        self.client.apply_batch([carla.command.DestroyActor(x)
                                for x in self.actor_list])

        # tick for changes to take effect
        self.world.tick()


def manual_iteration(env, number_of_iterations=1):
    """iterate manually over the environment"""
    print(env.reset())
    for i in range(number_of_iterations):
        # print("Sample:", env.action_space.sample())
        observation, reward, done, info = env.step(env.action_space.sample())
        print("Reward, Info:", reward, info)
        # env.render()
        time.sleep(0.5)
    env.close()


# def destroy_actors():
#     client = carla.Client('localhost', 2000)
#     client.set_timeout(10.0)
#     world = client.get_world()
#     actor_list = world.get_actors()
#     for actor in actor_list:
#         print(actor)
#         if str(actor.type_id).startswith('walker.pedestrian.'):
#             print(actor, actor.destroy())
#     print('actors killed')
if __name__ == '__main__':

    env = CustomEnv()
    # env_checker.check_env(env)
    print("checked?")
    manual_iteration(env, number_of_iterations=10)



