#!/usr/bin/env python

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

        # === Gym Variables ===
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        obs_size = 4
        high = np.array([-999] * obs_size)  # 360 degree scan to a max of 4.5 meters
        low = np.array([999] * obs_size)
        self.observation_space = spaces.Box(low=low, high=high,
                                            shape=(4,), dtype=np.float32)

        self.done = False
        self.reward = -100
        self.max_tick_count = 100
        self.ticks_near_car = 0

        # === Carla ===
        self.host = 'localhost'
        self.town = 'Town01'

        self.client = carla.Client(self.host, 2000)
        self.client.set_timeout(60.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.actor_list = []

        # === set correct map ===
        self.map = self.world.get_map()
        if not self.map.name.endswith(self.town):
            self.world = self.client.load_world(self.town)
            while not self.world.get_map().name.endswith(self.town):
                time.sleep(0.5)
            self.world = self.client.get_world()
            self.map = self.world.get_map()
        time.sleep(5)


    def draw_waypoint(self, location, index, life_time=120.0):

        self.world.debug.draw_string(location, str(index), draw_shadow=False,
                                         color=carla.Color(r=255, g=0, b=0), life_time=life_time,
                                         persistent_lines=True)

    def printPoints(self):
        allPoints = self.world.get_map().get_spawn_points()
        allP_array = []
        for i, point in enumerate(allPoints):
            x = point.location.x
            y = point.location.y
            allP_array.append((x,y))
            carla_point = carla.Location(x=x, y=y, z=0.5)
            self.draw_waypoint(carla_point, i)
        print(allP_array)


if __name__ == '__main__':
    env = CustomEnv()
    env.printPoints()