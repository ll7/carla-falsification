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
import copy
import gym
from gym import spaces
import numpy as np
from stable_baselines3.common import env_checker
import copy


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, time_steps_per_training):
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
        self.max_tick_count = time_steps_per_training
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
                time.sleep(0.2)
            self.world = self.client.get_world()
            self.map = self.world.get_map()
        time.sleep(5)

        # === set fixed time-step and synchronous mode
        # https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/#fixed-time-step
        # https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/#client-server-synchrony
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True  # Enables synchronous mode
        self.settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(self.settings)

        #Needed?
        # self.client.reload_world(False)

        # Set up the traffic manager
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(0)  # define TM seed for determinism



        # === Render Mode ===
        # settings = world.get_settings()
        # settings.no_rendering_mode = True
        # world.apply_settings(settings)
        # ...
        # settings.no_rendering_mode = False
        # world.apply_settings(settings)

        # === Spawn Points ===
        # 1. Spawn Point for Walker, 2. for Car
        self.spawn_points = []
        transform_walker_default = self.world.get_map().get_spawn_points()[35]
        transform_car_default = self.world.get_map().get_spawn_points()[89]

        # Modify car pos
        # x = transform_car_default.location.x + 0
        # y = 250
        # z = transform_car_default.location.z
        # carla_location = carla.Location(x=x, y=y, z=z)
        # transform_car_default = carla.Transform(carla_location, transform_walker_default.rotation)
        self.spawn_points.append(transform_walker_default)
        self.spawn_points.append(transform_car_default)
        # print(self.spawn_points[0], self.spawn_points[1])

        # === walker ===
        self.max_walking_speed = 15.0 / 3.6  # m/s
        self.pos_walker_default = [self.spawn_points[0].location.x, self.spawn_points[0].location.y]
        self.pos_walker = self.pos_walker_default
        self.__spawn_walker()

        # === car ===
        self.pos_car_default = [self.spawn_points[1].location.x, self.spawn_points[1].location.y]
        self.pos_car = self.pos_car_default

        self.__spawn_car()

        self.observation = [self.pos_car, self.pos_walker]

        # === Draw Start/ End Point ===
        carla_point = carla.Location(x=self.pos_walker[0], y=self.pos_walker[1], z=0.5)
        self.draw_waypoint(carla_point, 'o')
        carla_point = carla.Location(x=self.pos_car[0], y=self.pos_car[1], z=0.5)
        self.draw_waypoint(carla_point, 'x')

        self._set_camara_view()
        self.world.tick()
        print("Init success")
        time.sleep(0.2)

        self.extraReward = 0

    def __spawn_walker(self):
        self.walker_bp = self.blueprint_library.filter('0012')[0]
        self.walker_spawn_transform = self.spawn_points[0]
        self.walker = self.world.spawn_actor(
            self.walker_bp, self.walker_spawn_transform)
        self.actor_list.append(self.walker)

        try:
            self.collision_sensor = self.world.spawn_actor(
                self.blueprint_library.find('sensor.other.collision'),
                carla.Transform(), attach_to=self.walker)
        except:
            print("collision sensor failed")


    def _set_camara_view(self):


        # # ========== Whole View
        # spectator = self.world.get_spectator()
        #
        # location = self.spawn_points[0].location - \
        #            (self.spawn_points[0].location - self.spawn_points[1].location)
        # location.x += -20
        # transform = carla.Transform(location, self.spawn_points[0].rotation)
        #
        # # print('cam: ', transform.location)
        # spectator.set_transform(carla.Transform(transform.location + carla.Location(z=80),
        #                                         carla.Rotation(pitch=-60)))
        # time.sleep(0.1)

        # ========== Walker View
        spectator = self.world.get_spectator()

        location = carla.Location(x=143.119980, y=326.970001, z=0.300000)
        transform = carla.Transform(location, self.spawn_points[0].rotation)

        # print('cam: ', transform.location)
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=40),
                                                carla.Rotation(pitch=-60)))
        time.sleep(0.1)

    def __spawn_car(self):
        try:
            tm_port = self.set_tm_seed()

            self.car_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
            self.car_sp = self.spawn_points[1]
            self.car = self.world.spawn_actor(
                self.car_bp, self.car_sp)
            self.actor_list.append(self.car)


            # self.car.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
            # self.car.steer = 0.0
            # self.car.throttle = 1.0
            # self.car.brake = 0.0
            # self.car.hand_brake = False
            # self.car.reverse = False
            # self.car.manual_gear_shift = False
            # self.car.gear = 0

            self.car.set_autopilot(True, tm_port)
        except:
            print('spawn car error')

        try:
            self.collision_sensor = self.world.spawn_actor(
                self.blueprint_library.find('sensor.other.collision'),
                carla.Transform(), attach_to=self.car)
            self.collision_sensor.listen(lambda event: self.function_handler(event))
        except:
            print("collision sensor failed")

        try:
            if self.tick_count == 20:
                print(self.car.get_physics_control())
            if self.tick_count == 40:
                print(self.car.get_physics_control())
        except:
            print("Nope")

    def draw_waypoint(self, location, index, life_time=120.0):

        self.world.debug.draw_string(location, str(index), draw_shadow=False,
                                     color=carla.Color(r=255, g=0, b=0), life_time=life_time,
                                     persistent_lines=True)

    def __reward_calculation(self):
        reward = -math.dist(self.pos_walker, self.pos_car)+self.extraReward
        # if self.extraReward > 0:
        #     print(reward)
        return reward

        # distance = math.dist(self.pos_walker, self.pos_car)
        # better_first = (1-self.tick_count/self.max_tick_count)
        # if distance > 5:
        #     self.ticks_near_car -= 1
        #     return -distance/1000
        # self.ticks_near_car += 1
        # return self.ticks_near_car

    def step(self, action):
        # self.set_tm_seed()

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

        self.pos_car = [self.car.get_transform().location.x, self.car.get_transform().location.y]

        walker_control = carla.WalkerControl(
            direction, speed=self.max_walking_speed)

        self.walker.apply_control(walker_control)
        # print("start tick")
        #### TICK ####
        self.world.tick()
        self.tick_count += 1
        # time.sleep(0.0001)
        ##############

        self.reward = self.__reward_calculation()
        if self.reward > 0:
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

    def set_tm_seed(self):
        seed_value = 0
        tm = self.client.get_trafficmanager(8000)
        tm_port = tm.get_port()
        tm.set_random_device_seed(seed_value)
        return tm_port

    def function_handler(self, event):
        actor_we_collide_against = event.other_actor
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.extraReward = (intensity + 10) ** 2
        # if (actor_we_collide_against.type_id == "walker.pedestrian.0012"):
        #     print("Hit with", self.extraReward, "Extra Points")
        if intensity > 0:
            print("Good Hit")

    def reset(self):
        self.observation = self.spawn_points[0]
        self.tick_count = 0
        self.reward = 0
        self.extraReward = 0
        self.ticks_near_car = 0
        self.done = False
        self.info = {}
        self.pos_car = self.pos_car_default
        self.pos_walker = self.pos_walker_default
        # carla_point = carla.Location(x=self.pos_walker[0], y=self.pos_walker[1], z=1)
        self.walker.set_location(self.spawn_points[0].location)
        try:
            self.reset_car()
        except:
            print("Reset Error")

        self.world.tick()  # this has to be the last line in reset
        observation = self.pos_car + self.pos_walker
        observation = np.array(observation)
        return observation

    def reset_car(self):
        tm_port = self.set_tm_seed()
        self.car.set_autopilot(False, tm_port)
        self.collision_sensor.destroy()
        self.car.destroy()
        self.__spawn_car()

    def close(self):
        self.client = carla.Client(self.host, 2000)
        self.client.set_timeout(60.0)
        self.client.apply_batch([carla.command.DestroyActor(x)
                                 for x in self.actor_list])

        # tick for changes to take effect
        self.world.tick()
        self.world.tick()
