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


def get_round_values(list_float_values, decimal_places):
    value = []
    for i in list_float_values:
        value.append(round(i, decimal_places))
    return value


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, time_steps_per_training):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:

        # === Gym Variables ===
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        obs_size = 8
        # high = np.array([-500] * obs_size)
        high = np.array([500, 500, 500, 500, 50, 50, 50, 50])
        # low = np.array([500] * obs_size)
        low = np.array([0, 0, 0, 0, -50, -50, -50, -50])
        self.observation_space = spaces.Box(low=low, high=high,
                                            shape=(obs_size,), dtype=np.float32)

        self.done = False
        self.reward = 0
        self.tick_count = 0
        self.max_tick_count = time_steps_per_training
        self.ticks_near_car = 0

        # === Carla ===
        self.host = 'localhost'
        self.town = 'Town01'

        self.client = carla.Client(self.host, 2000)
        self.client.set_timeout(20.0)
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
        time.sleep(2)

        # === set fixed time-step and synchronous mode
        # https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/#fixed-time-step
        # https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/#client-server-synchrony
        settings = self.world.get_settings()
        settings.synchronous_mode = True  # Enables synchronous mode
        settings.fixed_delta_seconds = 0.05

        # === Physics substepping ===
        # In order to have an optimal physical simulation,
        # the substep delta time should at least be below 0.01666 and ideally below 0.01.
        # settings.substepping = True
        # settings.max_substep_delta_time = 0.01
        # settings.max_substeps = 10

        # === Render Mode ===
        # settings = world.get_settings()
        settings.no_rendering_mode = True
        # world.apply_settings(settings)
        # ...
        # settings.no_rendering_mode = False

        self.world.apply_settings(settings)

        # Needed?
        self.client.reload_world(False)

        # Set up the traffic manager
        self.traffic_manager = self.client.get_trafficmanager(8000)
        # self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(0)  # define TM seed for determinism


        # === Spawn Points ===
        # 1. Spawn Point for Walker, 2. for Car
        self.spawn_points = []
        transform_walker_default = self.world.get_map().get_spawn_points()[35]
        transform_car_default = self.world.get_map().get_spawn_points()[89]

        self.spawn_points.append(transform_walker_default)
        self.spawn_points.append(transform_car_default)

        # === walker ===
        self.max_walking_speed = 5   # 18/3,6 m/s
        self.pos_walker_default = [self.spawn_points[0].location.x, self.spawn_points[0].location.y]
        self.pos_walker = self.pos_walker_default
        self.walker, self.collision_sensor_walker = self.__spawn_walker()

        # === car ===
        self.pos_car_default = [self.spawn_points[1].location.x, self.spawn_points[1].location.y]
        self.pos_car = self.pos_car_default

        self.car, self.collision_sensor_car = self.__spawn_car()
        self.observation = self.get_obs()

        # === Draw Start/ End Point ===
        carla_point = carla.Location(x=self.pos_walker[0], y=self.pos_walker[1], z=0.5)
        self.draw_waypoint(carla_point, 'o')
        carla_point = carla.Location(x=self.pos_car[0], y=self.pos_car[1], z=0.5)
        self.draw_waypoint(carla_point, 'x')

        self.collisionReward = 0
        self.info = {"actions": []}

        self._set_camara_view()
        self.world.tick()



    def __spawn_walker(self):
        # === Load Blueprint and spawn walker ===
        walker_bp = self.blueprint_library.filter('0012')[0]
        walker_spawn_transform = self.spawn_points[0]
        walker = self.world.spawn_actor(walker_bp, walker_spawn_transform)
        self.actor_list.append(walker)
        # self.walker.set_transform(self.spawn_points[0])
        try:
            collision_sensor_walker = self.world.spawn_actor(
                self.blueprint_library.find('sensor.other.collision'),
                carla.Transform(), attach_to=walker)
        except:
            print("collision sensor failed")
        return walker, collision_sensor_walker

    def _set_camara_view(self):
        # === Walker View Camera ===
        spectator = self.world.get_spectator()

        location = self.spawn_points[0].location
        # location = carla.Location(x=143.119980, y=326.970001, z=0.300000)
        transform = carla.Transform(location, self.spawn_points[0].rotation)
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=90),
                                                carla.Rotation(pitch=-90)))

    def __spawn_car(self):
        tm_port = self.set_tm_seed()

        car_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        car_sp = self.spawn_points[1]
        car = self.world.spawn_actor(car_bp, car_sp)
        self.actor_list.append(car)
        velocity = carla.Vector3D(x=50, y=0, z=0)
        car.set_target_velocity(velocity)


        car.set_autopilot(True, tm_port)

        collision_sensor_car = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.collision'),
            carla.Transform(), attach_to=car)
        collision_sensor_car.listen(lambda event: self.collision_handler(event))
        return car, collision_sensor_car

    def draw_waypoint(self, location, index, life_time=120.0):

        self.world.debug.draw_string(location, str(index), draw_shadow=False,
                                     color=carla.Color(r=255, g=0, b=0), life_time=life_time,
                                     persistent_lines=True)

    def reward_calculation(self):
        # === Calculate Reward for RL-learning ===
        reward_distance = (-math.dist(self.pos_walker, self.pos_car))/1000

        # End training if collision happend
        if self.collisionReward > 0:
            self.done = True

        return reward_distance + self.collisionReward
    def render(self, mode="human"):
        # === Render Mode ===
        if mode == "human":
            settings = self.world.get_settings()
            settings.no_rendering_mode = False
            self.traffic_manager.set_synchronous_mode(False)
            self.world.apply_settings(settings)
        elif mode == "humanSync":
            settings = self.world.get_settings()
            settings.no_rendering_mode = False
            self.traffic_manager.set_synchronous_mode(True)
            self.world.apply_settings(settings)

        else:
            settings = self.world.get_settings()
            settings.no_rendering_mode = True
            self.world.apply_settings(settings)

    def step(self, action):
        # === Let the walker do a move ===
        # action = get_round_values(action, 1)
        action_length = np.linalg.norm(action)
        if action_length == 0.0:
            # the chances are slim, but theoretically both actions could be 0.0
            unit_action = np.array([0.0, 0.0], dtype=np.float32)
        elif action_length > 1.0:
            # create a vector for the action with the length of zero
            unit_action = np.array(action / action_length)
        else:
            unit_action = np.array(action)
        # unit_action = action
        # if self.tick_count < 100:
        #     print(self.tick_count, ":", self.walker.get_transform().location, self.walker.get_velocity(), action,
        #           self.walker.get_angular_velocity(), self.walker.get_acceleration(), action_length)

        direction = carla.Vector3D(x=float(unit_action[0]), y=float(unit_action[1]), z=0.0)
        walker_control = carla.WalkerControl(
            direction, speed=self.max_walking_speed, jump=False)
        self.info["actions"].append(unit_action.tolist())
        self.walker.apply_control(walker_control)

        # === Update position of walker and car
        self.pos_walker = [self.walker.get_transform().location.x,
                           self.walker.get_transform().location.y]
        self.pos_car = [self.car.get_transform().location.x, self.car.get_transform().location.y]

        # === Do a tick, an check if done ===
        self.world.tick()
        self.tick_count += 1
        if self.tick_count >= self.max_tick_count:
            self.done = True

        return self.get_obs(), self.reward_calculation(), self.done, self.info

    def set_tm_seed(self):
        # === Set Seed for TrafficManager ===
        seed_value = 0
        tm = self.client.get_trafficmanager(8000)
        tm_port = tm.get_port()
        tm.set_random_device_seed(seed_value)
        return tm_port

    def collision_handler(self, event):
        # === handle collisions and calculate extra reward ===
        actor_we_collide_against = event.other_actor
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        # To ensure that the initial force does not count as collision
        if self.tick_count > 100:
            self.collisionReward = min(abs(intensity)*100 + 0.1, 100)
        else:
            self.collisionReward = 0
        if (actor_we_collide_against.type_id == "walker.pedestrian.0012"):
            self.collisionReward = self.collisionReward + 1
            if intensity > 0:
                print("Good Hit:", self.collisionReward)
            else:
                print("Hit with Pedestrian: ", self.collisionReward)
        else:
            print("Car Collition with whatever:", self.collisionReward)

    def reset(self):
        self.tick_count = 0
        self.reward = 0
        self.collisionReward = 0
        self.ticks_near_car = 0
        self.done = False
        self.info = {"actions":[]}
        try:
            self.reset_walker()
        except:
            print("self.reset_walker() failed")
        try:
            self.reset_car()
        except:
            print("Reset Error")
        self.world.tick()
        return self.get_obs()

    def get_obs(self):
        get_vel = self.walker.get_velocity()
        vel_walker = [get_vel.x, get_vel.y]

        get_vel2 = self.car.get_velocity()
        vel_car = [get_vel2.x, get_vel2.y]
        observation = self.pos_car + self.pos_walker + vel_walker + vel_car
        observation = np.array(observation)
        return observation


    def reset_walker(self):
        self.pos_car = self.pos_car_default
        self.pos_walker = self.pos_walker_default
        self.collision_sensor_walker.destroy()
        self.walker.destroy()
        self.walker, self.collision_sensor_walker = self.__spawn_walker()
    def reset_car(self):
        tm_port = self.set_tm_seed()
        self.car.set_autopilot(False, tm_port)
        self.collision_sensor_car.destroy()
        self.car.destroy()
        self.car, self.collision_sensor_car = self.__spawn_car()

    def close(self):
        self.client = carla.Client(self.host, 2000)
        self.client.set_timeout(60.0)

        # Destroy Car
        self.car.set_autopilot(False)
        self.collision_sensor_car.destroy()
        self.car.destroy()

        # Destroy Walker
        self.collision_sensor_walker.destroy()
        self.walker.destroy()

        # Destroy all what isn't jet destroyed
        self.client.apply_batch([carla.command.DestroyActor(x)
                                 for x in self.actor_list])

        # tick for changes to take effect
        self.world.tick()
        self.world.tick()
        self.world.tick()
