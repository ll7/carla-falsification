#!/usr/bin/env python

"""
This code appears to define a custom reinforcement learning (RL) environment for 
training an RL agent to avoid a pedestrian in a simulated town. The environment is 
implemented as a subclass of the `gym.Env` class from the Gym library, which defines a 
standard interface for RL environments.

The environment initializes the action and observation spaces, which define the 
valid range of actions and observations that the RL agent can take and receive, 
respectively. The action space is defined as a 3-dimensional box with bounds 
[-1, -1, 1] and [1, 1, 1], representing the x and y components of the direction 
vector and the speed of the vehicle, respectively. The observation space is defined 
as a 10-dimensional box with bounds [100, 250, 100, 250, 0, 0, -30, -30, -1, -1] 
and [400, 360, 400, 360, 15, 15, 30, 30, 1, 1], representing the position and 
velocity of the vehicle and pedestrian, as well as the direction vector of the pedestrian.

Next, the environment sets up a connection to the CARLA simulation server, which 
runs the simulated town, and creates the necessary actors, such as the vehicle and 
pedestrian, and sensors, such as the camera and lidar, in the simulation. 
The environment also sets up a connection to the town map, 
which is used to compute distances and directions between actors in the town.

The environment defines several methods for interacting with the simulation and the 
RL agent. The `step()` method is called by the RL agent to take an action in the 
environment and receive the resulting observation, reward, and done flag. 
This method updates the position and velocity of the vehicle and pedestrian in the 
simulation, computes the reward based on the distance and relative velocities of the 
vehicle and pedestrian, and returns the resulting observation, reward, and done 
flag to the RL agent. The `reset()` method is called by the RL agent to reset the 
environment to its initial state, and the `render()` method is called to display 
the current state of the environment.

RL environment for training an RL agent to avoid a pedestrian in a simulated town. 
It sets up the connection to the CARLA simulation server and creates the necessary 
actors and sensors in the simulation. It also defines the action and observation spaces 
and the methods for interacting with the environment and the RL agent. 
This environment can be used as a basis for training an RL agent to avoid collisions 
with pedestrians in a simulated town.
"""

import datetime
import glob
import math
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
import time
from gym.spaces import Box
import gym
from gym import spaces
import numpy as np


def norm_angle(angle_rad):
    """Normalize the given angle within [-pi, +pi)"""
    if angle_rad > 0:
        return angle_rad % math.pi
    return - abs(angle_rad) % math.pi


def norm_angle_deg(deg_angle):
    """Normalize the given angle within [-pi, +pi)"""
    if deg_angle > 0:
        return deg_angle % 360
    return - abs(deg_angle) % 360

def vector_len(vec):
    """Compute the given vector's length"""
    return math.sqrt(vec[0] ** 2 + vec[1] ** 2)


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, time_steps_per_training=512):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:

        # === Gym Variables ===

        high_action = np.array([
            1, 1,  # direction vector (x/y)
            1  # speed from 0 to 100%
        ])
        low_action = np.array([
            -1, -1,  # dir_vec (x/y)
            1  # speed from 0 to 100%
        ])
        self.action_space = Box(low=low_action, high=high_action, shape=(3,), dtype=np.float32)

        high = np.array([
            400, 360,  # pos_car(x,y)
            400, 360,  # pos_walker(x,y)
            15, 15,    # vel_walker(x,y)
            30, 30,    # vel_car(x,y)
            1, 1       # direction vector walker (x, y)
        ])
        low = np.array([
            100, 250,  # pos_car(x,y)
            100, 250,  # pos_walker(x,y)
            0, 0,      # vel_walker(x,y)
            -30, -30,  # vel_car(x,y)
            -1, -1     # direction vector walker (x, y)
        ])
        obs_size = len(high)
        self.observation_space = spaces.Box(low=low, high=high,
                                            shape=(obs_size,), dtype=np.float32)

        self.done = False
        self.reward = 0
        self.tick_count = 0
        self.max_tick_count = time_steps_per_training

        # Get every sec the velocity to check if an emergency braking happened
        self.velocity_last_sec = 10

        # === Carla ===
        self.host = 'localhost'
        self.town = 'Town01'

        self.client = carla.Client(self.host, 2000)  # The Client connects CARLA to the server which runs the simulation
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()  # returns the world object currently active in the simulation
        self.blueprint_library = self.world.get_blueprint_library()  # Returns a list of actor blueprints available to
        # ease the spawn of these into the world.
        self.actor_list = [] # List of all actors
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
        self.spawn_points[0].location.y = self.spawn_points[0].location.y + 8
        self.spawn_points.append(transform_car_default)

        # === walker ===
        self.max_walking_speed = 5  # 18/3,6 m/s
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

        # === For Reward plot ===
        self.rewards = []
        self.colision_track = []
        self.drive_fast_near_walkers = []
        self.check_emergency_braking_track = []
        self.open_your_eyes_track = []
        self.reward_distance_track = []

    def __spawn_walker(self):
        """ Load Blueprint and spawn walker """
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
        """ Walker View Camera === """
        spectator = self.world.get_spectator()

        location = self.spawn_points[0].location
        location = carla.Location(x=147.119980, y=326.970001, z=0.300000)
        transform = carla.Transform(location, self.spawn_points[0].rotation)
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=80),
                                                carla.Rotation(pitch=-90)))

    def __spawn_car(self):
        """ Spawn Car on a given position and activate autopilot"""
        tm_port = self.set_tm_seed()

        car_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        car_sp = self.spawn_points[1]
        car = self.world.spawn_actor(car_bp, car_sp)
        self.actor_list.append(car)
        # Force to start with 8m/s
        velocity = carla.Vector3D(x=8, y=0, z=0)
        car.set_target_velocity(velocity)

        car.set_autopilot(True, tm_port)

        collision_sensor_car = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.collision'),
            carla.Transform(), attach_to=car)
        collision_sensor_car.listen(lambda event: self.collision_handler(event))
        return car, collision_sensor_car

    def draw_waypoint(self, location, index, life_time=120.0):
        """ Draw waypoints on a gien position"""
        self.world.debug.draw_string(location, str(index), draw_shadow=False,
                                     color=carla.Color(r=255, g=0, b=0), life_time=life_time,
                                     persistent_lines=True)

    def drive_fast_near_walker(self):
        """ Reward if the car drives near the walker"""
        v_car = self.car.get_velocity()
        v_ms = math.sqrt(v_car.x * v_car.x + v_car.y * v_car.y + v_car.z * v_car.z)
        if v_ms > 10 and math.dist(self.pos_walker, self.pos_car) < 3:
            print("drive_fast_near_walker")
            return 0.2
        return 0

    def check_emergency_braking(self):
        """ if the velocity change to fast -> Reward"""
        v_car = self.car.get_velocity()
        v_ms = math.sqrt(v_car.x * v_car.x + v_car.y * v_car.y + v_car.z * v_car.z)
        # Set old vel every secound
        if not self.tick_count % 20:
            self.velocity_last_sec = v_ms
        # TODO better formular then 80% loss of vel in 1 sec
        if self.velocity_last_sec > 8 and self.velocity_last_sec * 0.2 > v_ms:
            print("emergency_braking")
            return 0.1
        return 0

    def open_your_eyes(self):
        """ Punishment if the walker can see the car """
        # Ignore direction if the car is far away
        # if math.dist(self.pos_walker, self.pos_car) > 50:
        #     return 0

        direction = self.walker.get_control().direction
        dir_vec = [direction.x, direction.y]

        unit_location = [-self.pos_walker[0] + self.pos_car[0], -self.pos_walker[1] + self.pos_car[1]]
        a1 = math.degrees(self.vector_to_dir(unit_location))
        a2 = math.degrees(self.vector_to_dir(dir_vec))

        if norm_angle_deg(a1 + 90) > a2:
            return -0.02 * (norm_angle_deg(a1 + 90) - a2) / 90
        if norm_angle_deg(a1 - 90) < a2:
            return -0.02 * (a2 - norm_angle_deg(a1 - 90)) / 90

        return 0

    def reward_calculation(self):
        """ Calculate the whole reward"""
        # === Calculate Reward for RL-learning ===
        reward_distance = (-math.dist(self.pos_walker, self.pos_car)) / 1000
        coli = self.collisionReward
        self.collisionReward = 0
        dfnw = self.drive_fast_near_walker()
        ceb = self.check_emergency_braking()
        pye = self.open_your_eyes()
        reward_every_time = coli + dfnw + ceb + pye

        # For visualisation save all the previous rewards
        self.reward_distance_track.append(reward_distance)
        self.colision_track.append(coli)
        self.drive_fast_near_walkers.append(dfnw)
        self.check_emergency_braking_track.append(ceb)
        self.open_your_eyes_track.append(pye)
        self.rewards.append(reward_every_time)

        return reward_every_time

    def render(self, mode="human"):
        """ Set Render Mode in Carla"""
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
        """ Let the walker do a move/ step """

        # Factor of Max wlaker speed (0 - 1)
        walker_speed_action = action[2]
        action = np.array([action[0], action[1]])
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
        walker_speed = walker_speed_action * self.max_walking_speed
        walker_control = carla.WalkerControl(
            direction, speed=walker_speed, jump=False)
        unit_action = np.append(unit_action, [walker_speed])
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
        """ Set the seed of traffic manger"""
        # === Set Seed for TrafficManager ===
        seed_value = 0
        tm = self.client.get_trafficmanager(8000)
        tm_port = tm.get_port()
        tm.set_random_device_seed(seed_value)
        return tm_port

    def collision_handler(self, event):
        """ handle collisions and calculate extra reward """
        actor_we_collide_against = event.other_actor
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        # To ensure that the initial force does not count as collision
        if self.tick_count > 80:
            self.collisionReward = min(abs(intensity) * 100 + 0.1, 100)
        else:
            self.collisionReward = 0

        if (actor_we_collide_against.type_id == "walker.pedestrian.0012"):
            self.collisionReward = self.collisionReward + 1
            v_car = self.car.get_velocity()
            v_ms = math.sqrt(v_car.x * v_car.x + v_car.y * v_car.y + v_car.z * v_car.z)

            if v_ms > 1:
                self.collisionReward = self.collisionReward + v_ms / 10
                print("Car has velocity of: ", v_ms)
            self.done = True
            if intensity > 0:
                print("Good Hit:", self.collisionReward)
            else:
                print("Hit with Pedestrian: ", self.collisionReward)
        else:
            print("Car Collition with whatever:", self.collisionReward)

    def reset(self):
        """ Reset all values """
        self.tick_count = 0
        self.reward = 0
        self.collisionReward = 0
        self.done = False
        self.velocity_last_sec = 10
        self.info = {"actions": []}
        self.rewards = []
        self.colision_track = []
        self.drive_fast_near_walkers = []
        self.check_emergency_braking_track = []
        self.open_your_eyes_track = []
        self.reward_distance_track = []
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
        """ Combine obbervations and return it"""
        get_vel = self.walker.get_velocity()
        vel_walker = [get_vel.x, get_vel.y]

        get_vel2 = self.car.get_velocity()
        vel_car = [get_vel2.x, get_vel2.y]

        direction = self.walker.get_control().direction
        dir_vec = [direction.x, direction.y]
        observation = self.pos_car + self.pos_walker + vel_walker + vel_car + dir_vec
        observation = np.array(observation)
        return observation

    def vector_to_dir(self, vector):
        """Get the given vector's direction in radians within [-pi, pi)"""
        normed_vector = self.norm_vector(vector)
        angle = math.atan2(normed_vector[1], normed_vector[0])
        return norm_angle(angle)

    def norm_vector(self, vector):
        """Normalize the given vector to a proportional vector of length 1"""
        return self.scale_vector(vector, 1.0)

    def scale_vector(self, vector, new_len):
        """Amplify the length of the given vector"""
        old_len = vector_len(vector)
        if old_len == 0:
            return (0, 0)
        scaled_vector = (vector[0] * new_len / old_len,
                         vector[1] * new_len / old_len)
        return scaled_vector

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
        """Close environment"""
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
