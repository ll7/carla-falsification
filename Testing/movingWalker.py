#!/usr/bin/env python

import glob
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
import gym
import numpy as np

class MiniEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    collision_sensor: object

    def __init__(self):
        super(MiniEnv, self).__init__()

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

        # === set fixed time-step and synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True  # Enables synchronous mode
        settings.fixed_delta_seconds = 0.01
        self.world.apply_settings(settings)
        self.client.reload_world(False)

        self.transform_walker_default = self.world.get_map().get_spawn_points()[35]

        # === walker ===
        self.max_walking_speed = 5   # 18/3,6 m/s
        self.walker = self.__spawn_walker()
        self.collision_sensor = None

        # === Draw Start/ End Point ===
        self._set_camara_view()

    def __spawn_walker(self):
        # === Load Blueprint and spawn walker ===
        self.walker_bp = self.blueprint_library.filter('0012')[0]
        walker = self.world.spawn_actor(
            self.walker_bp, self.transform_walker_default)
        self.actor_list.append(walker)
        try:
            self.collision_sensor = self.world.spawn_actor(
                self.blueprint_library.find('sensor.other.collision'),
                carla.Transform(), attach_to=walker)
        except:
            print("collision sensor failed")
        return walker

    def _set_camara_view(self):
        # === Walker View Camera ===
        spectator = self.world.get_spectator()
        location = self.transform_walker_default.location
        # location = carla.Location(x=143.119980, y=326.970001, z=0.300000)
        transform = carla.Transform(location, self.transform_walker_default.rotation)
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=20),
                                                carla.Rotation(pitch=-90)))

    def reset_walker(self):
        try:
            self.collision_sensor.destroy()
        except:
            print("Collion Sensor")
        self.walker.destroy()
        self.walker = self.__spawn_walker()
        self.world.tick()

    def step(self, action):
        # === Let the walker do a move ===
        # action = get_round_values(action, 1)
        action_length = np.linalg.norm(action)
        if action_length == 0.0:
            # the chances are slim, but theoretically both actions could be 0.0
            unit_action = np.array([0.0, 0.0], dtype=np.float32)
        elif action_length > 1.0:
            # create a vector for the action with the length of zero
            unit_action = action / action_length
        else:
            unit_action = action

        direction = carla.Vector3D(x=float(unit_action[0]), y=float(unit_action[1]), z=0.0)
        walker_control = carla.WalkerControl(
            direction, speed=self.max_walking_speed, jump=False)
        self.walker.apply_control(walker_control)
        # === Do a tick, an check if done ===
        self.world.tick()


if __name__ == '__main__':
    env = MiniEnv()
    actions = [[-1.0, -0.3793123662471771], [-1.0, -0.17608408629894257], [-1.0, -0.5637321472167969], [-1.0, -0.47956418991088867], [-1.0, -0.48905491828918457], [-1.0, -0.5481595396995544], [-1.0, -0.5172244906425476], [-1.0, -0.6844726800918579], [-1.0, -0.9400148391723633], [-1.0, -0.27467668056488037], [-1.0, -0.27295565605163574], [-1.0, -0.22358492016792297], [-1.0, -0.705712616443634], [-1.0, -0.6426033973693848], [-1.0, -0.7309504151344299], [-1.0, -0.6361721158027649], [-1.0, -0.3716684579849243], [-1.0, -0.4094313085079193], [-1.0, -0.5057917237281799], [-1.0, -0.23639829456806183], [-1.0, -0.32823988795280457], [-1.0, -0.4487451910972595], [-1.0, -0.39464470744132996], [-1.0, -0.19187550246715546], [-1.0, -0.3388621211051941], [-1.0, -0.4014228582382202], [-1.0, -0.5363990068435669], [-1.0, -0.4235095679759979], [-1.0, -0.5569507479667664], [-1.0, -0.323492169380188], [-1.0, -0.09299829602241516], [-1.0, -0.5745701789855957], [-1.0, -0.014994174242019653], [-1.0, -0.22468581795692444], [-1.0, -0.20180265605449677], [-1.0, -0.5228874683380127], [-1.0, -0.16869549453258514], [-1.0, -0.5323735475540161], [-1.0, -0.23265765607357025], [-1.0, -0.5634411573410034], [-1.0, -0.26243215799331665], [-1.0, -0.5185160636901855], [-1.0, -0.6502723693847656], [-1.0, -0.4469561278820038], [-1.0, -0.7128753662109375], [-1.0, -0.4829714298248291], [-1.0, -0.3051680326461792], [-1.0, -0.4647364616394043], [-1.0, -0.23448999226093292], [-1.0, -0.3117094337940216], [-1.0, 0.0016753673553466797], [-1.0, -0.3930279016494751], [-1.0, -0.1387902796268463], [-1.0, -0.3807782530784607], [-1.0, -0.41243797540664673], [-1.0, -0.8231425881385803], [-1.0, -0.37870514392852783], [-1.0, -0.1899469792842865], [-1.0, -0.15304434299468994], [-1.0, -0.7598709464073181], [-1.0, -0.38118940591812134], [-1.0, -0.3759581446647644], [-1.0, -0.4784963130950928], [-1.0, -0.258968710899353], [-1.0, 0.02996009588241577], [-1.0, 0.028059229254722595], [-1.0, -0.3378080129623413], [-1.0, -0.1270674467086792], [-1.0, 0.0500432588160038], [-1.0, 0.17547041177749634], [-1.0, 0.1539633572101593], [-1.0, 0.3125724494457245], [-1.0, 0.36265233159065247], [-1.0, 0.3288215100765228], [-1.0, 0.513490617275238], [-1.0, 0.7323184609413147], [-1.0, 0.33130550384521484], [-1.0, 0.5244022607803345], [-1.0, 0.9939666986465454], [-1.0, 0.6822660565376282], [-1.0, 0.7094851136207581], [-1.0, 0.8617023229598999], [-1.0, 0.8724919557571411], [-1.0, 1.0], [-1.0, 0.9907684326171875], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 0.9550143480300903], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]

    # Ground truth
    env.reset_walker()
    positions = []
    for index, action in enumerate(actions):
        positions.append(env.walker.get_transform().location)
        env.step(action=action)
    for epoch in range(100):
        sprint = True
        env.reset_walker()
        for index, action in enumerate(actions):
            if positions[index].x != env.walker.get_transform().location.x:
                if sprint:
                    print("Oh no:", index, env.walker.get_transform().location.x, "soll:", positions[index].x)
                    sprint = False
            env.step(action=action)

        if epoch%10==0:
            print(epoch)
