#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a human agent to control the ego vehicle via keyboard
"""

import numpy as np
import json

try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

import carla

from leaderboard.autoagents import autonomous_agent, autonomous_agent_local
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

from collections import defaultdict
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from scene_descriptor.data_extractors.junction_handler import JunctionHandler
from scene_descriptor.data_extractors.lane_handler import LaneHandler, WaypointUtils

# Privileged route planner
from config import GlobalConfig
from privileged_route_planner import PrivilegedRoutePlanner

# Road handler
from scene_descriptor.data_extractors.road_handler import RoadHandler

color_red     = carla.Color(r=100, g=0,   b=0)
color_green   = carla.Color(r=0,   g=100, b=0)
color_blue    = carla.Color(r=0,   g=0,   b=100)
color_yellow  = carla.Color(r=100, g=100, b=0)

def get_entry_point():
    return 'HumanAgent'

class HumanInterface(object):

    """
    Class to control a vehicle manually for debugging purposes
    """

    def __init__(self, width, height, side_scale, left_mirror=False, right_mirror=False):
        self._width = width
        self._height = height
        self._scale = side_scale
        self._surface = None

        self._left_mirror = left_mirror
        self._right_mirror = right_mirror

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode((self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Human Agent")

    def run_interface(self, input_data):
        """
        Run the GUI
        """

        # Process sensor data
        image_center = input_data['Center'][1][:, :, -2::-1]
        self._surface = pygame.surfarray.make_surface(image_center.swapaxes(0, 1))

        # Add the left mirror
        if self._left_mirror:
            image_left = input_data['Left'][1][:, :, -2::-1]
            left_surface = pygame.surfarray.make_surface(image_left.swapaxes(0, 1))
            self._surface.blit(left_surface, (0, (1 - self._scale) * self._height))

        # Add the right mirror
        if self._right_mirror:
            image_right = input_data['Right'][1][:, :, -2::-1]
            right_surface = pygame.surfarray.make_surface(image_right.swapaxes(0, 1))
            self._surface.blit(right_surface, ((1 - self._scale) * self._width, (1 - self._scale) * self._height))

        # Display image
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        pygame.display.flip()

    def set_black_screen(self):
        """Set the surface to black"""
        black_array = np.zeros([self._width, self._height])
        self._surface = pygame.surfarray.make_surface(black_array)
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        pygame.display.flip()

    def _quit(self):
        pygame.quit()


class HumanAgent(autonomous_agent_local.AutonomousAgent):

    """
    Human agent to control the ego vehicle via keyboard
    """

    current_control = None
    agent_engaged = False

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        """
        Setup the agent parameters
        """
        self.track = Track.MAP

        self.agent_engaged = False
        self.camera_width = 1280
        self.camera_height = 720
        self._side_scale = 0.3
        self._left_mirror = False
        self._right_mirror = False

        self._hic = HumanInterface(
            self.camera_width,
            self.camera_height,
            self._side_scale,
            self._left_mirror,
            self._right_mirror
        )
        self._controller = KeyboardControl(path_to_conf_file)
        self._prev_timestamp = 0

        self._clock = pygame.time.Clock()

        self.world_map = CarlaDataProvider.get_map()
        self.ego_agent = CarlaDataProvider.get_hero_actor()
        self.world = self.ego_agent.get_world()

        # Setup privileged waypoint planner
        self.config = GlobalConfig()
        self.waypoint_planner = PrivilegedRoutePlanner(self.config)

        self.waypoint_planner.setup_route(
            self.org_dense_route_world_coord,
            self.world,
            self.world_map,
            False,
            self.ego_agent.get_location()
        )

        # Setup road handler
        self.road_handler = RoadHandler(self.config, self.world_map)

    def sensors(self):
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]
        """

        sensors = [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': self.camera_width, 'height': self.camera_height, 'fov': 100, 'id': 'Center'},
        ]

        if self._left_mirror:
            sensors.append(
                {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -1.0, 'z': 1, 'roll': 0.0, 'pitch': 0.0, 'yaw': 210.0,
                 'width': self.camera_width * self._side_scale, 'height': self.camera_height * self._side_scale,
                 'fov': 100, 'id': 'Left'})

        if self._right_mirror:
            sensors.append(
                {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 1.0, 'z': 1, 'roll': 0.0, 'pitch': 0.0, 'yaw': 150.0,
                 'width': self.camera_width * self._side_scale, 'height': self.camera_height * self._side_scale,
                 'fov': 100, 'id': 'Right'})

        return sensors

    def run_step(self, input_data, timestamp, sensors=None):
        """
        Execute one step of navigation.
        """
        cur_loc = self.ego_agent.get_location()
        ego_location = np.array([cur_loc.x, cur_loc.y, cur_loc.z])

        # cur_wp = self.world_map.get_waypoint(cur_loc)
        # _, junction_wp = JunctionHandler.get_next_junction(cur_wp)
        # if not junction_wp:
        #     print(f'No junction found in next 50.0m')

        # if junction_wp:
        #     distance = cur_wp.transform.location.distance(junction_wp.transform.location)
        #     print(f'Junction found {distance} m away')
        #     lane_wp = junction_wp.previous(2.0)[0]
        #     junction_map = JunctionHandler.create_junction_map(junction_wp)
        #     lanelets = JunctionHandler.get_junction_connections(junction_map, lane_wp)

        #     if lanelets:
        #         for lanelet in lanelets:
        #             self.world.debug.draw_line(lane_wp.transform.location, lanelet.entry_connection.transform.location, color=carla.Color(255, 255, 0))
        #             self.world.debug.draw_line(lanelet.entry_connection.transform.location, lanelet.entry_junction.transform.location)
        #             self.world.debug.draw_line(lanelet.entry_junction.transform.location, lanelet.exit_junction.transform.location, color=carla.Color(0, 255, 0))
        #             self.world.debug.draw_line(lanelet.exit_junction.transform.location, lanelet.exit_connection.transform.location, color=carla.Color(0, 0, 255))

        # Get the list of vehicles in the scene
        actors = self.world.get_actors()
        vehicles = list(actors.filter("*vehicle*"))
        npc_vehicles = [vehicle for vehicle in vehicles if vehicle.id != self.ego_agent.id]

        self.waypoint_planner.run_step(ego_location)
        planner_state = self.waypoint_planner.get_planner_state()
        leading_vehicles_grouped = self.road_handler.get_leading_vehicles(planner_state, npc_vehicles)
        trailing_vehicles_grouped = self.road_handler.get_trailing_vehicles(planner_state, npc_vehicles)
        oncoming_vehicles_grouped = self.road_handler.get_oncoming_vehicles(planner_state, npc_vehicles)
        cross_vehicles_grouped = self.road_handler.get_cross_vehicles(planner_state, npc_vehicles)

        # Draw route
        # route_points = planner_state.route_points[planner_state.route_index:]
        # for i in range(min(route_points.shape[0] - 1, self.config.draw_future_route_till_distance)):
        #     loc = route_points[i]
        #     loc = carla.Location(loc[0], loc[1], loc[2] + 0.1)
        #     self.world.debug.draw_point(location=loc,
        #                                 size=0.05,
        #                                 color=self.config.future_route_color,
        #                                 life_time=self.config.draw_life_time)

        print(f'Leading Vehicles')
        for lane_name, lane_vehicles_list in leading_vehicles_grouped.items():
            print(f'\tLane Name: {lane_name}')
            for lv in lane_vehicles_list:
                ll = lv.lanelet
                ll_wps = ll.waypoints_list()
                for wp_a, wp_b in zip(ll_wps[:-1], ll_wps[1:]):
                    loc_a = wp_a.transform.location
                    loc_a = carla.Location(loc_a.x, loc_a.y, loc_a.z + 0.1)

                    loc_b = wp_b.transform.location
                    loc_b = carla.Location(loc_b.x, loc_b.y, loc_b.z + 0.1)

                    self.world.debug.draw_line(
                        loc_a, loc_b, color=color_red, life_time=self.config.draw_life_time)

                    # self.world.debug.draw_string(
                    #     loc_a,
                    #     text = f'({loc_a.x}, {loc_a.y})',
                    #     life_time=0)
                    # self.world.debug.draw_string(
                    #     loc_b,
                    #     text = f'({loc_b.x}, {loc_b.y})',
                    #     life_time=0)

                vehicles = lv.vehicles
                for v in vehicles:
                    loc = v.get_location()
                    print(f'\t\tVehicle: {v.id}')

                    self.world.debug.draw_string(
                        location=loc,
                        color=color_red,
                        text=f'{v.id}',
                        life_time=0
                    )


        print(f'Trailing Vehicles')
        for lane_name, lane_vehicles_list in trailing_vehicles_grouped.items():
            print(f'\tLane Name: {lane_name}')
            for lv in lane_vehicles_list:
                ll = lv.lanelet
                ll_wps = ll.waypoints_list()
                for wp_a, wp_b in zip(ll_wps[:-1], ll_wps[1:]):
                    loc_a = wp_a.transform.location
                    loc_a = carla.Location(loc_a.x, loc_a.y, loc_a.z + 0.1)

                    loc_b = wp_b.transform.location
                    loc_b = carla.Location(loc_b.x, loc_b.y, loc_b.z + 0.1)

                    self.world.debug.draw_line(
                        loc_a, loc_b, color=color_blue, life_time=self.config.draw_life_time)

                vehicles = lv.vehicles
                for v in vehicles:
                    loc = v.get_location()
                    print(f'\t\tVehicle: {v.id}')

                    self.world.debug.draw_string(
                        location=loc,
                        text=f'{v.id}',
                        color=color_blue,
                        life_time=0
                    )

        print(f'Oncoming Vehicles')
        for lane_name, lane_vehicles_list in oncoming_vehicles_grouped.items():
            print(f'\tLane Name: {lane_name}')
            for lv in lane_vehicles_list:
                ll = lv.lanelet
                ll_wps = ll.waypoints_list()
                for wp_a, wp_b in zip(ll_wps[:-1], ll_wps[1:]):
                    loc_a = wp_a.transform.location
                    loc_a = carla.Location(loc_a.x, loc_a.y, loc_a.z + 0.1)

                    loc_b = wp_b.transform.location
                    loc_b = carla.Location(loc_b.x, loc_b.y, loc_b.z + 0.1)

                    self.world.debug.draw_line(
                        loc_a, loc_b, color=color_green, life_time=self.config.draw_life_time)

                vehicles = lv.vehicles
                for v in vehicles:
                    loc = v.get_location()
                    print(f'\t\tVehicle: {v.id}')

                    self.world.debug.draw_string(
                        location=loc,
                        text=f'{v.id}',
                        color=color_green,
                        life_time=0
                    )

        print(f'Cross Vehicles')
        for lane_name, lane_vehicles_list in cross_vehicles_grouped.items():
            print(f'\tLane Name: {lane_name}')
            for lv in lane_vehicles_list:
                ll = lv.lanelet
                ll_wps = ll.waypoints_list()
                for wp_a, wp_b in zip(ll_wps[:-1], ll_wps[1:]):
                    loc_a = wp_a.transform.location
                    loc_a = carla.Location(loc_a.x, loc_a.y, loc_a.z + 0.2)

                    loc_b = wp_b.transform.location
                    loc_b = carla.Location(loc_b.x, loc_b.y, loc_b.z + 0.2)

                    self.world.debug.draw_line(
                        loc_a, loc_b, color=color_yellow, life_time=0)

                vehicles = lv.vehicles
                for v in vehicles:
                    loc = v.get_location()
                    print(f'\t\tVehicle: {v.id}')

                    self.world.debug.draw_string(
                        location=loc,
                        text=f'{v.id}',
                        color=color_yellow,
                        life_time=0
                    )

        self._clock.tick_busy_loop(20)
        self.agent_engaged = True
        self._hic.run_interface(input_data)

        control = self._controller.parse_events(timestamp - self._prev_timestamp)
        self._prev_timestamp = timestamp

        return control

    def destroy(self, results=None):
        """
        Cleanup
        """
        self._hic.set_black_screen()
        self._hic._quit = True


class KeyboardControl(object):

    """
    Keyboard control for the human agent
    """

    def __init__(self, path_to_conf_file):
        """
        Init
        """
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0
        self._clock = pygame.time.Clock()

        # Get the mode
        if path_to_conf_file:

            with (open(path_to_conf_file, "r")) as f:
                lines = f.read().split("\n")
                self._mode = lines[0].split(" ")[1]
                self._endpoint = lines[1].split(" ")[1]

            # Get the needed vars
            if self._mode == "log":
                self._log_data = {'records': []}

            elif self._mode == "playback":
                self._index = 0
                self._control_list = []

                with open(self._endpoint) as fd:
                    try:
                        self._records = json.load(fd)
                        self._json_to_control()
                    except json.JSONDecodeError:
                        pass
        else:
            self._mode = "normal"
            self._endpoint = None

    def _json_to_control(self):

        # transform strs into VehicleControl commands
        for entry in self._records['records']:
            control = carla.VehicleControl(throttle=entry['control']['throttle'],
                                           steer=entry['control']['steer'],
                                           brake=entry['control']['brake'],
                                           hand_brake=entry['control']['hand_brake'],
                                           reverse=entry['control']['reverse'],
                                           manual_gear_shift=entry['control']['manual_gear_shift'],
                                           gear=entry['control']['gear'])
            self._control_list.append(control)

    def parse_events(self, timestamp):
        """
        Parse the keyboard events and set the vehicle controls accordingly
        """
        # Move the vehicle
        if self._mode == "playback":
            self._parse_json_control()
        else:
            self._parse_vehicle_keys(pygame.key.get_pressed(), timestamp*1000)

        # Record the control
        if self._mode == "log":
            self._record_control()

        return self._control

    def _parse_vehicle_keys(self, keys, milliseconds):
        """
        Calculate new vehicle controls based on input keys
        """

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.KEYUP:
                if event.key == K_q:
                    self._control.gear = 1 if self._control.reverse else -1
                    self._control.reverse = self._control.gear < 0

        if keys[K_UP] or keys[K_w]:
            self._control.throttle = 0.8
        else:
            self._control.throttle = 0.0

        steer_increment = 3e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0

        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_json_control(self):

        if self._index < len(self._control_list):
            self._control = self._control_list[self._index]
            self._index += 1
        else:
            print("JSON file has no more entries")

    def _record_control(self):
        new_record = {
            'control': {
                'throttle': self._control.throttle,
                'steer': self._control.steer,
                'brake': self._control.brake,
                'hand_brake': self._control.hand_brake,
                'reverse': self._control.reverse,
                'manual_gear_shift': self._control.manual_gear_shift,
                'gear': self._control.gear
            }
        }

        self._log_data['records'].append(new_record)

    def __del__(self):
        # Get ready to log user commands
        if self._mode == "log" and self._log_data:
            with open(self._endpoint, 'w') as fd:
                json.dump(self._log_data, fd, indent=4, sort_keys=True)
