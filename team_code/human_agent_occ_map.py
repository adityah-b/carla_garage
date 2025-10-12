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

# Scene descriptor
from scene_descriptor.scene_descriptor import SceneDescriptor
import cv2

# Trajectory planner
from trajectory_planner.trajectory_planner import TrajectoryPlanner
from trajectory_planner.occupancy_grid.grid_mapper import GridMapper, Maps

from pprint import pprint
from scene_analyzer.parsers.ego_plan_parser import *


import open3d as o3d

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

        # Setup scene descriptor
        self.scene_descriptor = SceneDescriptor(self.config, self.world_map)

        self.step = 0

        # Setup trajectory planner
        self.trajectory_planner = TrajectoryPlanner(self.config, self.world_map, self.ego_agent)
        self.trajectory_planner.setup_route(
            self.org_dense_route_world_coord,
            self.world,
            self.world_map,
            False,
            self.ego_agent.get_location()
        )

        self.grid_mapper = GridMapper(self.ego_agent)

        # OPEN3D VISUALIZATION
        # self.vis = o3d.visualization.Visualizer()
        # self.vis.create_window(
        #     window_name='Carla Lidar',
        #     width=960,
        #     height=540,
        #     left=480,
        #     top=270)
        # self.vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        # self.vis.get_render_option().point_size = 1
        # self.vis.get_render_option().show_coordinate_frame = True

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
            {
                'type': 'sensor.camera.rgb',
                'x': self.config.camera_pos[0],
                'y': self.config.camera_pos[1],
                'z': self.config.camera_pos[2],
                'roll': self.config.camera_rot_0[0],
                'pitch': self.config.camera_rot_0[1],
                'yaw': self.config.camera_rot_0[2],
                'width': self.config.camera_width,
                'height': self.config.camera_height,
                'fov': self.config.camera_fov,
                'id': 'rgb'
            }, {
                'type': 'sensor.camera.rgb',
                'x': self.config.camera_pos[0],
                'y': self.config.camera_pos[1],
                'z': 20.0,
                'roll': self.config.camera_rot_0[0],
                'pitch': -90,
                'yaw': 0,
                'width': self.config.camera_width,
                'height': self.config.camera_height * 2,
                'fov': self.config.camera_fov,
                'id': 'rgb_bev'
            },
            # {
            #     'type': 'sensor.camera.instance_segmentation',
            #     'x': 0.0,
            #     'y': 0.0,
            #     'z': self.config.occupancy_camera_z,
            #     'roll': 0.0,
            #     'pitch': 0.0,
            #     'yaw': 0.0,
            #     'width': self.config.occupancy_camera_height,
            #     'height': self.config.occupancy_camera_height,
            #     'fov': 90,
            #     'id': 'instance_front'
            # },
            # {
            #     'type': 'sensor.camera.depth',
            #     'x': 0.0,
            #     'y': 0.0,
            #     'z': self.config.occupancy_camera_z,
            #     'roll': 0.0,
            #     'pitch': 0.0,
            #     'yaw': 0.0,
            #     'width': self.config.occupancy_camera_height,
            #     'height': self.config.occupancy_camera_height,
            #     'fov': 90,
            #     'id': 'depth_front'
            # },
            # {
            #     'type': 'sensor.camera.instance_segmentation',
            #     'x': 0.0,
            #     'y': 0.0,
            #     'z': self.config.occupancy_camera_z,
            #     'roll': 0.0,
            #     'pitch': 0.0,
            #     'yaw': 90.0,
            #     'width': self.config.occupancy_camera_height,
            #     'height': self.config.occupancy_camera_height,
            #     'fov': 90,
            #     'id': 'instance_right'
            # }, {
            #     'type': 'sensor.camera.depth',
            #     'x': 0.0,
            #     'y': 0.0,
            #     'z': self.config.occupancy_camera_z,
            #     'roll': 0.0,
            #     'pitch': 0.0,
            #     'yaw': 90.0,
            #     'width': self.config.occupancy_camera_height,
            #     'height': self.config.occupancy_camera_height,
            #     'fov': 90,
            #     'id': 'depth_right'
            # },
            # {
            #     'type': 'sensor.camera.instance_segmentation',
            #     'x': 0.0,
            #     'y': 0.0,
            #     'z': self.config.occupancy_camera_z,
            #     'roll': 0.0,
            #     'pitch': 0.0,
            #     'yaw': 180.0,
            #     'width': self.config.occupancy_camera_height,
            #     'height': self.config.occupancy_camera_height,
            #     'fov': 90,
            #     'id': 'instance_back'
            # }, {
            #     'type': 'sensor.camera.depth',
            #     'x': 0.0,
            #     'y': 0.0,
            #     'z': self.config.occupancy_camera_z,
            #     'roll': 0.0,
            #     'pitch': 0.0,
            #     'yaw': 180.0,
            #     'width': self.config.occupancy_camera_height,
            #     'height': self.config.occupancy_camera_height,
            #     'fov': 90,
            #     'id': 'depth_back'
            # },
            # {
            #     'type': 'sensor.camera.instance_segmentation',
            #     'x': 0.0,
            #     'y': 0.0,
            #     'z': self.config.occupancy_camera_z,
            #     'roll': 0.0,
            #     'pitch': 0.0,
            #     'yaw': 270.0,
            #     'width': self.config.occupancy_camera_height,
            #     'height': self.config.occupancy_camera_height,
            #     'fov': 90,
            #     'id': 'instance_left'
            # }, {
            #     'type': 'sensor.camera.depth',
            #     'x': 0.0,
            #     'y': 0.0,
            #     'z': self.config.occupancy_camera_z,
            #     'roll': 0.0,
            #     'pitch': 0.0,
            #     'yaw': 270.0,
            #     'width': self.config.occupancy_camera_height,
            #     'height': self.config.occupancy_camera_height,
            #     'fov': 90,
            #     'id': 'depth_left'
            # },
            # {
            #     'type': 'sensor.camera.instance_segmentation',
            #     'x': 0.0,
            #     'y': 0.0,
            #     'z': 20.0,
            #     'roll': 0.0,
            #     'pitch': -90.0,
            #     'yaw': 0.0,
            #     'width': self.config.occupancy_camera_height,
            #     'height': self.config.occupancy_camera_height,
            #     'fov': self.config.camera_fov,
            #     'id': 'instance_bev'
            # }, {
            #     'type': 'sensor.camera.depth',
            #     'x': 0.0,
            #     'y': 0.0,
            #     'z': 20.0,
            #     'roll': 0.0,
            #     'pitch': -90.0,
            #     'yaw': 0.0,
            #     'width': self.config.occupancy_camera_height,
            #     'height': self.config.occupancy_camera_height,
            #     'fov': self.config.camera_fov,
            #     'id': 'depth_bev'
            # },
            {
                'type': 'sensor.lidar.ray_cast_semantic',
                'x': self.config.lidar_pos[0],
                'y': self.config.lidar_pos[1],
                'z': self.config.lidar_pos[2],
                'roll': self.config.lidar_rot[0],
                'pitch': self.config.lidar_rot[1],
                'yaw': self.config.lidar_rot[2],
                'id': 'lidar_semantic'
            }
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
        self.step +=1

        if not self.agent_engaged:
            # Setup cameras
            self.camera_tags = ['rgb', 'rgb_bev']
            cameras = [(tag, self.sensor_interface._sensors_objects[tag]) for tag in self.camera_tags]
            self.scene_descriptor.setup_cameras(cameras)

            # self.grid_mapper_cam_tags = []
            # for cam_type in ['instance', 'depth']:
            #     for cam_dir in ['front', 'left', 'right', 'back', 'bev']:
            #     # for cam_dir in ['front', 'back']:
            #         tag = cam_type + "_" + cam_dir
            #         self.grid_mapper_cam_tags.append(tag)
            # grid_mapper_cams = [(tag, self.sensor_interface._sensors_objects[tag]) for tag in self.grid_mapper_cam_tags]

            # self.grid_mapper.setup_sensors(
            #     grid_mapper_cams,
            #     self.sensor_interface._sensors_objects['lidar_semantic']
            # )

        cur_loc = self.ego_agent.get_location()
        ego_location = np.array([cur_loc.x, cur_loc.y, cur_loc.z])
        cur_wp = self.world_map.get_waypoint(cur_loc)

        # Get the list of vehicles in the scene
        actors = self.world.get_actors()
        vehicles = list(actors.filter("*vehicle*"))
        npc_vehicles = [vehicle for vehicle in vehicles if vehicle.id != self.ego_agent.id]

        # Trajectory planner
        self.trajectory_planner.update_planner(ego_location)
        planner_state = self.trajectory_planner.get_planner_state()

        # Get camera sensor object
        image_obvs = []
        for tag in self.camera_tags:
            image_obvs.append((tag, input_data[tag][1][:, :, :3]))
        self.scene_descriptor.set_camera_observations(image_obvs)

        bb_images = self.scene_descriptor.draw_actor_bounding_boxes(self.ego_agent, actors)

        bb_rgb = bb_images['rgb']
        bb_rgb_bev = bb_images['rgb_bev']

        # bb_rgb = cv2.cvtColor(bb_rgb, cv2.COLOR_BGR2RGB)
        # bb_rgb_bev = cv2.cvtColor(bb_rgb_bev, cv2.COLOR_BGR2RGB)
        bb_final = np.concatenate((bb_rgb, bb_rgb_bev), axis=0)

        cv2.namedWindow("BirdView RGB", cv2.WINDOW_NORMAL)
        cv2.imshow("BirdView RGB", bb_final)
        cv2.waitKey(1)

        # Set grid mapper camera data
        # grid_mapper_image_obvs = []
        # for tag in self.grid_mapper_cam_tags:
        #     grid_mapper_image_obvs.append((tag, input_data[tag][1][:, :, :3]))
        # self.grid_mapper.set_camera_observations(grid_mapper_image_obvs)

        semantic_lidar_data = input_data['lidar_semantic']

        maps : Maps = self.grid_mapper.update_maps(
            semantic_lidar_data[1]
        )

        occ_img = maps.occupancy_map
        visu = (occ_img * 255).astype(np.uint8)
        cv2.namedWindow("BirdView Occupancy", cv2.WINDOW_NORMAL)
        cv2.imshow('BirdView Occupancy', visu)
        cv2.waitKey(1)

        # OPEN3D VISUALIZATION
        # if self.step == 2:
        #     self.vis.add_geometry(maps.instance_map)
        # self.vis.update_geometry(maps.instance_map)

        # self.vis.poll_events()
        # self.vis.update_renderer()

        # instance_img = None
        # depth_img = None
        # for tag, img in grid_mapper_image_obvs:
        #     if "depth" in tag:
        #         if depth_img is None:
        #             depth_img = img
        #         else:
        #             depth_img = np.concatenate((depth_img, img), axis=1)
        #     else:
        #         if instance_img is None:
        #             instance_img = img
        #         else:
        #             instance_img = np.concatenate((instance_img, img), axis=1)

        # surround_img = np.concatenate((instance_img, depth_img), axis=0)
        # cv2.namedWindow("Instance + Depth", cv2.WINDOW_NORMAL)
        # cv2.imshow("Instance + Depth", surround_img)
        # cv2.waitKey(1)

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
