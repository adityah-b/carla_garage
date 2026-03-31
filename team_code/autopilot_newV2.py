"""
Privileged driving agent used for data collection.
Drives by accessing the simulator directly.
"""

import os
import ujson
import pathlib
import gzip
import cv2
import math
import numpy as np
import carla
import logging

import transfuser_utils as t_u

from typing import List, Dict

from collections import deque
from agents.navigation.local_planner import RoadOption
from scipy.integrate import RK45
from datetime import datetime

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.autoagents import autonomous_agent, autonomous_agent_local
from nav_planner import RoutePlanner
from lateral_controller import LateralPIDController
from privileged_route_planner import PrivilegedRoutePlanner
from config import GlobalConfig
from scenario_logger_v2 import ScenarioLogger
from longitudinal_controller import LongitudinalLinearRegressionController, LongitudinalPIDController
from kinematic_bicycle_model import KinematicBicycleModel

logger = logging.getLogger(__name__)

# ------------- NEW CODE ------------- #

########################################
# PERCEPTION
########################################

# Scene descriptor
from scene_descriptor.scene_descriptor import SceneDescriptor

# Scene analyzer
from scene_analyzer.scene_analyzer import SceneAnalyzer

########################################
# PREDICTION
########################################

# Motion predictor
from actor_prediction.motion_prediction import MotionPrediction

# Collision checker
from actor_prediction.collision_checker import CollisionChecker

########################################
# PLANNING
########################################

# Trajectory planner
from trajectory_planner.trajectory_planner import TrajectoryPlanner

# Longitudinal planner
from local_planner.longitudinal.config_specs import *
from local_planner.longitudinal.long_planner import LongPlanner

# Lateral planner
from local_planner.lateral.config_specs import *
from local_planner.lateral.lat_planner import LatPlanner

# ------------- NEW CODE ------------- #

# from scene_analyzer.parsers.ego_plan_parser import *
from scene_analyzer.parsers.ego_plan_pydantic_models import *

def get_entry_point():
  return "AutoPilot"

class AutoPilot(autonomous_agent_local.AutonomousAgent):
  """
      Privileged driving agent used for data collection.
      Drives by accessing the simulator directly.
      """

  def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
    """
        Set up the autonomous agent for the CARLA simulation.

        Args:
            config_file_path (str): Path to the configuration file.
            route_index (int, optional): Index of the route to follow.
            traffic_manager (object, optional): The traffic manager object.

        """
    self.recording = False
    self.track = autonomous_agent.Track.MAP
    self.config_path = path_to_conf_file
    self.step = -1
    self.initialized = False
    self.save_path = None
    self.route_index = route_index

    self.datagen = int(os.environ.get("DATAGEN", 0)) == 1

    self.config = GlobalConfig()

    self.speed_histogram = []
    self.make_histogram = int(os.environ.get("HISTOGRAM", 0))

    self.tp_stats = False
    self.tp_sign_agrees_with_angle = []
    if int(os.environ.get("TP_STATS", 0)):
      self.tp_stats = True

    # Dynamics models
    self.ego_model = KinematicBicycleModel(self.config)
    self.vehicle_model = KinematicBicycleModel(self.config)

    # Configuration
    self.visualize = int(os.environ.get("DEBUG_CHALLENGE", 0))

    self.walker_close = False
    self.distance_to_walker = np.inf
    self.stop_sign_close = False
    self.waiting_ticks_at_stop_sign = 0

    # To avoid failing the ActorBlockedTest, the agent has to move at least 0.1 m/s every 179 ticks
    self.ego_blocked_for_ticks = 0

    # Controllers
    self._turn_controller = LateralPIDController(self.config)

    self.list_traffic_lights = []

    # Navigation command buffer, needed because the correct command comes from the last cleared waypoint
    self.commands = deque(maxlen=2)
    self.commands.append(4)
    self.commands.append(4)
    self.next_commands = deque(maxlen=2)
    self.next_commands.append(4)
    self.next_commands.append(4)
    self.target_point_prev = [1e5, 1e5, 1e5]

    # Initialize controls
    self.steer = 0.0
    self.throttle = 0.0
    self.brake = 0.0

    self.augmentation_translation = 0
    self.augmentation_rotation = 0

    # Angle to the next waypoint, normalized in [-1, 1] corresponding to [-90, 90]
    self.angle = 0.0
    self.stop_sign_hazard = False
    self.traffic_light_hazard = False
    self.walker_hazard = False
    self.vehicle_hazard = False
    self.junction = False
    self.aim_wp = None  # Waypoint the expert is steering towards
    self.remaining_route = None  # Remaining route
    self.remaining_route_original = None  # Remaining original route
    self.close_traffic_lights = []
    self.close_stop_signs = []
    self.was_at_stop_sign = False
    self.cleared_stop_sign = False
    self.visible_walker_ids = []
    self.walker_past_pos = {}  # Position of walker in the last frame

    self._vehicle_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam

    # Get the world map and the ego vehicle
    self.world_map = CarlaDataProvider.get_map()

    # ------------- NEW CODE ------------- #
    self.traffic_manager = traffic_manager

    # ------------- NEW CODE ------------- #

    # Set up the save path if specified
    if os.environ.get("SAVE_PATH", None) is not None:
      self.save_path = pathlib.Path(os.environ["SAVE_PATH"]) / f"route{self.route_index}"
      self.save_path.mkdir(parents=True, exist_ok=True)

      if self.datagen:
        (self.save_path / "measurements").mkdir()

      self.lon_logger = ScenarioLogger(
          save_path=self.save_path,
          route_index=route_index,
          logging_freq=self.config.logging_freq,
          log_only=True,
          route_only=False,  # with vehicles
          roi=self.config.logger_region_of_interest,
      )

  def toggle_recording(self, force_stop=False):
    """
        Toggle the recording of the simulation data.

        Args:
            force_stop (bool, optional): If True, stop the recording regardless of the current state.
        """
    # Toggle the recording state and determine the text
    self.recording = not self.recording

    if self.recording and not force_stop:
      self.client = CarlaDataProvider.get_client()

      # Determine the scenario name and number
      scenario_name = pathlib.Path(self.config_path).parent.stem
      scenario_number = pathlib.Path(self.config_path).stem

      # Construct the log file path
      log_file_name = f"{scenario_name}_{scenario_number}"
      log_path = f"{self.save_path / log_file_name}.log"
      # log_path = f"{pathlib.Path(os.environ['SAVE_PATH'])}/{scenario_name}/{scenario_number}.log"

      print(f"Saving to {log_path}")
      pathlib.Path(os.path.dirname(log_path)).mkdir(parents=True, exist_ok=True)

      # Start the recorder with the specified log path
      # self.client.start_recorder(log_path, True)
      self.client.start_recorder(log_path, False)
    else:
      # Stop the recorder
      self.client.stop_recorder()

  def _init(self, hd_map):
    """
        Initialize the agent by setting up the route planner, longitudinal controller,
        command planner, and other necessary components.

        Args:
            hd_map (carla.Map): The map object of the CARLA world.
        """
    print("Sparse Waypoints:", len(self._global_plan))
    print("Dense Waypoints:", len(self.org_dense_route_world_coord))

    # Get the hero vehicle and the CARLA world
    self._vehicle = CarlaDataProvider.get_hero_actor()
    self._world = self._vehicle.get_world()

    logger.debug('Generating waypoints...')

    self.map_waypoints = self.world_map.generate_waypoints(2.0)
    self.crosswalks = self.world_map.get_crosswalks()

    self._world.tick()

    logger.debug('Waypoints generated.')

    # Visualizer
    # self.visualizer = Visualizer(self._vehicle)

    # Check if the vehicle starts from a parking spot
    distance_to_road = self.org_dense_route_world_coord[0][0].location.distance(self._vehicle.get_location())
    # The first waypoint starts at the lane center, hence it's more than 2 m away from the center of the
    # ego vehicle at the beginning.
    starts_with_parking_exit = distance_to_road > 2

    # Set up the route planner and extrapolation
    self._waypoint_planner = PrivilegedRoutePlanner(self.config)
    self._waypoint_planner.setup_route(self.org_dense_route_world_coord, self._world, self.world_map,
                                       starts_with_parking_exit, self._vehicle.get_location())

    # ------------- NEW CODE ------------- #

    ########################################
    # PERCEPTION
    ########################################

    # Scene descriptor
    self.scene_descriptor = SceneDescriptor(self.config, self.world_map)

    # Scene analyzer
    scene_analyzer_config = self.config.scene_analyzer_config
    self.scene_analyzer = SceneAnalyzer(
      provider=scene_analyzer_config['provider'],
      model_name=scene_analyzer_config['model_name'],
      temperature=scene_analyzer_config['temperature'],
      max_output_tokens=scene_analyzer_config['max_output_tokens'],
    )

    ########################################
    # PREDICTION
    ########################################

    # Motion prediction
    self.motion_prediction = MotionPrediction(self.config, self.world_map)

    ########################################
    # PLANNING
    ########################################

    # Trajectory planner
    self.trajectory_planner = TrajectoryPlanner(self.config, self.world_map, self._vehicle)

    self.trajectory_planner.setup_route(
      self.org_dense_route_world_coord, self._world, self.world_map, starts_with_parking_exit, self._vehicle.get_location()
    )
    # ------------- NEW CODE ------------- #

    self._waypoint_planner.save()

    # Set up the longitudinal controller and command planner
    self._longitudinal_controller = LongitudinalLinearRegressionController(self.config)
    self._command_planner = RoutePlanner(self.config.route_planner_min_distance, self.config.route_planner_max_distance)
    self._command_planner.set_route(self._global_plan_world_coord)

    # Set up logging
    if self.save_path is not None:
      self.lon_logger.ego_vehicle = self._vehicle
      self.lon_logger.world = self._world

    # Preprocess traffic lights
    all_actors = self._world.get_actors()
    for actor in all_actors:
      if "traffic_light" in actor.type_id:
        center, waypoints = t_u.get_traffic_light_waypoints(actor, self.world_map)
        self.list_traffic_lights.append((actor, center, waypoints))

    # Remove bugged 2-wheelers
    # https://github.com/carla-simulator/carla/issues/3670
    for actor in all_actors:
      if "vehicle" in actor.type_id:
        extent = actor.bounding_box.extent
        if extent.x < 0.001 or extent.y < 0.001 or extent.z < 0.001:
          actor.destroy()

    # ------------- NEW CODE ------------- #
    # Setup cameras
    self.camera_tags = ['rgb', 'rgb_bev']
    cameras = [(tag, self.sensor_interface._sensors_objects[tag]) for tag in self.camera_tags]
    self.scene_descriptor.setup_cameras(cameras)

    # ------------- NEW CODE ------------- #

    self.initialized = True

  def sensors(self):
    """
        Returns a list of sensor specifications for the ego vehicle.

        Each sensor specification is a dictionary containing the sensor type,
        reading frequency, position, and other relevant parameters.

        Returns:
            list: A list of sensor specification dictionaries.
        """
    sensor_specs = [
      {
        "type": "sensor.opendrive_map",
        "reading_frequency": 1e-6,
        "id": "hd_map"
      }, {
        "type": "sensor.other.imu",
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "sensor_tick": 0.05,
        "id": "imu"
      }, {
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
      }, {
        'type': 'sensor.lidar.ray_cast_semantic',
        'x': self.config.lidar_pos[0],
        'y': self.config.lidar_pos[1],
        'z': self.config.lidar_pos[2],
        'roll': self.config.lidar_rot[0],
        'pitch': self.config.lidar_rot[1],
        'yaw': self.config.lidar_rot[2],
        'id': 'lidar_semantic'
      }, {
        "type": "sensor.speedometer",
        "reading_frequency": 20,
        "id": "speed"
      }, {
        "type": "sensor.other.collision",
        "id": "collision"
      }

    ]

    return sensor_specs

  def tick_autopilot(self, input_data):
    """
        Get the current state of the vehicle from the input data and the vehicle's sensors.

        Args:
            input_data (dict): Input data containing sensor information.

        Returns:
            dict: A dictionary containing the vehicle's position (GPS), speed, and compass heading.
        """
    # Get the vehicle's speed from its velocity vector
    speed = self._vehicle.get_velocity().length()

    # Get the IMU data from the input data
    imu_data = input_data["imu"][1][-1]

    # Preprocess the compass data from the IMU
    compass = t_u.preprocess_compass(imu_data)

    # Get the vehicle's position from its location
    position = self._vehicle.get_location()
    gps = np.array([position.x, position.y, position.z])

    # Create a dictionary containing the vehicle's state
    vehicle_state = {
        "gps": gps,
        "speed": speed,
        "compass": compass,
    }

    return vehicle_state

  def run_step(self, input_data, timestamp, sensors=None, plant=False):
    """
        Run a single step of the agent's control loop.

        Args:
            input_data (dict): Input data for the current step.
            timestamp (float): Timestamp of the current step.
            sensors (list, optional): List of sensor objects. Default is None.
            plant (bool, optional): Flag indicating whether to run the plant simulation or not. Default is False.

        Returns:
            If plant is False, it returns the control commands (steer, throttle, brake).
            If plant is True, it returns the driving data for the current step.
        """
    self.step += 1

    # Initialize the agent if not done yet
    if not self.initialized:
      client = CarlaDataProvider.get_client()
      world_map = client.get_world().get_map()
      self._init(world_map)

    # Get the control commands and driving data for the current step
    control = self._get_control(input_data, plant)

    return control

  def _get_control(self, input_data, plant):
    """
        Compute the control commands and save the driving data for the current frame.

        Args:
            input_data (dict): Input data for the current frame.
            plant (object): The plant object representing the vehicle dynamics.

        Returns:
            tuple: A tuple containing the control commands (steer, throttle, brake) and the driving data.
        """
    tick_data = self.tick_autopilot(input_data)
    ego_position = tick_data["gps"]

    # Waypoint planning and route generation
    route_np, route_wp, _, distance_to_next_traffic_light, next_traffic_light, distance_to_next_stop_sign,\
                                    next_stop_sign, speed_limit = self._waypoint_planner.run_step(ego_position)

    # Extract relevant route information
    self.remaining_route = route_np[self.config.tf_first_checkpoint_distance:][::self.config.points_per_meter]
    self.remaining_route_original = self._waypoint_planner.original_route_points[self._waypoint_planner.route_index:][
        self.config.tf_first_checkpoint_distance:][::self.config.points_per_meter]

    # Get the current speed and target speed
    ego_speed = tick_data["speed"]
    target_speed_initial = min(speed_limit * self.config.ratio_target_speed_limit, 72. / 3.6)  # merge the two last speed bins

    # Reduce target speed if there is a junction ahead
    for i in range(min(self.config.max_lookahead_to_check_for_junction, len(route_wp))):
      if route_wp[i].is_junction:
        target_speed_initial = min(target_speed_initial, self.config.max_speed_in_junction)
        break

    # ------------- NEW CODE ------------- #

    # Update planner state
    self.trajectory_planner.update_planner(ego_position)

    # Get planner state
    planner_state = self.trajectory_planner.get_planner_state()

    ########################################
    # PERCEPTION
    ########################################

    # Get all actors in the scene
    actors = self._world.get_actors()

    # Compile lidar sensor measurements
    lidar_data = {
        'sensor' : self.sensor_interface._sensors_objects['lidar_semantic'],
        'raw_data' : input_data['lidar_semantic'][1]
    }

    # Compile collision sensor measurements
    collision_data = None
    collision_sensor_meas = input_data.get('collision')
    if collision_sensor_meas:
        collision_data = collision_sensor_meas[1]

    # Process scene
    scene_context = self.scene_descriptor.process_complete_scene(
      self._vehicle,
      actors,
      planner_state,
      lidar_data,
      collision_sensor_data=collision_data,
    )

    # Get formatted text (dense) and scene summary (simplified, for VLM)
    scene_text = scene_context.formatted_text
    scene_summary = scene_context.scene_summary

    # _______ IMAGE RENDERING AND SAVING _______

    # Get camera sensor object
    image_obvs = []
    for tag in self.camera_tags:
        image_obvs.append((tag, input_data[tag][1][:, :, :3]))
    self.scene_descriptor.set_camera_observations(image_obvs)

    bb_images = self.scene_descriptor.draw_actor_bounding_boxes(self._vehicle, scene_context.scene_data)

    bb_rgb = bb_images['rgb']
    bb_rgb_bev = bb_images['rgb_bev']

    bb_final = np.concatenate((bb_rgb, bb_rgb_bev), axis=0)

    cv2.namedWindow("BirdView RGB", cv2.WINDOW_NORMAL)
    cv2.imshow("BirdView RGB", bb_final)
    cv2.waitKey(1)

    if self.step % int(self.config.carla_fps // 3) == 0:
      # Ensure the output directory exists only once
      if not hasattr(self, '_output_dir'):
          current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
          self._output_dir = f"runs/run_{current_time}"
          os.makedirs(self._output_dir, exist_ok=True)
      output_dir = self._output_dir

      # Save the image to the output directory
      if not hasattr(self, '_image_counter'):
          self._image_counter = 0
      self._image_counter += 1
      output_path = os.path.join(output_dir, f"rgb_bounding_boxes_{self._image_counter:04d}.png")
      cv2.imwrite(output_path, bb_final)

      text_output_path = os.path.join(output_dir, f"scene_context_{self._image_counter:04d}.txt")
      with open(text_output_path, 'w') as f:
        f.write(scene_text)

    # _______ IMAGE RENDERING AND SAVING _______

    print(f'\n\nStructured Data\n\n')
    print(f'{scene_text}')

    # Update scene data
    self.trajectory_planner.update_scene_data(scene_context, lidar_data['raw_data'])

    plan_with_reasoning = False
    ego_plan = None
    hl_beh   = None
    if self.step <= 20:
      target_speed = 0.0
      brake = True
    else:
      # Check if LLM reasoning required
      plan_with_reasoning = self.trajectory_planner.plan_with_reasoning()
      if plan_with_reasoning:
        if self.trajectory_planner.cur_plan is None and self.trajectory_planner.needs_replan is True:
          # Log previous episode to memory if it completed cleanly (no collisions)
          prev_plan_state = self.trajectory_planner.prev_plan_state
          if (prev_plan_state is not None
                and prev_plan_state.status == PlanStatus.FINISHED
                and not prev_plan_state.collision_events):
            self.scene_analyzer.log_episode(prev_plan_state)

          failed_prev_plan = (
            prev_plan_state
            if prev_plan_state is not None and prev_plan_state.status == PlanStatus.FAILED
            else None
          )
          hl_beh = self.scene_analyzer.get_high_level_behaviour(
            scene_summary=scene_summary,
            scene_image=bb_final,
            prev_plan_state=failed_prev_plan,
          )
          print(f'\n\nHigh Level Behaviour:')
          print(f'{hl_beh.to_string()}')

          # Generate ego plan
          ego_plan = self.scene_analyzer.get_ego_plan(
              text=hl_beh.to_string(),
              hl_beh=hl_beh,
              image=bb_rgb,
              prev_plan=self.trajectory_planner.prev_plan_state,
          )

          # ego_plan = self.scene_analyzer.get_ego_plan(
          #     text=hl_beh.to_string(),
          #     image=bb_final,
          #     prev_plan=self.trajectory_planner.prev_plan_execution,
          # )
          # NOTE: BYPASSING HIGH LEVEL ANALYZER, FEEDING TEXT AND IMAGE DIRECTLY TO EGO PLANNER
          # ego_plan = self.scene_analyzer.get_ego_plan(
          #     text=scene_text,
          #     prev_plan=self.trajectory_planner.prev_plan_execution,
          # )

          # hl_beh = HighLevelBehaviour(
          #     key_actors=[],
          #     traffic_objects=[],
          #     obstacles=[],
          #     next_action="Follow route",
          #     reasoning=["Placeholder high-level behaviour."]
          # )

          # ego_plan = EgoPlan(
          #     action=Action.OVERTAKE_LEFT,
          #     target_speed=25.5,
          #     conditions=[],
          #     reasoning=[
          #         "Detected oncoming vehicle at intersection.",
          #         "Waiting for clear gap before initiating left turn.",
          #         "Maintaining safety margin based on target speed."
          #     ]
          # )

          print(f'\n\nEgo Plan:')
          print(f'{ego_plan.to_string()}')

          # Set the plan — capture scene state at decision time for later memory logging
          self.trajectory_planner.set_plan(ego_plan, hl_beh=hl_beh, scene_text=scene_summary, scene_image=bb_rgb)

      target_speed, brake, route_np, route_wp = self.trajectory_planner.execute_plan(self.step)

    max_route_length = len(planner_state.route_waypoints)
    look_ahead_points = int(self.config.draw_future_route_till_distance)
    to_index = min(max_route_length - 1, planner_state.route_index + look_ahead_points)

    for i in range(planner_state.route_index, to_index, self.config.points_per_meter):
        loc = planner_state.route_points[i]
        loc = carla.Location(loc[0], loc[1], loc[2] + 0.1)
        self._world.debug.draw_point(location=loc,
                                    size=0.05,
                                    color=self.config.future_route_color,
                                    life_time=self.config.draw_life_time)

        # route_cmd = planner_state.route_commands[i]
        # cmd_str = ''
        # if route_cmd == RoadOption.LEFT:
        #   cmd_str = 'left'
        # elif route_cmd == RoadOption.RIGHT:
        #   cmd_str = 'right'
        # elif route_cmd == RoadOption.STRAIGHT:
        #   cmd_str = 'straight'
        # # elif route_cmd == RoadOption.LANEFOLLOW:
        # #   cmd_str = 'lanefollow'
        # elif route_cmd == RoadOption.CHANGELANELEFT:
        #   cmd_str = 'CHANGELANELEFT'
        # elif route_cmd == RoadOption.CHANGELANERIGHT:
        #   cmd_str = 'CHANGELANERIGHT'

        # self._world.debug.draw_string(
        #   location=loc,
        #   text=cmd_str,
        #   color=self.config.other_vehicles_forecasted_bbs_color,
        #   life_time=self.config.draw_life_time
        # )

    # ------------- NEW CODE ------------- #

    # Determine if the ego vehicle is at a junction
    ego_vehicle_waypoint = self.world_map.get_waypoint(self._vehicle.get_location())
    self.junction = ego_vehicle_waypoint.is_junction

    # Compute throttle and brake control
    throttle, control_brake = self._longitudinal_controller.get_throttle_and_brake(brake, target_speed, ego_speed)

    # Compute steering control
    steer = self._get_steer(route_np, ego_position, tick_data["compass"], ego_speed)

    print(f'\n\nAUTOPILOT')
    print(f'\tCurrent Speed: {ego_speed}')
    print(f'\tTarget Speed: {target_speed}')
    print(f'\tThrottle: {throttle}')
    print(f'\tBrake: {brake}')
    print(f'\tControl Brake: {control_brake}')
    print(f'\tSteer: {steer}')

    # Create the control command
    control = carla.VehicleControl()
    control.steer = steer + self.config.steer_noise * np.random.randn()
    control.throttle = throttle
    control.brake = float(brake or control_brake)

    # Apply brake if the vehicle is stopped to prevent rolling back
    if control.throttle == 0 and ego_speed < self.config.minimum_speed_to_prevent_rolling_back:
      control.brake = 1

    # Apply throttle if the vehicle is blocked for too long
    ego_velocity = CarlaDataProvider.get_velocity(self._vehicle)
    if ego_velocity < 0.1:
      self.ego_blocked_for_ticks += 1
    else:
      self.ego_blocked_for_ticks = 0

    if self.ego_blocked_for_ticks >= self.config.max_blocked_ticks:
      control.throttle = 1
      control.brake = 0

    # Save control commands and target speed
    self.steer = control.steer
    self.throttle = control.throttle
    self.brake = control.brake

    # Get the target and next target points from the command planner
    command_route = self._command_planner.run_step(ego_position)
    if len(command_route) > 2:
      target_point, far_command = command_route[1]
      next_target_point, next_far_command = command_route[2]
    elif len(command_route) > 1:
      target_point, far_command = command_route[1]
      next_target_point, next_far_command = command_route[1]
    else:
      target_point, far_command = command_route[0]
      next_target_point, next_far_command = command_route[0]

    # Update command history and save driving datas
    if (target_point != self.target_point_prev).all():
      self.target_point_prev = target_point
      self.commands.append(far_command.value)
      self.next_commands.append(next_far_command.value)

    # Log this step
    force_log = (
      (collision_data is not None) or
      (hl_beh is not None) or
      (ego_plan is not None)
    )

    if self.save_path is not None:
      self.lon_logger.log_step(
          planner_state=planner_state,
          scene_context=scene_context,
          prediction_data=self.trajectory_planner.prediction_data,
          long_planner_result=self.trajectory_planner.long_planner.current_plan,
          lat_planner_result=self.trajectory_planner.lat_planner.current_plan,
          ego_control=control,
          ego_plan=ego_plan or self.trajectory_planner.cur_plan,
          high_level_beh=hl_beh,
          force=force_log,
          plan_with_reasoning=plan_with_reasoning,
      )

    return control

  def save(self, target_point, next_target_point, steering, throttle, brake, control_brake, target_speed, speed_limit,
           tick_data, speed_reduced_by_obj):
    """
        Save the driving data for the current frame.

        Args:
            target_point (numpy.ndarray): Coordinates of the target point.
            next_target_point (numpy.ndarray): Coordinates of the next target point.
            steering (float): The steering angle for the current frame.
            throttle (float): The throttle value for the current frame.
            brake (float): The brake value for the current frame.
            control_brake (bool): Whether the brake is controlled by the agent or not.
            target_speed (float): The target speed for the current frame.
            speed_limit (float): The speed limit for the current frame.
            tick_data (dict): Dictionary containing the current state of the vehicle.
            speed_reduced_by_obj (tuple): Tuple containing information about the object that caused speed reduction.

        Returns:
            dict: A dictionary containing the driving data for the current frame.
        """
    frame = self.step // self.config.data_save_freq

    # Extract relevant data from inputs
    target_point_2d = target_point[:2]
    next_target_point_2d = next_target_point[:2]
    ego_position = tick_data["gps"][:2]
    ego_orientation = tick_data["compass"]
    ego_speed = tick_data["speed"]

    # Convert target points to ego vehicle's local coordinate frame
    ego_target_point = t_u.inverse_conversion_2d(target_point_2d, ego_position, ego_orientation).tolist()
    ego_next_target_point = t_u.inverse_conversion_2d(next_target_point_2d, ego_position, ego_orientation).tolist()
    ego_aim_point = t_u.inverse_conversion_2d(self.aim_wp[:2], ego_position, ego_orientation).tolist()

    # Get the remaining route points in the local coordinate frame
    dense_route = []
    dense_route_original = []
    remaining_route = self.remaining_route[:self.config.num_route_points_saved]
    remaining_route_original = self.remaining_route_original[:self.config.num_route_points_saved]

    changed_route = bool((self._waypoint_planner.route_points[self._waypoint_planner.route_index] !=
                          self._waypoint_planner.original_route_points[self._waypoint_planner.route_index]).any())
    for (checkpoint, checkpoint_original) in zip(remaining_route, remaining_route_original):
      dense_route.append(t_u.inverse_conversion_2d(checkpoint[:2], ego_position[:2], ego_orientation).tolist())
      dense_route_original.append(
          t_u.inverse_conversion_2d(checkpoint_original[:2], ego_position[:2], ego_orientation).tolist())

    # Extract speed reduction object information
    speed_reduced_by_obj_type, speed_reduced_by_obj_id, speed_reduced_by_obj_distance = None, None, None
    if speed_reduced_by_obj is not None:
      speed_reduced_by_obj_type, speed_reduced_by_obj_id, speed_reduced_by_obj_distance = speed_reduced_by_obj[1:]
      # Convert numpy to float so that it can be saved to json.
      if speed_reduced_by_obj_distance is not None:
        speed_reduced_by_obj_distance = float(speed_reduced_by_obj_distance)

    data = {
        "pos_global": ego_position.tolist(),
        "theta": ego_orientation,
        "speed": ego_speed,
        "target_speed": target_speed,
        "speed_limit": speed_limit,
        "target_point": ego_target_point,
        "target_point_next": ego_next_target_point,
        "command": self.commands[-2],
        "next_command": self.next_commands[-2],
        "aim_wp": ego_aim_point,
        "route": dense_route,
        "route_original": dense_route_original,
        "changed_route": changed_route,
        "speed_reduced_by_obj_type": speed_reduced_by_obj_type,
        "speed_reduced_by_obj_id": speed_reduced_by_obj_id,
        "speed_reduced_by_obj_distance": speed_reduced_by_obj_distance,
        "steer": steering,
        "throttle": throttle,
        "brake": bool(brake),
        "control_brake": bool(control_brake),
        "junction": bool(self.junction),
        "vehicle_hazard": bool(self.vehicle_hazard),
        "vehicle_affecting_id": self.vehicle_affecting_id,
        "light_hazard": bool(self.traffic_light_hazard),
        "walker_hazard": bool(self.walker_hazard),
        "walker_affecting_id": self.walker_affecting_id,
        "stop_sign_hazard": bool(self.stop_sign_hazard),
        "stop_sign_close": bool(self.stop_sign_close),
        "walker_close": bool(self.walker_close),
        "walker_close_id": self.walker_close_id,
        "angle": self.angle,
        "augmentation_translation": self.augmentation_translation,
        "augmentation_rotation": self.augmentation_rotation,
        "ego_matrix": self._vehicle.get_transform().get_matrix()
    }

    if self.tp_stats:
      deg_pred_angle = -math.degrees(math.atan2(-ego_aim_point[1], ego_aim_point[0]))

      tp_angle = -math.degrees(math.atan2(-ego_target_point[1], ego_target_point[0]))
      if abs(tp_angle) > 1.0 and abs(deg_pred_angle) > 1.0:
        same_direction = float(tp_angle * deg_pred_angle >= 0.0)
        self.tp_sign_agrees_with_angle.append(same_direction)

    if ((self.step % self.config.data_save_freq == 0) and (self.save_path is not None) and self.datagen):
      measurements_file = self.save_path / "measurements" / f"{frame:04}.json.gz"
      with gzip.open(measurements_file, "wt", encoding="utf-8") as f:
        ujson.dump(data, f, indent=4)

    return data

  def destroy(self, results=None):
    """
        Save the collected data and statistics to files, and clean up the data structures.
        This method should be called at the end of the data collection process.

        Args:
            results (optional): Any additional results to be processed or saved.
        """
    if self.save_path is not None:
      self.lon_logger.dump_to_json()

      # Save the target speed histogram to a compressed JSON file
      if len(self.speed_histogram) > 0:
        with gzip.open(self.save_path / "target_speeds.json.gz", "wt", encoding="utf-8") as f:
          ujson.dump(self.speed_histogram, f, indent=4)

      del self.speed_histogram

      if self.tp_stats:
        if len(self.tp_sign_agrees_with_angle) > 0:
          print("Agreement between TP and steering: ",
                sum(self.tp_sign_agrees_with_angle) / len(self.tp_sign_agrees_with_angle))
          with gzip.open(self.save_path / "tp_agreements.json.gz", "wt", encoding="utf-8") as f:
            ujson.dump(self.tp_sign_agrees_with_angle, f, indent=4)

    del self.tp_sign_agrees_with_angle
    del self.visible_walker_ids
    del self.walker_past_pos

  def _get_steer(self, route_points, current_position, current_heading, current_speed):
    """
        Calculate the steering angle based on the current position, heading, speed, and the route points.

        Args:
            route_points (numpy.ndarray): An array of (x, y) coordinates representing the route points.
            current_position (tuple): The current position (x, y) of the vehicle.
            current_heading (float): The current heading angle (in radians) of the vehicle.
            current_speed (float): The current speed of the vehicle (in m/s).

        Returns:
            float: The calculated steering angle.
        """
    speed_scale = self.config.lateral_pid_speed_scale
    speed_offset = self.config.lateral_pid_speed_offset

    # Calculate the lookahead index based on the current speed
    speed_in_kmph = current_speed * 3.6
    lookahead_distance = speed_scale * speed_in_kmph + speed_offset
    lookahead_distance = np.clip(lookahead_distance, self.config.lateral_pid_default_lookahead,
                                 self.config.lateral_pid_maximum_lookahead_distance)
    lookahead_index = int(min(lookahead_distance, route_points.shape[0] - 1))

    # Get the target point from the route points
    target_point = route_points[lookahead_index]

    # Calculate the angle between the current heading and the target point
    angle_unnorm = self._get_angle_to(current_position, current_heading, target_point)
    normalized_angle = angle_unnorm / 90

    self.aim_wp = target_point
    self.angle = normalized_angle

    # Calculate the steering angle using the turn controller
    steering_angle = self._turn_controller.step(route_points, current_speed, current_position, current_heading)
    steering_angle = round(steering_angle, 3)

    return steering_angle

  def _get_angle_to(self, current_position, current_heading, target_position):
    """
        Calculate the angle (in degrees) from the current position and heading to a target position.

        Args:
            current_position (list): A list of (x, y) coordinates representing the current position.
            current_heading (float): The current heading angle in radians.
            target_position (tuple or list): A tuple or list of (x, y) coordinates representing the target position.

        Returns:
            float: The angle (in degrees) from the current position and heading to the target position.
        """
    cos_heading = math.cos(current_heading)
    sin_heading = math.sin(current_heading)

    # Calculate the vector from the current position to the target position
    position_delta = target_position - current_position

    # Calculate the dot product of the position delta vector and the current heading vector
    aim_x = cos_heading * position_delta[0] + sin_heading * position_delta[1]
    aim_y = -sin_heading * position_delta[0] + cos_heading * position_delta[1]

    # Calculate the angle (in radians) from the current heading to the target position
    angle_radians = -math.atan2(-aim_y, aim_x)

    # Convert the angle from radians to degrees
    angle_degrees = np.float_(math.degrees(angle_radians))

    return angle_degrees