import carla
import numpy as np

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

from agents.navigation.local_planner import RoadOption
from config import GlobalConfig
from .base_actor_extractor import BaseActorExtractor
from privileged_route_planner import PlannerState
from actor_prediction.collision_checker import CollisionChecker

@dataclass(frozen=True, slots=True)
class ObstacleDataEntry:
    obstacle: carla.Actor
    relative_position: Tuple[float, float]
    relative_distance: float
    obstructs_ego : bool
    intrusion_idx : Optional[int]
    is_near_junction: bool

    @property
    def id(self) -> int:
        return self.obstacle.id

@dataclass
class ObstacleData:
    all_obstacles : List[ObstacleDataEntry] = field(default_factory=list)
    ego_obstacles : List[ObstacleDataEntry] = field(default_factory=list)

class ObstacleDataExtractor(BaseActorExtractor):
    def __init__(self, config: GlobalConfig, carla_map: carla.Map):
        super().__init__(config)
        self.carla_map = carla_map

    # def extract_obstacle_data(
    #     self,
    #     ego_transform: carla.Transform,
    #     ego_wp: carla.Waypoint,
    #     obstacles: List[carla.Actor],
    #     planner_state : PlannerState,
    #     lidar_data : Dict
    # ) -> Tuple[List[ObstacleData], Optional[ObstacleData]]:
    #     if not obstacles:
    #         return [], None

    #     ego_location = ego_transform.location
    #     ego_forward = ego_transform.get_forward_vector()

    #     lidar_sensor = lidar_data['sensor']
    #     lidar_raw_data = lidar_data['raw_data']

    #     lidar_pts_3d = lidar_raw_data['xyz'].astype(np.float32) # [Nx3]
    #     lidar_semantic_tags = lidar_raw_data['object_tag'].astype(np.int32) # [Nx1]
    #     lidar_instance_tags = lidar_raw_data['object_idx'].astype(np.int32) # [Nx1]

    #     # Convert to homogenous form [Nx4]
    #     ones = np.ones((lidar_pts_3d.shape[0], 1), dtype=np.float32)
    #     lidar_pts_4d = np.hstack([lidar_pts_3d, ones])

    #     # Transform points from lidar-frame to world
    #     T_lidar_wrt_world = np.array(lidar_sensor.get_transform().get_matrix(), dtype=np.float64) # [4x4]
    #     lidar_pts_world_3d = (lidar_pts_4d @ T_lidar_wrt_world.T)[:, :3]

    #     route_index = planner_state.route_index
    #     route_points = planner_state.route_points
    #     lookahead_index = min(
    #         route_points.shape[0],
    #         route_index + int(self.config.obstacle_detection_radius * self.config.points_per_meter)
    #     )
    #     route_subset = route_points[route_index:lookahead_index] # [Mx3]
    #     diff = lidar_pts_world_3d[:, None, :] - route_subset[None, :, :]
    #     dists = np.linalg.norm(diff, axis=2)   # (N, M)
    #     dists_to_route = dists.min(axis=1)     # (N,)

    #     # TODO: SET ACTUAL DISTANCE THRESHOLD
    #     dist_mask = dists_to_route <= 1.5

    #     lane_obstacles: List[carla.Actor] = []
    #     for obstacle in obstacles:
    #         # Ignore non-disruptive obstacles
    #         if any(map(obstacle.type_id.__contains__, ['dirtdebris'])):
    #             continue

    #         if ego_location.distance(obstacle.get_location()) > self.config.obstacle_detection_radius:
    #             continue

    #         obstacle_wp = self.carla_map.get_waypoint(
    #             obstacle.get_location(), lane_type=carla.LaneType.Any
    #         )
    #         if not obstacle_wp:
    #             continue

    #         process_obstacle = False
    #         lidar_id_mask = lidar_instance_tags == obstacle.id

    #         obstacle_mask = lidar_id_mask & dist_mask
    #         if np.any(obstacle_mask):
    #             process_obstacle = True

    #         if not process_obstacle and (obstacle_wp.road_id != ego_wp.road_id or obstacle_wp.lane_id != ego_wp.lane_id):
    #             continue

    #         relative_vector = obstacle_wp.transform.location - ego_location
    #         if not process_obstacle and (ego_forward.dot(relative_vector) <= 0.0):
    #             continue

    #         lane_obstacles.append(obstacle)

    #     if not lane_obstacles:
    #         return [], None

    #     ego_transform_matrix = self._get_ego_transform_matrix(ego_transform)
    #     obstacle_matrices = np.array([obs.get_transform().get_matrix() for obs in lane_obstacles])

    #     relative_positions = self._calculate_relative_positions(obstacle_matrices, ego_transform_matrix)
    #     relative_distances = np.linalg.norm(relative_positions, axis=1)

    #     obstacle_data: List[ObstacleData] = []
    #     for idx, obstacle in enumerate(lane_obstacles):
    #         rel_pos = relative_positions[idx][:2].round(2)
    #         obstacle_data.append(
    #             ObstacleData(
    #                 obstacle=obstacle,
    #                 id=int(obstacle.id),
    #                 relative_position=tuple(rel_pos),
    #                 relative_distance=round(float(relative_distances[idx]), 2),
    #             )
    #         )

    #     obstacle_data.sort(key=lambda o: o.relative_distance)
    #     closest_obstacle = None if not obstacle_data else obstacle_data[0]
    #     return obstacle_data, closest_obstacle

    # def extract_obstacle_data(
    #     self,
    #     ego_transform: carla.Transform,
    #     ego_wp: carla.Waypoint,
    #     obstacles: List[carla.Actor],
    #     planner_state : PlannerState,
    #     lidar_data : Dict
    # ) -> Tuple[List[ObstacleData], Optional[ObstacleData]]:
    #     if not obstacles:
    #         return [], None

    #     ego_location = ego_transform.location
    #     ego_forward = ego_transform.get_forward_vector()

    #     route_index = planner_state.route_index
    #     route_points = planner_state.route_points
    #     route_commands = planner_state.route_commands

    #     lookahead_index = min(
    #         route_points.shape[0],
    #         route_index + int(self.config.obstacle_detection_radius * self.config.points_per_meter)
    #     )
    #     route_xy = route_points[route_index:lookahead_index, :2] # [Mx2]
    #     relevant_cmds = route_commands[route_index : lookahead_index]
    #     junction_cmds = np.array([RoadOption.LEFT, RoadOption.RIGHT, RoadOption.STRAIGHT])

    #     obstacle_locs = np.array([[obstacle.get_location().x, obstacle.get_location().y] for obstacle in obstacles])

    #     # Calculate static obstacle squared distances to route points
    #     rel_pos = obstacle_locs[:, None, :] - route_xy[None, :, :]
    #     dist_sq = np.sum(rel_pos ** 2, axis=2)
    #     argmin_route_idx = dist_sq.argmin(axis=1)
    #     min_dist_sq = dist_sq.min(axis=1)

    #     # Distance to ego
    #     ego_xy = np.array([ego_location.x, ego_location.y])
    #     rel_to_ego = obstacle_locs - ego_xy
    #     veh_to_ego_dist_sq = np.sum(rel_to_ego ** 2, axis=1)

    #     # In-front dot-product
    #     ego_fwd_vec = np.array([ego_forward.x, ego_forward.y])
    #     loc_dots = rel_to_ego @ ego_fwd_vec

    #     # Identify obstacles near junction
    #     is_junction_point = np.isin(relevant_cmds, junction_cmds).astype(np.int32)
    #     NUM_CMDS = relevant_cmds.shape[0]
    #     junction_search_dist = int(10.0 * self.config.points_per_meter) # TODO: store this in config
    #     obs_start_idxs = np.maximum(argmin_route_idx - junction_search_dist, 0)
    #     obs_end_idxs = np.minimum(argmin_route_idx + junction_search_dist + 1, NUM_CMDS)

    #     prefix = np.concatenate(([0], np.cumsum(is_junction_point, dtype=np.int32)))
    #     obs_near_junction = (prefix[obs_end_idxs] - prefix[obs_start_idxs]) > 0

    #     # Build mask
    #     # TODO: store this in config
    #     route_thresh_sq = 2.0 ** 2
    #     radius_thresh_sq = self.config.obstacle_detection_radius ** 2

    #     mask = (min_dist_sq <= route_thresh_sq) & \
    #            (veh_to_ego_dist_sq <= radius_thresh_sq) & \
    #            (loc_dots >= 0)

    #     excluded_types = ['dirtdebris']

    #     lane_obstacles: List[carla.Actor] = []
    #     lane_obs_near_junction: List[bool] = []
    #     for obs, m, near_j in zip(obstacles, mask, obs_near_junction):
    #         if not m:
    #             continue
    #         if any(ext in obs.type_id for ext in excluded_types):
    #             continue
    #         lane_obstacles.append(obs)
    #         lane_obs_near_junction.append(bool(near_j))

    #     if not lane_obstacles:
    #         return [], None

    #     ego_transform_matrix = self._get_ego_transform_matrix(ego_transform)
    #     obstacle_matrices = np.array([obs.get_transform().get_matrix() for obs in lane_obstacles])

    #     relative_positions = self._calculate_relative_positions(obstacle_matrices, ego_transform_matrix)
    #     relative_distances = np.linalg.norm(relative_positions, axis=1)

    #     obstacle_data: List[ObstacleData] = []
    #     for idx, obstacle in enumerate(lane_obstacles):
    #         rel_pos = relative_positions[idx][:2].round(2)
    #         obstacle_data.append(
    #             ObstacleData(
    #                 obstacle=obstacle,
    #                 id=int(obstacle.id),
    #                 relative_position=tuple(rel_pos),
    #                 relative_distance=round(float(relative_distances[idx]), 2),
    #                 is_near_junction=lane_obs_near_junction[idx],
    #             )
    #         )

    #     obstacle_data.sort(key=lambda o: o.relative_distance)
    #     closest_obstacle = obstacle_data[0] if obstacle_data else None
    #     return obstacle_data, closest_obstacle

    def extract_obstacle_data(
        self,
        ego_transform: carla.Transform,
        ego_wp: carla.Waypoint,
        obstacles: List[carla.Actor],
        planner_state : PlannerState,
        lidar_data : Dict
    ) -> ObstacleData:
        if not obstacles:
            return ObstacleData()

        route_index = planner_state.route_index
        route_points = planner_state.route_points
        route_commands = planner_state.route_commands

        lookahead_index = min(
            route_points.shape[0],
            route_index + int(self.config.obstacle_detection_radius * self.config.points_per_meter)
        )

        # Slice route
        relevant_cmds = route_commands[route_index : lookahead_index]

        ego_loc = ego_transform.location

        def is_on_driving_lane(
            actor : carla.Actor,
            lane_type : carla.LaneType = carla.LaneType.Driving | carla.LaneType.Parking
        ) -> bool:
            wp = self.carla_map.get_waypoint(
                actor.get_location(),
                project_to_road=False,
                lane_type=lane_type
            )
            return wp is not None

        # Find obstacles on road and filter out unwanted types
        excluded_types = ['dirtdebris', 'mesh']
        driving_obstacles : List[carla.Actor] = [
            obs
            for obs in obstacles if \
                obs.is_active and \
                is_on_driving_lane(obs) and \
                not any(t in obs.type_id for t in excluded_types) and \
                ego_loc.distance(obs.get_location()) <= self.config.obstacle_detection_radius
        ]
        if not driving_obstacles:
            return ObstacleData()

        # Junction prefix-sum for fast "near junction" query
        junction_cmds = (RoadOption.LEFT, RoadOption.RIGHT, RoadOption.STRAIGHT)
        is_junction = np.isin(relevant_cmds, np.array(junction_cmds)).astype(np.int32)
        prefix = np.concatenate(([0], np.cumsum(is_junction, dtype=np.int32)))

        junction_search = int(10.0 * self.config.points_per_meter)  # TODO: config
        num_cmds = len(relevant_cmds)

        def near_junction(global_intrusion_idx: int) -> bool:
            # Convert global route index -> local index into relevant_cmds
            local_idx = global_intrusion_idx - route_index

            # If the intrusion lies outside the sliced lookahead window, it can't be "near a junction"
            if local_idx < 0 or local_idx >= num_cmds:
                return False

            s = max(local_idx - junction_search, 0)
            e = min(local_idx + junction_search + 1, num_cmds)
            return (prefix[e] - prefix[s]) > 0

        # Find obstacles obstructing ego route
        intruders = self._check_lane_intrusions(vehicles=driving_obstacles, planner_state=planner_state, bbox_inflation=1.0)

        # Extract obstacle positional data
        ego_transform_matrix = self._get_ego_transform_matrix(ego_transform)
        obstacle_matrices = np.array([obs.get_transform().get_matrix() for obs in driving_obstacles])

        relative_positions = self._calculate_relative_positions(obstacle_matrices, ego_transform_matrix)
        relative_distances = np.linalg.norm(relative_positions, axis=1)

        all_obstacles : List[ObstacleDataEntry] = []
        ego_obstacles : List[ObstacleDataEntry] = []

        for idx, obstacle in enumerate(driving_obstacles):
            rel_dist = relative_distances[idx].round(2)
            rel_pos = relative_positions[idx][:2].round(2)

            intrusion_idx = intruders.get(obstacle.id)
            obstructs_ego = intrusion_idx is not None
            is_near_junction = near_junction(intrusion_idx) if obstructs_ego else False

            obstacle_data_entry = ObstacleDataEntry(
                obstacle=obstacle,
                relative_position=tuple(rel_pos),
                relative_distance=rel_dist,
                obstructs_ego=obstructs_ego,
                intrusion_idx=intrusion_idx,
                is_near_junction=is_near_junction,
            )

            all_obstacles.append(obstacle_data_entry)
            if obstructs_ego:
                ego_obstacles.append(obstacle_data_entry)

        all_obstacles.sort(key=lambda o: o.relative_distance)
        ego_obstacles.sort(key=lambda o: o.relative_distance)

        return ObstacleData(
            all_obstacles=all_obstacles,
            ego_obstacles=ego_obstacles,
        )