import carla
import numpy as np

from dataclasses import dataclass
from typing import List, Tuple, Dict

from config import GlobalConfig
from .base_actor_extractor import BaseActorExtractor
from privileged_route_planner import PlannerState


@dataclass(frozen=True, slots=True)
class ObstacleData:
    obstacle: carla.Actor
    id: int
    relative_position: Tuple[float, float]
    relative_distance: float


class ObstacleDataExtractor(BaseActorExtractor):
    def __init__(self, config: GlobalConfig, carla_map: carla.Map):
        self.config = config
        self.carla_map = carla_map

    def extract_obstacle_data(
        self,
        ego_wp: carla.Waypoint,
        obstacles: List[carla.Actor],
        planner_state : PlannerState,
        lidar_data : Dict
    ) -> List[ObstacleData]:
        if not obstacles:
            return []

        ego_location = ego_wp.transform.location
        ego_forward = ego_wp.transform.get_forward_vector()

        lidar_sensor = lidar_data['sensor']
        lidar_raw_data = lidar_data['raw_data']

        lidar_pts_3d = lidar_raw_data['xyz'].astype(np.float32) # [Nx3]
        lidar_semantic_tags = lidar_raw_data['object_tag'].astype(np.int32) # [Nx1]
        lidar_instance_tags = lidar_raw_data['object_idx'].astype(np.int32) # [Nx1]

        # Convert to homogenous form [Nx4]
        ones = np.ones((lidar_pts_3d.shape[0], 1), dtype=np.float32)
        lidar_pts_4d = np.hstack([lidar_pts_3d, ones])

        # Transform points from lidar-frame to world
        T_lidar_wrt_world = np.array(lidar_sensor.get_transform().get_matrix(), dtype=np.float64) # [4x4]
        lidar_pts_world_3d = (lidar_pts_4d @ T_lidar_wrt_world.T)[:, :3]

        route_index = planner_state.route_index
        route_points = planner_state.route_points
        lookahead_index = min(
            route_points.shape[0],
            route_index + int(self.config.detection_radius * self.config.points_per_meter)
        )
        route_subset = route_points[route_index:lookahead_index] # [Mx3]
        diff = lidar_pts_world_3d[:, None, :] - route_subset[None, :, :]
        dists = np.linalg.norm(diff, axis=2)   # (N, M)
        dists_to_route = dists.min(axis=1)     # (N,)

        # TODO: SET ACTUAL DISTANCE THRESHOLD
        dist_mask = dists_to_route <= 1.5

        lane_obstacles: List[carla.Actor] = []
        for obstacle in obstacles:
            if ego_location.distance(obstacle.get_location()) > self.config.detection_radius:
                continue

            obstacle_wp = self.carla_map.get_waypoint(
                obstacle.get_location(), lane_type=carla.LaneType.Any
            )
            if not obstacle_wp:
                continue

            process_obstacle = False
            lidar_id_mask = lidar_instance_tags == obstacle.id

            obstacle_mask = lidar_id_mask & dist_mask
            if np.any(obstacle_mask):
                process_obstacle = True

            if not process_obstacle and (obstacle_wp.road_id != ego_wp.road_id or obstacle_wp.lane_id != ego_wp.lane_id):
                continue

            relative_vector = obstacle_wp.transform.location - ego_location
            if not process_obstacle and (ego_forward.dot(relative_vector) <= 0.0):
                continue

            lane_obstacles.append(obstacle)

        if not lane_obstacles:
            return []

        ego_transform_matrix = self._get_ego_transform_matrix(ego_wp)
        obstacle_matrices = np.array([obs.get_transform().get_matrix() for obs in lane_obstacles])

        relative_positions = self._calculate_relative_positions(obstacle_matrices, ego_transform_matrix)
        relative_distances = np.linalg.norm(relative_positions, axis=1)

        obstacle_data: List[ObstacleData] = []
        for idx, obstacle in enumerate(lane_obstacles):
            rel_pos = relative_positions[idx][:2].round(2)
            obstacle_data.append(
                ObstacleData(
                    obstacle=obstacle,
                    id=int(obstacle.id),
                    relative_position=tuple(rel_pos),
                    relative_distance=round(float(relative_distances[idx]), 2),
                )
            )

        obstacle_data.sort(key=lambda o: o.relative_distance)
        return obstacle_data