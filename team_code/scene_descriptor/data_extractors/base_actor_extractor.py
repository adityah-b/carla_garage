import carla
import numpy as np

from typing import List, Dict

from agents.navigation.local_planner import RoadOption
from privileged_route_planner import PlannerState
from actor_prediction.collision_checker import CollisionChecker

class BaseActorExtractor:
    def __init__(self, config):
        self.config = config

    def _get_ego_transform_matrix(self, ego_transform: carla.Transform) -> np.ndarray:
        """
        Extract ego vehicle transformation matrix.
        """
        return np.array(ego_transform.get_matrix())

    def _get_ego_yaw(self, ego_transform: carla.Transform) -> float:
        """
        Extract ego vehicle yaw angle in radians.
        """
        return np.deg2rad(ego_transform.rotation.yaw)

    def _calculate_relative_positions(
        self,
        actor_matrices: np.ndarray,
        ego_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate relative positions of actors with respect to ego vehicle.

        Args:
            actor_matrices: Nx4x4 array of actor transformation matrices
            ego_matrix: 4x4 ego vehicle transformation matrix

        Returns:
            Nx3 array of relative positions
        """
        # Get positions from transformation matrices
        actor_positions = actor_matrices[:, :3, 3]
        ego_position = ego_matrix[:3, 3]

        # Calculate relative positions in world coordinates
        relative_world = actor_positions - ego_position[np.newaxis, :]

        # Transform to ego vehicle coordinate system
        ego_rotation = ego_matrix[:3, :3]
        relative_ego = (ego_rotation.T @ relative_world.T).T

        return relative_ego

    def _normalize_angles(self, angles: np.ndarray) -> np.ndarray:
        """Normalize angles to [-π, π] range."""
        return (angles + np.pi) % (2 * np.pi) - np.pi

    def _check_lane_intrusions(
        self,
        vehicles : List[carla.Vehicle],
        planner_state : PlannerState,
        *,
        bbox_inflation : float = 1.0
    ) -> Dict[int, int]: # K = actor_id, V = global intrusion_idx
        all_intruding_vehicles = {}

        # Get route data
        route_index = planner_state.route_index
        route_bbs = planner_state.route_bbs
        route_cmds = planner_state.route_commands

        # TODO: UPDATE INDEX TRANSFORMATION WITH CONFIG NUMBERS
        # Convert indices to route_bb index
        route_index_spaced = route_index // self.config.points_per_meter
        route_index_spaced = route_index_spaced // 2

        # TODO: UPDATE INDEX TRANSORMATION WITH CONFIG NUMBERS
        max_route_bbs = len(route_bbs)
        lookahead_distance = int(80.0 // 2)
        to_index_spaced = min(max_route_bbs, route_index_spaced + lookahead_distance + 1)

        route_bbs_subset = route_bbs[route_index_spaced : to_index_spaced]

        # Get all lane intrusions
        intruding_vehicles = CollisionChecker.get_lane_intrusions(
            actors=vehicles,
            route_bboxes=route_bbs_subset,
            bbox_inflation=bbox_inflation,
        )

        # Filter out lane intrusions at junctions
        for veh_id, intrusion_idx_offset in intruding_vehicles.items():
            # Convert to actual route bb index
            intrusion_idx = route_index_spaced + intrusion_idx_offset

            # Convert route_bb index to dense route index
            intrusion_idx_dense = int(np.clip(intrusion_idx * 2 * self.config.points_per_meter, 0, route_cmds.shape[0] - 1))

            # Add to overall dict if intrusion index is not in a junction
            if route_cmds[intrusion_idx_dense] not in [RoadOption.LEFT, RoadOption.RIGHT, RoadOption.STRAIGHT]:
                all_intruding_vehicles[veh_id] = intrusion_idx_dense

        return all_intruding_vehicles
