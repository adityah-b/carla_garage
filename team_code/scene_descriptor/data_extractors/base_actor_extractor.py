import carla
import numpy as np


class BaseActorExtractor:
    def _get_ego_transform_matrix(self, ego_wp: carla.Waypoint) -> np.ndarray:
        """
        Extract ego vehicle transformation matrix.
        """
        return np.array(ego_wp.transform.get_matrix())

    def _get_ego_yaw(self, ego_wp: carla.Waypoint) -> float:
        """
        Extract ego vehicle yaw angle in radians.
        """
        return np.deg2rad(ego_wp.transform.rotation.yaw)

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
