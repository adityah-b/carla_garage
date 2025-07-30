import numpy as np
import carla

from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """
    Camera intrinsic parameters
    """
    width : int
    height : int
    fov : float
    K : np.ndarray
    K_behind : np.ndarray

    @classmethod
    def build_camera_intrinsics(cls, camera_actor_obj : carla.Sensor):
        """
        Extract camera configuration from CARLA actor.
        """
        width = int(camera_actor_obj.attributes['image_size_x'])
        height = int(camera_actor_obj.attributes['image_size_y'])
        fov = float(camera_actor_obj.attributes['fov'])

        def build_projection_matrix(is_behind_camera : bool = False) -> np.ndarray:
            """
            Build camera projection matrix.

            Args:
                is_behind_camera: Whether to build matrix for points behind camera

            Returns:
                3x3 projection matrix
            """
            focal = width / (2.0 * np.tan(fov * np.pi / 360.0))
            focal = -focal if is_behind_camera else focal

            K = np.eye(3, dtype=np.float64)
            K[0, 0] = K[1, 1] = focal
            K[0, 2] = width  / 2.0
            K[1, 2] = height / 2.0

            return K

        return cls(
            width=width,
            height=height,
            fov=fov,
            K=build_projection_matrix(),
            K_behind=build_projection_matrix(is_behind_camera=True)
        )


class CameraInterface:
    def __init__(self, camera_actor_obj : carla.Sensor):
        self.obj = camera_actor_obj
        self.config = CameraIntrinsics.build_camera_intrinsics(camera_actor_obj)

        self._image: Optional[np.ndarray] = None
        self._world_to_camera_matrix: Optional[np.ndarray] = None

    # -------------------------------------------------------------------- #
    #  Setters and Getters
    # -------------------------------------------------------------------- #

    def reset_state(self) -> None:
        self._world_to_camera_matrix = None
        self._image = None

    def set_image(self, image : np.ndarray) -> None:
        """
        Update the camera observation with the new image data.
        """
        self._image = np.copy(image) if image is not None else None

    @property
    def image(self) -> Optional[np.ndarray]:
        """
        Get current camera image.
        """
        return self._image

    @property
    def transform(self) -> carla.Transform:
        return self.obj.get_transform()

    @property
    def world_to_camera_matrix(self) -> np.ndarray:
        """
        Get current world to camera transformation matrix
        """
        if self._world_to_camera_matrix is None:
            self._world_to_camera_matrix = np.array(self.transform.get_inverse_matrix(), dtype=np.float64)
        return self._world_to_camera_matrix

    # -------------------------------------------------------------------- #
    #  Utility
    # -------------------------------------------------------------------- #

    def is_point_in_canvas(self, point : np.ndarray) -> bool:
        """
        Check if 2D point is within camera canvas bounds.
        """
        x, y = point[0], point[1]
        return (0 <= x < self.config.width) and (0 <= y < self.config.height)

    def project_3d_to_2d(
            self,
            location : carla.Location,
            is_behind_camera : bool = False,
        ) -> np.ndarray:
        """
        Project 3D world coordinates to 2D image coordinates.

        Args:
            location: CARLA location object
            is_behind_camera: Whether to use projection matrix for points behind the camera

        Returns:
            2D image coordinates [x, y]
        """
        # Convert to homogeneous coordinates
        point_3d = np.array([location.x, location.y, location.z, 1], dtype=np.float64)

        # Transform to camera coordinates
        point_camera = self.world_to_camera_matrix @ point_3d

        # Convert from UE4 coordinate system to standard
        # (x, y, z) -> (y, -z, x)
        point_camera_std = np.array([
            point_camera[1],
            -point_camera[2],
            point_camera[0]
        ])

        # Project to 2D using camera matrix
        point_2d_homogeneous = (self.config.K_behind if is_behind_camera else self.config.K) @ point_camera_std

        # Normalize by depth
        if point_2d_homogeneous[2] != 0:
            point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]
        else:
            point_2d = point_2d_homogeneous[:2]

        return point_2d
