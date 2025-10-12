import carla
import cv2
import numpy as np

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Set, Dict, List

from team_code.trajectory_planner.occupancy_grid.sensor_data_processor import WorldMappingProcessor, BEVGrid
from team_code.scene_descriptor.camera_interface import CameraInterface

import open3d as o3d


@dataclass(frozen=True, slots=True)
class Maps:
    occupancy_map       : np.ndarray
    instance_map        : np.ndarray
    semantic_map        : np.ndarray
    base_cost_map       : np.ndarray
    dynamic_cost_map    : np.ndarray
    total_cost_map      : np.ndarray

class GridMapper:
    def __init__(
        self,
        ego_vehicle : carla.Vehicle,
        x_min : float = -30.0,
        x_max : float = 30.0,
        y_min : float = -30.0,
        y_max : float = 30.0,
        grid_resolution: float = 0.25,
    ):
        self.ego_vehicle = ego_vehicle
        self.grid = BEVGrid(
            x_min = x_min,
            x_max = x_max,
            y_min = y_min,
            y_max = y_max,
            resolution = grid_resolution
        )

        # Initialize processor with world coordinates
        vehicle_transform = self.ego_vehicle.get_transform()
        world_origin = (vehicle_transform.location.x, vehicle_transform.location.y)

        self.processor = WorldMappingProcessor(
            self.ego_vehicle,
            grid_size=(self.grid.H, self.grid.W),
            grid_resolution=grid_resolution,
            world_origin=world_origin
        )

        # Sensors
        self.cameras : Dict[str, CameraInterface] = {}
        self.camera_sensor : Optional[CameraInterface] = None
        self.lidar_sensor : Optional[carla.Sensor] = None

        self.point_list = o3d.geometry.PointCloud()

    def setup_sensors(
        self,
        cameras : List[Tuple[str, carla.Sensor]],
        lidar : carla.Sensor
    ) -> None:
        for tag, camera_obj in cameras:
            self.cameras[tag] = CameraInterface(camera_obj)

        self.lidar_sensor = lidar

    def set_camera_observations(
        self,
        images: List[Tuple[str, np.ndarray]]
    ) -> None:
        """
        Update camera observations with new image data.

        Args:
            images: List of tuples containing (camera_tag, image_data)
        """
        for tag, image in images:
            if tag in self.cameras:
                self.cameras[tag].set_image(image)
            else:
                print(f"Warning: Camera tag '{tag}' not found in registered cameras.")

    def update_maps(
        self,
        lidar_data : Dict
    ) -> Maps:
        ego_tf = self.ego_vehicle.get_transform()
        # occupancy_map = self.processor.get_occupancy_map_visualize_lidar(
        #     self.cameras,
        #     self.lidar_sensor,
        #     lidar_data,
        #     ego_tf,
        #     self.grid,
        #     point_list=self.point_list
        # )
        occupancy_map = self.processor.get_occupancy_map_lidar(
            self.cameras,
            self.lidar_sensor,
            lidar_data,
            ego_tf,
            self.grid,
        )
        # occupancy_map = self.processor.get_occupancy_map_multiview_camera(
        #     self.cameras,
        #     self.lidar_sensor,
        #     lidar_data,
        #     ego_tf,
        #     self.grid,
        # )

        return Maps(
            occupancy_map=occupancy_map,
            instance_map=None,
            semantic_map=None,
            base_cost_map=None,
            dynamic_cost_map=None,
            total_cost_map=None
        )

        # return Maps(
        #     occupancy_map=occupancy_map,
        #     instance_map=self.point_list,
        #     semantic_map=None,
        #     base_cost_map=None,
        #     dynamic_cost_map=None,
        #     total_cost_map=None
        # )