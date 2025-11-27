import carla
import cv2
import heapq
import numpy as np
import open3d as o3d

from math import sqrt
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Set, Dict, List, Any

from .sensor_data_processor import WorldMappingProcessor, BEVGrid
from scene_descriptor.camera_interface import CameraInterface

@dataclass(frozen=True, slots=True)
class Maps:
    occupancy_map       : np.ndarray
    instance_map        : np.ndarray
    semantic_map        : np.ndarray
    base_cost_map       : np.ndarray
    dynamic_cost_map    : np.ndarray
    total_cost_map      : np.ndarray

class GridMapper:
    def __init__(self, ego_vehicle : carla.Vehicle, lat_grid_spec):
        self.ego_vehicle = ego_vehicle
        self.lat_grid_spec = lat_grid_spec

        # Initialize processor with world coordinates
        vehicle_transform = self.ego_vehicle.get_transform()
        world_origin = (vehicle_transform.location.x, vehicle_transform.location.y)

        self.processor = WorldMappingProcessor(
            self.ego_vehicle,
            grid_size=(self.lat_grid_spec.H, self.lat_grid_spec.W),
            grid_resolution=self.lat_grid_spec.resolution,
            world_origin=world_origin
        )

        # Sensors
        self.cameras : Dict[str, CameraInterface] = {}
        self.camera_sensor : Optional[CameraInterface] = None
        self.lidar_sensor : Optional[carla.Sensor] = None

        self.point_list = o3d.geometry.PointCloud()

        # TESTING PAYLOAD
        self.payload :Dict[str, Any] = {
            'static_occupancy_map' : None,
            'static_cost_map' : None,
            'start_point_ego' : None,
            'goal_point_ego' : None
        }

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
        lidar_data: Dict,
        route_points_world: np.ndarray
    ) -> Maps:
        ego_tf = self.ego_vehicle.get_transform()

        # Get base cost maps
        road_cost_map, static_cost_map, dynamic_cost_map = self.processor.get_base_cost_maps(
            lidar_data,
            ego_tf,
            self.lat_grid_spec
        )
        occupancy_map = np.ones_like(road_cost_map, dtype=np.uint8)
        occupancy_map[road_cost_map > 0] = 0
        occupancy_map[static_cost_map > 0] = 0
        # occupancy_map[dynamic_cost_map > 0] = 0

        # Augment static cost map with route information
        static_cost_map_route = self.processor.augment_static_cost_map(
            road_cost_map,
            static_cost_map,
            self.ego_vehicle,
            route_points_world,
            self.lat_grid_spec
        )

        # cost_map = self.processor.get_cost_map(
        #     occupancy_map
        # )

        # TESTING PAYLOAD
        self.payload['static_occupancy_map'] = occupancy_map
        self.payload['static_cost_map'] = static_cost_map_route
        self.payload['dynamic_cost_map'] = dynamic_cost_map

        return Maps(
            occupancy_map=occupancy_map,
            instance_map=None,
            semantic_map=None,
            base_cost_map=None,
            dynamic_cost_map=dynamic_cost_map,
            total_cost_map=static_cost_map_route
        )
