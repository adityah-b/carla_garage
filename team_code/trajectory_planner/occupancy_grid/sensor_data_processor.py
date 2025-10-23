import carla
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import time
import queue
import h5py
import os

from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
from matplotlib.colors import ListedColormap
from enum import IntEnum

from team_code.scene_descriptor.camera_interface import CameraInterface


from matplotlib import cm
import open3d as o3d

@dataclass
class BEVGrid:
    x_min : float = -50.0
    x_max : float = 50.0
    y_min : float = -50.0
    y_max : float = 50.0
    resolution : float = 0.5

    @property
    def H(self) -> int: # rows (x axis)
        return int(round((self.x_max - self.x_min) / self.resolution))

    @property
    def W(self) -> int: # cols (y axis)
        return int(round((self.y_max - self.y_min) / self.resolution))

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.H, self.W)

    def world_to_grid(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # i = np.floor((x - self.x_min) / self.resolution).astype(np.int32)
        i = np.floor((self.x_max - x) / self.resolution).astype(np.int32)
        j = np.floor((y - self.y_min) / self.resolution).astype(np.int32)
        return i, j

    def grid_to_world(self, grid_x: np.ndarray, grid_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        world_x = self.x_max - (grid_x + 0.5) * self.resolution
        world_y = self.y_min + (grid_y + 0.5) * self.resolution
        return world_x, world_y

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
LABEL_COLORS = np.array([
    (0, 0, 0),       # Unlabeled
    (128, 64, 128),  # Roads
    (244, 35, 232),  # Sidewalks
    (70, 70, 70),    # Building
    (102, 102, 156), # Wall
    (190, 153, 153), # Fences
    (153, 153, 153), # Pole
    (250, 170, 30),  # TrafficLight
    (220, 220, 0),   # TrafficSign
    (107, 142, 35),  # Vegetation
    (152, 251, 152), # Terrain
    (70, 130, 180),  # Sky
    (220, 20, 60),   # Pedestrian
    (255, 0, 0),     # Rider
    (0, 0, 142),     # Car
    (0, 0, 70),      # Truck
    (0, 60, 100),    # Bus
    (0, 80, 100),    # Train
    (0, 0, 230),     # Motorcycle
    (119, 11, 32),   # Bicycle
    (110, 190, 160), # Static
    (170, 120, 50),  # Dynamic
    (55, 90, 80),    # Other
    (45, 60, 150),   # Water
    (157, 234, 50),  # RoadLines
    (81, 0, 81),     # Ground
    (150, 100, 100), # Bridge
    (230, 150, 140), # RailTrack
    (180, 165, 180), # GuardRail
]) / 255.0 # normalize each channel [0-1] since is what Open3D uses


# Semantic labels for CARLA objects
class ActorClass(IntEnum):
    UNLABELED       = 0
    ROADS           = 1
    SIDEWALKS       = 2
    BUILDING        = 3
    WALL            = 4
    FENCE           = 5
    POLE            = 6
    TRAFFICLIGHT    = 7
    TRAFFICSIGN     = 8
    VEGETATION      = 9
    TERRAIN         = 10
    SKY             = 11
    PEDESTRIAN      = 12
    RIDER           = 13
    CAR             = 14
    TRUCK           = 15
    BUS             = 16
    TRAIN           = 17
    MOTORCYCLE      = 18
    BICYCLE         = 19
    STATIC          = 20
    DYNAMIC         = 21
    OTHER           = 22
    WATER           = 23
    ROADLINE        = 24
    GROUND          = 25
    BRIDGE          = 26
    RAILTRACK       = 27
    GUARDRAIL       = 28

@dataclass
class InstanceInfo:
    """Information about a tracked instance"""
    instance_id: int
    semantic_tag: int
    world_positions: Set[Tuple[float, float]]  # World coordinates
    grid_positions: Set[Tuple[int, int]]       # Grid coordinates
    last_seen: float
    confidence: float
    is_critical: bool = False

class WorldMappingProcessor:
    """Grid-based mapping with proper world coordinate system"""

    def __init__(
        self,
        ego_vehicle : carla.Vehicle,
        grid_size: Tuple[int, int] = (400, 400),
        grid_resolution: float = 0.25,  # meters per grid cell
        world_origin: Tuple[float, float] = (0.0, 0.0)
    ):
        """
        Initialize mapping processor with world coordinates

        Args:
            grid_size: Grid dimensions (width, height) in cells
            grid_resolution: Resolution in meters per cell
            world_origin: World coordinates of grid center (x, y) in meters
        """
        self.ego_vehicle = ego_vehicle
        self.world = ego_vehicle.get_world()

        self.grid_size = grid_size
        self.grid_resolution = grid_resolution
        self.world_origin = world_origin

        # Calculate world bounds
        half_width = (grid_size[0] * grid_resolution) / 2
        half_height = (grid_size[1] * grid_resolution) / 2

        self.world_bounds = {
            'min_x': world_origin[0] - half_width,
            'max_x': world_origin[0] + half_width,
            'min_y': world_origin[1] - half_height,
            'max_y': world_origin[1] + half_height
        }

        print(f"Grid covers world area:")
        print(f"  X: [{self.world_bounds['min_x']:.1f}, {self.world_bounds['max_x']:.1f}] meters")
        print(f"  Y: [{self.world_bounds['min_y']:.1f}, {self.world_bounds['max_y']:.1f}] meters")
        print(f"  Resolution: {grid_resolution:.2f} m/cell")

        # Separate maps
        self.occupancy_map = np.zeros(grid_size, dtype=np.float32)
        self.instance_map = np.zeros(grid_size, dtype=np.uint32)
        self.semantic_map = np.zeros(grid_size, dtype=np.uint8)
        self.base_cost_map = np.zeros(grid_size, dtype=np.float32)
        self.dynamic_cost_map = np.zeros(grid_size, dtype=np.float32)
        self.total_cost_map = np.zeros(grid_size, dtype=np.float32)

        # Cameras
        self.cameras : Dict[str, CameraInterface] = {}

        # Confidence maps
        self.camera_confidence = np.zeros(grid_size, dtype=np.float32)
        self.lidar_confidence = np.zeros(grid_size, dtype=np.float32)

        # Camera properties
        self.camera_width = 800
        self.camera_height = 600
        self.camera_fov_meters = 100.0  # Field of view in meters

        # Instance tracking
        self.tracked_instances: Dict[int, InstanceInfo] = {}
        self.critical_instances: Set[int] = set()

        # Vehicle state for coordinate transforms
        self.ego_world_pos = (0.0, 0.0)
        self.ego_yaw = 0.0

        # Cost configuration
        self.base_costs = {
            ActorClass.UNLABELED        : 50,
            ActorClass.ROADS            : 50,
            ActorClass.SIDEWALKS        : 50,
            ActorClass.BUILDING         : 50,
            ActorClass.WALL             : 50,
            ActorClass.FENCE            : 50,
            ActorClass.POLE             : 50,
            ActorClass.TRAFFICLIGHT     : 50,
            ActorClass.TRAFFICSIGN      : 50,
            ActorClass.VEGETATION       : 50,
            ActorClass.TERRAIN          : 50,
            ActorClass.SKY              : 50,
            ActorClass.PEDESTRIAN       : 50,
            ActorClass.RIDER            : 50,
            ActorClass.CAR              : 50,
            ActorClass.TRUCK            : 50,
            ActorClass.BUS              : 50,
            ActorClass.TRAIN            : 50,
            ActorClass.MOTORCYCLE       : 50,
            ActorClass.BICYCLE          : 50,
            ActorClass.STATIC           : 50,
            ActorClass.DYNAMIC          : 50,
            ActorClass.OTHER            : 50,
            ActorClass.WATER            : 50,
            ActorClass.ROADLINE         : 50,
            ActorClass.GROUND           : 50,
            ActorClass.BRIDGE           : 50,
            ActorClass.RAILTRACK        : 50,
            ActorClass.GUARDRAIL        : 50,
        }

        self.critical_cost_multiplier = 2.0
        self.critical_inflation_radius = int(2.0 / grid_resolution)  # 2 meters in grid cells

        # Dynamic object classes
        self.dynamic_classes = {
            ActorClass.PEDESTRIAN,
            ActorClass.RIDER,
            ActorClass.CAR,
            ActorClass.TRUCK,
            ActorClass.BUS,
            ActorClass.TRAIN,
            ActorClass.MOTORCYCLE,
            ActorClass.BICYCLE,
            ActorClass.DYNAMIC,
        }

        # Latest sensor data
        self.latest_camera_data = None
        self.latest_lidar_data = None

        self.H_pixels_to_grid : Optional[np.ndarray] = None

        ws_dir = Path(os.environ['WORK_DIR'])
        map_dir = ws_dir.joinpath('team_code/birds_eye_view/maps_2ppm_cv')
        map_name = self.world.get_map().name.split('/')[-1]

        map_file = map_dir / (map_name + '.h5')
        print(f"\nLoading map from {map_file}")

        with h5py.File(map_file, 'r', libver='latest', swmr=True) as hf:
            self.pixels_per_meter = float(hf.attrs['pixels_per_meter'])
            self.world_origin = np.array(hf.attrs['world_offset_in_meters'], dtype=np.float32)

            road_mask_pixels = np.array(hf['road'], dtype=np.uint8)
            print(f'road_mask_pixels shape: {road_mask_pixels.shape}')

            # Cache the static road layer (binary) and its smoothed variant for reuse every tick
            self.road_mask_world = (road_mask_pixels > 0).astype(np.uint8)
            kernel = np.ones((3, 3), dtype=np.uint8)
            road_mask_world_counts = cv2.filter2D(
                self.road_mask_world,
                -1,
                kernel,
                borderType=cv2.BORDER_CONSTANT,
            )
            self.road_mask_world_dilated = (road_mask_world_counts >= 5).astype(np.uint8)

        print(f'self.pixels_per_meter: {self.pixels_per_meter}')
        print(f'self.world_origin: {self.world_origin}')
        print(f'self.road_mask_world_dilated shape: {self.road_mask_world_dilated.shape}')

        # Costs
        self.free_space_cost = 0
        self.static_obstacle_cost = 200
        self.dynamic_min_cost = 220
        self.dynamic_max_cost = 255

        # road_img = (self.road_mask_world_dilated * 255).astype(np.uint8)
        # cv2.namedWindow("BirdView World Map", cv2.WINDOW_NORMAL)
        # cv2.imshow('BirdView World Map', road_img)
        # cv2.waitKey(1)

    def project_camera_pts_to_ego(
        self,
        instance_camera : CameraInterface,
        depth_camera : CameraInterface,
        T_world_wrt_ego : np.ndarray,
    ) -> np.ndarray:
        instance_rgb = instance_camera.image[:, :, :][:, :, ::-1].astype(np.float32)
        print(f'instance_rgb shape: {instance_rgb.shape}')

        depth_rgb = depth_camera.image[:, :, :][:, :, ::-1].astype(np.float32)
        norm_depths = (
            depth_rgb[:, :, 0] +
            depth_rgb[:, :, 1] * 256.0 +
            depth_rgb[:, :, 2] * (256.0 ** 2)) / (256.0 ** 3 - 1.0)
        depths_m = 1000.0 * norm_depths
        print(f'depths_m shape: {depths_m.shape}')

        H = depth_camera.config.height
        W = depth_camera.config.width

        u, v = np.meshgrid(np.arange(W), np.arange(H))
        K = depth_camera.config.K
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        Z = depths_m
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        # pts_camera_3d = np.stack([X, Y, Z], axis=-1) # [HxWx3]
        # Convert from standard to UE4 coordinate system (y, -z, x) -> (x, y, z) = (z, x, -y) [4x4]
        pts_camera_3d = np.stack([Z, X, -Y], axis=-1) # [HxWx3]
        print(f'pts_camera_3d shape: {pts_camera_3d.shape}')

        # A = np.array([
        #     [0, 0, 1, 0],
        #     [1, 0, 0, 0],
        #     [0, -1, 0, 0],
        #     [0, 0, 0, 1]
        # ], dtype=np.float64)

        T_camera_wrt_world = np.array(depth_camera.obj.get_transform().get_matrix(), dtype=np.float64) # [4x4]
        T_camera_wrt_ego = T_world_wrt_ego @ T_camera_wrt_world

        pts_camera_4d = np.concatenate([pts_camera_3d.reshape(-1, 3), np.ones((H * W, 1))], axis=-1) # [Nx4]
        print(f'pts_camera_4d shape: {pts_camera_4d.shape}')

        pts_ego_4d = pts_camera_4d @ T_camera_wrt_ego.T # [Nx4] @ [4x4]^T -> [Nx4]
        pts_ego_3d = pts_ego_4d[:, :3].reshape(H, W, 3) # [HxWx3]
        print(f'pts_ego_3d shape: {pts_ego_3d.shape}')

        return pts_ego_3d

    def get_bb_corners_world(
        self,
        actor : carla.Actor
    ) -> np.ndarray:
        """
        Returns (4,3) ground-plane corners (x,y,z) in WORLD frame for a CARLA BoundingBox,
        respecting its rotation (yaw) and extents.
        Corner order: [front-right, front-left, back-left, back-right] in the box's local frame.
        """
        actor_tf = actor.get_transform()
        bb = actor.bounding_box
        # print(f'bounding box: {str(bb)}')
        cx, cy, cz = bb.location.x + actor_tf.location.x, bb.location.y + actor_tf.location.y, bb.location.z + actor_tf.location.z
        ex, ey, ez = bb.extent.x, bb.extent.y, bb.extent.z
        yaw = np.deg2rad(bb.rotation.yaw + actor_tf.rotation.yaw)  # CARLA stores degrees

        # local corners at z = cz (middle face) – any constant z works since we drop z later
        local = np.array([
            [ +ex, +ey, cz, 1.0],
            [ +ex, -ey, cz, 1.0],
            [ -ex, -ey, cz, 1.0],
            [ -ex, +ey, cz, 1.0],
        ], dtype=np.float32)

        c, s = np.cos(yaw), np.sin(yaw)
        T_box_world = np.array([
            [ c, -s, 0, cx],
            [ s,  c, 0, cy],
            [ 0,  0, 1, cz],
            [ 0,  0, 0,  1],
        ], dtype=np.float32)

        corners_world_4d = (T_box_world @ local.T).T    # [4x4]
        return corners_world_4d[:, :3]

    def get_base_cost_maps(
        self,
        lidar_data : Dict,
        ego_tf : carla.Transform,
        grid : BEVGrid
    ) -> Tuple[np.ndarray, np.ndarray]:
        ego_loc = ego_tf.location
        ego_yaw_rad = np.deg2rad(ego_tf.rotation.yaw)

        # ############################################
        # # Drivable Area Filtering
        # ############################################

        R = 100.0
        C = np.cos(ego_yaw_rad)
        S = np.sin(ego_yaw_rad)

        grid_res = grid.resolution
        ppm = self.pixels_per_meter
        origin_x, origin_y = self.world_origin

        warp_matrix = np.array([
            [
                ppm * (-S * grid_res),
                ppm * (-C * grid_res),
                ppm * ((ego_loc.x + C*grid.x_max - S*grid.y_min) - 0.5*grid_res*(C + S) - origin_x)
            ],
            [
                ppm * ( C * grid_res),
                ppm * (-S * grid_res),
                ppm * ((ego_loc.y + S*grid.x_max + C*grid.y_min) + 0.5*grid_res*(C - S) - origin_y)
            ],
        ], dtype=np.float32)

        road_mask_local = cv2.warpAffine(
            self.road_mask_world_dilated,
            warp_matrix,
            (grid.W, grid.H),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.uint8)

        static_cost_map = np.zeros((grid.H, grid.W), dtype=np.uint8)
        dynamic_cost_map = np.zeros((grid.H, grid.W), dtype=np.uint8)

        static_cost_map[road_mask_local == 0] = self.static_obstacle_cost

        # ############################################
        # # Actor Processing
        # ############################################

        actors = self.world.get_actors()
        vehicles = list(actors.filter('*vehicle*'))
        pedestrians = list(actors.filter('*walker*'))
        dynamic_actors = vehicles + pedestrians

        for actor in dynamic_actors:
            if actor.id == self.ego_vehicle.id:
                continue

            if not actor.is_active:
                continue

            if ego_loc.distance(actor.get_location()) > R:
                continue

            actor_corners_world_3d = self.get_bb_corners_world(actor)
            actor_dx = actor_corners_world_3d[:, 0] - ego_loc.x
            actor_dy = actor_corners_world_3d[:, 1] - ego_loc.y

            actor_corners_ego_x = C * actor_dx + S * actor_dy
            actor_corners_ego_y = -S * actor_dx + C * actor_dy

            crop_box_filter = (
                (actor_corners_ego_x >= grid.x_min) & (actor_corners_ego_x < grid.x_max) &
                (actor_corners_ego_y >= grid.y_min) & (actor_corners_ego_y < grid.y_max)
            )

            actor_corners_ego_x = actor_corners_ego_x[crop_box_filter]
            actor_corners_ego_y = actor_corners_ego_y[crop_box_filter]

            if actor_corners_ego_x.size == 0:
                continue

            i, j = grid.world_to_grid(actor_corners_ego_x, actor_corners_ego_y)
            poly = np.stack([j, i], axis=1)
            if poly.size == 0:
                continue

            hull = cv2.convexHull(poly.astype(np.float32))
            hull_pts = np.squeeze(hull).astype(np.int32)
            if hull_pts.ndim != 2 or hull_pts.shape[0] < 3:
                continue
            cv2.fillConvexPoly(dynamic_cost_map, hull_pts, self.static_obstacle_cost)

        # ############################################
        # # Lidar Processing
        # ############################################

        # Get 3D lidar points and semantic tags
        lidar_pts_3d = lidar_data['xyz'].astype(np.float32) # [Nx3]
        lidar_semantic_tags = lidar_data['object_tag'].astype(np.int32) # [Nx1]

        # print(f'lidar_pts_3d shape: {lidar_pts_3d.shape}')

        # Convert to homogenous form [Nx4]
        ones = np.ones((lidar_pts_3d.shape[0], 1), dtype=np.float32)
        lidar_pts_world_4d = np.hstack([lidar_pts_3d, ones])

        # Transform points from lidar-frame to ego
        # T_lidar_wrt_world = np.array(lidar_sensor.get_transform().get_matrix(), dtype=np.float64) # [4x4]
        # T_world_wrt_ego = np.array(ego_tf.get_inverse_matrix(), dtype=np.float64) # [4x4]
        # T_lidar_wrt_ego = T_world_wrt_ego @ T_lidar_wrt_world

        T_lidar_wrt_ego = np.array([
            [ 0.0,  1.0,  0.0, 0.0],
            [-1.0,  0.0,  0.0, 0.0],
            [ 0.0,  0.0,  1.0, 2.5],
            [ 0.0,  0.0,  0.0, 1.0],
        ], dtype=np.float32)

        lidar_pts_ego_4d = lidar_pts_world_4d @ T_lidar_wrt_ego.T # [Nx4] @ [4x4]^T -> [Nx4]

        # Get 3D lidar points in ego frame [Nx3]
        lidar_pts_ego_3d = lidar_pts_ego_4d[:, :3]

        # print(f'lidar_pts_ego_3d shape: {lidar_pts_ego_3d.shape}')

        # Lidar points bounds filtering
        x_ok = (lidar_pts_ego_3d[:, 0] >= grid.x_min) & (lidar_pts_ego_3d[:, 0] < grid.x_max)
        y_ok = (lidar_pts_ego_3d[:, 1] >= grid.y_min) & (lidar_pts_ego_3d[:, 1] < grid.y_max)
        z_ok = (lidar_pts_ego_3d[:, 2] >= 0.5) & (lidar_pts_ego_3d[:, 2] < 3.0)

        # Crop box filter
        crop_box_filter = x_ok & y_ok & z_ok

        # Semantic filter (filter drivable regions)
        semantic_filter = np.isin(
            lidar_semantic_tags,
            [
                ActorClass.ROADS,
                ActorClass.ROADLINE,
                ActorClass.GROUND,
                ActorClass.UNLABELED,
                ActorClass.OTHER
            ],
            invert=True)

        # print(f'x_ok shape: {x_ok.shape}')
        # print(f'y_ok shape: {y_ok.shape}')
        # print(f'z_ok shape: {z_ok.shape}')
        # print(f'semantic_filter shape: {semantic_filter.shape}')
        # print(f'crop box filter shape: {crop_box_filter.shape}')

        # Create occupancy mask
        occ_mask = crop_box_filter & semantic_filter
        if np.any(occ_mask):
            lidar_pts_ego_3d = lidar_pts_ego_3d[occ_mask] # (M,3)
            lidar_tags_filtered = lidar_semantic_tags[occ_mask]

            dynamic_semantic_mask = np.isin(lidar_tags_filtered, list(self.dynamic_classes))

            static_pts = lidar_pts_ego_3d[~dynamic_semantic_mask]
            if static_pts.size > 0:
                i_static, j_static = grid.world_to_grid(static_pts[:, 0], static_pts[:, 1])
                static_cost_map[i_static, j_static] = self.static_obstacle_cost

            dynamic_pts = lidar_pts_ego_3d[dynamic_semantic_mask]
            if dynamic_pts.size > 0:
                i_dyn, j_dyn = grid.world_to_grid(dynamic_pts[:, 0], dynamic_pts[:, 1])
                dynamic_cost_map[i_dyn, j_dyn] = np.maximum(dynamic_cost_map[i_dyn, j_dyn], self.static_obstacle_cost)

        return static_cost_map, dynamic_cost_map

    def get_cost_map(
        self,
        occupancy_map : np.ndarray,
        max_cost : float = 255
    ) -> np.ndarray:
        """
        Converts a binary occupancy grid to a smooth cost map.
        """
        # 0=occupied, 1=free
        occ = (occupancy_map == 0).astype(np.uint8)

        # Compute distance transform (distance to nearest obstacle)
        dist = cv2.distanceTransform(1 - occ, distanceType=cv2.DIST_L2, maskSize=5)

        # Normalize so 0=obstacle, 1=far away
        dist_norm = dist / dist.max()

        # Invert so cost is *higher* near obstacles
        cost = (1 - dist_norm ** 0.5) * max_cost

        return cost.astype(np.uint8)

    def get_occupancy_map_multiview_camera(
        self,
        cameras : Dict[str, CameraInterface],
        lidar_sensor : carla.Sensor,
        lidar_data : Dict,
        ego_tf : carla.Transform,
        grid : BEVGrid
    ) -> np.ndarray:
        occupancy_grid = np.zeros((grid.H, grid.W), dtype=np.float16)
        # occupancy_grid = np.full((grid.H, grid.W), 0.5, dtype=np.float16)

        T_world_wrt_ego = np.array(ego_tf.get_inverse_matrix(), dtype=np.float64) # [4x4]
        for cam_dir in ['front', 'left', 'right', 'back', 'bev']:
        # for cam_dir in ['front', 'left', 'right', 'back']:
        # for cam_dir in ['front', 'back']:
            instance_camera = cameras[f'instance_{cam_dir}']
            depth_camera = cameras[f'depth_{cam_dir}']

            instance_img = instance_camera.image[:, :, ::-1].astype(np.float32)
            semantic_img = instance_img[:, :, 0].astype(np.uint8)

            # Get 3D ego points
            pts_ego_3d = self.project_camera_pts_to_ego(instance_camera, depth_camera, T_world_wrt_ego) # [HxWx3]
            x, y, z = pts_ego_3d[:, :, 0], pts_ego_3d[:, :, 1], pts_ego_3d[:, :, 2]
            print(f'x shape: {x.shape}')

            # 3D points bounds filtering
            crop_box_filter = \
                (x >= grid.x_min) & (x < grid.x_max) & \
                (y >= grid.y_min) & (y < grid.y_max) & \
                (z >= -1.0) & (z < 3.0)

            if not np.any(crop_box_filter):
                continue

            # Semantic filter
            semantic_filter = np.isin(
                semantic_img,
                [
                    ActorClass.ROADS,
                    ActorClass.ROADLINE,
                    ActorClass.GROUND,
                    ActorClass.UNLABELED
                ],
                invert=True
            )

            if cam_dir == 'bev':
                overhead_filter = np.isin(
                    semantic_img,
                    [
                        ActorClass.VEGETATION,
                        ActorClass.POLE
                    ],
                    invert=True
                )
                semantic_filter &= overhead_filter

            # Occupancy mask
            occupancy_mask = crop_box_filter & semantic_filter
            free_mask = crop_box_filter & (~semantic_filter)

            x_occ, y_occ = x[occupancy_mask], y[occupancy_mask]
            x_free, y_free = x[free_mask], y[free_mask]
            print(f'filtered x shape: {x_occ.shape}')

            i, j = grid.world_to_grid(x_occ, y_occ)
            valid = (i >= 0) & (i < grid.H) & (j >= 0) & (j < grid.W)
            occupancy_grid[i[valid], j[valid]] = 1

            # print(f'filtered x shape: {x_free.shape}')

            # i, j = grid.world_to_grid(x_free, y_free)
            # valid = (i >= 0) & (i < grid.H) & (j >= 0) & (j < grid.W)
            # occupancy_grid[i[valid], j[valid]] = 0


        return occupancy_grid

    # def get_occupancy_map_bev_instance(
    #     self,
    #     cameras : Dict[str, CameraInterface],
    #     lidar_sensor : carla.Sensor,
    #     lidar_data : Dict,
    #     ego_tf : carla.Transform,
    #     grid : BEVGrid
    # ) -> np.ndarray:
    #     occupancy_grid = np.zeros((grid.H, grid.W), dtype=np.uint8)
    #     # Rearrange camera data to RGB from BGR
    #     camera_data_rgb = camera_sensor.image[:, :, :][:, :, ::-1].astype(np.float32)
    #     camera_data_depth_rgb = depth_camera_sensor.image[:, :, :][:, :, ::-1].astype(np.float32)

    #     # Extract channels
    #     # Red = Semantic tags
    #     # Green + Blue = Instance ID
    #     semantic_image = camera_data_rgb[:, :, 0].astype(np.uint8)
    #     instance_image = camera_data_rgb[:, :, 1].astype(np.uint32) + (camera_data_rgb[:, :, 2].astype(np.uint32) << 8)

    #     # Get depth data
    #     normalized_depths = (
    #         camera_data_depth_rgb[:, :, 0] +
    #         camera_data_depth_rgb[:, :, 1] * 256.0 +
    #         camera_data_depth_rgb[:, :, 2] * 256.0 * 256.0) / (256.0 ** 3 - 1.0)
    #     depths_m = 1000.0 * normalized_depths
    #     h_above_ground = (depth_camera_sensor.obj.get_location().z - ego_tf.location.z) - depths_m
    #     print(f'h_above_ground: {h_above_ground}')

    #     # Get BEV camera to BEV grid transformation
    #     if self.H_pixels_to_grid is None:
    #         self.H_pixels_to_grid = self.compute_pixel_to_ego_grid_H(
    #             camera_sensor,
    #             ego_tf,
    #             grid
    #         )

    #     # Filter drivable regions with a height constraint
    #     height_mask = (h_above_ground >= 0.0) & (h_above_ground <= 2.0)
    #     semantic_mask = np.isin(semantic_image, [ActorClass.ROADS, ActorClass.ROADLINE, ActorClass.GROUND, ActorClass.UNLABELED], invert=True)

    #     occupancy_image = (height_mask & semantic_mask).astype(np.uint8)

    #     # Warp BEV image to BEV grid
    #     bev_from_cam = cv2.warpPerspective(
    #         occupancy_image,
    #         self.H_pixels_to_grid,
    #         (grid.W, grid.H),
    #         flags=cv2.INTER_NEAREST,
    #         borderMode=cv2.BORDER_CONSTANT,
    #         borderValue=0
    #     ).astype(np.uint8)

    #     occupancy_grid = np.full((grid.H, grid.W), 255, dtype=np.uint8)
    #     occupancy_grid[(bev_from_cam > 0)] = 1

    def get_occupancy_map_visualize_lidar(
        self,
        cameras : Dict[str, CameraInterface],
        lidar_sensor : carla.Sensor,
        lidar_data : Dict,
        ego_tf : carla.Transform,
        grid : BEVGrid,
        point_list = None
    ) -> np.ndarray:
        occupancy_grid = np.zeros((grid.H, grid.W), dtype=np.uint8)

        # ############################################
        # # Lidar Processing
        # ############################################

        # Get 3D lidar points and semantic tags
        lidar_pts_3d = lidar_data['xyz'].astype(np.float32) # [Nx3]
        # lidar_semantic_tags = lidar_data['object_tag'].astype(np.int32) # [Nx1]
        lidar_semantic_tags = np.array(lidar_data['object_tag']) # [Nx1]

        print(f'lidar_semantic_tags shape: {lidar_semantic_tags.shape}')

        # We're negating the y to correclty visualize a world that matches
        # what we see in Unreal since Open3D uses a right-handed coordinate system
        lidar_pts_3d[:, :1] = -lidar_pts_3d[:, :1]

        # Colorize the pointcloud based on the CityScapes color palette
        int_color = LABEL_COLORS[lidar_semantic_tags]

        point_list.points = o3d.utility.Vector3dVector(lidar_pts_3d)
        point_list.colors = o3d.utility.Vector3dVector(int_color)

        # print(f'lidar_pts_3d shape: {lidar_pts_3d.shape}')

        # Convert to homogenous form [Nx4]
        ones = np.ones((lidar_pts_3d.shape[0], 1), dtype=np.float32)
        lidar_pts_world_4d = np.hstack([lidar_pts_3d, ones])

        # Transform points from lidar-frame to ego
        # T_lidar_wrt_world = np.array(lidar_sensor.get_transform().get_matrix(), dtype=np.float64) # [4x4]
        # T_world_wrt_ego = np.array(ego_tf.get_inverse_matrix(), dtype=np.float64) # [4x4]
        # T_lidar_wrt_ego = T_world_wrt_ego @ T_lidar_wrt_world

        T_lidar_wrt_ego = np.array([
            [ 0.0,  1.0,  0.0, 0.0],
            [-1.0,  0.0,  0.0, 0.0],
            [ 0.0,  0.0,  1.0, 2.5],
            [ 0.0,  0.0,  0.0, 1.0],
        ], dtype=np.float32)

        lidar_pts_ego_4d = lidar_pts_world_4d @ T_lidar_wrt_ego.T # [Nx4] @ [4x4]^T -> [Nx4]

        # Get 3D lidar points in ego frame [Nx3]
        lidar_pts_ego_3d = lidar_pts_ego_4d[:, :3]

        # print(f'lidar_pts_ego_3d shape: {lidar_pts_ego_3d.shape}')

        # Lidar points bounds filtering
        x_ok = (lidar_pts_ego_3d[:, 0] >= grid.x_min) & (lidar_pts_ego_3d[:, 0] < grid.x_max)
        y_ok = (lidar_pts_ego_3d[:, 1] >= grid.y_min) & (lidar_pts_ego_3d[:, 1] < grid.y_max)
        z_ok = (lidar_pts_ego_3d[:, 2] >= -2.0) & (lidar_pts_ego_3d[:, 2] < 3.0)

        # Crop box filter
        crop_box_filter = x_ok & y_ok & z_ok

        # Semantic filter (filter drivable regions)
        semantic_filter = np.isin(
            lidar_semantic_tags,
            [
                ActorClass.ROADS,
                ActorClass.ROADLINE,
                ActorClass.GROUND,
                ActorClass.UNLABELED,
                ActorClass.TERRAIN
            ],
            invert=True)

        # print(f'x_ok shape: {x_ok.shape}')
        # print(f'y_ok shape: {y_ok.shape}')
        # print(f'z_ok shape: {z_ok.shape}')
        # print(f'semantic_filter shape: {semantic_filter.shape}')
        # print(f'crop box filter shape: {crop_box_filter.shape}')

        # Create occupancy mask
        occ_mask = crop_box_filter & semantic_filter
        lidar_pts_ego_3d = lidar_pts_ego_3d[occ_mask]   # (M,3)

        # print(f'occ_mask shape: {occ_mask.shape}')
        # occ_mask = occ_mask.astype(bool)          # (N,1)

        # print(f'occ_mask post filter shape: {occ_mask.shape}')

        # # rows = np.where(occ_mask[:, 0])[0]        # 1-D row indices from the 2-D mask
        # # lidar_pts_ego_3d = lidar_pts_ego_3d[rows, :]   # (M,3)

        # print(f'lidar_pts_ego_3d filtered shape: {lidar_pts_ego_3d.shape}')
        # # Convert to grid coordinates and set occupancy cells
        i, j = grid.world_to_grid(lidar_pts_ego_3d[:, 0], lidar_pts_ego_3d[:, 1])
        valid = (i >= 0) & (i < grid.H) & (j >= 0) & (j < grid.W)

        # occ_mask_lidar = np.ones(occupancy_grid.shape, dtype=np.uint8)

        # # gx = i[valid]
        # # gy = j[valid]

        # # for cx, cy in zip(gx, gy):
        # #     # FREE along the ray (except endpoint)
        # #     for rx, ry in self.bresenham(50, 25, cx, cy):
        # #         if rx == cx and ry == cy:   # endpoint -> stop before marking FREE
        # #             break
        # #         occ_mask_lidar[rx, ry] = 1
        # #     # OCCUPIED at the hit
        # #     occ_mask_lidar[cx, cy] = 0

        occupancy_grid[i[valid], j[valid]] = 1

        # fill_unknown = (occupancy_grid == 255)
        # occupancy_grid[fill_unknown] = occ_mask_lidar[fill_unknown]

        # # cam_write = (bev_from_cam > 0) & (occupancy_grid != 1)
        # # occupancy_grid[cam_write] = 1

        # # fill_unknown = (occupancy_grid == 255)  & (bev_from_cam > 0)
        # # occupancy_grid[fill_unknown] = 1

        return occupancy_grid

    def get_occupancy_map(
        self,
        cameras : Dict[str, CameraInterface],
        lidar_sensor : carla.Sensor,
        lidar_data : Dict,
        ego_tf : carla.Transform,
        grid : BEVGrid
    ) -> np.ndarray:
        occupancy_grid = np.zeros((grid.H, grid.W), dtype=np.uint8)

        # ############################################
        # # Lidar Processing
        # ############################################

        # Get 3D lidar points and semantic tags
        lidar_pts_3d = lidar_data['xyz'].astype(np.float32) # [Nx3]
        lidar_semantic_tags = lidar_data['object_tag'].astype(np.int32) # [Nx1]

        # print(f'lidar_pts_3d shape: {lidar_pts_3d.shape}')

        # Convert to homogenous form [Nx4]
        ones = np.ones((lidar_pts_3d.shape[0], 1), dtype=np.float32)
        lidar_pts_world_4d = np.hstack([lidar_pts_3d, ones])

        # Transform points from lidar-frame to ego
        # T_lidar_wrt_world = np.array(lidar_sensor.get_transform().get_matrix(), dtype=np.float64) # [4x4]
        # T_world_wrt_ego = np.array(ego_tf.get_inverse_matrix(), dtype=np.float64) # [4x4]
        # T_lidar_wrt_ego = T_world_wrt_ego @ T_lidar_wrt_world

        T_lidar_wrt_ego = np.array([
            [ 0.0,  1.0,  0.0, 0.0],
            [-1.0,  0.0,  0.0, 0.0],
            [ 0.0,  0.0,  1.0, 2.5],
            [ 0.0,  0.0,  0.0, 1.0],
        ], dtype=np.float32)

        lidar_pts_ego_4d = lidar_pts_world_4d @ T_lidar_wrt_ego.T # [Nx4] @ [4x4]^T -> [Nx4]

        # Get 3D lidar points in ego frame [Nx3]
        lidar_pts_ego_3d = lidar_pts_ego_4d[:, :3]

        # print(f'lidar_pts_ego_3d shape: {lidar_pts_ego_3d.shape}')

        # Lidar points bounds filtering
        x_ok = (lidar_pts_ego_3d[:, 0] >= grid.x_min) & (lidar_pts_ego_3d[:, 0] < grid.x_max)
        y_ok = (lidar_pts_ego_3d[:, 1] >= grid.y_min) & (lidar_pts_ego_3d[:, 1] < grid.y_max)
        z_ok = (lidar_pts_ego_3d[:, 2] >= -2.0) & (lidar_pts_ego_3d[:, 2] < 3.0)

        # Crop box filter
        crop_box_filter = x_ok & y_ok & z_ok

        # Semantic filter (filter drivable regions)
        semantic_filter = np.isin(
            lidar_semantic_tags,
            [
                ActorClass.ROADS,
                ActorClass.ROADLINE,
                ActorClass.GROUND,
                ActorClass.UNLABELED,
                ActorClass.OTHER
            ],
            invert=True)

        # print(f'x_ok shape: {x_ok.shape}')
        # print(f'y_ok shape: {y_ok.shape}')
        # print(f'z_ok shape: {z_ok.shape}')
        # print(f'semantic_filter shape: {semantic_filter.shape}')
        # print(f'crop box filter shape: {crop_box_filter.shape}')

        # Create occupancy mask
        occ_mask = crop_box_filter & semantic_filter
        lidar_pts_ego_3d = lidar_pts_ego_3d[occ_mask]   # (M,3)

        # print(f'occ_mask shape: {occ_mask.shape}')
        # occ_mask = occ_mask.astype(bool)          # (N,1)

        # print(f'occ_mask post filter shape: {occ_mask.shape}')

        # # rows = np.where(occ_mask[:, 0])[0]        # 1-D row indices from the 2-D mask
        # # lidar_pts_ego_3d = lidar_pts_ego_3d[rows, :]   # (M,3)

        # print(f'lidar_pts_ego_3d filtered shape: {lidar_pts_ego_3d.shape}')
        # # Convert to grid coordinates and set occupancy cells
        i, j = grid.world_to_grid(lidar_pts_ego_3d[:, 0], lidar_pts_ego_3d[:, 1])
        valid = (i >= 0) & (i < grid.H) & (j >= 0) & (j < grid.W)

        # occ_mask_lidar = np.ones(occupancy_grid.shape, dtype=np.uint8)

        # # gx = i[valid]
        # # gy = j[valid]

        # # for cx, cy in zip(gx, gy):
        # #     # FREE along the ray (except endpoint)
        # #     for rx, ry in self.bresenham(50, 25, cx, cy):
        # #         if rx == cx and ry == cy:   # endpoint -> stop before marking FREE
        # #             break
        # #         occ_mask_lidar[rx, ry] = 1
        # #     # OCCUPIED at the hit
        # #     occ_mask_lidar[cx, cy] = 0

        occupancy_grid[i[valid], j[valid]] = 1

        # fill_unknown = (occupancy_grid == 255)
        # occupancy_grid[fill_unknown] = occ_mask_lidar[fill_unknown]

        # # cam_write = (bev_from_cam > 0) & (occupancy_grid != 1)
        # # occupancy_grid[cam_write] = 1

        # # fill_unknown = (occupancy_grid == 255)  & (bev_from_cam > 0)
        # # occupancy_grid[fill_unknown] = 1

        return occupancy_grid

    def update_ego_state(self, vehicle_transform):
        """Update ego vehicle state for coordinate transforms"""
        self.ego_world_pos = (vehicle_transform.location.x, vehicle_transform.location.y)
        self.ego_yaw = math.radians(vehicle_transform.rotation.yaw)

    def setup_cameras(
        self,
        cameras: List[Tuple[str, carla.Sensor]]
    ) -> None:
        """
        Initialize camera interfaces from CARLA camera actors.

        Args:
            cameras: List of tuples containing (tag, camera_actor_obj)
        """
        self.cameras.clear()

        for tag, camera_obj in cameras:
            self.cameras[tag] = CameraInterface(camera_obj)

    def world_to_grid(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        grid_x = int((world_x - self.world_bounds['min_x']) / self.grid_resolution)
        grid_y = int((world_y - self.world_bounds['min_y']) / self.grid_resolution)

        grid_x = np.clip(grid_x, 0, self.grid_size[0] - 1)
        grid_y = np.clip(grid_y, 0, self.grid_size[1] - 1)

        return grid_x, grid_y

    def camera_pixel_to_world_homography(
        self,
        camera : CameraInterface,
        ground_z : float
    ) -> Tuple[np.ndarray, np.ndarray]:
        T_cam_to_world = camera.world_to_camera_matrix
        R_cam_to_world = T_cam_to_world[:3, :3]
        t_cam_to_world = T_cam_to_world[:3, 3:4]

        r1 = R_cam_to_world[:, 0:1]
        r2 = R_cam_to_world[:, 1:2]
        r3 = R_cam_to_world[:, 2:3]

        A = np.hstack([r1, r2, t_cam_to_world + r3 * ground_z])
        H = camera.config.K @ A
        H_inv = np.linalg.inv(H).astype(np.float32)

        return (H, H_inv)

    def ego_to_grid_affine(
        self
    ) -> np.ndarray:
        S = np.array([
            [1.0 / self.grid_resolution, 0.0, -self.world_bounds['min_x'] / self.grid_resolution],
            [0.0, 1.0 / self.grid_resolution, -self.world_bounds['min_y'] / self.grid_resolution],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        return S

    def compute_pixel_to_ego_grid_H(
        self,
        camera : CameraInterface,
        ego_tf : carla.Transform,
        grid : BEVGrid,
        z_ground_world : Optional[float] = None
    ) -> np.ndarray:
        H_rows = grid.H
        W_cols = grid.W

        ############################################
        # Ego -> World coordinate transformation
        ############################################

        # Choose ground plane (world frame)
        camera_loc = camera.obj.get_location()
        if z_ground_world is None:
            # TODO: Replace 20.0 with known camera height
            z_ground_world = camera_loc.z - 20.0
            print(f'ground_world_z: {z_ground_world}')

        # Set ego-centric grid corners in meters (ego frame) [Nx3]
        corners_ego = np.array([
            [grid.x_min, grid.y_min, 0.0], # top-left (rear-left if x_min < 0)
            [grid.x_min, grid.y_max, 0.0], # top-right
            [grid.x_max, grid.y_max, 0.0], # bottom-right (front-right)
            [grid.x_max, grid.y_min, 0.0], # bottom-left
        ], dtype=np.float64)

        # Convert to homogenous form [Nx4]
        ones = np.ones((corners_ego.shape[0], 1), dtype=np.float64)
        corners_ego_4d = np.hstack([corners_ego, ones])

        # Transform ego-frame grid points to world frame
        T_ego_wrt_world = np.array(ego_tf.get_matrix(), dtype=np.float64) # [4x4]
        pts_world_4d = corners_ego_4d @ T_ego_wrt_world.T # [Nx4] @ [4x4]^T -> [Nx4]

        # Get 3D world points
        corners_world_3d = pts_world_4d[:, :3]

        # Set z-coordinates to ground plane
        corners_world_3d[:, 2] = z_ground_world

        # Convert to pixel coordinates (pixel frame of BEV camera) [Nx2]
        corners_pixels_2d = camera.project_world_to_pixels_arr(corners_world_3d)

        # Set ego-centric grid corners in pixels (grid frame) [Nx2]
        dst = np.array([
            [0.0, 0.0], # top-left
            [W_cols - 1.0, 0.0], # top-right
            [W_cols - 1.0, H_rows - 1.0], # bottom-right
            [0.0, H_rows - 1.0], # bottom-left
        ], dtype=np.float32)

        src = corners_pixels_2d.astype(np.float32)

        # Get perspective transform matrix [3x3]
        H_pixels_to_grid = cv2.getPerspectiveTransform(src, dst)
        return H_pixels_to_grid

    def camera_pixel_to_grid_homography(
        self,
        camera : CameraInterface,
        ground_z : float,
        ego_tf : carla.Transform
    ) -> np.ndarray:
        _, H_pixel_to_world = self.camera_pixel_to_world_homography(camera, ground_z)
        T_world_to_ego = np.array(ego_tf.get_inverse_matrix(), dtype=np.float32)      # (4,4)
        R_we = T_world_to_ego[:3, :3]                                                # (3,3)
        t_we = T_world_to_ego[:3, 3:4]                                               # (3,1)

        # Build 3×3 affine for the plane Z = ground_z:
        # [x_e; y_e; 1] = H_we @ [X_w; Y_w; 1]
        # where:
        # x_e = r11*X + r12*Y + (r13*ground_z + t_x)
        # y_e = r21*X + r22*Y + (r23*ground_z + t_y)
        H_world_to_ego = np.array([
            [R_we[0, 0], R_we[0, 1], R_we[0, 2] * ground_z + t_we[0, 0]],
            [R_we[1, 0], R_we[1, 1], R_we[1, 2] * ground_z + t_we[1, 0]],
            [0.0,        0.0,        1.0],
        ], dtype=np.float32)  # (3,3)

        # 3×3 ego->grid affine (scale, origin shift, optional flip)
        S_ego_to_grid = self.ego_to_grid_affine().astype(np.float32)                 # (3,3)

        # Compose: pixels -> world -> ego -> grid
        H_pixel_to_grid = S_ego_to_grid @ H_pixel_to_world          # (3,3)
        return H_pixel_to_grid

    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates (center of cell)"""
        world_x = self.world_bounds['min_x'] + (grid_x + 0.5) * self.grid_resolution
        world_y = self.world_bounds['min_y'] + (grid_y + 0.5) * self.grid_resolution

        return world_x, world_y

    def camera_pixel_to_world(self, u: int, v: int) -> Tuple[float, float]:
        """Convert BEV camera pixel to world coordinates"""
        # Convert pixel to vehicle-relative coordinates (BEV camera is centered on ego)
        ego_rel_x = (u / self.camera_width - 0.5) * self.camera_fov_meters
        ego_rel_y = (v / self.camera_height - 0.5) * self.camera_fov_meters

        # Transform to world coordinates using ego vehicle pose
        cos_yaw = math.cos(self.ego_yaw)
        sin_yaw = math.sin(self.ego_yaw)

        world_x = self.ego_world_pos[0] + ego_rel_x * cos_yaw - ego_rel_y * sin_yaw
        world_y = self.ego_world_pos[1] + ego_rel_x * sin_yaw + ego_rel_y * cos_yaw

        return world_x, world_y

    def lidar_point_to_world(self, point: np.ndarray) -> Tuple[float, float]:
        """Convert LIDAR point to world coordinates"""
        # LIDAR point is in vehicle coordinate system
        x, y, z = point

        # Transform to world coordinates
        cos_yaw = math.cos(self.ego_yaw)
        sin_yaw = math.sin(self.ego_yaw)

        world_x = self.ego_world_pos[0] + x * cos_yaw - y * sin_yaw
        world_y = self.ego_world_pos[1] + x * sin_yaw + y * cos_yaw

        return world_x, world_y

    def is_in_bounds(self, world_x: float, world_y: float) -> bool:
        """Check if world coordinates are within grid bounds"""
        return (self.world_bounds['min_x'] <= world_x <= self.world_bounds['max_x'] and
                self.world_bounds['min_y'] <= world_y <= self.world_bounds['max_y'])

    def process_camera_data(self, image_data, vehicle_transform) -> Dict:
        """Process BEV instance segmentation camera data"""
        self.update_ego_state(vehicle_transform)

        # Convert CARLA image
        array = np.frombuffer(image_data.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image_data.height, image_data.width, 4))

        self.camera_height, self.camera_width = image_data.height, image_data.width

        # Extract channels
        semantic_image = array[:, :, 2]
        instance_image = array[:, :, 0].astype(np.uint32) + (array[:, :, 1].astype(np.uint32) << 8)

        self.latest_camera_data = array[:, :, :3][:, :, ::-1]

        # TODO: Fix ground Z, need ego vehicle object
        grid = BEVGrid()
        H_pixels_to_grid = self.compute_pixel_to_ego_grid_H(
            self.cameras['rgb_bev_instance'],
            vehicle_transform,
            grid
        )
        semantic_image = semantic_image.astype(np.uint8)
        occupancy_image = cv2.warpPerspective(
            semantic_image, H_pixels_to_grid, (grid.W, grid.H), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        occ = np.isin(occupancy_image, [ActorClass.ROADS, ActorClass.ROADLINE], invert=True)
        occupancy_mask = occ.astype(np.uint8)
        print(f'Occ mask shape: {occupancy_mask.shape}')

        vis = (occupancy_mask * 255).astype(np.uint8)
        cv2.namedWindow("BirdView Occupancy", cv2.WINDOW_NORMAL)
        cv2.imshow('BirdView Occupancy', vis)
        cv2.waitKey(1)

        cv2.namedWindow("BirdView Img", cv2.WINDOW_NORMAL)
        cv2.imshow('BirdView Img', self.latest_camera_data)
        cv2.waitKey(1)

        # H_pixel_to_grid = self.camera_pixel_to_grid_homography(self.cameras['rgb_bev_instance'], ground_z=1.0, ego_tf=vehicle_transform)
        # occupancy_grid = cv2.warpPerspective(
        #     mask_u8,
        #     H_pixel_to_grid,
        #     dsize=(self.grid_size[0], self.grid_size[1]),
        #     flags=cv2.INTER_NEAREST,
        #     borderMode=cv2.BORDER_CONSTANT,
        #     borderValue=0
        # )

        # self.occupancy_map = occupancy_grid

        camera_instances = {}
        current_time = time.time()

        # Subsample for performance
        for v in range(0, image_data.height, 4):
            for u in range(0, image_data.width, 4):
                semantic_tag = int(semantic_image[v, u])
                instance_id = int(instance_image[v, u])

                if instance_id > 0:
                    # Convert to world coordinates
                    world_x, world_y = self.camera_pixel_to_world(u, v)

                    # Check if within grid bounds
                    if not self.is_in_bounds(world_x, world_y):
                        continue

                    # Convert to grid coordinates
                    grid_x, grid_y = self.world_to_grid(world_x, world_y)

                    # Update occupancy
                    if semantic_tag in [ActorClass.ROADS, ActorClass.ROADLINE]:
                        occupancy = 0.0
                    else:
                        occupancy = 1.0
                    # if semantic_tag in [1, 2, 4, 5, 10, 11]:
                    #     occupancy = 1.0
                    # elif semantic_tag in [7, 8, 14]:
                    #     occupancy = 0.0
                    # else:
                    #     occupancy = 0.3

                    # self.occupancy_map[grid_x, grid_y] = occupancy
                    # Weighted update
                    camera_weight = 0.7
                    total_weight = self.camera_confidence[grid_x, grid_y] + camera_weight

                    if total_weight > 0:
                        self.occupancy_map[grid_x, grid_y] = (
                            self.occupancy_map[grid_x, grid_y] * self.camera_confidence[grid_x, grid_y] +
                            occupancy * camera_weight
                        ) / total_weight

                    # Update maps
                    self.instance_map[grid_x, grid_y] = instance_id
                    self.semantic_map[grid_x, grid_y] = semantic_tag
                    self.camera_confidence[grid_x, grid_y] = min(total_weight, 1.0)

                    # Track instances
                    if instance_id not in camera_instances:
                        camera_instances[instance_id] = {
                            'semantic_tag': semantic_tag,
                            'world_positions': set(),
                            'grid_positions': set(),
                            'pixel_count': 0
                        }

                    camera_instances[instance_id]['world_positions'].add((world_x, world_y))
                    camera_instances[instance_id]['grid_positions'].add((grid_x, grid_y))
                    camera_instances[instance_id]['pixel_count'] += 1

        return camera_instances

    def process_lidar_data(self, lidar_data, vehicle_transform) -> Dict:
        """Process semantic LIDAR data"""
        self.update_ego_state(vehicle_transform)

        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)
        ]))

        point_cloud = np.array([[p[0], p[1], p[2]] for p in points])
        instance_ids = np.array([p[4] for p in points])
        semantic_tags = np.array([p[5] for p in points])

        self.latest_lidar_data = {
            'points': point_cloud,
            'instance_ids': instance_ids,
            'semantic_tags': semantic_tags
        }

        lidar_instances = {}
        current_time = time.time()

        for point, instance_id, semantic_tag in zip(point_cloud, instance_ids, semantic_tags):
            if instance_id > 0:
                # Convert to world coordinates
                world_x, world_y = self.lidar_point_to_world(point)

                # Check if within grid bounds
                if not self.is_in_bounds(world_x, world_y):
                    continue

                # Convert to grid coordinates
                grid_x, grid_y = self.world_to_grid(world_x, world_y)

                # Update occupancy
                if semantic_tag in [ActorClass.ROADS, ActorClass.ROADLINE]:
                    occupancy = 0.0
                else:
                    occupancy = 1.0
                # if semantic_tag in [1, 2, 4, 5, 10, 11]:
                #     occupancy = 1.0
                # elif semantic_tag in [7, 8, 14]:
                #     occupancy = 0.0
                # else:
                #     occupancy = 0.3

                lidar_weight = 0.8

                # Fuse with camera data
                if self.camera_confidence[grid_x, grid_y] > 0:
                    total_weight = (self.camera_confidence[grid_x, grid_y] +
                                  self.lidar_confidence[grid_x, grid_y] + lidar_weight)
                    self.occupancy_map[grid_x, grid_y] = (
                        self.occupancy_map[grid_x, grid_y] *
                        (self.camera_confidence[grid_x, grid_y] + self.lidar_confidence[grid_x, grid_y]) +
                        occupancy * lidar_weight
                    ) / total_weight
                else:
                    self.occupancy_map[grid_x, grid_y] = occupancy

                # LIDAR priority for instance assignment
                self.instance_map[grid_x, grid_y] = instance_id
                self.semantic_map[grid_x, grid_y] = semantic_tag
                self.lidar_confidence[grid_x, grid_y] = min(
                    self.lidar_confidence[grid_x, grid_y] + lidar_weight, 1.0
                )

                # Track instances
                if instance_id not in lidar_instances:
                    lidar_instances[instance_id] = {
                        'semantic_tag': semantic_tag,
                        'world_positions': set(),
                        'grid_positions': set(),
                        'point_count': 0
                    }

                lidar_instances[instance_id]['world_positions'].add((world_x, world_y))
                lidar_instances[instance_id]['grid_positions'].add((grid_x, grid_y))
                lidar_instances[instance_id]['point_count'] += 1

        return lidar_instances

    def update_instance_tracking(self, camera_instances: Dict, lidar_instances: Dict):
        """Update tracked instances with world coordinates"""
        current_time = time.time()
        all_instances = {**camera_instances, **lidar_instances}

        for instance_id, data in all_instances.items():
            if instance_id in self.tracked_instances:
                instance = self.tracked_instances[instance_id]
                instance.world_positions = data['world_positions']
                instance.grid_positions = data['grid_positions']
                instance.last_seen = current_time
                instance.confidence = min(instance.confidence + 0.1, 1.0)
            else:
                self.tracked_instances[instance_id] = InstanceInfo(
                    instance_id=instance_id,
                    semantic_tag=data['semantic_tag'],
                    world_positions=data['world_positions'],
                    grid_positions=data['grid_positions'],
                    last_seen=current_time,
                    confidence=0.7
                )

        # Remove old instances
        instances_to_remove = []
        for instance_id, instance in self.tracked_instances.items():
            if current_time - instance.last_seen > 3.0:
                instances_to_remove.append(instance_id)

        for instance_id in instances_to_remove:
            self.critical_instances.discard(instance_id)
            del self.tracked_instances[instance_id]

    def update_base_cost_map(self):
        """Update base cost map from semantic information"""
        self.base_cost_map.fill(0.0)

        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                semantic_tag = self.semantic_map[i, j]
                if semantic_tag > 0:
                    self.base_cost_map[i, j] = self.base_costs.get(semantic_tag, 50)

    def mark_critical_instance(self, instance_id: int, is_critical: bool = True):
        """Mark instance as critical with world coordinate context"""
        if instance_id in self.tracked_instances:
            self.tracked_instances[instance_id].is_critical = is_critical
            if is_critical:
                self.critical_instances.add(instance_id)
            else:
                self.critical_instances.discard(instance_id)

    def update_dynamic_cost_map(self):
        """Update dynamic cost map for critical instances"""
        self.dynamic_cost_map.fill(0.0)

        for instance_id, instance in self.tracked_instances.items():
            if instance.is_critical or instance_id in self.critical_instances:
                # Apply high cost to critical instances
                for grid_x, grid_y in instance.grid_positions:
                    if (0 <= grid_x < self.grid_size[0] and 0 <= grid_y < self.grid_size[1]):
                        base_cost = self.base_cost_map[grid_x, grid_y]
                        critical_cost = base_cost * self.critical_cost_multiplier
                        self.dynamic_cost_map[grid_x, grid_y] = max(
                            self.dynamic_cost_map[grid_x, grid_y], critical_cost
                        )

                # Inflate around critical instances
                self._inflate_critical_instance(instance)

            elif instance.semantic_tag in self.dynamic_classes:
                for grid_x, grid_y in instance.grid_positions:
                    if (0 <= grid_x < self.grid_size[0] and 0 <= grid_y < self.grid_size[1]):
                        dynamic_cost = self.base_cost_map[grid_x, grid_y] * 1.3
                        self.dynamic_cost_map[grid_x, grid_y] = max(
                            self.dynamic_cost_map[grid_x, grid_y], dynamic_cost
                        )

    def _inflate_critical_instance(self, instance: InstanceInfo):
        """Apply cost inflation around critical instances"""
        for grid_x, grid_y in instance.grid_positions:
            for dx in range(-self.critical_inflation_radius, self.critical_inflation_radius + 1):
                for dy in range(-self.critical_inflation_radius, self.critical_inflation_radius + 1):
                    new_x, new_y = grid_x + dx, grid_y + dy

                    if (0 <= new_x < self.grid_size[0] and 0 <= new_y < self.grid_size[1]):
                        distance = np.sqrt(dx*dx + dy*dy)
                        if distance <= self.critical_inflation_radius:
                            inflation_factor = 1.0 - (distance / self.critical_inflation_radius)
                            inflation_cost = 80 * inflation_factor
                            self.dynamic_cost_map[new_x, new_y] = max(
                                self.dynamic_cost_map[new_x, new_y], inflation_cost
                            )

    def update_total_cost_map(self):
        """Combine base and dynamic costs"""
        self.total_cost_map = np.maximum(self.base_cost_map, self.dynamic_cost_map)

    def identify_critical_actors(self):
        """Identify critical actors based on proximity to ego"""
        ego_grid_x, ego_grid_y = self.world_to_grid(*self.ego_world_pos)

        for instance_id, instance in self.tracked_instances.items():
            # Mark pedestrians as critical
            if instance.semantic_tag == 4:
                self.mark_critical_instance(instance_id, True)

            # Mark close vehicles as critical
            elif instance.semantic_tag == 10:
                min_distance = float('inf')
                for grid_x, grid_y in instance.grid_positions:
                    distance = np.sqrt((grid_x - ego_grid_x)**2 + (grid_y - ego_grid_y)**2)
                    min_distance = min(min_distance, distance)

                # Convert to world distance
                world_distance = min_distance * self.grid_resolution
                if world_distance < 15.0:  # Within 15 meters
                    self.mark_critical_instance(instance_id, True)

    def get_world_bounds(self) -> Dict[str, float]:
        """Get world coordinate bounds of the grid"""
        return self.world_bounds.copy()

    def get_grid_info(self) -> Dict:
        """Get grid information for path planning"""
        return {
            'grid_size': self.grid_size,
            'resolution': self.grid_resolution,
            'world_bounds': self.world_bounds,
            'ego_world_pos': self.ego_world_pos,
            'ego_grid_pos': self.world_to_grid(*self.ego_world_pos)
        }

    def query_cost_at_world_pos(self, world_x: float, world_y: float) -> float:
        """Query cost at world coordinates"""
        if not self.is_in_bounds(world_x, world_y):
            return float('inf')  # Out of bounds

        grid_x, grid_y = self.world_to_grid(world_x, world_y)
        return self.total_cost_map[grid_x, grid_y]

    def get_occupancy_at_world_pos(self, world_x: float, world_y: float) -> float:
        """Query occupancy at world coordinates"""
        if not self.is_in_bounds(world_x, world_y):
            return 1.0  # Assume occupied if out of bounds

        grid_x, grid_y = self.world_to_grid(world_x, world_y)
        return self.occupancy_map[grid_x, grid_y]

class WorldMappingSystem:
    """CARLA mapping system with proper world coordinate integration"""

    def __init__(self, carla_client, world, vehicle,
                 grid_resolution: float = 0.25, grid_size: Tuple[int, int] = (400, 400)):
        self.client = carla_client
        self.world = world
        self.vehicle = vehicle

        # Initialize processor with world coordinates
        vehicle_transform = vehicle.get_transform()
        world_origin = (vehicle_transform.location.x, vehicle_transform.location.y)

        self.processor = WorldMappingProcessor(
            grid_size=grid_size,
            grid_resolution=grid_resolution,
            world_origin=world_origin
        )

        self.camera_sensor = None
        self.lidar_sensor = None
        self.camera_queue = queue.Queue()
        self.lidar_queue = queue.Queue()

        self.setup_sensors()

    def setup_sensors(self):
        """Setup sensors with proper world positioning"""
        blueprint_library = self.world.get_blueprint_library()

        # BEV Camera
        camera_bp = blueprint_library.find('sensor.camera.instance_segmentation')
        camera_bp.set_attribute('image_size_x', str(self.processor.camera_width))
        camera_bp.set_attribute('image_size_y', str(self.processor.camera_height))
        camera_bp.set_attribute('fov', '90')

        camera_transform = carla.Transform(
            carla.Location(x=0, y=0, z=50),
            carla.Rotation(pitch=-90, yaw=0, roll=0)
        )

        self.camera_sensor = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle
        )
        self.camera_sensor.listen(lambda data: self.camera_queue.put(data))

        camera_name = "rgb_bev_instance"
        self.processor.setup_cameras([(camera_name, self.camera_sensor)])

        # Semantic LIDAR
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast_semantic')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('range', '50')
        lidar_bp.set_attribute('points_per_second', '50000')
        lidar_bp.set_attribute('rotation_frequency', '20')

        lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=2))

        self.lidar_sensor = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.vehicle
        )
        self.lidar_sensor.listen(lambda data: self.lidar_queue.put(data))

    def update_maps(self, instance_cost_overrides: Dict[int, float] = None):
        """Update maps with world coordinate awareness"""
        camera_instances = {}
        lidar_instances = {}
        vehicle_transform = self.vehicle.get_transform()

        # Process sensor data with vehicle pose
        if not self.camera_queue.empty():
            camera_data = self.camera_queue.get()
            camera_instances = self.processor.process_camera_data(camera_data, vehicle_transform)

        if not self.lidar_queue.empty():
            lidar_data = self.lidar_queue.get()
            lidar_instances = self.processor.process_lidar_data(lidar_data, vehicle_transform)

        # Update tracking and costs
        self.processor.update_instance_tracking(camera_instances, lidar_instances)
        self.processor.identify_critical_actors()
        self.processor.update_base_cost_map()
        self.processor.update_dynamic_cost_map()
        self.processor.update_total_cost_map()

        return camera_instances, lidar_instances

    def get_path_planning_data(self) -> Dict:
        """Get data formatted for path planning algorithms"""
        grid_info = self.processor.get_grid_info()

        return {
            'occupancy_grid': self.processor.occupancy_map.copy(),
            'cost_map': self.processor.total_cost_map.copy(),
            'grid_resolution': grid_info['resolution'],
            'world_bounds': grid_info['world_bounds'],
            'ego_world_pos': grid_info['ego_world_pos'],
            'ego_grid_pos': grid_info['ego_grid_pos'],
            'world_to_grid_func': self.processor.world_to_grid,
            'grid_to_world_func': self.processor.grid_to_world
        }

    def plan_path_to_goal(self, goal_world_x: float, goal_world_y: float) -> List[Tuple[float, float]]:
        """Simple A* path planning to demonstrate world coordinate usage"""
        from heapq import heappush, heappop

        # Get current state
        start_world = self.processor.ego_world_pos
        start_grid = self.processor.world_to_grid(*start_world)
        goal_grid = self.processor.world_to_grid(goal_world_x, goal_world_y)

        if not self.processor.is_in_bounds(goal_world_x, goal_world_y):
            print(f"Goal ({goal_world_x:.1f}, {goal_world_y:.1f}) is out of bounds!")
            return []

        print(f"Planning path from world {start_world} to ({goal_world_x:.1f}, {goal_world_y:.1f})")
        print(f"Grid coordinates: {start_grid} to {goal_grid}")

        # Simple A* implementation
        open_set = []
        heappush(open_set, (0, start_grid))

        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self._heuristic(start_grid, goal_grid)}

        directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]

        while open_set:
            current = heappop(open_set)[1]

            if current == goal_grid:
                # Reconstruct path in world coordinates
                path_grid = []
                node = goal_grid
                while node in came_from:
                    path_grid.append(node)
                    node = came_from[node]
                path_grid.append(start_grid)
                path_grid.reverse()

                # Convert to world coordinates
                path_world = [self.processor.grid_to_world(gx, gy) for gx, gy in path_grid]
                return path_world

            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)

                if (neighbor[0] < 0 or neighbor[0] >= self.processor.grid_size[0] or
                    neighbor[1] < 0 or neighbor[1] >= self.processor.grid_size[1]):
                    continue

                # Check occupancy and cost
                if self.processor.occupancy_map[neighbor] > 0.8:  # Occupied
                    continue

                move_cost = 1.0 if dx == 0 or dy == 0 else 1.414  # Diagonal cost
                cost_penalty = self.processor.total_cost_map[neighbor] * 0.01
                tentative_g = g_score[current] + move_cost + cost_penalty

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_grid)
                    heappush(open_set, (f_score[neighbor], neighbor))

        print("No path found!")
        return []

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Heuristic function for A* (Euclidean distance)"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def visualize_maps(self, save_path: Optional[str] = None,
                      planned_path: List[Tuple[float, float]] = None):
        """Visualize maps with world coordinate overlays"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))

        # BEV Camera
        if self.processor.latest_camera_data is not None:
            axes[0, 0].imshow(self.processor.latest_camera_data)
            axes[0, 0].set_title('BEV Instance Segmentation')
            axes[0, 0].axis('off')

        # LIDAR Point Cloud with world coordinates
        if self.processor.latest_lidar_data is not None:
            points = self.processor.latest_lidar_data['points']
            instance_ids = self.processor.latest_lidar_data['instance_ids']

            # Transform to world coordinates for display
            world_points = []
            for point in points:
                wx, wy = self.processor.lidar_point_to_world(point)
                world_points.append([wx, wy])
            world_points = np.array(world_points)

            scatter = axes[0, 1].scatter(world_points[:, 0], world_points[:, 1],
                                       c=instance_ids, s=0.5, cmap='tab20')
            axes[0, 1].set_title('LIDAR (World Coordinates)')
            axes[0, 1].set_xlabel('World X (m)')
            axes[0, 1].set_ylabel('World Y (m)')
            axes[0, 1].grid(True)

            # Mark ego position
            ego_x, ego_y = self.processor.ego_world_pos
            axes[0, 1].plot(ego_x, ego_y, 'r*', markersize=15, label='Ego Vehicle')
            axes[0, 1].legend()

        # Occupancy Map
        im_occ = axes[0, 2].imshow(self.processor.occupancy_map.T,
                                  origin='lower', cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[0, 2].set_title('Occupancy Map')
        axes[0, 2].set_xlabel('Grid X')
        axes[0, 2].set_ylabel('Grid Y')
        plt.colorbar(im_occ, ax=axes[0, 2], label='Occupancy')

        # Add world coordinate annotations
        # self._add_world_coordinate_ticks(axes[0, 2])

        # Instance Map
        im_inst = axes[1, 0].imshow(self.processor.instance_map.T,
                                   origin='lower', cmap='tab20')
        axes[1, 0].set_title('Instance Map')
        axes[1, 0].set_xlabel('Grid X')
        axes[1, 0].set_ylabel('Grid Y')
        plt.colorbar(im_inst, ax=axes[1, 0], label='Instance ID')
        # self._add_world_coordinate_ticks(axes[1, 0])

        # Semantic Map
        im_sem = axes[1, 1].imshow(self.processor.semantic_map.T,
                                  origin='lower', cmap='tab10')
        axes[1, 1].set_title('Semantic Map')
        axes[1, 1].set_xlabel('Grid X')
        axes[1, 1].set_ylabel('Grid Y')
        plt.colorbar(im_sem, ax=axes[1, 1], label='Semantic Class')
        # self._add_world_coordinate_ticks(axes[1, 1])

        # Base Cost Map
        im_base = axes[1, 2].imshow(self.processor.base_cost_map.T,
                                   origin='lower', cmap='hot')
        axes[1, 2].set_title('Base Cost Map')
        axes[1, 2].set_xlabel('Grid X')
        axes[1, 2].set_ylabel('Grid Y')
        plt.colorbar(im_base, ax=axes[1, 2], label='Base Cost')
        # self._add_world_coordinate_ticks(axes[1, 2])

        # Dynamic Cost Map
        im_dyn = axes[2, 0].imshow(self.processor.dynamic_cost_map.T,
                                  origin='lower', cmap='plasma')
        axes[2, 0].set_title('Dynamic Cost Map')
        axes[2, 0].set_xlabel('Grid X')
        axes[2, 0].set_ylabel('Grid Y')
        plt.colorbar(im_dyn, ax=axes[2, 0], label='Dynamic Cost')
        # self._add_world_coordinate_ticks(axes[2, 0])

        # Total Cost Map with Path
        im_total = axes[2, 1].imshow(self.processor.total_cost_map.T,
                                    origin='lower', cmap='hot')
        axes[2, 1].set_title('Total Cost Map + Path')
        axes[2, 1].set_xlabel('Grid X')
        axes[2, 1].set_ylabel('Grid Y')
        plt.colorbar(im_total, ax=axes[2, 1], label='Total Cost')
        # self._add_world_coordinate_ticks(axes[2, 1])

        # Plot planned path if provided
        if planned_path:
            path_grid_x = []
            path_grid_y = []
            for wx, wy in planned_path:
                gx, gy = self.processor.world_to_grid(wx, wy)
                path_grid_x.append(gx)
                path_grid_y.append(gy)

            axes[2, 1].plot(path_grid_x, path_grid_y, 'cyan', linewidth=3,
                          label=f'Planned Path ({len(planned_path)} waypoints)')
            axes[2, 1].plot(path_grid_x[0], path_grid_y[0], 'go', markersize=8, label='Start')
            axes[2, 1].plot(path_grid_x[-1], path_grid_y[-1], 'ro', markersize=8, label='Goal')
            axes[2, 1].legend()

        # World Coordinate Reference
        axes[2, 2].text(0.1, 0.9, 'World Coordinate System:', fontsize=12, fontweight='bold',
                       transform=axes[2, 2].transAxes)

        bounds = self.processor.get_world_bounds()
        ego_pos = self.processor.ego_world_pos
        grid_info = self.processor.get_grid_info()

        info_text = f"""
Grid Size: {self.processor.grid_size[0]} x {self.processor.grid_size[1]}
Resolution: {self.processor.grid_resolution:.2f} m/cell
World Bounds:
  X: [{bounds['min_x']:.1f}, {bounds['max_x']:.1f}] m
  Y: [{bounds['min_y']:.1f}, {bounds['max_y']:.1f}] m
Ego Position:
  World: ({ego_pos[0]:.1f}, {ego_pos[1]:.1f}) m
  Grid: {grid_info['ego_grid_pos']}
        """

        axes[2, 2].text(0.1, 0.8, info_text, fontsize=10, fontfamily='monospace',
                       transform=axes[2, 2].transAxes, verticalalignment='top')
        axes[2, 2].axis('off')

        plt.tight_layout()

        # if save_path:
        #     plt.savefig(save_path, dpi=200, bbox_inches='tight')

        return fig

    def _add_world_coordinate_ticks(self, ax):
        """Add world coordinate tick labels to grid plots"""
        # Get current ticks
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()

        # Convert to world coordinates
        x_world_labels = []
        for x in xticks:
            if 0 <= x < self.processor.grid_size[0]:
                wx, _ = self.processor.grid_to_world(int(x), 0)
                x_world_labels.append(f'{wx:.0f}m')
            else:
                x_world_labels.append('')

        y_world_labels = []
        for y in yticks:
            if 0 <= y < self.processor.grid_size[1]:
                _, wy = self.processor.grid_to_world(0, int(y))
                y_world_labels.append(f'{wy:.0f}m')
            else:
                y_world_labels.append('')

        # Create secondary axes for world coordinates
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(x_world_labels, fontsize=8)
        ax2.set_xlabel('World X (m)', fontsize=8)

        ax3 = ax.twinx()
        ax3.set_ylim(ax.get_ylim())
        ax3.set_yticks(yticks)
        ax3.set_yticklabels(y_world_labels, fontsize=8)
        ax3.set_ylabel('World Y (m)', fontsize=8)

    def cleanup(self):
        """Cleanup sensors"""
        if self.camera_sensor:
            self.camera_sensor.destroy()
        if self.lidar_sensor:
            self.lidar_sensor.destroy()

def main_example():
    """Example with world coordinate integration and path planning"""
    try:
        # Connect to CARLA
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        # Spawn vehicle
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])

        # Create world-aware mapping system
        mapping_system = WorldMappingSystem(
            client, world, vehicle,
            grid_resolution=0.25,  # 25cm per cell
            grid_size=(100, 100)   # 100m x 100m coverage
        )

        print("World-Coordinate CARLA Mapping System initialized...")

        # Get initial world bounds
        bounds = mapping_system.processor.get_world_bounds()
        print(f"Mapping area: X[{bounds['min_x']:.1f}, {bounds['max_x']:.1f}], "
              f"Y[{bounds['min_y']:.1f}, {bounds['max_y']:.1f}] meters")

        # Run mapping
        planned_path = None
        for i in range(500):
            # Update maps
            camera_instances, lidar_instances = mapping_system.update_maps()

            # Demonstrate path planning every 30 iterations
            # if i == 30:
            #     ego_pos = mapping_system.processor.ego_world_pos
            #     goal_x = ego_pos[0] + 20.0  # 20 meters ahead
            #     goal_y = ego_pos[1] + 10.0  # 10 meters to the side

            #     print(f"\nPlanning path to goal: ({goal_x:.1f}, {goal_y:.1f})")
            #     # planned_path = mapping_system.plan_path_to_goal(goal_x, goal_y)
            #     planned_path = None

            #     if planned_path:
            #         print(f"Path found with {len(planned_path)} waypoints")
            #         print("First few waypoints:")
            #         for j, (px, py) in enumerate(planned_path[:3]):
            #             print(f"  {j}: ({px:.2f}, {py:.2f}) m")
            #     else:
            #         print("No path found to goal")

            # Visualize with world coordinates
            if i % 25 == 0:
                critical_instances = len([inst for inst in mapping_system.processor.tracked_instances.values()
                                        if inst.is_critical])
                print(f"Iteration {i}: Critical instances: {critical_instances}")

                # fig = mapping_system.visualize_maps(f"world_mapping_{i}.png", planned_path)
                # plt.show(block=True)
                # plt.pause(3.0)
                # plt.close(fig)

            # Move vehicle
            # if i % 35 == 0:
                # vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=0.1))
            vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=0.1))

            # time.sleep(0.1)

        # Final demonstration
        print("\nFinal mapping results...")

        # Get path planning data
        planning_data = mapping_system.get_path_planning_data()
        print(f"Path planning data ready:")
        print(f"  Grid resolution: {planning_data['grid_resolution']:.2f} m/cell")
        print(f"  Ego position: {planning_data['ego_world_pos']}")
        print(f"  Grid bounds: {planning_data['world_bounds']}")

        # Demonstrate coordinate conversion
        test_world_x, test_world_y = planning_data['ego_world_pos']
        test_grid_x, test_grid_y = planning_data['world_to_grid_func'](test_world_x, test_world_y)
        back_world_x, back_world_y = planning_data['grid_to_world_func'](test_grid_x, test_grid_y)

        print(f"\nCoordinate conversion test:")
        print(f"  World: ({test_world_x:.2f}, {test_world_y:.2f}) m")
        print(f"  Grid:  ({test_grid_x}, {test_grid_y})")
        print(f"  Back:  ({back_world_x:.2f}, {back_world_y:.2f}) m")

        # Final visualization
        # fig = mapping_system.visualize_maps("final_world_mapping.png", planned_path)
        # plt.show()

        # Save world-coordinate maps
        maps = {
            'occupancy': mapping_system.processor.occupancy_map,
            'cost': mapping_system.processor.total_cost_map,
            'instance': mapping_system.processor.instance_map,
            'semantic': mapping_system.processor.semantic_map
        }

        for name, map_data in maps.items():
            np.save(f'world_{name}_map.npy', map_data)

        # Save metadata for path planning
        metadata = {
            'grid_size': mapping_system.processor.grid_size,
            'resolution': mapping_system.processor.grid_resolution,
            'world_bounds': bounds,
            'ego_world_pos': mapping_system.processor.ego_world_pos
        }
        # np.save('world_map_metadata.npy', metadata)

        print(f"\nSaved world-coordinate maps and metadata")
        print("Maps can now be used for path planning with proper world coordinates!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if 'mapping_system' in locals():
            mapping_system.cleanup()
        if 'vehicle' in locals():
            vehicle.destroy()

if __name__ == "__main__":
    main_example()