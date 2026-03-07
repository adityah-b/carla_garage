import carla.libcarla
import carla
import numpy as np
import cv2
import h5py
import os

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, Union
from enum import IntEnum

from scene_descriptor.camera_interface import CameraInterface

from privileged_route_planner import PlannerState
from scene_descriptor.scene_descriptor import SceneData
from actor_prediction.motion_prediction import PredictionData
from actor_prediction.collision_checker import CollisionInterval
from team_code.scene_analyzer.parsers.ego_plan_pydantic_models import *

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

        i = np.clip(i, 0, self.H - 1)
        j = np.clip(j, 0, self.W - 1)

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
        }

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
            shoulder_mask_pixels = np.array(hf['shoulder'], dtype=np.uint8)
            parking_mask_pixels = np.array(hf['parking'], dtype=np.uint8)
            print(f'road_mask_pixels shape: {road_mask_pixels.shape}')

            # Cache the static road layer (binary) and its smoothed variant for reuse every tick
            self.road_mask_world = (
                (road_mask_pixels > 0) |
                (shoulder_mask_pixels > 0) |
                (parking_mask_pixels > 0)
            ).astype(np.uint8)
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
        self.static_obstacle_cost = 255
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
        obj : Union[carla.Actor, carla.BoundingBox],
    ) -> np.ndarray:
        """
        Returns (4,3) ground-plane corners (x,y,z) in WORLD frame for a CARLA BoundingBox,
        respecting its rotation (yaw) and extents.
        Corner order: [front-right, front-left, back-left, back-right] in the box's local frame.
        """
        if isinstance(obj, carla.Actor):
            actor_tf = obj.get_transform()
            bb = obj.bounding_box

            cx = bb.location.x + actor_tf.location.x
            cy = bb.location.y + actor_tf.location.y
            cz = bb.location.z + actor_tf.location.z

            yaw_deg = actor_tf.rotation.yaw
        elif isinstance(obj, carla.BoundingBox):
            bb = obj

            cx = bb.location.x
            cy = bb.location.y
            cz = bb.location.z

            yaw_deg = bb.rotation.yaw
        else:
            raise TypeError(f"get_bb_corners_world expects carla.Actor or carla.BoundingBox, got {type(obj)}")

        ex, ey, ez = bb.extent.x, bb.extent.y, bb.extent.z

        # Enforce that x extent corresponds to length (usually extents are swapped for parked static vehicles)
        if ex < ey:
            ex = bb.extent.y
            ey = bb.extent.x

        # yaw = np.deg2rad(bb.rotation.yaw + actor_tf.rotation.yaw)  # CARLA stores degrees
        yaw = np.deg2rad(yaw_deg)  # CARLA stores degrees

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

    def convert_frame_to_ego(
        self,
        frame_pts : np.ndarray,
        T_frame_wrt_ego : np.ndarray,
    ) -> np.ndarray:
        # Convert frame points to homogenous form
        ones = np.ones((frame_pts.shape[0], 1), dtype=np.float32) # [Nx1]
        frame_pts_4d = np.hstack([frame_pts, ones])

        # Transform to ego frame
        ego_pts_3d = (frame_pts_4d @ T_frame_wrt_ego.T)[:, :3]

        return ego_pts_3d

    def draw_actor_on_map(
        self,
        ego_tf : carla.Transform,
        grid : BEVGrid,
        actor : Union[carla.Actor, carla.BoundingBox],
        map : np.ndarray,
        cost : float
    ) -> None:
        ego_loc = ego_tf.location
        ego_yaw_rad = np.deg2rad(ego_tf.rotation.yaw)

        C = np.cos(ego_yaw_rad)
        S = np.sin(ego_yaw_rad)

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
            return

        i, j = grid.world_to_grid(actor_corners_ego_x, actor_corners_ego_y)
        poly = np.stack([j, i], dtype=np.int32, axis=1)

        hull = cv2.convexHull(poly)

        cv2.fillConvexPoly(map, hull, cost)

    # def get_base_cost_maps(
    #     self,
    #     lidar_data : Dict,
    #     ego_tf : carla.Transform,
    #     grid : BEVGrid
    # ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     ego_loc = ego_tf.location
    #     ego_yaw_rad = np.deg2rad(ego_tf.rotation.yaw)

    #     # ############################################
    #     # # Drivable Area Filtering
    #     # ############################################

    #     R = 100.0
    #     C = np.cos(ego_yaw_rad)
    #     S = np.sin(ego_yaw_rad)

    #     grid_res = grid.resolution
    #     ppm = self.pixels_per_meter
    #     origin_x, origin_y = self.world_origin

    #     warp_matrix = np.array([
    #         [
    #             ppm * (-S * grid_res),
    #             ppm * (-C * grid_res),
    #             ppm * ((ego_loc.x + C*grid.x_max - S*grid.y_min) - 0.5*grid_res*(C + S) - origin_x)
    #         ],
    #         [
    #             ppm * ( C * grid_res),
    #             ppm * (-S * grid_res),
    #             ppm * ((ego_loc.y + S*grid.x_max + C*grid.y_min) + 0.5*grid_res*(C - S) - origin_y)
    #         ],
    #     ], dtype=np.float32)

    #     road_mask_local = cv2.warpAffine(
    #         self.road_mask_world_dilated,
    #         warp_matrix,
    #         (grid.W, grid.H),
    #         flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
    #         borderMode=cv2.BORDER_CONSTANT,
    #         borderValue=0,
    #     ).astype(np.uint8)

    #     road_occupancy_map = np.zeros((grid.H, grid.W), dtype=np.uint8)
    #     road_occupancy_map[road_mask_local == 0] = 1

    #     # ############################################
    #     # # Actor Processing
    #     # ############################################
    #     static_cost_map = np.zeros((grid.H, grid.W), dtype=np.uint8)
    #     dynamic_cost_map = np.zeros((grid.H, grid.W), dtype=np.uint8)

    #     actors = self.world.get_actors()
    #     vehicles = list(actors.filter('*vehicle*'))
    #     pedestrians = list(actors.filter('*walker*'))

    #     static_actors = list(actors.filter('*static*'))
    #     dynamic_actors = vehicles + pedestrians

    #     parked_dynamic_vehicles = set()

    #     # Static actor processing
    #     for actor in static_actors:
    #         if not actor.is_active:
    #             continue

    #         if ego_loc.distance(actor.get_location()) > R:
    #             continue

    #         should_draw_actor = False
    #         # Draw static vehicle (parked)
    #         if actor.type_id == 'static.prop.mesh':
    #             if 'mesh_path' in actor.attributes and 'Car' in actor.attributes['mesh_path']:
    #                 should_draw_actor = True

    #         # Draw traffic cones
    #         elif actor.type_id == 'static.prop.constructioncone':
    #             should_draw_actor = True

    #         if should_draw_actor:
    #             self.draw_actor_on_map(ego_tf, grid, actor, static_cost_map, self.static_obstacle_cost)

    #     # Dynamic actor processing
    #     for actor in dynamic_actors:
    #         if actor.id == self.ego_vehicle.id:
    #             continue

    #         if not actor.is_active:
    #             continue

    #         if ego_loc.distance(actor.get_location()) > R:
    #             continue

    #         # Check if actor is dynamically spawned non-moving vehicle (treat as static object)
    #         if isinstance(actor, carla.Vehicle):
    #             veh_control = actor.get_control()
    #             if veh_control.hand_brake:
    #                 parked_dynamic_vehicles.add(actor.id)
    #                 self.draw_actor_on_map(ego_tf, grid, actor, static_cost_map, self.static_obstacle_cost)
    #                 continue

    #         self.draw_actor_on_map(
    #             ego_tf, grid, actor, dynamic_cost_map, self.static_obstacle_cost
    #         )

    #     # ############################################
    #     # # Lidar Processing
    #     # ############################################

    #     # Get 3D lidar points and semantic tags
    #     lidar_pts_3d = lidar_data['xyz'].astype(np.float32) # [Nx3]
    #     lidar_semantic_tags = lidar_data['object_tag'].astype(np.int32) # [Nx1]
    #     lidar_instance_tags = lidar_data['object_idx'].astype(np.int32) # [Nx1]

    #     # Convert to homogenous form [Nx4]
    #     ones = np.ones((lidar_pts_3d.shape[0], 1), dtype=np.float32)
    #     lidar_pts_world_4d = np.hstack([lidar_pts_3d, ones])

    #     # Transform points from lidar-frame to ego
    #     # T_lidar_wrt_world = np.array(lidar_sensor.get_transform().get_matrix(), dtype=np.float64) # [4x4]
    #     # T_world_wrt_ego = np.array(ego_tf.get_inverse_matrix(), dtype=np.float64) # [4x4]
    #     # T_lidar_wrt_ego = T_world_wrt_ego @ T_lidar_wrt_world

    #     T_lidar_wrt_ego = np.array([
    #         [ 0.0,  1.0,  0.0, 0.0],
    #         [-1.0,  0.0,  0.0, 0.0],
    #         [ 0.0,  0.0,  1.0, 2.5],
    #         [ 0.0,  0.0,  0.0, 1.0],
    #     ], dtype=np.float32)

    #     lidar_pts_ego_4d = lidar_pts_world_4d @ T_lidar_wrt_ego.T # [Nx4] @ [4x4]^T -> [Nx4]

    #     # Get 3D lidar points in ego frame [Nx3]
    #     lidar_pts_ego_3d = lidar_pts_ego_4d[:, :3]

    #     # Lidar points bounds filtering
    #     x_ok = (lidar_pts_ego_3d[:, 0] >= grid.x_min) & (lidar_pts_ego_3d[:, 0] < grid.x_max)
    #     y_ok = (lidar_pts_ego_3d[:, 1] >= grid.y_min) & (lidar_pts_ego_3d[:, 1] < grid.y_max)
    #     z_ok = (lidar_pts_ego_3d[:, 2] >= 0.5) & (lidar_pts_ego_3d[:, 2] < 3.0)

    #     # Crop box filter
    #     crop_box_filter = x_ok & y_ok & z_ok

    #     # Semantic filter (filter drivable regions)
    #     semantic_filter = np.isin(
    #         lidar_semantic_tags,
    #         [
    #             ActorClass.ROADS,
    #             ActorClass.ROADLINE,
    #             ActorClass.GROUND,
    #             ActorClass.UNLABELED,
    #             ActorClass.OTHER
    #         ],
    #         invert=True)

    #     # Create occupancy mask
    #     occ_mask = crop_box_filter & semantic_filter
    #     if np.any(occ_mask):
    #         lidar_pts_ego_3d = lidar_pts_ego_3d[occ_mask] # (M,3)
    #         lidar_tags_filtered = lidar_semantic_tags[occ_mask]
    #         lidar_instances_filtered = lidar_instance_tags[occ_mask]

    #         static_vehicle_mask = np.isin(lidar_instances_filtered, list(parked_dynamic_vehicles))
    #         dynamic_semantic_mask = np.isin(lidar_tags_filtered, list(self.dynamic_classes))

    #         static_pts = lidar_pts_ego_3d[~dynamic_semantic_mask | static_vehicle_mask]
    #         if static_pts.size > 0:
    #             i_static, j_static = grid.world_to_grid(static_pts[:, 0], static_pts[:, 1])
    #             static_cost_map[i_static, j_static] = self.static_obstacle_cost

    #         dynamic_pts = lidar_pts_ego_3d[dynamic_semantic_mask]
    #         if dynamic_pts.size > 0:
    #             i_dyn, j_dyn = grid.world_to_grid(dynamic_pts[:, 0], dynamic_pts[:, 1])
    #             dynamic_cost_map[i_dyn, j_dyn] = np.maximum(dynamic_cost_map[i_dyn, j_dyn], self.static_obstacle_cost)

    #     return road_occupancy_map, static_cost_map, dynamic_cost_map

    def get_base_cost_maps(
        self,
        planner_state : PlannerState,
        scene_data : SceneData,
        prediction_data : PredictionData,
        lidar_data : Dict,
        ego_tf : carla.Transform,
        grid : BEVGrid,
        ego_plan : EgoPlan,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        # road_occupancy_map: 1 = non-drivable, 0 = drivable
        road_occupancy_map = np.zeros((grid.H, grid.W), dtype=np.uint8)
        road_occupancy_map[road_mask_local == 0] = 1

        # ############################################
        # # Actor Processing
        # ############################################
        static_cost_map = np.zeros((grid.H, grid.W), dtype=np.uint8)
        dynamic_cost_map = np.zeros((grid.H, grid.W), dtype=np.uint8)

        actors = self.world.get_actors()
        vehicles = list(actors.filter('*vehicle*'))
        pedestrians = list(actors.filter('*walker*'))

        dynamic_actors = vehicles + pedestrians

        static_obstacles = set()

        # Static actor processing
        all_obstacles = scene_data.obstacle_data.all_obstacles
        for obs in all_obstacles:
            self.draw_actor_on_map(ego_tf=ego_tf, grid=grid, actor=obs.obstacle, map=static_cost_map, cost=self.static_obstacle_cost)
            static_obstacles.add(obs.id)

        # TODO: CURRENTLY SELECTING TARGETS MANUALLY, IDEALLY SHOULD BE DONE BY LLMS
        if ego_plan.action == Action.SHARE_LANE:
            # Intruding vehicle processing
            all_intruding_vehicles = scene_data.vehicle_data.get(traffic_type="oncoming", get_intruders=True)
            # print(f"\n\nINTRUDING VEHICLES LENGTH")
            # print(f"\t\tlen: {len(all_intruding_vehicles)}")
            for intruder in all_intruding_vehicles:
                print(f'\n\nSCENE DRAWING')
                print(f'\n\nINTRUDER ID: {intruder.id}, INTRUSION IDX: {intruder.intrusion_idx}, ROUTE IDX: {planner_state.route_index}')
                print(f'\n\nINTRUSION BB IDX: {intruder.intrusion_idx // 20}, INTRUSION BB IDX OFFSET: {(intruder.intrusion_idx // 20) - (planner_state.route_index // 20)}')
                intruder_overlap = prediction_data.all_actor_overlaps.get(intruder.id)[0]

                # Draw predicted bounding boxes for approximately 2s
                num_frames = 40
                intrusion_idx = intruder.intrusion_idx

                intrusion_bb : carla.libcarla.BoundingBox = planner_state.original_route_bbs[intrusion_idx // 20]
                bb_right_vec = intrusion_bb.rotation.get_right_vector()

                intruder_loc = intruder.vehicle.get_location()
                route_bb_to_intruder_vec = intruder_loc - intrusion_bb.location
                lateral_disp = bb_right_vec.dot(route_bb_to_intruder_vec)

                print(f'\n\nLATERAL DISP: {lateral_disp}')

                # TODO: TEMPORRARY
                # self.draw_actor_on_map(
                #     ego_tf, grid, intruder.vehicle, dynamic_cost_map, self.static_obstacle_cost
                # )
                self.draw_actor_on_map(
                    ego_tf, grid, intruder.vehicle, static_cost_map, self.static_obstacle_cost
                )

                # bb_start_idx = intruder_overlap.space_start_idx + (planner_state.route_index // 20)
                # bb_end_idx = intruder_overlap.space_end_idx + (planner_state.route_index // 20)

                # print(f'\n\nBB INDICES: start: {bb_start_idx}, end: {bb_end_idx}')

                cost = 150.0
                # for bb in planner_state.original_route_bbs[bb_start_idx:bb_end_idx + 1]:\
                for bb in planner_state.original_route_bbs[planner_state.route_index // 20 : intrusion_idx // 20 + 1]:
                    right_vec = bb.rotation.get_right_vector()

                    shifted_loc = bb.location + lateral_disp * right_vec
                    new_extent = bb.extent
                    new_extent.y = intruder.vehicle.bounding_box.extent.y

                    shifted_bb = carla.BoundingBox(shifted_loc, new_extent)
                    shifted_bb.rotation = bb.rotation

                    cost *= 1.1
                    self.draw_actor_on_map(
                        ego_tf, grid, shifted_bb, dynamic_cost_map, cost
                    )
                    # self.draw_actor_on_map(
                    #     ego_tf, grid, shifted_bb, static_cost_map, cost
                    # )

        # Cyclist processing
        all_cyclists = scene_data.vehicle_data.get(vehicle_types={"cyclist"})
        # print(f"\n\nCYCLISTS LENGTH")
        # print(f"\t\tlen: {len(all_cyclists)}")
        for cyclist in all_cyclists:
            forecasted_bbs = prediction_data.veh_forecasted_bbs.get(cyclist.id, [])
            # print(f'forecasted_bbs: {len(forecasted_bbs)}')
            num_frames = 40
            print(f'\n\nDRAWING CYCLIST ID: {cyclist.id} LAST BB DISTANCE: {self.ego_vehicle.get_location().distance(forecasted_bbs[num_frames - 1].location)}')

            # TODO: TEMPORRARY
            self.draw_actor_on_map(
                ego_tf, grid, cyclist.vehicle, static_cost_map, self.static_obstacle_cost
            )

            for bb in forecasted_bbs[:num_frames]:
                # self.draw_actor_on_map(
                #     ego_tf, grid, bb, dynamic_cost_map, self.static_obstacle_cost
                # )
                self.draw_actor_on_map(
                    ego_tf, grid, bb, dynamic_cost_map, self.static_obstacle_cost
                )

        # ############################################
        # # Lidar Processing
        # ############################################

        # Get 3D lidar points and semantic tags
        lidar_pts_3d = lidar_data['xyz'].astype(np.float32)   # [N,3]
        lidar_semantic_tags = lidar_data['object_tag'].astype(np.int32)   # [N,1]
        lidar_instance_tags = lidar_data['object_idx'].astype(np.int32)   # [N,1]

        # Convert to homogenous form [N,4]
        ones = np.ones((lidar_pts_3d.shape[0], 1), dtype=np.float32)
        lidar_pts_world_4d = np.hstack([lidar_pts_3d, ones])

        # Transform points from lidar-frame to ego
        # T_lidar_wrt_world = np.array(lidar_sensor.get_transform().get_matrix(), dtype=np.float64) # [4x4]
        # T_world_wrt_ego = np.array(ego_tf.get_inverse_matrix(), dtype=np.float64) # [4x4]
        # T_lidar_wrt_ego = T_world_wrt_ego @ T_lidar_wrt_world

        # TODO: CHANGE THIS HARD CODING
        T_lidar_wrt_ego = np.array([
            [ 0.0,  1.0,  0.0, 0.0],
            [-1.0,  0.0,  0.0, 0.0],
            [ 0.0,  0.0,  1.0, 2.5],
            [ 0.0,  0.0,  0.0, 1.0],
        ], dtype=np.float32)

        lidar_pts_ego_4d = lidar_pts_world_4d @ T_lidar_wrt_ego.T   # [N,4]
        lidar_pts_ego_3d = lidar_pts_ego_4d[:, :3]                  # [N,3]

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
            invert=True
        )

        # Create occupancy mask
        occ_mask = crop_box_filter & semantic_filter
        if np.any(occ_mask):
            # Filter LiDAR arrays by occupancy mask
            lidar_pts_ego_3d = lidar_pts_ego_3d[occ_mask]                   # (M,3)
            lidar_tags_filtered = lidar_semantic_tags[occ_mask]
            lidar_instances_filtered = lidar_instance_tags[occ_mask]

            # ------------------------------
            # Split static vs dynamic points
            # ------------------------------
            static_obstacle_mask = np.isin(lidar_instances_filtered, list(static_obstacles))
            dynamic_semantic_mask = np.isin(lidar_tags_filtered, list(self.dynamic_classes))

            # ---------- STATIC POINTS ----------
            static_pts = lidar_pts_ego_3d[static_obstacle_mask]
            if static_pts.size > 0:
                i_static, j_static = grid.world_to_grid(static_pts[:, 0], static_pts[:, 1])

                # Keep only points that lie on drivable road mask
                road_ok_static = road_mask_local[i_static, j_static] > 0  # or == 255, depending on mask
                i_static = i_static[road_ok_static]
                j_static = j_static[road_ok_static]

                static_cost_map[i_static, j_static] = self.static_obstacle_cost

            # ---------- DYNAMIC POINTS ----------
            # dynamic_pts = lidar_pts_ego_3d[dynamic_semantic_mask]
            # if dynamic_pts.size > 0:
            #     i_dyn, j_dyn = grid.world_to_grid(dynamic_pts[:, 0], dynamic_pts[:, 1])

            #     # Again, keep only points that lie on drivable road mask
            #     road_ok_dyn = road_mask_local[i_dyn, j_dyn] > 0
            #     i_dyn = i_dyn[road_ok_dyn]
            #     j_dyn = j_dyn[road_ok_dyn]

            #     dynamic_cost_map[i_dyn, j_dyn] = np.maximum(
            #         dynamic_cost_map[i_dyn, j_dyn],
            #         self.static_obstacle_cost
            #     )

        return road_occupancy_map, static_cost_map, dynamic_cost_map

    def _compute_road_cost(
        self,
        road_occupancy_map : np.ndarray,
        grid : BEVGrid,
        offroad_tau_m : float = 0.5,
        max_offroad_cost : float = 200.0
    ) -> np.ndarray:
        # Compute distance transform (distance to nearest road boundary)
        dist_cells = cv2.distanceTransform(road_occupancy_map, distanceType=cv2.DIST_L2, maskSize=5)

        # Convert to meters
        dist_m = dist_cells * grid.resolution

        # Apply road inflation (Exponential decay from max cost offroad to 0 cost on road)
        road_cost = max_offroad_cost * (1.0 - np.exp(-dist_m / max(offroad_tau_m, 1e-6)))

        return road_cost.astype(np.float32)

    def _compute_static_obstacle_cost(
        self,
        static_cost_map : np.ndarray,
        grid : BEVGrid,
        obstacle_inflation_radius_m : float = 1.75,
        obstacle_max_cost : float = 255.0
    ) -> np.ndarray:
        # Convert to binary occupancy (0 cost = free space)
        occ = (static_cost_map != 0).astype(np.uint8)

        # Compute distance transform (distance to nearest obstacle)
        dist = cv2.distanceTransform(1 - occ, distanceType=cv2.DIST_L2, maskSize=5)

        # Convert to meters
        dist_m = dist * grid.resolution

        # Apply obstacle inflation
        norm = np.clip(1.0 - (dist_m / obstacle_inflation_radius_m), 0.0, 1.0)
        obstacle_cost = (norm ** 0.8) * obstacle_max_cost

        # Hard block actual occupied cells
        obstacle_cost[static_cost_map != 0] = obstacle_max_cost

        return obstacle_cost.astype(np.float32)

    def _compute_route_cost(
        self,
        route_points_grid : Tuple[np.ndarray, np.ndarray],
        grid : BEVGrid,
        route_inflation_radius_m : float = 3.0,
        route_min_cost : float = 0.0,
        route_max_cost : float = 100.0
    ) -> np.ndarray:
        grid_x, grid_y = route_points_grid

        # Create route occupancy mask (0 = route)
        route_mask = np.ones((grid.H, grid.W), dtype=np.uint8)
        route_mask[grid_x, grid_y] = 0

        # Compute distance transform (distance to nearest route point)
        dist = cv2.distanceTransform(route_mask, distanceType=cv2.DIST_L2, maskSize=5)

        # Convert to meters
        dist_m = dist * grid.resolution

        # Apply route inflation (0 near route, 1 when far)
        norm = np.clip(dist_m / route_inflation_radius_m, 0.0, 1.0)
        route_cost = route_min_cost + norm * (route_max_cost - route_min_cost)

        return route_cost.astype(np.float32)

    def augment_static_cost_map(
        self,
        road_cost_map : np.ndarray,
        static_cost_map : np.ndarray,
        ego_vehicle : carla.Actor,
        route_points_world : np.ndarray,
        grid : BEVGrid,
        w_obst : float = 1.0,
        w_route : float = 1.0,
        w_road : float = 1.0,
        max_cost : float = 255.0
    ) -> np.ndarray:
        # ############################################
        # # Base Cost
        # ############################################
        road_cost = self._compute_road_cost(road_cost_map, grid)

        # ############################################
        # # Static Obstacle Inflation
        # ############################################
        obstacle_cost = self._compute_static_obstacle_cost(static_cost_map, grid)

        # ############################################
        # # Global Route Bias
        # ############################################

        # Transform world route points into ego frame
        ego_tf = ego_vehicle.get_transform()
        T_world_wrt_ego = np.array(ego_tf.get_inverse_matrix(), dtype=np.float32)
        route_pts_ego = self.convert_frame_to_ego(route_points_world, T_world_wrt_ego)

        # Convert to grid points
        grid_x, grid_y = grid.world_to_grid(route_pts_ego[:, 0], route_pts_ego[:, 1])

        # Get route cost
        route_cost = self._compute_route_cost((grid_x, grid_y), grid)

        # ############################################
        # # Costmap Construction
        # ############################################

        # Weighted sum
        cost_map = w_obst * obstacle_cost + w_route * route_cost + w_road * road_cost

        # Hard block actual occupied cells
        cost_map[static_cost_map != 0] = max_cost

        cost_map = np.clip(cost_map, 0.0, max_cost)

        return cost_map.astype(np.uint8)

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
