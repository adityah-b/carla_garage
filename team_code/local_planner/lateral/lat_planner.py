import cv2
import carla
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple, Optional

from .config_specs import *
from .planner_algo import PlannerAlgo
from .grid_mapper import GridMapper

class LatPlanner:
    def __init__(
        self,
        ego_vehicle : carla.Vehicle,
        lat_grid_spec : LatGridSpec,
        lat_algo_spec : LatAlgoSpec,
        algo_name : str = 'astar',
        sim_freq : float = 20.0,
        plan_freq : float = 10.0,
    ):
        self.ego_vehicle = ego_vehicle

        self.lat_grid_spec = lat_grid_spec
        self.lat_algo_spec = lat_algo_spec

        self.grid_mapper = GridMapper(ego_vehicle, lat_grid_spec=lat_grid_spec)
        self.planner = PlannerAlgo(algo_name=algo_name, lat_algo_spec=lat_algo_spec)

    def validate_plan_against_occupancy(
        self,
        occupancy_map: np.ndarray,
        path: np.ndarray,
    ) -> Tuple[bool, float]:
        pass

    def run_step(
        self,
        route_points_world_3d : np.ndarray,
        lidar_data : Dict,
        start_point_world_3d : np.ndarray,
        goal_point_world_3d : np.ndarray,
        all_conditions : Dict = {}
    ) -> np.ndarray:
        # Get occupancy and cost maps
        maps = self.grid_mapper.update_maps(
            lidar_data=lidar_data,
            route_points_world=route_points_world_3d
        )

        occupancy_map = maps.occupancy_map
        static_cost_map = maps.total_cost_map

        # Get ego transformation matrices
        ego_tf = self.ego_vehicle.get_transform()
        T_world_wrt_ego = np.array(ego_tf.get_inverse_matrix(), dtype=np.float32) # [4x4]
        T_ego_wrt_world = np.array(ego_tf.get_matrix(), dtype=np.float32) # [4x4]

        # Convert world points to ego frame
        world_pts_3d = np.vstack([start_point_world_3d, goal_point_world_3d])

        # print(f'world_pts_3d: {world_pts_3d}')
        # print(f'world_pts_3d shape: {world_pts_3d.shape}')

        ones = np.ones((world_pts_3d.shape[0], 1), dtype=np.float32)
        world_pts_4d = np.hstack([world_pts_3d, ones])

        # print(f'world_pts_4d: {world_pts_4d}')
        # print(f'world_pts_4d shape: {world_pts_4d.shape}')

        # Convert ego points to grid frame
        ego_pts_2d = (world_pts_4d @ T_world_wrt_ego.T)[:, :2]

        start_point_grid = ego_pts_2d[0, :]
        goal_point_grid = ego_pts_2d[1, :]

        # Convert to grid indices
        # TODO: CLEAN UP CODE
        start_node = self.lat_grid_spec.world_to_grid(start_point_grid[0], start_point_grid[1])
        goal_node = self.lat_grid_spec.world_to_grid(goal_point_grid[0], goal_point_grid[1])

        # Run planner
        path_grid_list, cost = self.planner.run(
            occupancy_map=occupancy_map,
            cost_map=static_cost_map,
            start_node=start_node,
            goal_node=goal_node
        )

        # NOTE: Visualization
        occupancy_map = maps.occupancy_map
        occ_img = (occupancy_map * 255).astype(np.uint8)
        occ_bgr = cv2.cvtColor(occ_img, cv2.COLOR_GRAY2BGR)

        dynamic_cost_map = maps.dynamic_cost_map
        cost_map = maps.total_cost_map
        heat_map = cv2.applyColorMap(cost_map, cv2.COLORMAP_TURBO)
        dynamic_heat_map = cv2.applyColorMap(dynamic_cost_map, cv2.COLORMAP_TURBO)

        # # Draw on occupancy image
        H, W = occupancy_map.shape

        pts = np.asarray([(c_, r_) for (r_, c_) in path_grid_list], dtype=np.int32)  # (x=col, y=row)

        if pts.size != 0:
            # Line thickness
            thickness = max(1, int(round((0.4 / 0.5) * 1.0)))

            # --- Draw on copies ---
            heat_with_path = heat_map.copy()
            occ_with_path  = occ_bgr.copy()

            # Path polyline (green), start (red), goal (blue)
            for img in (heat_with_path, occ_with_path):
                cv2.polylines(img, [pts], isClosed=False, color=(0,255,0),
                            thickness=thickness, lineType=cv2.LINE_AA)
                cv2.circle(img, tuple(pts[0]),  radius=thickness*2, color=(0,0,255), thickness=-1)  # start
                cv2.circle(img, tuple(pts[-1]), radius=thickness*2, color=(255,0,0), thickness=-1)  # goal

            # cv2.namedWindow("BirdView Occupancy", cv2.WINDOW_NORMAL)
            # # cv2.imshow('BirdView Occupancy', occ_img)
            # cv2.imshow('BirdView Occupancy', occ_with_path)
            # cv2.waitKey(1)

            # cv2.namedWindow("BirdView Cost Map", cv2.WINDOW_NORMAL)
            # cv2.imshow('BirdView Cost Map', heat_map)
            # cv2.waitKey(1)

            map_with_path = np.hstack([occ_with_path, heat_with_path, dynamic_heat_map])
            cv2.namedWindow("BirdView Maps", cv2.WINDOW_NORMAL)
            cv2.imshow('BirdView Maps', map_with_path)
            cv2.waitKey(1)

        # NOTE: Visualization

        if len(path_grid_list) > 0:
            path_grid = np.array(path_grid_list, dtype=np.float32)
            x, y = self.lat_grid_spec.grid_to_world(path_grid[:, 0], path_grid[:, 1])
            ones = np.ones(x.shape[0], dtype=np.float32)
            path_ego_4d = np.stack([x, y, ones, ones], axis=-1)
            # print(f'path_ego_4d shape: {path_ego_4d.shape}')

            # Convert ego points to world frame
            path_world_3d = (path_ego_4d @ T_ego_wrt_world.T)[:, :3]

            # print(f'path_world_3d shape: {path_world_3d.shape}')

            return path_world_3d

        return np.array([])

    ########################################
    # Visualization methods for debugging
    ########################################

# TODO: Add plotting/visualization methods for debugging