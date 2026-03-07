import cv2
import carla
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from config import GlobalConfig

# Perception modules
from privileged_route_planner import PlannerState
from scene_descriptor.scene_descriptor import SceneData

# Prediction modules
from actor_prediction.motion_prediction import PredictionData
from actor_prediction.collision_checker import CollisionInterval

# Behavioural planner modules
from team_code.scene_analyzer.parsers.ego_plan_pydantic_models import *

from .config_specs import *
from .planner_algo import PlannerAlgo
from .grid_mapper import GridMapper

@dataclass
class LatPlannerResult:
    start_idx : int = 0
    goal_idx : int = 0

    start_point_world : np.ndarray = field(default_factory=lambda : np.array([]))
    goal_point_world : np.ndarray = field(default_factory=lambda : np.array([]))

    planned_path : np.ndarray = field(default_factory=lambda : np.array([]))

    is_new_plan : bool = False

    @property
    def is_empty_plan(self) -> bool:
        return self.planned_path.size == 0

    def clear(self):
        self.start_idx = 0
        self.goal_idx = 0
        self.start_point_world = np.array([])
        self.goal_point_world = np.array([])
        self.planned_path = np.array([])
        self.is_new_plan = False

class LatPlanner:
    def __init__(
        self,
        config : GlobalConfig,
        ego_vehicle : carla.Vehicle,
        algo_name : str = 'astar',
    ):
        self.config = config
        self.ego_vehicle = ego_vehicle

        self.lat_grid_spec : LatGridSpec = self.config.lat_grid_spec
        self.lat_algo_spec : LatAlgoSpec = self.config.lat_algo_spec

        sim_freq = self.config.fps
        plan_freq_lo = self.config.lat_planning_frequency
        plan_freq_hi = self.config.lat_planning_frequency_high

        self.dt_sim = 1 / sim_freq
        self.sim_ticks_per_plan_lo = max(1, int(round(sim_freq / plan_freq_lo)))
        self.sim_ticks_per_plan_hi = max(1, int(round(sim_freq / plan_freq_hi)))

        self.grid_mapper = GridMapper(ego_vehicle, lat_grid_spec=self.lat_grid_spec)
        self.planner = PlannerAlgo(algo_name=algo_name, lat_algo_spec=self.lat_algo_spec)

        # Planner state
        self.current_plan = LatPlannerResult()

    @property
    def has_active_plan(self) -> bool:
        return not self.current_plan.is_empty_plan

    def reset_plan(self) -> None:
        self.current_plan.clear()

    def validate_plan_against_occupancy(
        self,
        occupancy_map: np.ndarray,
        path: np.ndarray,
    ) -> Tuple[bool, float]:
        pass

    # def run_step(
    #     self,
    #     route_points_world_3d : np.ndarray,
    #     lidar_data : Dict,
    #     start_point_world_3d : np.ndarray,
    #     goal_point_world_3d : np.ndarray,
    #     actor_collisions : Dict[int, List[CollisionInterval]] = {}, # K=actor id, V=collision intervals,
    #     all_conditions : Dict = {},
    #     actor_predictions : Dict[int, List[carla.BoundingBox]] = {}, # K=actor id, V=predicted BBs
    # ) -> np.ndarray:
    #     # Get occupancy and cost maps
    #     maps = self.grid_mapper.update_maps(
    #         lidar_data=lidar_data,
    #         route_points_world=route_points_world_3d,
    #         actor_collisions=actor_collisions,
    #         all_conditions=all_conditions,
    #         actor_predictions=actor_predictions
    #     )

    #     occupancy_map = maps.occupancy_map
    #     static_cost_map = maps.total_cost_map

    #     # Get ego transformation matrices
    #     ego_tf = self.ego_vehicle.get_transform()
    #     T_world_wrt_ego = np.array(ego_tf.get_inverse_matrix(), dtype=np.float32) # [4x4]
    #     T_ego_wrt_world = np.array(ego_tf.get_matrix(), dtype=np.float32) # [4x4]

    #     # Convert world points to ego frame
    #     world_pts_3d = np.vstack([start_point_world_3d, goal_point_world_3d])

    #     # print(f'world_pts_3d: {world_pts_3d}')
    #     # print(f'world_pts_3d shape: {world_pts_3d.shape}')

    #     ones = np.ones((world_pts_3d.shape[0], 1), dtype=np.float32)
    #     world_pts_4d = np.hstack([world_pts_3d, ones])

    #     # print(f'world_pts_4d: {world_pts_4d}')
    #     # print(f'world_pts_4d shape: {world_pts_4d.shape}')

    #     # Convert ego points to grid frame
    #     ego_pts_2d = (world_pts_4d @ T_world_wrt_ego.T)[:, :2]

    #     start_point_grid = ego_pts_2d[0, :]
    #     goal_point_grid = ego_pts_2d[1, :]

    #     # Convert to grid indices
    #     # TODO: CLEAN UP CODE
    #     start_node = self.lat_grid_spec.world_to_grid(start_point_grid[0], start_point_grid[1])
    #     goal_node = self.lat_grid_spec.world_to_grid(goal_point_grid[0], goal_point_grid[1])

    #     # NOTE: DEBUG
    #     self.start_node = start_node
    #     self.goal_node = goal_node

    #     # Run planner
    #     path_grid_list, cost = self.planner.run(
    #         occupancy_map=occupancy_map,
    #         cost_map=static_cost_map,
    #         start_node=start_node,
    #         goal_node=goal_node
    #     )

    #     # NOTE: Visualization
    #     occupancy_map = maps.occupancy_map
    #     occ_img = (occupancy_map * 255).astype(np.uint8)
    #     occ_bgr = cv2.cvtColor(occ_img, cv2.COLOR_GRAY2BGR)

    #     dynamic_cost_map = maps.dynamic_cost_map
    #     cost_map = maps.total_cost_map
    #     heat_map = cv2.applyColorMap(cost_map, cv2.COLORMAP_TURBO)
    #     dynamic_heat_map = cv2.applyColorMap(dynamic_cost_map, cv2.COLORMAP_TURBO)

    #     # # Draw on occupancy image
    #     H, W = occupancy_map.shape

    #     pts = np.asarray([(c_, r_) for (r_, c_) in path_grid_list], dtype=np.int32)  # (x=col, y=row)

    #     if pts.size != 0:
    #         # Line thickness
    #         thickness = max(1, int(round((0.4 / 0.5) * 1.0)))

    #         # --- Draw on copies ---
    #         heat_with_path = heat_map.copy()
    #         occ_with_path  = occ_bgr.copy()

    #         # Path polyline (green), start (red), goal (blue)
    #         for img in (heat_with_path, occ_with_path):
    #             cv2.polylines(img, [pts], isClosed=False, color=(0,255,0),
    #                         thickness=thickness, lineType=cv2.LINE_AA)
    #             cv2.circle(img, tuple(pts[0]),  radius=thickness*2, color=(0,0,255), thickness=-1)  # start
    #             cv2.circle(img, tuple(pts[-1]), radius=thickness*2, color=(255,0,0), thickness=-1)  # goal

    #         # cv2.namedWindow("BirdView Occupancy", cv2.WINDOW_NORMAL)
    #         # # cv2.imshow('BirdView Occupancy', occ_img)
    #         # cv2.imshow('BirdView Occupancy', occ_with_path)
    #         # cv2.waitKey(1)

    #         # cv2.namedWindow("BirdView Cost Map", cv2.WINDOW_NORMAL)
    #         # cv2.imshow('BirdView Cost Map', heat_map)
    #         # cv2.waitKey(1)

    #         map_with_path = np.hstack([occ_with_path, heat_with_path, dynamic_heat_map])
    #         cv2.namedWindow("BirdView Maps", cv2.WINDOW_NORMAL)
    #         cv2.imshow('BirdView Maps', map_with_path)
    #         cv2.waitKey(1)

    #     # NOTE: Visualization

    #     if len(path_grid_list) > 0:
    #         path_grid = np.array(path_grid_list, dtype=np.float32)
    #         x, y = self.lat_grid_spec.grid_to_world(path_grid[:, 0], path_grid[:, 1])
    #         ones = np.ones(x.shape[0], dtype=np.float32)
    #         path_ego_4d = np.stack([x, y, ones, ones], axis=-1)
    #         # print(f'path_ego_4d shape: {path_ego_4d.shape}')

    #         # Convert ego points to world frame
    #         path_world_3d = (path_ego_4d @ T_ego_wrt_world.T)[:, :3]

    #         # print(f'path_world_3d shape: {path_world_3d.shape}')

    #         return path_world_3d

    #     return np.array([])

    def run_step(
        self,
        state_machine,
        plan_tick_counter : int,
        planner_state : PlannerState,
        lidar_data : Dict,
        scene_data : SceneData,
        prediction_data : PredictionData,
        ego_plan : EgoPlan,
        *,
        key_actor_registry : Dict = {},
        buffer_distance : float = 10,
        all_conditions : Dict = {},
    ) -> LatPlannerResult:
        self.current_plan.is_new_plan = False

        print(f'\n\nLAT PLANNER')

        plan_lo = (plan_tick_counter % self.sim_ticks_per_plan_lo) == 1
        plan_hi = state_machine.needs_route_adjustment and (plan_tick_counter % self.sim_ticks_per_plan_hi) == 1
        should_plan_now = plan_lo or plan_hi

        print(f'\n\nNEEDS ROUTE ADJUSTMENT: {state_machine.needs_route_adjustment}')
        if should_plan_now:
            route_index = planner_state.route_index
            route_points = planner_state.route_points

            start_idx = route_index
            # TODO: NEED TO FIX HOW WE CHOOSE GOAL POINT (IDEALLY SHOULD BE PROVIDED BY LLM)
            # NOTE FEB 26: USING LLM SELECTED ACTOR REGISTRY TO SELECT TARGET DISTANCES
            goal_idx = state_machine.get_target_end_idx(
                config=self.config,
                planner_state=planner_state,
                scene_data=scene_data,
                prediction_data=prediction_data,
                actor_registry=key_actor_registry,
                target_distance_initial=self.config.lat_planner_max_distance,
                buffer_distance=buffer_distance,
            )

            print(f'\tstart_idx: {route_index}, goal_idx: {goal_idx}')
            print(f'\tSTATE MACHINE')
            print(f'\t\ttarget_end_idx: {state_machine.target_end_idx}, cur_route_changes: {state_machine.cur_route_changes}')

            start_point_world_3d = route_points[start_idx]
            goal_point_world_3d = route_points[goal_idx]

            print(f'\n\nLAT PLANNER POINT CONVERSIONS')
            print(f'\t\tWORLD FRAME GOAL PRIOR TO CONVERSION: {goal_point_world_3d}')
            # Get occupancy and cost maps
            maps = self.grid_mapper.update_maps(
                planner_state=planner_state,
                start_idx=start_idx,
                goal_idx=goal_idx,
                scene_data=scene_data,
                lidar_data=lidar_data,
                prediction_data=prediction_data,
                ego_plan=ego_plan,
                all_conditions=all_conditions,
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

            print(f'\t\tEGO FRAME GOAL PRIOR TO CONVERSION: {goal_point_grid}')

            # Convert to grid indices
            # TODO: CLEAN UP CODE
            start_node = self.lat_grid_spec.world_to_grid(start_point_grid[0], start_point_grid[1])
            goal_node = self.lat_grid_spec.world_to_grid(goal_point_grid[0], goal_point_grid[1])

            print(f'\t\tEGO GRID GOAL PRIOR TO CONVERSION: {goal_node}')

            # NOTE: DEBUG
            self.start_node = start_node
            self.goal_node = goal_node

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

            # Line thickness
            thickness = max(1, int(round((0.4 / 0.5) * 1.0)))

            # --- Draw on copies ---
            heat_with_path = heat_map.copy()
            occ_with_path  = occ_bgr.copy()

            # Path polyline (green), start (red), goal (blue)
            for img in (heat_with_path, occ_with_path):
                if pts.size != 0:
                    cv2.polylines(img, [pts], isClosed=False, color=(0,255,0),
                                thickness=thickness, lineType=cv2.LINE_AA)

                cv2.circle(img, tuple(start_node[::-1]),  radius=thickness*2, color=(0,0,255), thickness=-1)  # start
                cv2.circle(img, tuple(goal_node[::-1]), radius=thickness*2, color=(255,0,0), thickness=-1)  # goal

            map_with_path = np.hstack([occ_with_path, heat_with_path, dynamic_heat_map])
            cv2.namedWindow("BirdView Maps", cv2.WINDOW_NORMAL)
            cv2.imshow('BirdView Maps', map_with_path)
            cv2.waitKey(1)

            if len(path_grid_list) > 0:
                path_grid = np.array(path_grid_list, dtype=np.float32)
                print(f'\t\tASTAR EGO GRID GOAL: {path_grid[-1, :]}')
                x, y = self.lat_grid_spec.grid_to_world(path_grid[:, 0], path_grid[:, 1])
                ones = np.ones(x.shape[0], dtype=np.float32)
                path_ego_4d = np.stack([x, y, ones * 0.0, ones], axis=-1)
                print(f'\t\tASTAR EGO FRAME GOAL AFTER CONVERSION: {path_ego_4d[-1, :3]}')
                # print(f'path_ego_4d shape: {path_ego_4d.shape}')

                # Convert ego points to world frame
                path_world_3d = (path_ego_4d @ T_ego_wrt_world.T)[:, :3]
                print(f'\t\tASTAR WORLD FRAME GOAL AFTER CONVERSION: {path_world_3d[-1, :]}')

                # TODO: TEMPORARY CHECKING TO SEE IF UNCHANGED START AND GOAL POINTS FIX ANYTHING
                # REWRITE START AND END POINTS WITH ORIGINAL POINTS
                path_world_3d[0, :] = start_point_world_3d
                path_world_3d[-1, :] = goal_point_world_3d

                self.current_plan = LatPlannerResult(
                    start_idx=start_idx,
                    goal_idx=goal_idx,
                    start_point_world=start_point_world_3d,
                    goal_point_world=goal_point_world_3d,
                    planned_path=path_world_3d,
                    is_new_plan=True,
                )

                # print(f'path_world_3d shape: {path_world_3d.shape}')

            else:
                self.current_plan = LatPlannerResult()

        return self.current_plan

    ########################################
    # Visualization methods for debugging
    ########################################

# TODO: Add plotting/visualization methods for debugging