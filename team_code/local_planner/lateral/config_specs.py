import numpy as np

from dataclasses import dataclass
from typing import Tuple

from config import GlobalConfig

@dataclass
class LatGridSpec:
    # Lateral XY planning grid specs
    x_min : float = -5.0
    x_max : float = 50.0
    y_min : float = -10.0
    y_max : float = 10.0
    resolution : float = 0.25

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
        # TODO: FIGURE OUT AXIS ORDERING
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

@dataclass
class LatAlgoSpec:
    # TODO: MOVE THIS TO config.py
    config = GlobalConfig()

    # Grid resolution
    # TODO: MOVE THIS TO CONFIG AND LINK WITH ACTUAL GRID RESOLUTION INSTEAD OF MANUAL TYPING
    grid_res : float = 0.25

    # Vehicle grid dimensions
    # veh_half_len_grid : float = np.floor(config.ego_extent_x / grid_res)
    # veh_half_width_grid : float = np.floor(config.ego_extent_y / grid_res)
    veh_half_len_grid : float = np.floor(config.ego_extent_x * 0.5 / grid_res)
    veh_half_width_grid : float = np.floor(config.ego_extent_y * 0.5 / grid_res)
    print(f'veh_half_len_grid: {veh_half_len_grid}')
    print(f'veh_half_width_grid: {veh_half_width_grid}')
    # veh_half_len_grid : float = 2.0 / grid_res
    # veh_half_width_grid : float = 1.0 / grid_res

    # Goal node tolerance
    goal_tol_m : float = 0.5
    goal_tol_grid : float = goal_tol_m / grid_res

    # Costmap weight
    w_cost : float = 1.0
