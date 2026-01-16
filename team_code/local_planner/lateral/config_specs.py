import numpy as np

from dataclasses import dataclass
from typing import Tuple

@dataclass
class LatGridSpec:
    # Lateral XY planning grid specs
    x_min : float
    x_max : float
    y_min : float
    y_max : float
    resolution : float

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
    # Grid resolution
    grid_res : float

    # Vehicle dimensions
    veh_half_len : float
    veh_half_width : float

    # Goal node tolerance
    goal_tol_m : float

    # Costmap weight
    w_cost : float

    # Vehicle grid dimensions
    # TODO: REPLACE WITH 3 CIRCLE COLLISION CHECK
    @property
    def veh_half_len_grid(self) -> float:
        return np.floor(self.veh_half_len * 0.5 / self.grid_res)

    @property
    def veh_half_width_grid(self) -> float:
        return np.floor(self.veh_half_width * 0.5 / self.grid_res)

    # Goal tolerance grid radius
    @property
    def goal_tol_grid(self) -> float:
        return self.goal_tol_m / self.grid_res
