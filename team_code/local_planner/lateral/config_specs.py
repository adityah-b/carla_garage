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


@dataclass
class SLGridSpec:
    L_max : float
    L_min : float
    dl : float

    S_max : float
    ds : float

    lane_width_buffer_m : float

    # Costs
    collision_cost : float
    source_lane_cost : float
    target_lane_cost : float


@dataclass
class SLAlgoSpec:
    L_min : float
    L_max : float

    dL_max : float

    ddL_max : float

    # Kinematic costs
    W_ref_offset: float
    W_heading: float
    W_curvature : float
    W_obstacle: float

    # Grid resolution
    ds_grid : float
    dl_grid : float

    # Algorithm resolution
    ds_algo : float
    dl_algo : float

    @property
    def dl_quant_factor(self) -> float:
        """
        Multiplier for quantizing continuous velocity in the search state key.
        Derived from the max allowable velocity difference that would keep
        kinematic drift bounded within a single occupancy cell (ds_grid)
        over one planner step (dt_algo).
        """
        return self.dl_algo / self.ds_grid

@dataclass
class SLQPSpec:
    num_vars : int
    num_samples : int

    # Heading and curvature limits
    dL_max : float
    ddL_max : float

    # Kinematic costs
    W_L: float
    W_dL: float
    W_ddL : float
    W_dddL : float
