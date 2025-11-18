import numpy as np
from dataclasses import dataclass

@dataclass
class STGridSpec:
    S_max : float = 40.0
    ds : float = 0.1

    T_max : float = 3.0
    dt : float = 0.05

@dataclass
class STAlgoSpec:
    # Kinematic constraints
    A_max: float = 4.0
    J_max: float = 6.0

    # Kinematic costs
    W_vel: float = 1.0
    W_acc: float = 1.0
    W_jerk: float = 1.0

    # Grid resolution
    ds_grid : float = 0.1
    dt_grid : float = 0.05

    # Algorithm resolution
    @property
    def dt_algo_res(self) -> float:
        return 2 * np.sqrt((2 * self.ds_grid) / self.A_max).round(2)

    @property
    def ds_algo_res(self) -> float:
        return self.ds_grid
