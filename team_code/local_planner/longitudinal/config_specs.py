import numpy as np
from dataclasses import dataclass

@dataclass
class STGridSpec:
    S_max : float
    ds : float

    T_max : float
    dt : float

@dataclass
class STAlgoSpec:
    # Kinematic constraints
    A_max: float
    J_max: float

    # Kinematic costs
    W_vel: float
    W_acc: float
    W_jerk: float

    # Grid resolution
    ds_grid : float
    dt_grid : float

    # Algorithm resolution
    ds_algo : float

    @property
    def dt_algo(self) -> float:
        # TODO: CHOOSE HOW MUCH TIME BUFFER WE"RE ADDING
        return 1.25 * np.sqrt((2 * self.ds_algo) / self.A_max).round(2)
