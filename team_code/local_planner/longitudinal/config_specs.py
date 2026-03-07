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
    A_min: float
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
        # dt_acc = 1.25 * np.sqrt((2 * self.ds_algo) / self.A_max)
        dt_max_acc = 1.1 * np.sqrt((2 * self.ds_algo) / self.A_max)
        dt_min_acc = 1.1 * np.sqrt((2 * self.ds_algo) / self.A_min)

        # Time required by Jerk limit (s = 1/6 * j * t^3)
        # We need enough time to ramp up acceleration to move 1 ds
        # Derived from: ds = 1/6 * J_max * t^3  ->  t = (6 * ds / J)^(1/3)
        # dt_jerk = 1.25 * np.cbrt((6 * self.ds_algo) / self.J_max)
        dt_jerk = 1.1 * np.cbrt((6 * self.ds_algo) / self.J_max)

        return float(max(dt_max_acc, dt_min_acc, dt_jerk))
