import numpy as np

from scipy.integrate import RK45
from config import GlobalConfig

class IDM:
    def __init__(self, config : GlobalConfig):
        self.config = config

    def compute_target_speed_idm(
        self,
        desired_speed : float,
        leading_actor_length : float,
        ego_speed : float,
        leading_actor_speed : float,
        distance_to_leading_actor : float,
        s0 : float = 4.,
        T : float = 0.5
    ) -> float:
        """
            Compute the target speed for the ego vehicle using the Intelligent Driver Model (IDM).

            Args:
                desired_speed (float): The desired speed of the ego vehicle.
                leading_actor_length (float): The length of the leading actor (vehicle or obstacle).
                ego_speed (float): The current speed of the ego vehicle.
                leading_actor_speed (float): The speed of the leading actor.
                distance_to_leading_actor (float): The distance to the leading actor.
                s0 (float, optional): The minimum desired net distance.
                T (float, optional): The desired time headway.

            Returns:
                float: The computed target speed for the ego vehicle.
        """

        a = self.config.idm_maximum_acceleration  # Maximum acceleration [m/s²]
        b = self.config.idm_comfortable_braking_deceleration_high_speed if ego_speed > \
                        self.config.idm_comfortable_braking_deceleration_threshold else \
                        self.config.idm_comfortable_braking_deceleration_low_speed # Comfortable deceleration [m/s²]
        delta = self.config.idm_acceleration_exponent  # Acceleration exponent

        t_bound = self.config.idm_t_bound

        def idm_equations(t, x):
            """
                    Differential equations for the Intelligent Driver Model.

                    Args:
                        t (float): Time.
                        x (list): State variables [position, speed].

                    Returns:
                        list: Derivatives of the state variables.
                    """
            ego_position, ego_speed = x

            speed_diff = ego_speed - leading_actor_speed
            s_star = s0 + ego_speed * T + ego_speed * speed_diff / 2. / np.sqrt(a * b)
            # The maximum is needed to avoid numerical unstabilities
            s = max(0.1, distance_to_leading_actor + t * leading_actor_speed - ego_position - leading_actor_length)
            dvdt = a * (1. - (ego_speed / desired_speed)**delta - (s_star / s)**2)

            return [ego_speed, dvdt]

        # Set the initial conditions
        y0 = [0., ego_speed]

        # Integrate the differential equations using RK45
        rk45 = RK45(fun=idm_equations, t0=0., y0=y0, t_bound=t_bound)
        while rk45.status == "running":
            rk45.step()

        # The target speed is the final speed obtained from the integration
        target_speed = rk45.y[1]

        # Clip the target speed to non-negative values
        return np.clip(target_speed, 0, np.inf)
