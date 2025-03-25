import carla
import numpy as np
import traceback

from scipy.integrate import RK45
from scene_interpreter import *

class TrajectoryPlanner:
    def __init__(self, config):
        self.config = config
        self.command_mapping = {
            "longitudinal": {
                "maintain_speed": self._maintain_speed,
                # "stop": self._stop,
                "accelerate": self._accelerate,
                "decelerate": self._decelerate,
            },
            # "lateral": {
            #     "turn_left": self._turn_left,
            #     "turn_right": self._turn_right,
            #     "change_lane_left": self._change_lane_left,
            #     "change_lane_right": self._change_lane_right,
            # },
        }
        self.all_actors = {
            "vehicle": {},
            "pedestrian": {},
            "cyclist": {},
            "traffic_light": {},
            "stop_sign": {},
        }
        self.target_speed = 0.0

    def _compute_target_speed_idm(
            self,
            desired_speed,
            leading_actor_length,
            ego_speed,
            leading_actor_speed,
            distance_to_leading_actor,
            s0=4.,
            T=0.5):
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

    """
    Interface for translating high-level text commands into low-level outputs.
    """

    def update_state(self, vehicle_context, traffic_context, ego_context):
        self.vehicle_context = vehicle_context
        self.traffic_context = traffic_context
        self.ego_context = ego_context

        # Extract NPC vehicles
        npc_vehicles_list = vehicle_context['npc_vehicles']
        npc_vehicles_dict = {vehicle.id: vehicle for vehicle in npc_vehicles_list}
        self.all_actors['vehicle'] = npc_vehicles_dict

        # Extract traffic lights
        next_traffic_light = traffic_context['next_traffic_light']
        if next_traffic_light:
            self.all_actors['traffic_light'] = {next_traffic_light.id: next_traffic_light}


    def run_command(self, high_level_command: HighLevelCommand):
        # TODO: Currently returns None, fallback to default rule-based IDM behaviour
        if not high_level_command:
            return None

        # TODO: Only longitudinal commands are supported for now, add lateral ones
        command_type = "longitudinal"
        command = high_level_command.command.value
        params = high_level_command.params
        key_actors = high_level_command.key_actors
        reasoning = high_level_command.reasoning

        command_mapping = self.command_mapping[command_type]
        try:
            print(f'Executing command: {command} with params: {params} and key actors: {key_actors} with reasoning: {reasoning}')
            return command_mapping[command](params, key_actors)
        except Exception as e:
            print(f'Exception while executing command "{command}": {e}')
            traceback.print_exc()
            return None

    def _get_key_actors(self, key_actors):
        extracted_actors = []
        for key_actor in key_actors:
            actor_type = key_actor.actor_type.value
            actor_id = key_actor.actor_id
            try:
                actor = self.all_actors[actor_type][actor_id]
                extracted_actors.append((actor_type, actor))
            except KeyError:
                print(f'A key actor with ID {actor_id} of type {actor_type} was not found.')
                continue

        return extracted_actors

    def _accelerate(self, params: LongitudinalCommandParams, key_actors_llm: list[KeyActor]):
        key_actors = self._get_key_actors(key_actors_llm)
        target_speed_initial = params.target_speed \
            if params.target_speed \
            else self.ego_context['speed']

        print(f'Ego Speed: {self.ego_context["speed"]}\n')
        print(f'Target Speed Initial: {target_speed_initial}\n')

        target_speeds = [target_speed_initial]
        if key_actors:
            for key_actor in key_actors:
                actor_type, actor = key_actor
                if actor_type == "vehicle":
                    leading_actor_speed = actor.get_velocity().length()
                    leading_actor_length = actor.bounding_box.extent.x * 2

                    ego_location = self.ego_context['location']
                    distance_to_leading_actor = ego_location.distance(actor.get_location())

                    desired_following_distance = params.desired_following_distance \
                        if params.desired_following_distance \
                        else self.config.idm_leading_vehicle_minimum_distance

                    target_speed = self._compute_target_speed_idm(
                        desired_speed = target_speed_initial,
                        leading_actor_length = leading_actor_length,
                        ego_speed = self.ego_context['speed'],
                        leading_actor_speed = leading_actor_speed,
                        distance_to_leading_actor = distance_to_leading_actor,
                        s0 = desired_following_distance
                    )

                    print(f'Leading Vehicle Speed: {target_speed}\n')
                elif actor_type == "traffic_light":
                    if actor.state == carla.TrafficLightState.Red:
                        leading_actor_speed = 0.0
                        leading_actor_length = 0.0

                        ego_location = self.ego_context['location']
                        ego_speed = self.ego_context['speed']
                        distance_to_leading_actor = self.traffic_context['distance_to_next_traffic_light']

                        desired_following_distance = params.desired_following_distance \
                            if params.desired_following_distance \
                            else self.config.idm_red_light_minimum_distance

                        desired_time_headway = self.config.idm_red_light_desired_time_headway

                        target_speed = self._compute_target_speed_idm(
                            desired_speed = target_speed_initial,
                            leading_actor_length = leading_actor_length,
                            ego_speed = ego_speed,
                            leading_actor_speed = leading_actor_speed,
                            distance_to_leading_actor = distance_to_leading_actor,
                            s0 = desired_following_distance,
                            T=desired_time_headway
                        )
                    else:
                        target_speed = target_speed_initial
                    print(f'Traffic Light Speed: {target_speed}\n')

                target_speeds.append(target_speed)

        self.target_speed = min(target_speeds)
        return self.target_speed


    def _maintain_speed(self, params: LongitudinalCommandParams, key_actors_llm: list[KeyActor]):
        key_actors = self._get_key_actors(key_actors_llm)
        target_speed_initial = self.ego_context['speed']

        target_speeds = [target_speed_initial]
        if key_actors:
            for key_actor in key_actors:
                actor_type, actor = key_actor
                if actor_type == "vehicle":
                    leading_actor_speed = actor.get_velocity().length()
                    leading_actor_length = actor.bounding_box.extent.x * 2

                    ego_location = self.ego_context['location']
                    distance_to_leading_actor = ego_location.distance(actor.get_location())

                    desired_following_distance = params.desired_following_distance \
                        if params.desired_following_distance \
                        else self.config.idm_leading_vehicle_minimum_distance

                    target_speed = self._compute_target_speed_idm(
                        desired_speed = target_speed_initial,
                        leading_actor_length = leading_actor_length,
                        ego_speed = self.ego_context['speed'],
                        leading_actor_speed = leading_actor_speed,
                        distance_to_leading_actor = distance_to_leading_actor,
                        s0 = desired_following_distance
                    )
                elif actor_type == "traffic_light":
                    leading_actor_speed = 0.0
                    leading_actor_length = 0.0

                    ego_location = self.ego_context['location']
                    ego_speed = self.ego_context['speed']
                    distance_to_leading_actor = self.traffic_context['distance_to_next_traffic_light']

                    desired_following_distance = params.desired_following_distance \
                        if params.desired_following_distance \
                        else self.config.idm_red_light_minimum_distance

                    desired_time_headway = self.config.idm_red_light_desired_time_headway

                    target_speed = self._compute_target_speed_idm(
                        desired_speed = target_speed_initial,
                        leading_actor_length = leading_actor_length,
                        ego_speed = ego_speed,
                        leading_actor_speed = leading_actor_speed,
                        distance_to_leading_actor = distance_to_leading_actor,
                        s0 = desired_following_distance,
                        T=desired_time_headway
                    )

                target_speeds.append(target_speed)

        self.target_speed = min(target_speeds)
        return self.target_speed


    def _decelerate(self, params: LongitudinalCommandParams, key_actors_llm: list[KeyActor]):
        key_actors = self._get_key_actors(key_actors_llm)
        target_speed_initial = params.target_speed \
            if params.target_speed \
            else self.ego_context['speed']

        target_speeds = [target_speed_initial]
        if key_actors:
            for key_actor in key_actors:
                actor_type, actor = key_actor
                if actor_type == "vehicle":
                    leading_actor_speed = actor.get_velocity().length()
                    leading_actor_length = actor.bounding_box.extent.x * 2

                    ego_location = self.ego_context['location']
                    ego_speed = self.ego_context['speed']
                    distance_to_leading_actor = ego_location.distance(actor.get_location())

                    desired_following_distance = params.desired_following_distance \
                        if params.desired_following_distance \
                        else self.config.idm_leading_vehicle_minimum_distance

                    desired_time_headway = self.config.idm_leading_vehicle_time_headway

                    target_speed = self._compute_target_speed_idm(
                        desired_speed = target_speed_initial,
                        leading_actor_length = leading_actor_length,
                        ego_speed = ego_speed,
                        leading_actor_speed = leading_actor_speed,
                        distance_to_leading_actor = distance_to_leading_actor,
                        s0 = desired_following_distance,
                        T=desired_time_headway
                    )
                elif actor_type == "traffic_light":
                    leading_actor_speed = 0.0
                    leading_actor_length = 0.0

                    ego_location = self.ego_context['location']
                    ego_speed = self.ego_context['speed']
                    distance_to_leading_actor = self.traffic_context['distance_to_next_traffic_light']

                    desired_following_distance = params.desired_following_distance \
                        if params.desired_following_distance \
                        else self.config.idm_red_light_minimum_distance

                    desired_time_headway = self.config.idm_red_light_desired_time_headway

                    target_speed = self._compute_target_speed_idm(
                        desired_speed = target_speed_initial,
                        leading_actor_length = leading_actor_length,
                        ego_speed = ego_speed,
                        leading_actor_speed = leading_actor_speed,
                        distance_to_leading_actor = distance_to_leading_actor,
                        s0 = desired_following_distance,
                        T=desired_time_headway
                    )

                target_speeds.append(target_speed)

        self.target_speed = min(target_speeds)
        return self.target_speed

