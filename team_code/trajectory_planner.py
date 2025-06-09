import carla
import numpy as np
import traceback
import transfuser_utils as t_u

from scipy.integrate import RK45
from scene_interpreter import *
from privileged_route_planner import PrivilegedRoutePlanner

from agent_utils import AgentPrediction

class TrajectoryPlanner:
    def __init__(self, config):
        self.config = config
        self.agent_prediction = AgentPrediction(config)
        # self.command_mapping = {
        #     "longitudinal": {
        #         "maintain_speed": self._maintain_speed,
        #         # "stop": self._stop,
        #         "accelerate": self._accelerate,
        #         "decelerate": self._decelerate,
        #     },
        #     # "lateral": {
        #     #     "turn_left": self._turn_left,
        #     #     "turn_right": self._turn_right,
        #     #     "change_lane_left": self._change_lane_left,
        #     #     "change_lane_right": self._change_lane_right,
        #     # },
        # }
        self.command_mapping = {
            "maintain_speed": self._maintain_speed,
            # "stop": self._stop,
            "accelerate": self._accelerate,
            "decelerate": self._decelerate,
            "change_lane_left": self._change_lane,
            "change_lane_right": self._change_lane,
        }
        self.all_actors = {
            "vehicle": {},
            "pedestrian": {},
            "cyclist": {},
            "traffic_light": {},
            "stop_sign": {},
        }
        self.target_speed = 0.0

        # Dummy waypoint planner
        self._waypoint_planner = PrivilegedRoutePlanner(self.config)

    def setup_agent_prediction(self, traffic_manager, world_map, global_route_planner, ego_vehicle):
        self.agent_prediction.setup(traffic_manager, world_map, global_route_planner, ego_vehicle)

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

    def update_state(self, vehicle_context, traffic_context, ego_context, waypoint_planner):
        self.agent_prediction.update_state(vehicle_context, ego_context)

        self._waypoint_planner = waypoint_planner
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


    def rollout_trajectory(self, command, params, key_actors, structured_data):
        forecast_length = self.config.default_forecast_length
        ego_context = self.ego_context
        vehicle_context = self.vehicle_context
        command_mapping = self.command_mapping
        near_lane_change = False


        if command in 'change_lane_left':
            forecast_length = self.config.forecast_length_lane_change
            near_lane_change = True

        num_future_frames = int(self.config.bicycle_frame_rate * forecast_length)

        # Run the command
        target_speed, route_points, route_waypoints = command_mapping[command](params, key_actors)

        ego_context["route"] = route_waypoints
        ego_context["route_points"] = route_points

        # Rollout the trajectory
        npc_vehicles = vehicle_context['npc_vehicles']
        npc_vehicles_predicted_paths, npc_vehicles_dict = self.agent_prediction.predict_npc_vehicle_waypoints(ego_context, npc_vehicles)
        npc_vehicles_predicted_bounding_boxes = self.agent_prediction.forecast_npc_vehicle_bounding_boxes(npc_vehicles_dict, npc_vehicles_predicted_paths, num_future_frames)

        ego_predicted_bounding_boxes = self.agent_prediction.forecast_ego_vehicle_bounding_boxes(ego_context, target_speed, num_future_frames)

        world = ego_context['ego_actor'].get_world()
        # for actor_idx, actors_forecasted_bounding_boxes in npc_vehicles_predicted_bounding_boxes.items():
        #     for bb in actors_forecasted_bounding_boxes:
        #         world.debug.draw_box(box=bb,
        #                                 rotation=bb.rotation,
        #                                 thickness=0.1,
        #                                 color=self.config.other_vehicles_forecasted_bbs_color,
        #                                 life_time=self.config.draw_life_time)

        # for bb in ego_predicted_bounding_boxes:
        #     world.debug.draw_box(box=bb,
        #                             rotation=bb.rotation,
        #                             thickness=0.1,
        #                             color=self.config.ego_vehicle_forecasted_bbs_normal_color)

        npc_vehicle_collisions = self.agent_prediction.check_ego_collision_vehicles(near_lane_change=near_lane_change,
                                                                                    ego_predicted_bounding_boxes=ego_predicted_bounding_boxes,
                                                                                    npc_vehicles_predicted_bounding_boxes_dict=npc_vehicles_predicted_bounding_boxes,
                                                                                    npc_vehicles_lanes=structured_data['agent'])

        if len(npc_vehicle_collisions) > 0:
            for vehicle_id, collision_data in npc_vehicle_collisions.items():
                ego_bounding_box = collision_data['ego_bounding_box']
                npc_vehicle_bounding_box = collision_data['npc_vehicle_bounding_box']

                world.debug.draw_box(box=ego_bounding_box,
                                    rotation=ego_bounding_box.rotation,
                                    thickness=0.1,
                                    color=self.config.ego_vehicle_forecasted_bbs_hazard_color,
                                    life_time=self.config.draw_life_time)
                world.debug.draw_box(box=npc_vehicle_bounding_box,
                                    rotation=npc_vehicle_bounding_box.rotation,
                                    thickness=0.1,
                                    color=self.config.leading_vehicle_color,
                                    life_time=self.config.draw_life_time)
        print(f'NPC Vehicle Collisions: {npc_vehicle_collisions}')

    def run_command(self, high_level_command: HighLevelCommand, structured_data):
        # TODO: Currently returns None, fallback to default rule-based IDM behaviour
        if not high_level_command:
            return None

        # TODO: Only longitudinal commands are supported for now, add lateral ones
        # command_type = "longitudinal"
        command = high_level_command.command.value
        params = high_level_command.params
        key_actors = high_level_command.key_actors
        reasoning = high_level_command.reasoning

        # command_mapping = self.command_mapping[command_type]
        command_mapping = self.command_mapping
        try:
            print(f'Executing command: {command} with params: {params} and key actors: {key_actors} with reasoning: {reasoning}')
            self.rollout_trajectory(command, params, key_actors, structured_data)
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

    # def _accelerate(self, params: LongitudinalCommandParams, key_actors_llm: list[KeyActor]):
    #     key_actors = self._get_key_actors(key_actors_llm)
    #     target_speed_initial = params.target_speed \
    #         if params.target_speed \
    #         else self.ego_context['speed']

    #     print(f'Ego Speed: {self.ego_context["speed"]}\n')
    #     print(f'Target Speed Initial: {target_speed_initial}\n')

    #     target_speeds = [target_speed_initial]
    #     if key_actors:
    #         for key_actor in key_actors:
    #             actor_type, actor = key_actor
    #             if actor_type == "vehicle":
    #                 leading_actor_speed = actor.get_velocity().length()
    #                 leading_actor_length = actor.bounding_box.extent.x * 2

    #                 ego_location = self.ego_context['location']
    #                 distance_to_leading_actor = ego_location.distance(actor.get_location())

    #                 desired_following_distance = params.desired_following_distance \
    #                     if params.desired_following_distance \
    #                     else self.config.idm_leading_vehicle_minimum_distance

    #                 target_speed = self._compute_target_speed_idm(
    #                     desired_speed = target_speed_initial,
    #                     leading_actor_length = leading_actor_length,
    #                     ego_speed = self.ego_context['speed'],
    #                     leading_actor_speed = leading_actor_speed,
    #                     distance_to_leading_actor = distance_to_leading_actor,
    #                     s0 = desired_following_distance
    #                 )

    #                 print(f'Leading Vehicle Speed: {target_speed}\n')
    #             elif actor_type == "traffic_light":
    #                 if actor.state == carla.TrafficLightState.Red:
    #                     leading_actor_speed = 0.0
    #                     leading_actor_length = 0.0

    #                     ego_location = self.ego_context['location']
    #                     ego_speed = self.ego_context['speed']
    #                     distance_to_leading_actor = self.traffic_context['distance_to_next_traffic_light']

    #                     desired_following_distance = params.desired_following_distance \
    #                         if params.desired_following_distance \
    #                         else self.config.idm_red_light_minimum_distance

    #                     desired_time_headway = self.config.idm_red_light_desired_time_headway

    #                     target_speed = self._compute_target_speed_idm(
    #                         desired_speed = target_speed_initial,
    #                         leading_actor_length = leading_actor_length,
    #                         ego_speed = ego_speed,
    #                         leading_actor_speed = leading_actor_speed,
    #                         distance_to_leading_actor = distance_to_leading_actor,
    #                         s0 = desired_following_distance,
    #                         T=desired_time_headway
    #                     )
    #                 else:
    #                     target_speed = target_speed_initial
    #                 print(f'Traffic Light Speed: {target_speed}\n')

    #             target_speeds.append(target_speed)

    #     self.target_speed = min(target_speeds)
    #     start_index = self.ego_context['route_index']
    #     return self.target_speed, self._waypoint_planner.route_points[start_index:], self._waypoint_planner.route_waypoints[start_index:]

    def _get_traffic_light_speed(self, target_speed_initial):
        ego_speed = self.ego_context['speed']
        next_traffic_light = self.traffic_context['next_traffic_light']

        target_speed = target_speed_initial
        if next_traffic_light:
            # Check if the traffic light is red and within a certain distance
            distance_to_traffic_light = self.traffic_context['distance_to_next_traffic_light']
            if distance_to_traffic_light <= self.config.light_radius and \
               next_traffic_light.state == carla.TrafficLightState.Red:
                leading_actor_speed = 0.0
                leading_actor_length = 0.0

                distance_to_leading_actor = distance_to_traffic_light
                desired_following_distance = self.config.idm_red_light_minimum_distance
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

        return target_speed

    def _get_target_speed(self, target_speed_initial, params: LongitudinalCommandParams):
        ego_location = self.ego_context['location']
        ego_speed = self.ego_context['speed']
        ego_wp = self.ego_context['waypoint']
        ego_lane_id = ego_wp.lane_id

        print(f'Ego Speed: {ego_speed}\n')
        print(f'Target Speed Initial: {target_speed_initial}\n')

        target_speeds = [target_speed_initial]
        ongoing_leading_vehicles = self.vehicle_context['ongoing_leading_vehicles']

        # Check for leading vehicles in the ego lane
        if ego_lane_id in ongoing_leading_vehicles:
            ego_lane_leading_vehicles_list = ongoing_leading_vehicles[ego_lane_id]
            for leading_vehicle in ego_lane_leading_vehicles_list:
                leading_actor_speed = leading_vehicle.get_velocity().length()
                leading_actor_length = leading_vehicle.bounding_box.extent.x * 2

                distance_to_leading_actor = ego_location.distance(leading_vehicle.get_location())

                desired_following_distance = self.config.idm_leading_vehicle_minimum_distance
                desired_time_headway = self.config.idm_leading_vehicle_time_headway

                target_speed_vehicle = self._compute_target_speed_idm(
                    desired_speed = target_speed_initial,
                    leading_actor_length = leading_actor_length,
                    ego_speed = ego_speed,
                    leading_actor_speed = leading_actor_speed,
                    distance_to_leading_actor = distance_to_leading_actor,
                    s0 = desired_following_distance,
                    T=desired_time_headway
                )
                print(f'Leading Vehicle ID: {leading_vehicle.id}, IDM Speed: {target_speed_vehicle}\n')
                target_speeds.append(target_speed_vehicle)

        # Check for traffic lights
        target_speed_traffic_light = self._get_traffic_light_speed(target_speed_initial)
        print(f'Traffic Light Speed: {target_speed_traffic_light}\n')
        target_speeds.append(target_speed_traffic_light)

        return min(target_speeds)

    def _accelerate(self, params: LongitudinalCommandParams, key_actors_llm: list[KeyActor]):
        key_actors = self._get_key_actors(key_actors_llm)
        target_speed_initial = params.target_speed \
            if params.target_speed \
            else min(self.ego_context['speed'] * 1.1, self.traffic_context['speed_limit'])

        self.target_speed = self._get_target_speed(target_speed_initial, params)
        start_index = self.ego_context['route_index']
        return self.target_speed, self._waypoint_planner.route_points[start_index:], self._waypoint_planner.route_waypoints[start_index:]

    def _decelerate(self, params: LongitudinalCommandParams, key_actors_llm: list[KeyActor]):
        key_actors = self._get_key_actors(key_actors_llm)
        target_speed_initial = params.target_speed \
            if params.target_speed \
            else max(0.0, min(self.ego_context['speed'] * 0.9, 0.25))

        self.target_speed = self._get_target_speed(target_speed_initial, params)
        start_index = self.ego_context['route_index']
        return self.target_speed, self._waypoint_planner.route_points[start_index:], self._waypoint_planner.route_waypoints[start_index:]

    def _maintain_speed(self, params: LongitudinalCommandParams, key_actors_llm: list[KeyActor]):
        key_actors = self._get_key_actors(key_actors_llm)
        target_speed_initial = self.ego_context['speed']

        self.target_speed = self._get_target_speed(target_speed_initial, params)
        start_index = self.ego_context['route_index']
        return self.target_speed, self._waypoint_planner.route_points[start_index:], self._waypoint_planner.route_waypoints[start_index:]

    # def _maintain_speed(self, params: LongitudinalCommandParams, key_actors_llm: list[KeyActor]):
    #     key_actors = self._get_key_actors(key_actors_llm)
    #     target_speed_initial = self.ego_context['speed']

    #     target_speeds = [target_speed_initial]
    #     if key_actors:
    #         for key_actor in key_actors:
    #             actor_type, actor = key_actor
    #             if actor_type == "vehicle":
    #                 leading_actor_speed = actor.get_velocity().length()
    #                 leading_actor_length = actor.bounding_box.extent.x * 2

    #                 ego_location = self.ego_context['location']
    #                 distance_to_leading_actor = ego_location.distance(actor.get_location())

    #                 desired_following_distance = params.desired_following_distance \
    #                     if params.desired_following_distance \
    #                     else self.config.idm_leading_vehicle_minimum_distance

    #                 target_speed = self._compute_target_speed_idm(
    #                     desired_speed = target_speed_initial,
    #                     leading_actor_length = leading_actor_length,
    #                     ego_speed = self.ego_context['speed'],
    #                     leading_actor_speed = leading_actor_speed,
    #                     distance_to_leading_actor = distance_to_leading_actor,
    #                     s0 = desired_following_distance
    #                 )
    #             elif actor_type == "traffic_light":
    #                 leading_actor_speed = 0.0
    #                 leading_actor_length = 0.0

    #                 ego_location = self.ego_context['location']
    #                 ego_speed = self.ego_context['speed']
    #                 distance_to_leading_actor = self.traffic_context['distance_to_next_traffic_light']

    #                 desired_following_distance = params.desired_following_distance \
    #                     if params.desired_following_distance \
    #                     else self.config.idm_red_light_minimum_distance

    #                 desired_time_headway = self.config.idm_red_light_desired_time_headway

    #                 target_speed = self._compute_target_speed_idm(
    #                     desired_speed = target_speed_initial,
    #                     leading_actor_length = leading_actor_length,
    #                     ego_speed = ego_speed,
    #                     leading_actor_speed = leading_actor_speed,
    #                     distance_to_leading_actor = distance_to_leading_actor,
    #                     s0 = desired_following_distance,
    #                     T=desired_time_headway
    #                 )

    #             target_speeds.append(target_speed)

    #     self.target_speed = min(target_speeds)
    #     start_index = self.ego_context['route_index']
    #     return self.target_speed, self._waypoint_planner.route_points[start_index:], self._waypoint_planner.route_waypoints[start_index:]

    # def _decelerate(self, params: LongitudinalCommandParams, key_actors_llm: list[KeyActor]):
    #     key_actors = self._get_key_actors(key_actors_llm)
    #     target_speed_initial = params.target_speed \
    #         if params.target_speed \
    #         else self.ego_context['speed']

    #     target_speeds = [target_speed_initial]
    #     if key_actors:
    #         for key_actor in key_actors:
    #             actor_type, actor = key_actor
    #             if actor_type == "vehicle":
    #                 leading_actor_speed = actor.get_velocity().length()
    #                 leading_actor_length = actor.bounding_box.extent.x * 2

    #                 ego_location = self.ego_context['location']
    #                 ego_speed = self.ego_context['speed']
    #                 distance_to_leading_actor = ego_location.distance(actor.get_location())

    #                 desired_following_distance = params.desired_following_distance \
    #                     if params.desired_following_distance \
    #                     else self.config.idm_leading_vehicle_minimum_distance

    #                 desired_time_headway = self.config.idm_leading_vehicle_time_headway

    #                 target_speed = self._compute_target_speed_idm(
    #                     desired_speed = target_speed_initial,
    #                     leading_actor_length = leading_actor_length,
    #                     ego_speed = ego_speed,
    #                     leading_actor_speed = leading_actor_speed,
    #                     distance_to_leading_actor = distance_to_leading_actor,
    #                     s0 = desired_following_distance,
    #                     T=desired_time_headway
    #                 )
    #             elif actor_type == "traffic_light":
    #                 leading_actor_speed = 0.0
    #                 leading_actor_length = 0.0

    #                 ego_location = self.ego_context['location']
    #                 ego_speed = self.ego_context['speed']
    #                 distance_to_leading_actor = self.traffic_context['distance_to_next_traffic_light']

    #                 desired_following_distance = params.desired_following_distance \
    #                     if params.desired_following_distance \
    #                     else self.config.idm_red_light_minimum_distance

    #                 desired_time_headway = self.config.idm_red_light_desired_time_headway

    #                 target_speed = self._compute_target_speed_idm(
    #                     desired_speed = target_speed_initial,
    #                     leading_actor_length = leading_actor_length,
    #                     ego_speed = ego_speed,
    #                     leading_actor_speed = leading_actor_speed,
    #                     distance_to_leading_actor = distance_to_leading_actor,
    #                     s0 = desired_following_distance,
    #                     T=desired_time_headway
    #                 )

    #             target_speeds.append(target_speed)

    #     self.target_speed = min(target_speeds)
    #     start_index = self.ego_context['route_index']
    #     return self.target_speed, self._waypoint_planner.route_points[start_index:], self._waypoint_planner.route_waypoints[start_index:]

    def _change_lane(self, params: LongitudinalCommandParams, key_actors_llm: list[KeyActor], shift_to_left_lane = True):
        target_speed_initial = params.target_speed \
            if params.target_speed \
            else self.traffic_context['speed_limit']
        target_speed = target_speed_initial

        lane_change = self.ego_context['lane_change']
        lane_change_direction = lane_change['lane_change_direction']

        # Implement sanity check for no lane change
        if lane_change is None:
            pass

        from_index = self.ego_context['route_index']
        start_wp = self.ego_context['route'][0]

        lane_change_late_start_wp = lane_change['lane_change_late_start_point']
        lane_change_end_wp = lane_change['lane_change_end_point']
        lane_change_end_point_loc = lane_change_end_wp.transform.location

        lane_change_index = from_index
        to_index = from_index
        while to_index < len(self._waypoint_planner.route_waypoints) and \
          lane_change_end_wp != self._waypoint_planner.route_waypoints[to_index]:
            if lane_change_late_start_wp == self._waypoint_planner.route_waypoints[to_index]:
                lane_change_index = to_index
            to_index += 1

        available_lane_change_distance = lane_change_end_point_loc.distance(start_wp.transform.location)
        # transition_length = self.config.transition_smoothness_distance
        transition_length = max(self.config.transition_smoothness_distance, available_lane_change_distance * self.config.points_per_meter)
        lane_transition_factor = max(1.0, np.abs(lane_change_late_start_wp.lane_id - lane_change_end_wp.lane_id))

        print(f'From Index: {from_index}, Lane Change Index: {lane_change_index}, To Index: {to_index}')
        self._waypoint_planner.change_lane(from_index, lane_change_index, to_index, lane_change_direction, transition_length, lane_transition_factor)

        ongoing_leading_vehicles = self.vehicle_context['ongoing_leading_vehicles']
        target_lane_id = lane_change_end_wp.lane_id

        if target_lane_id in ongoing_leading_vehicles:
            lane_leading_vehicles_list = ongoing_leading_vehicles[target_lane_id]
            for leading_vehicle in lane_leading_vehicles_list:
                leading_actor_speed = leading_vehicle.get_velocity().length()
                leading_actor_length = leading_vehicle.bounding_box.extent.x * 2

                ego_location = self.ego_context['location']
                distance_to_leading_actor = ego_location.distance(leading_vehicle.get_location())

                desired_following_distance = self.config.idm_leading_vehicle_minimum_distance / 2.0
                desired_time_headway = self.config.idm_leading_vehicle_time_headway / 2.0

                target_speed = min(target_speed, self._compute_target_speed_idm(
                    desired_speed = target_speed_initial,
                    leading_actor_length = leading_actor_length,
                    ego_speed = self.ego_context['speed'],
                    leading_actor_speed = leading_actor_speed,
                    distance_to_leading_actor = distance_to_leading_actor,
                    s0 = desired_following_distance,
                    T = desired_time_headway
                ))

        self.target_speed = target_speed
        return self.target_speed, self._waypoint_planner.route_points[from_index:], self._waypoint_planner.route_waypoints[from_index:]