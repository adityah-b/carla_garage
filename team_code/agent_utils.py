import numpy as np
import transfuser_utils as t_u

class AgentPrediction:
    def __init__(self, config):
        self.config = config

    def setup(self, traffic_manager, world_map, global_route_planner):
        self._traffic_manager = traffic_manager
        self._world_map = world_map
        self._global_route_planner = global_route_planner

    def _group_npc_vehicles_by_road_and_lane(self, vehicles_list):
        """
        Group NPC vehicles by road and lane.

        Args:
            vehicles_list (list): List of NPC vehicles.

        Returns:
            dict: A dictionary with road IDs as keys and dictionaries as values,
                  where each dictionary has lane IDs as keys and lists of NPC vehicles in each lane as values.
        """
        vehicles_by_road_and_lane = {}
        for vehicle in vehicles_list:
            # Locate the NPC vehicle in the map
            vehicle_loc = vehicle.get_location()
            vehicle_wp = self._world_map.get_waypoint(vehicle_loc)

            if not vehicle_wp:
                print(f'Could not find waypoint for vehicle {vehicle.id}')
                continue

            # Get the road and lane ID of the NPC vehicle
            road_id = vehicle_wp.road_id
            lane_id = vehicle_wp.lane_id

            if road_id not in vehicles_by_road_and_lane:
                vehicles_by_road_and_lane[road_id] = {}
            if lane_id not in vehicles_by_road_and_lane[road_id]:
                vehicles_by_road_and_lane[road_id][lane_id] = []

            vehicles_by_road_and_lane[road_id][lane_id].append(vehicle)

        return vehicles_by_road_and_lane

    def _find_waypoint_entry_exit(self, waypoint):
        """
        Find the entry and exit points of the waypoint's corresponding road segment.
        Args:
            waypoint (carla.Waypoint): The waypoint.

        Returns:
            tuple: A tuple containing the entry and exit points of the road segment.
        """
        entry_xyz, exit_xyz = None, None
        try:
            entry_xyz, exit_xyz = self._global_route_planner._road_id_to_edge[waypoint.road_id][waypoint.section_id][waypoint.lane_id]
        except KeyError:
            print(f'Could not find edge for waypoint {waypoint.transform.location}')
            pass
        return (entry_xyz, exit_xyz)

    def _is_connected_edge(self, entry_exit_pair, next_entry_exit_pair):
        """
        Check if two edges are connected.

        Args:
            entry_exit_pair (tuple): The entry and exit points of the first edge.
            next_entry_exit_pair (tuple): The entry and exit points of the second edge.

        Returns:
            bool: True if the edges are connected, False otherwise.
        """
        edge = self._global_route_planner._graph.edges[entry_exit_pair[0], entry_exit_pair[1]]
        next_edge = self._global_route_planner._graph.edges[next_entry_exit_pair[0], next_entry_exit_pair[1]]

        exit_wp = edge['exit_waypoint']
        next_entry_wp = next_edge['entry_waypoint']

        return exit_wp.road_id == next_entry_wp.road_id and \
               exit_wp.section_id == next_entry_wp.section_id

    def _group_waypoints_by_road_ids(self, waypoints):
        """
        Prune a list of waypoints to only contain unique road IDs.

        Args:
            waypoints (list): List of waypoints.

        Returns:
            pruned_waypoints (list): List of waypoints containing unique road IDs.
        """
        if not waypoints:
            return []

        pruned_waypoints = [waypoints[0]]
        for waypoint in waypoints:
            if waypoint.road_id != pruned_waypoints[-1].road_id:
                pruned_waypoints.append(waypoint)

        return pruned_waypoints

    def _get_next_ego_waypoints(self, ego_route, ego_speed):
        """
        Get the next waypoints of the ego vehicle.

        Args:
            ego_route (list): List of waypoints of the ego vehicle.
            ego_speed (float): Speed of the ego vehicle.

        Returns:
            list: List of waypoints of the ego vehicle up to the specified distance.
        """
        distance = ego_speed * self.config.prediction_horizon
        traveled_distance = 0.0
        cur_index = 0
        cur_position = ego_route[cur_index].transform.location

        while traveled_distance < distance and cur_index < len(ego_route):
            traveled_distance = cur_position.distance(ego_route[cur_index + 1].transform.location)
            cur_index += 1

        return ego_route[:cur_index]

    def _get_ego_route_entry_exit_pairs(self, ego_waypoints):
        """
        Get the entry and exit points of the road segments of the ego vehicle's route.

        Args:
            ego_waypoints (list): List of waypoints of the ego vehicle.

        Returns:
            list: List of tuples containing the entry and exit points of the road segments
        """
        ego_entry_exit_pairs = []
        pruned_ego_waypoints = self._group_waypoints_by_road_ids(ego_waypoints)
        for wp in pruned_ego_waypoints:
            ego_entry_exit_pairs.append(self._find_waypoint_entry_exit(wp))

        return ego_entry_exit_pairs

    def _choose_at_junction(self, waypoint_choices, ego_entry_exit_pairs):
        """
        Choose the next waypoint at a junction.

        Args:
            waypoint_choices (list): List of waypoint choices at a junction.
            ego_entry_exit_pairs (list): List of entry and exit points of the ego vehicle's route.

        Returns:
            carla.Waypoint: The chosen waypoint.
        """
        for wp in waypoint_choices:
            entry_exit_pair = self._find_waypoint_entry_exit(wp)
            for ego_entry_exit_pair in ego_entry_exit_pairs:
                if self._is_connected_edge(entry_exit_pair, ego_entry_exit_pair):
                    return wp

        return waypoint_choices[np.random.randint(len(waypoint_choices))] # Randomly choose a waypoint


    def _get_waypoint_list_at_distance(self, waypoint, distance, ego_entry_exit_pairs):
        """
        Get a list of waypoints at a certain distance from the input waypoint.

        Args:
            waypoint (carla.Waypoint): The input waypoint.
            distance (float): The distance to travel.
            ego_entry_exit_pairs (list): List of entry and exit points of the ego vehicle's route.

        Returns:
            list: List of waypoints at the specified distance.
        """
        print(f'Getting waypoints at distance {distance} from {waypoint.transform.location}')
        traveled_distance = 0.0
        plan = []

        cur_wp = waypoint
        while traveled_distance < distance:
            wp_choice = cur_wp.next(self.config.sampling_resolution)
            if len(wp_choice) > 1:
                print(f'Junction detected at waypoint {cur_wp.transform.location}')
                cur_wp = self._choose_at_junction(wp_choice, ego_entry_exit_pairs)
            else:
                cur_wp = wp_choice[0]

            plan.append(cur_wp)
            traveled_distance += cur_wp.transform.location.distance(waypoint.transform.location)

        return plan

    def run_step(self, ego_data, npc_vehicles_list):
        """
        Run one step of the agent prediction.

        Args:
            ego_data (dict): Dictionary containing the ego vehicle's data.
            npc_vehicles_list (list): List of NPC vehicles.

        Returns:
            dict: A dictionary containing the predicted positions of NPC vehicles.
        """
        ego_route = ego_data['route']
        ego_speed = ego_data['speed']

        print(f'Getting next waypoints for ego vehicle with speed {ego_speed}')
        ego_waypoints = self._get_next_ego_waypoints(ego_route, ego_speed)

        print(f'Getting entry and exit pairs for ego vehicle')
        ego_entry_exit_pairs = self._get_ego_route_entry_exit_pairs(ego_waypoints)

        predicted_positions = {}

        # print(f'Grouping NPC vehicles by road and lane')
        # npc_vehicles_by_road_and_lane = self._group_npc_vehicles_by_road_and_lane(npc_vehicles_list)
        # predicted_positions = {}

        # print(f'Grouped NPC vehicles: {npc_vehicles_by_road_and_lane}')
        # lanes = npc_vehicles_by_road_and_lane.values()
        # for lane_id, vehicles in lanes.items():

        for vehicle in npc_vehicles_list:
            vehicle_plan = []
            vehicle_loc = vehicle.get_location()

            # Try to get the next actions of the NPC vehicle from TrafficManager
            next_vehicle_actions = None
            vehicle_wp = None
            try:
                next_vehicle_actions = self._traffic_manager.get_all_actions(vehicle)
            except Exception as e:
                print(f'Error getting next actions for vehicle {vehicle.id}: {e}')
                print(f'Using current waypoint instead')
                vehicle_wp = self._world_map.get_waypoint(vehicle_loc)
                if not vehicle_wp:
                    print(f'Could not find waypoint for vehicle {vehicle.id}')
                    continue

            # Vehicle not controlled by TrafficManager (either static vehicle or scenario vehicle)
            if not next_vehicle_actions:
                # Check if scenario vehicle
                # TODO: Implement scenario vehicle check
                pass

            elif len(next_vehicle_actions) > 1:
                print(f'Multiple actions for vehicle {vehicle.id}')
                next_vehicle_wps = [action[1] for action in next_vehicle_actions]
                vehicle_plan.extend(next_vehicle_wps)
                vehicle_wp = next_vehicle_wps[-1]

            else:
                vehicle_wp = next_vehicle_actions[0][1]

            cur_speed = np.sqrt(vehicle.get_velocity().x**2 + vehicle.get_velocity().y**2)
            distance = cur_speed * self.config.prediction_horizon
            vehicle_plan = self._get_waypoint_list_at_distance(vehicle_wp, distance, ego_entry_exit_pairs)

            if vehicle.id not in predicted_positions:
                predicted_positions[vehicle.id] = []
            predicted_positions[vehicle.id].extend(vehicle_plan)

        return predicted_positions

class SceneDescriptor:
    """
    Interface class to convert privileged simulator data into a structured JSON-like format.
    """
    def __init__(self, config):
        """
        Initialize the SceneDescriptor object.

        Args:
            config (object): The configuration object.
        """
        self.config = config

    def get_traffic_data(self, traffic_context):
        """
        Get the traffic data from the privileged simulator data.

        Returns:
            dict: A dictionary containing the traffic data.
        """
        def __get_traffic_light_data(traffic_light, distance_to_light):
            """
            Get the traffic light data from the privileged simulator data.

            Args:
                traffic_light (carla.Actor): The traffic light actor.

            Returns:
                dict: A dictionary containing the traffic light data.
            """
            traffic_light_data = None

            if traffic_light and distance_to_light < self.config.traffic_light_distance_threshold:
              state = traffic_light.get_state()

              if state == carla.TrafficLightState.Red:
                light_state = "RED"
              elif state == carla.TrafficLightState.Yellow:
                light_state = "YELLOW"
              elif state == carla.TrafficLightState.Green:
                light_state = "GREEN"
              else:
                light_state = "UNKNOWN"

              traffic_light_data = {
                  "distance_to_light": distance_to_light,
                  "state": light_state,
              }
            return traffic_light_data

        def __get_stop_sign_data(stop_sign, distance_to_stop_sign):
            """
            Get the stop sign data from the privileged simulator data.

            Args:
                stop_sign (carla.Actor): The stop sign actor.

            Returns:
                dict: A dictionary containing the stop sign data.
            """
            stop_sign_data = None

            if stop_sign and distance_to_stop_sign < self.config.stop_sign_distance_threshold:
              stop_sign_data = {
                  "distance_to_stop_sign": distance_to_stop_sign
              }
            return stop_sign_data

        traffic_data = {
            "next_traffic_light": __get_traffic_light_data(traffic_context["next_traffic_light"], traffic_context["distance_to_next_traffic_light"]),
            "next_stop_sign": __get_stop_sign_data(traffic_context["next_stop_sign"], traffic_context["distance_to_next_stop_sign"]),
            "speed_limit": traffic_context["speed_limit"]
        }
        return traffic_data

    def get_ego_data(self, ego_context):
        """
        Get the ego vehicle data from the privileged simulator data.

        Returns:
            dict: A dictionary containing the ego vehicle data.
        """
        ego_data = {
            "speed": ego_context["speed"],
            "orientation": ego_context["compass"],
            "position": ego_context["gps"][:2].tolist()
        }
        return ego_data

    def get_agent_data(self, agent_context, ego_data):
        """
        Get the agent data from the privileged simulator data.

        Returns:
            dict: A dictionary containing the agent data.
        """
        def __get_npc_vehicle_data(vehicles):
            """
            Get the non-player vehicle data from the privileged simulator data.

            Args:
                vehicles (list): A list of non-player vehicle actors.

            Returns:
                list: A list of dictionaries containing the non-player vehicle data.
            """
            npc_vehicle_data = []

            for vehicle in vehicles:
              vehicle_position = np.array([vehicle.get_location().x, vehicle.get_location().y], dtype=np.float32)
              relative_position_veh_wrt_ego = t_u.inverse_conversion_2d(vehicle_position, ego_data["position"], -ego_data["orientation"]).tolist()

              print(f"Vehicle Position: {vehicle.get_location().x}, {vehicle.get_location().y}")
              print(f"Relative Vehicle Position Ego Frame: {relative_position_veh_wrt_ego}")

              relative_distance = np.linalg.norm(relative_position_veh_wrt_ego)

              vehicle_data = {
                  "vehicle_id": vehicle.id,
                  "data": {
                    "speed": vehicle.get_velocity().length(),
                    "relative orientation": t_u.normalize_angle(np.deg2rad(vehicle.get_transform().rotation.yaw) - ego_data["orientation"]),
                    "relative position": relative_position_veh_wrt_ego,
                    "relative distance": relative_distance
                  }
              }
              npc_vehicle_data.append(vehicle_data)
            return npc_vehicle_data

        agent_data = {
            "leading_vehicles": __get_npc_vehicle_data(agent_context["leading_vehicles"]),
            "trailing_vehicles": __get_npc_vehicle_data(agent_context["trailing_vehicles"]),
        }
        return agent_data

    def get_structured_data(self, traffic_context, ego_context, agent_context):
        """
        Convert the privileged simulator data into a structured JSON-like format.

        Returns:
            dict: A dictionary containing the structured data.
        """
        ego_data = self.get_ego_data(ego_context)

        data = {
            "traffic": self.get_traffic_data(traffic_context),
            "ego": ego_data,
            "agent": self.get_agent_data(agent_context, ego_data)
        }
        return data

    def to_json(self, structured_data):
        """
        Convert the structured data into a JSON string.

        Returns:
            str: A JSON string containing the structured data.
        """
        return json.dumps(structured_data, indent=4)

        # ...existing code...

    def to_formatted_string(self, structured_data):
      """
      Convert the structured data to a formatted string.

      Args:
        structured_data (dict): The structured data.

      Returns:
        str: A formatted string representation of the data.
      """
      traffic_data = structured_data['traffic']
      ego_data = structured_data['ego']
      agent_data = structured_data['agent']

      formatted_string = "Traffic Data:\n"
      formatted_string += "    Next Traffic Light:\n"
      if traffic_data['next_traffic_light']:
        formatted_string += f"        Distance to Light: {traffic_data['next_traffic_light'].get('distance_to_light', 'N/A')}\n"
        formatted_string += f"        State: {traffic_data['next_traffic_light'].get('state', 'N/A')}\n"
      else:
        formatted_string += "        No data available\n"
      formatted_string += "    Next Stop Sign:\n"
      if traffic_data['next_stop_sign']:
        formatted_string += f"        Distance to Stop Sign: {traffic_data['next_stop_sign'].get('distance_to_stop_sign', 'N/A')}\n"
      else:
        formatted_string += "        No data available\n"
      formatted_string += f"    Speed Limit: {traffic_data.get('speed_limit', 'N/A')}\n"

      formatted_string += "Ego Data:\n"
      formatted_string += f"    Speed: {ego_data.get('speed', 'N/A')}\n"
      formatted_string += f"    Orientation: {ego_data.get('orientation', 'N/A')}\n"
      formatted_string += f"    Position: {ego_data.get('position', 'N/A')}\n"

      formatted_string += "Agent Data:\n"
      formatted_string += "    Leading Vehicles:\n"
      if agent_data['leading_vehicles']:
        for vehicle in agent_data['leading_vehicles']:
          formatted_string += f"        Vehicle ID: {vehicle.get('vehicle_id', 'N/A')}, Relative Position: {vehicle['data']['relative position']}, Relative Orientation: {vehicle['data']['relative orientation']}, Speed: {vehicle['data']['speed']}, Relative Distance: {vehicle['data']['relative distance']}\n"
      else:
        formatted_string += "        No data available\n"
      formatted_string += "    Trailing Vehicles:\n"
      if agent_data['trailing_vehicles']:
        for vehicle in agent_data['trailing_vehicles']:
          formatted_string += f"        Vehicle ID: {vehicle.get('vehicle_id', 'N/A')}, Relative Position: {vehicle['data']['relative position']}, Relative Orientation: {vehicle['data']['relative orientation']}, Speed: {vehicle['data']['speed']}, Relative Distance: {vehicle['data']['relative distance']}\n"
      else:
        formatted_string += "        No data available\n"

      return formatted_string

    # Example usage:
    # autopilot_instance = AutoPilot(...)
    # interface = SimulatorDataInterface(autopilot_instance)
    # structured_data = interface.get_structured_data(traffic_context, ego_context, agent_context)
    # formatted_string = interface.to_formatted_string(structured_data)
    # print(formatted_string)

    # ...existing code...
