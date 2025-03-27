import numpy as np
import carla
import transfuser_utils as t_u

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

    def _get_npc_vehicle_data(self, ego_context, vehicles):
        """
        Get the non-player vehicle data from the privileged simulator data.

        Args:
            ego_context (dict): Dictionary containing the ego context data.
            vehicles (list): A list of non-player vehicle actors.

        Returns:
            list: A list of dictionaries containing the non-player vehicle data.
        """
        ego_wp = ego_context['waypoint']
        ego_transform = ego_wp.transform
        ego_yaw = np.deg2rad(ego_transform.rotation.yaw)
        ego_matrix = np.array(ego_transform.get_matrix())

        npc_vehicle_data = []

        for vehicle in vehicles:
            # Get the vehicle data
            vehicle_id = vehicle.id
            vehicle_transform = vehicle.get_transform()
            vehicle_yaw = np.deg2rad(vehicle_transform.rotation.yaw)
            vehicle_matrix = np.array(vehicle_transform.get_matrix())
            vehicle_velocity = vehicle.get_velocity()

            vehicle_speed = vehicle_velocity.length()
            vehicle_speed = np.round(vehicle_speed, 2)

            # Calculate the relative position and orientation of the vehicle
            relative_yaw = t_u.normalize_angle(vehicle_yaw - ego_yaw)
            relative_yaw = np.round(relative_yaw, 2)

            relative_pos = t_u.get_relative_transform(ego_matrix, vehicle_matrix)[:2]
            relative_pos = np.round(relative_pos, 2)
            # vehicle_speed = self._get_forward_speed(transform=vehicle_transform, velocity=vehicle_velocity)

            # print(f"Vehicle Position: {vehicle.get_location().x}, {vehicle.get_location().y}")
            # print(f"Relative Vehicle Position Ego Frame: {relative_pos}")

            relative_distance = np.linalg.norm(relative_pos)
            relative_distance = np.round(relative_distance, 2)

            vehicle_data = {
                "vehicle_id": vehicle_id,
                "data": {
                    "speed": vehicle_speed,
                    "relative orientation": relative_yaw,
                    "relative position": relative_pos,
                    "relative distance": relative_distance
                }
            }
            npc_vehicle_data.append(vehicle_data)
        return npc_vehicle_data

    def _group_npc_vehicles(self, agent_context, ego_context, npc_vehicles):
        """
        Group the NPC vehicles by road and lane.

        Args:
            agent_context (dict): Dictionary containing the agent context data.
            ego_context (dict): Dictionary containing the ego context data.
            npc_vehicles (list): List of NPC vehicles.

        Returns:
            dict: A dictionary containing the NPC vehicles grouped by road and lane.
        """
        # Get the ego vehicle data
        ego_wp = ego_context['waypoint']
        ego_loc = ego_wp.transform.location
        ego_yaw = np.deg2rad(ego_wp.transform.rotation.yaw)
        ego_lane_id = ego_wp.lane_id
        # print(f'Ego Lane ID: {ego_wp.lane_id}, Road ID: {ego_wp.road_id}, Location: {ego_wp.transform.location}')

        # Setup NPC vehicle data dictionary
        grouped_npc_vehicles = {
            "Ongoing Traffic": {},
            "Oncoming Traffic": {},
            "Cross Traffic": {}
        }

        ongoing_leading_vehicles = agent_context["ongoing_leading_vehicles"]
        ongoing_trailing_vehicles = agent_context["ongoing_trailing_vehicles"]
        oncoming_leading_vehicles = agent_context["oncoming_leading_vehicles"]
        oncoming_trailing_vehicles = agent_context["oncoming_trailing_vehicles"]

        for lane_id, lane_vehicles in ongoing_leading_vehicles.items():
            # print(f'Vehicles in lane {lane_id}: {lane_vehicles}')
            if lane_id == ego_lane_id:
                key = "Ego"
            else:
                offset = lane_id - ego_lane_id
                key = f"Left-{abs(offset)}" if offset > 0 else f"Right-{abs(offset)}"

            grouped_npc_vehicles["Ongoing Traffic"][key] = {
                "leading_vehicles": self._get_npc_vehicle_data(ego_context, lane_vehicles),
                "trailing_vehicles": self._get_npc_vehicle_data(ego_context, ongoing_trailing_vehicles[lane_id]),
            }

        for lane_id, lane_vehicles in oncoming_leading_vehicles.items():
            # print(f'Vehicles in lane {lane_id}: {lane_vehicles}')
            offset = lane_id - ego_lane_id
            key = f"Left-{abs(offset)}" if offset > 0 else f"Right-{abs(offset)}"
            grouped_npc_vehicles["Oncoming Traffic"][key] = {
                "leading_vehicles": self._get_npc_vehicle_data(ego_context, lane_vehicles),
                "trailing_vehicles": self._get_npc_vehicle_data(ego_context, oncoming_trailing_vehicles[lane_id]),
            }

        return grouped_npc_vehicles


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
                  "id": traffic_light.id,
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
            "position": ego_context["gps"][:2].tolist(),
            "route": ego_context["route"],
            "waypoint": ego_context["waypoint"]
        }
        return ego_data

    def get_agent_data(self, agent_context, ego_context):
        """
        Get the agent data from the privileged simulator data.

        Returns:
            dict: A dictionary containing the agent data.
        """

        agent_data = self._group_npc_vehicles(agent_context, ego_context, agent_context["ongoing_leading_vehicles"])
        # agent_data = {
        #     "leading_vehicles": __get_npc_vehicle_data(agent_context["leading_vehicles"]),
        #     "trailing_vehicles": __get_npc_vehicle_data(agent_context["trailing_vehicles"]),
        # }
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
        formatted_string += f"        Traffic Light ID: {traffic_data['next_traffic_light'].get('id', 'N/A')}, State: {traffic_data['next_traffic_light'].get('state', 'N/A')}, Relative Distance: {traffic_data['next_traffic_light'].get('distance_to_light', 'N/A')}\n"
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
      formatted_string += "    Ongoing Traffic:\n"
      for lane, vehicles in agent_data["Ongoing Traffic"].items():
        formatted_string += f"        {lane}:\n"
        formatted_string += "            Leading Vehicles:\n"
        if vehicles["leading_vehicles"]:
            for vehicle in vehicles["leading_vehicles"]:
                formatted_string += f"                Vehicle ID: {vehicle.get('vehicle_id', 'N/A')}, Relative Position: {vehicle['data']['relative position']}, Relative Orientation: {vehicle['data']['relative orientation']}, Speed: {vehicle['data']['speed']}, Relative Distance: {vehicle['data']['relative distance']}\n"
        else:
            formatted_string += "                No data available\n"

        formatted_string += "            Trailing Vehicles:\n"
        if vehicles["trailing_vehicles"]:
            for vehicle in vehicles["trailing_vehicles"]:
                formatted_string += f"                Vehicle ID: {vehicle.get('vehicle_id', 'N/A')}, Relative Position: {vehicle['data']['relative position']}, Relative Orientation: {vehicle['data']['relative orientation']}, Speed: {vehicle['data']['speed']}, Relative Distance: {vehicle['data']['relative distance']}\n"
        else:
            formatted_string += "                No data available\n"

      formatted_string += "    Oncoming Traffic:\n"
      for lane, vehicles in agent_data["Oncoming Traffic"].items():
        formatted_string += f"        {lane}:\n"
        formatted_string += "            Leading Vehicles:\n"
        for vehicle in vehicles["leading_vehicles"]:
            formatted_string += f"                Vehicle ID: {vehicle.get('vehicle_id', 'N/A')}, Relative Position: {vehicle['data']['relative position']}, Relative Orientation: {vehicle['data']['relative orientation']}, Speed: {vehicle['data']['speed']}, Relative Distance: {vehicle['data']['relative distance']}\n"
        formatted_string += "            Trailing Vehicles:\n"
        for vehicle in vehicles["trailing_vehicles"]:
            formatted_string += f"                Vehicle ID: {vehicle.get('vehicle_id', 'N/A')}, Relative Position: {vehicle['data']['relative position']}, Relative Orientation: {vehicle['data']['relative orientation']}, Speed: {vehicle['data']['speed']}, Relative Distance: {vehicle['data']['relative distance']}\n"

    #   formatted_string += "    Cross Traffic:\n"
    #   for lane, vehicles in agent_data["Cross Traffic"].items():
    #     formatted_string += f"        {lane}:\n"
    #     formatted_string += "            Leading Vehicles:\n"
    #     for vehicle in vehicles["leading_vehicles"]:
    #         formatted_string += f"                Vehicle ID: {vehicle.get('vehicle_id', 'N/A')}, Relative Position: {vehicle['data']['relative position']}, Relative Orientation: {vehicle['data']['relative orientation']}, Speed: {vehicle['data']['speed']}, Relative Distance: {vehicle['data']['relative distance']}\n"
    #     formatted_string += "            Trailing Vehicles:\n"
    #     for vehicle in vehicles["trailing_vehicles"]:
    #         formatted_string += f"                Vehicle ID: {vehicle.get('vehicle_id', 'N/A')}, Relative Position: {vehicle['data']['relative position']}, Relative Orientation: {vehicle['data']['relative orientation']}, Speed: {vehicle['data']['speed']}, Relative Distance: {vehicle['data']['relative distance']}\n"

      return formatted_string

    # Example usage:
    # autopilot_instance = AutoPilot(...)
    # interface = SimulatorDataInterface(autopilot_instance)
    # structured_data = interface.get_structured_data(traffic_context, ego_context, agent_context)
    # formatted_string = interface.to_formatted_string(structured_data)
    # print(formatted_string)

    # ...existing code...
