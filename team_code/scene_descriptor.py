import numpy as np
import carla
import transfuser_utils as t_u
import cv2

class SceneDescriptor:
    """
    Interface class to convert privileged simulator data into a structured JSON-like format.
    """
    # Dictionary of CameraInterface objects organized by their sensor tags
    camera_objs = {}
    camera_tags = []

    class CameraInterface:
        """
        Class to handle camera data and projection matrices.
        """
        def __init__(self, camera_actor_obj):
            self.obj = camera_actor_obj
            self.width = int(camera_actor_obj.attributes['image_size_x'])
            self.height = int(camera_actor_obj.attributes['image_size_y'])
            self.fov = float(camera_actor_obj.attributes['fov'])

            print(f"Camera Width: {self.width} type: {type(self.width)}, Height: {self.height} type: {type(self.height)}, FOV: {self.fov} type: {type(self.fov)}")

            self.K = self._build_projection_matrix(self.width, self.height, self.fov)
            self.K_behind = self._build_projection_matrix(self.width, self.height, self.fov, is_behind_camera=True)

            self.image = None

        def _build_projection_matrix(self, w, h, fov, is_behind_camera=False):
            focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
            K = np.identity(3)

            if is_behind_camera:
                K[0, 0] = K[1, 1] = -focal
            else:
                K[0, 0] = K[1, 1] = focal

            K[0, 2] = w / 2.0
            K[1, 2] = h / 2.0
            return K

        def _point_in_canvas(self, pos, img_h, img_w):
            """Return true if point is in canvas"""
            if (pos[0] >= 0) and (pos[0] < img_w) and (pos[1] >= 0) and (pos[1] < img_h):
                return True
            return False

        def _get_image_point(self, loc, K, w2c):
            # Calculate 2D projection of 3D coordinate

            # Format the input coordinate (loc is a carla.Position object)
            point = np.array([loc.x, loc.y, loc.z, 1])
            # transform to camera coordinates
            point_camera = np.dot(w2c, point)

            # New we must change from UE4's coordinate system to an "standard"
            # (x, y ,z) -> (y, -z, x)
            # and we remove the fourth componebonent also
            point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

            # now project 3D->2D using the camera matrix
            point_img = np.dot(K, point_camera)
            # normalize
            point_img[0] /= point_img[2]
            point_img[1] /= point_img[2]

            return point_img[0:2]

        def _set_image(self, image):
            """
            Update the camera observation with the new image data.

            Args:
                image (numpy.ndarray): The image data from the camera.
            """
            self.image = np.copy(image)

    def __init__(self, config):
        """
        Initialize the SceneDescriptor object.

        Args:
            config (object): The configuration object.
        """
        self.config = config

    def setup_cameras(self, cameras):
        for tag, camera_obj in cameras:
           self.camera_objs[tag] = self.CameraInterface(camera_obj)
           self.camera_tags.append(tag)

    def set_camera_observations(self, images):
        """
        Set the camera observations with the new image data.

        Args:
            images (list): A list of tuples containing the camera tag and the image data.
        """
        for tag, image in images:
            if tag in self.camera_tags:
                self.camera_objs[tag]._set_image(image)
            else:
                print(f"Camera tag {tag} not found in camera tags.")

    def _draw_bounding_box(self, camera, vehicle):
        world_to_camera = np.array(camera.obj.get_transform().get_inverse_matrix())
        edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]
        verts = [v for v in vehicle.bounding_box.get_world_vertices(vehicle.get_transform())]

        camera_transform = camera.obj.get_transform()
        camera_fwd_vec = camera_transform.get_forward_vector()
        camera_loc = camera_transform.location

        for edge in edges:
            p1 = camera._get_image_point(verts[edge[0]], camera.K, world_to_camera)
            p2 = camera._get_image_point(verts[edge[1]], camera.K, world_to_camera)

            p1_in_canvas = camera._point_in_canvas(p1, camera.height, camera.width)
            p2_in_canvas = camera._point_in_canvas(p2, camera.height, camera.width)

            if not p1_in_canvas and not p2_in_canvas:
                continue

            ray0 = verts[edge[0]] - camera_loc
            ray1 = verts[edge[1]] - camera_loc

            # One of the vertex is behind the camera
            if not (camera_fwd_vec.dot(ray0) > 0):
                p1 = camera._get_image_point(verts[edge[0]], camera.K_behind, world_to_camera)
            if not (camera_fwd_vec.dot(ray1) > 0):
                p2 = camera._get_image_point(verts[edge[1]], camera.K_behind, world_to_camera)

            cv2.line(camera.image, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (0,0,255), 1)

    def _draw_label(self, camera, vehicle):
        world_to_camera = np.array(camera.obj.get_transform().get_inverse_matrix())
        vehicle_loc = vehicle.get_transform().location
        vehicle_camera_loc = camera._get_image_point(vehicle_loc, camera.K, world_to_camera)

        if camera._point_in_canvas(vehicle_camera_loc, camera.height, camera.width):
            cx, cy = int(vehicle_camera_loc[0]), int(vehicle_camera_loc[1])

            box_width, box_height = 60, 20
            top_left = (cx - box_width // 2, cy - box_height // 2)
            bottom_right = (cx + box_width // 2, cy + box_height // 2)

            # Draw blue rectangle
            cv2.rectangle(camera.image, top_left, bottom_right, (0, 0, 255), thickness=-1)

            # Put the vehicle ID as text inside the box
            text = str(vehicle.id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

            # Center the text within the box
            text_x = cx - text_size[0] // 2
            text_y = cy + text_size[1] // 2

            cv2.putText(camera.image, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    def draw_actor_bbs(self, ego_context, agent_context):
        ego_actor = ego_context['ego_actor']
        ego_transform = ego_actor.get_transform()
        ego_loc = ego_transform.location
        ego_fwd_vec = ego_transform.get_forward_vector()

        npc_vehicles = agent_context['npc_vehicles']
        bb_images = {}

        for tag, camera in self.camera_objs.items():
            for vehicle in npc_vehicles:
                vehicle_loc = vehicle.get_transform().location
                ego_to_vehicle_vec = vehicle_loc - ego_loc
                dist_to_vehicle = vehicle_loc.distance(ego_loc)

                draw_boxes = (
                    "bev" in tag or
                    (ego_to_vehicle_vec.dot(ego_fwd_vec) > 0 and dist_to_vehicle < self.config.max_front_cam_draw_distance)
                )

                if draw_boxes:
                    self._draw_bounding_box(camera, vehicle)
                    self._draw_label(camera, vehicle)

            bb_images[tag] = camera.image

        return bb_images

    # def _get_npc_vehicle_data(self, ego_context, vehicles):
    #     """
    #     Get the non-player vehicle data from the privileged simulator data.

    #     Args:
    #         ego_context (dict): Dictionary containing the ego context data.
    #         vehicles (list): A list of non-player vehicle actors.

    #     Returns:
    #         list: A list of dictionaries containing the non-player vehicle data.
    #     """
    #     ego_wp = ego_context['waypoint']
    #     ego_transform = ego_wp.transform
    #     ego_yaw = np.deg2rad(ego_transform.rotation.yaw)
    #     ego_matrix = np.array(ego_transform.get_matrix())

    #     npc_vehicle_data = []

    #     for vehicle in vehicles:
    #         # Get the vehicle data
    #         vehicle_id = vehicle.id
    #         vehicle_transform = vehicle.get_transform()
    #         vehicle_yaw = np.deg2rad(vehicle_transform.rotation.yaw)
    #         vehicle_matrix = np.array(vehicle_transform.get_matrix())
    #         vehicle_velocity = vehicle.get_velocity()

    #         vehicle_speed = vehicle_velocity.length()
    #         vehicle_speed = np.round(vehicle_speed, 2)

    #         # Calculate the relative position and orientation of the vehicle
    #         relative_yaw = t_u.normalize_angle(vehicle_yaw - ego_yaw)
    #         relative_yaw = np.round(relative_yaw, 2)

    #         relative_pos = t_u.get_relative_transform(ego_matrix, vehicle_matrix)[:2]
    #         relative_pos = np.round(relative_pos, 2)
    #         # vehicle_speed = self._get_forward_speed(transform=vehicle_transform, velocity=vehicle_velocity)

    #         # print(f"Vehicle Position: {vehicle.get_location().x}, {vehicle.get_location().y}")
    #         # print(f"Relative Vehicle Position Ego Frame: {relative_pos}")

    #         relative_distance = np.linalg.norm(relative_pos)
    #         relative_distance = np.round(relative_distance, 2)

    #         vehicle_data = {
    #             "vehicle_id": vehicle_id,
    #             "data": {
    #                 "speed": vehicle_speed,
    #                 "relative orientation": relative_yaw,
    #                 "relative position": relative_pos,
    #                 "relative distance": relative_distance
    #             }
    #         }
    #         npc_vehicle_data.append(vehicle_data)
    #     return npc_vehicle_data

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

        if len(vehicles) > 0:
            # Get the transformation matrices of the vehicles
            vehicle_transform_matrices = np.array([vehicle.get_transform().get_matrix() for vehicle in vehicles])

            # Get the yaw angles of the vehicles
            vehicle_yaws = np.array([np.deg2rad(vehicle.get_transform().rotation.yaw) for vehicle in vehicles])

            # Get the speeds of the vehicles
            vehicle_speeds = np.array([vehicle.get_velocity().length() for vehicle in vehicles])
            vehicle_speeds = np.round(vehicle_speeds, 2)

            # Get the IDs of the vehicles
            vehicle_ids = np.array([vehicle.id for vehicle in vehicles])

            # Calculate the relative position and orientation of the vehicle wrt the ego vehicle
            # Returns a 2D array of shape (N, 3) where N is the number of vehicles
            relative_positions = vehicle_transform_matrices[:, :3, 3] - ego_matrix[np.newaxis, :3, 3]
            relative_positions = ego_matrix[:3, :3].T @ relative_positions.T
            relative_positions = np.round(relative_positions.T, 2)

            # Calculate the relative yaw angles of the vehicles
            relative_yaws = (vehicle_yaws - ego_yaw + np.pi) % (2 * np.pi) - np.pi
            relative_yaws = np.round(relative_yaws, 2)
            
            # Calculate the relative distances of the vehicles from the ego vehicle
            relative_distances = np.linalg.norm(relative_positions, axis=1)
            relative_distances = np.round(relative_distances, 2)

            # Find the closest vehicle to the ego
            min_distance_index = np.argmin(relative_distances)

            vehicle_data = {
                "vehicle_id": vehicle_ids[min_distance_index],
                "data": {
                    "speed": vehicle_speeds[min_distance_index],
                    "relative orientation": relative_yaws[min_distance_index],
                    "relative position": relative_positions[min_distance_index][:2].tolist(),
                    "relative distance": relative_distances[min_distance_index]
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
                key = "Ego Lane"
            else:
                left_wp = ego_wp.get_left_lane()
                right_wp = ego_wp.get_right_lane()
                key = None
                if left_wp:
                    if (left_wp.lane_id < ego_lane_id and lane_id <= left_wp.lane_id) or \
                       (left_wp.lane_id > ego_lane_id and lane_id >= left_wp.lane_id):
                        key = f"Left-{abs(lane_id - ego_lane_id)} Lane"
                if key is None and right_wp:
                    if (right_wp.lane_id < ego_lane_id and lane_id <= right_wp.lane_id) or \
                       (right_wp.lane_id > ego_lane_id and lane_id >= right_wp.lane_id):
                        key = f"Right-{abs(lane_id - ego_lane_id)} Lane"
                if key is None:
                    offset = lane_id - ego_lane_id
                    key = f"Other-{abs(offset)}"

            leading_vehicles = self._get_npc_vehicle_data(ego_context, lane_vehicles)
            trailing_vehicles = self._get_npc_vehicle_data(ego_context, ongoing_trailing_vehicles[lane_id]) if lane_id in ongoing_trailing_vehicles else []

            if leading_vehicles or trailing_vehicles:
                grouped_npc_vehicles["Ongoing Traffic"][key] = {
                    "leading_vehicles": leading_vehicles,
                    "trailing_vehicles": trailing_vehicles,
                }

        for lane_id, lane_vehicles in oncoming_leading_vehicles.items():
            # print(f'Vehicles in lane {lane_id}: {lane_vehicles}')
            if lane_id == ego_lane_id:
                key = "Ego Lane"
            else:
                left_wp = ego_wp.get_left_lane()
                right_wp = ego_wp.get_right_lane()
                key = None
                if left_wp:
                    if (left_wp.lane_id < ego_lane_id and lane_id <= left_wp.lane_id) or \
                       (left_wp.lane_id > ego_lane_id and lane_id >= left_wp.lane_id):
                        key = f"Left-{abs(lane_id - ego_lane_id)} Lane"
                if key is None and right_wp:
                    if (right_wp.lane_id < ego_lane_id and lane_id <= right_wp.lane_id) or \
                       (right_wp.lane_id > ego_lane_id and lane_id >= right_wp.lane_id):
                        key = f"Right-{abs(lane_id - ego_lane_id)} Lane"
                if key is None:
                    offset = lane_id - ego_lane_id
                    key = f"Other-{abs(offset)}"

            leading_vehicles = self._get_npc_vehicle_data(ego_context, lane_vehicles)
            trailing_vehicles = self._get_npc_vehicle_data(ego_context, oncoming_trailing_vehicles[lane_id]) if lane_id in oncoming_trailing_vehicles else []

            if leading_vehicles or trailing_vehicles:
                grouped_npc_vehicles["Oncoming Traffic"][key] = {
                    "leading_vehicles": leading_vehicles,
                    "trailing_vehicles": trailing_vehicles,
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
        def __get_lane_change_data(lane_change_data):
            has_lane_change = False
            lane_change_direction = None
            can_change_lane = False
            available_lane_change_distance = -1.0

            if lane_change_data:
                has_lane_change = lane_change_data["has_lane_change"]
                if has_lane_change:
                    lane_change_direction = lane_change_data["lane_change_direction"]
                    lane_change_early_start_point = lane_change_data["lane_change_early_start_point"]
                    lane_change_late_start_point = lane_change_data["lane_change_late_start_point"]
                    lane_change_end_point = lane_change_data["lane_change_end_point"]

                    ego_wp = ego_context["route"][0]
                    lane_change_early_start_point_loc = lane_change_early_start_point.transform.location
                    lane_change_late_start_point_loc = lane_change_late_start_point.transform.location

                    # Check if ego has crossed the lane change early start point
                    has_passed_start_point = False
                    print(f'Distance to early start point: {lane_change_early_start_point_loc.distance(ego_wp.transform.location)}')
                    ego_transform = ego_wp.transform
                    ego_heading = ego_transform.get_forward_vector()
                    ego_actor_vec = lane_change_early_start_point_loc - ego_transform.location
                    if ego_heading.dot(ego_actor_vec) < 0:
                        has_passed_start_point = True

                    # Check if road network allows for lane change
                    target_lane = ego_wp.get_left_lane() if lane_change_direction == "left" else ego_wp.get_right_lane()
                    if target_lane and has_passed_start_point:
                        available_lane_change_distance = lane_change_late_start_point_loc.distance(ego_transform.location)
                        if available_lane_change_distance > 5.0:
                            can_change_lane = True

            lane_change_info = {
                "has_upcoming_lane_change": has_lane_change,
                "lane_change_direction": lane_change_direction,
                "can_change_lane": can_change_lane,
                "available_lane_change_distance": available_lane_change_distance,
            }
            return lane_change_info

        ego_data = {
            "speed": ego_context["speed"],
            "orientation": ego_context["compass"],
            "position": ego_context["gps"][:2].tolist(),
            "route": ego_context["route"],
            "waypoint": ego_context["waypoint"],
            "lane_change": __get_lane_change_data(ego_context["lane_change"]),
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
      formatted_string += f"    Lane Change:\n"
      lane_change = ego_data.get('lane_change', {})
      formatted_string += f"        Has Upcoming Lane Change: {lane_change.get('has_upcoming_lane_change', 'N/A')}\n"
      formatted_string += f"        Lane Change Direction: {lane_change.get('lane_change_direction', 'N/A')}\n"
      formatted_string += f"        Can Change Lane: {lane_change.get('can_change_lane', 'N/A')}\n"
      formatted_string += f"        Available Lane Change Distance: {lane_change.get('available_lane_change_distance', 'N/A')}\n"

      formatted_string += "Agent Data:\n"
      if agent_data["Ongoing Traffic"]:
        formatted_string += "    Ongoing Traffic:\n"
        for lane, vehicles in agent_data["Ongoing Traffic"].items():
            if not vehicles["leading_vehicles"] and not vehicles["trailing_vehicles"]:
                continue
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

      if agent_data["Oncoming Traffic"]:          
        formatted_string += "    Oncoming Traffic:\n"
        for lane, vehicles in agent_data["Oncoming Traffic"].items():
            if not vehicles["leading_vehicles"] and not vehicles["trailing_vehicles"]:
                continue
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
