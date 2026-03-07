import cv2
import numpy as np
import carla

from dataclasses import dataclass
from typing import Dict, Tuple

from .camera_interface import CameraInterface
from .data_extractors.scene_extractor import SceneData

@dataclass(frozen=True, slots=True)
class ImageRendererConfig:
    # Rendering configuration
    VEHICLE_BBOX_COLOR = (255, 0, 0) # Blue in BGR
    # EGO_BBOX_COLOR = (0, 255, 0) # Green
    EGO_BBOX_COLOR = (43, 64, 6) #  Dark Green
    # CYCLIST_BBOX_COLOR = (255, 255, 0) # Cyan
    CYCLIST_BBOX_COLOR = (255, 0, 127) # Violet
    # PED_BBOX_COLOR = (0, 255, 255) # Yellow
    PED_BBOX_COLOR = (34, 119, 204) # Ochre
    OBSTACLE_BBOX_COLOR = (0, 0, 255) # Red

    BBOX_THICKNESS = 2
    LABEL_COLOR = (255, 255, 255)  # White text

    VEHICLE_LABEL_BG_COLOR = (255, 0, 0)  # Blue background
    # EGO_LABEL_BG_COLOR = (0, 255, 0) # Green
    EGO_LABEL_BG_COLOR = (43, 64, 6) #  Dark Green
    # CYCLIST_LABEL_BG_COLOR = (255, 255, 0) # Cyan
    CYCLIST_LABEL_BG_COLOR = (255, 0, 127) # Violet
    # PED_LABEL_BG_COLOR = (0, 255, 255) # Yellow
    PED_LABEL_BG_COLOR = (34, 119, 204) # Ochre
    OBSTACLE_LABEL_BG_COLOR = (0, 0, 255) # Red

    LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
    LABEL_FONT_SCALE = 0.5
    LABEL_FONT_THICKNESS = 1
    LABEL_BOX_SIZE = (60, 20)

    # Draw distance threshold
    MAX_FRONT_CAM_DRAW_DISTANCE = 50.0


class ImageRenderer:
    # Bounding box edge connections (cube vertices)
    BBOX_EDGES = [
        [0, 1], [1, 3], [3, 2], [2, 0],  # Bottom face
        [0, 4], [4, 5], [5, 1], [5, 7],  # Vertical edges and top partial
        [7, 6], [6, 4], [6, 2], [7, 3]   # Top face and remaining edges
    ]

    def __init__(self):
        self.config = ImageRendererConfig()

    # -------------------------------------------------------------------- #
    #  Utility
    # -------------------------------------------------------------------- #

    def render_actor_bounding_boxes(
        self,
        cameras : Dict[str, CameraInterface],
        ego_vehicle : carla.Vehicle,
        scene_data : SceneData,
    ) -> Dict[str, np.ndarray]:
        """
        Render bounding boxes for all actors on all camera images.

        Args:
            cameras: Dictionary of camera interfaces by tag
            ego_context: Ego vehicle context data
            agent_context: Agent context data

        Returns:
            Dictionary of rendered images by camera tag
        """
        ego_transform = ego_vehicle.get_transform()
        ego_location = ego_transform.location
        ego_forward_vec = ego_transform.get_forward_vector()

        # Get all pedestrians
        all_peds = [] if scene_data.ped_data is None else scene_data.ped_data

        # Get all obstacles
        all_obstacles = scene_data.obstacle_data.all_obstacles

        # Get all vehicles (cyclists included)
        all_vehicles = scene_data.vehicle_data.all_vehicles_flat

        rendered_images = {}

        for tag, camera in cameras.items():
            if camera.image is None:
                continue

            # Create copy of image for rendering
            rendered_image = camera.image.copy()

            # Render ego vehicle
            bb_color = self.config.EGO_BBOX_COLOR
            label_bg_color = self.config.EGO_LABEL_BG_COLOR
            label_text = "EGO"
            if self._should_render_actor(
                ego_vehicle, ego_location, ego_forward_vec, tag
            ):
                self._render_single_actor(
                    rendered_image,
                    camera,
                    ego_vehicle,
                    bb_color,
                    label_bg_color,
                    label_text
                )

            for vehicle_entry in all_vehicles:
                vehicle = vehicle_entry.vehicle

                if self._should_render_actor(
                    vehicle, ego_location, ego_forward_vec, tag
                ):
                    is_cyclist = vehicle_entry.vehicle_type == "cyclist"
                    if is_cyclist:
                        self._render_single_actor(
                            rendered_image,
                            camera,
                            vehicle,
                            bb_color=self.config.CYCLIST_BBOX_COLOR,
                            label_bg_color=self.config.CYCLIST_LABEL_BG_COLOR,
                            label_text=str(vehicle.id)
                        )
                    else:
                        bb_color = self.config.VEHICLE_BBOX_COLOR
                        label_bg_color = self.config.VEHICLE_LABEL_BG_COLOR
                        label_text = str(vehicle.id)
                        self._render_single_actor(
                            rendered_image,
                            camera,
                            vehicle,
                            bb_color,
                            label_bg_color,
                            label_text
                        )

            for ped_entry in all_peds:
                ped = ped_entry.pedestrian
                if self._should_render_actor(
                    ped, ego_location, ego_forward_vec, tag
                ):
                    self._render_single_actor(
                        rendered_image,
                        camera,
                        ped,
                        bb_color=self.config.PED_BBOX_COLOR,
                        label_bg_color=self.config.PED_LABEL_BG_COLOR,
                        label_text=str(ped.id)
                    )

            for obstacle_entry in all_obstacles:
                obstacle = obstacle_entry.obstacle
                if self._should_render_actor(
                    obstacle, ego_location, ego_forward_vec, tag
                ):
                    self._render_single_actor(
                        rendered_image,
                        camera,
                        obstacle,
                        bb_color=self.config.OBSTACLE_BBOX_COLOR,
                        label_bg_color=self.config.OBSTACLE_LABEL_BG_COLOR,
                        label_text=str(obstacle.id)
                    )

            rendered_images[tag] = rendered_image

        return rendered_images

    # -------------------------------------------------------------------- #
    #  Private
    # -------------------------------------------------------------------- #

    def _should_render_actor(
        self,
        actor : carla.Actor,
        ego_location : carla.Location,
        ego_forward_vec : carla.Vector3D,
        camera_tag: str,
    ) -> bool:
        """
        Determine if actor should be rendered based on distance and camera type.

        Args:
            actor: CARLA actor
            ego_location: Ego vehicle location
            ego_forward_vec: Ego vehicle forward vector
            camera_tag: Camera identifier tag

        Returns:
            True if actor should be rendered
        """
        # Always render for bird's eye view cameras
        if "bev" in camera_tag.lower():
            return True

        return False
        # actor_location = actor.get_location()
        # ego_to_actor_vec = actor_location - ego_location
        # distance = actor_location.distance(ego_location)

        # # Only render actors in front and within distance threshold
        # is_in_front = ego_to_actor_vec.dot(ego_forward_vec) > 0
        # is_within_range = distance < self.config.MAX_FRONT_CAM_DRAW_DISTANCE

        # return is_in_front and is_within_range

    def _render_single_actor(
        self,
        image: np.ndarray,
        camera: CameraInterface,
        actor : carla.Actor,
        bb_color : Tuple,
        label_bg_color : Tuple,
        label_text : str,
    ) -> None:
        """
        Render bounding box and label for a single actor.

        Args:
            image: Image array to render on
            camera: Camera interface object
            actor: CARLA actor
        """

        # Render bounding box
        self._draw_bounding_box(image, camera, actor, bb_color)

        # Render label
        self._draw_actor_label(image, camera, actor, label_bg_color, label_text)

    def _draw_bounding_box(
        self,
        image: np.ndarray,
        camera: CameraInterface,
        actor: carla.Actor,
        bb_color: Tuple,
    ) -> None:
        """
        Draw a 2D axis-aligned bounding box that encloses the projected 3D actor bbox.

        Args:
            image: Image array to draw on
            camera: Camera interface object
            actor: CARLA actor
        """
        # Get bounding box vertices in world coordinates
        bbox_vertices = actor.bounding_box.get_world_vertices(actor.get_transform())

        camera_transform = camera.transform
        camera_forward_vec = camera_transform.get_forward_vector()
        camera_location = camera_transform.location

        projected_points = []

        # Project all vertices to 2D
        for vertex in bbox_vertices:
            p_2d = self._project_vertex_with_occlusion_handling(
                vertex, camera, camera_forward_vec, camera_location
            )
            if p_2d is not None and camera.is_point_in_canvas(p_2d):
                projected_points.append(p_2d)

        # If nothing valid was projected, skip drawing
        if len(projected_points) == 0:
            return

        projected_points = np.array(projected_points)

        min_x = int(np.min(projected_points[:, 0]))
        max_x = int(np.max(projected_points[:, 0]))
        min_y = int(np.min(projected_points[:, 1]))
        max_y = int(np.max(projected_points[:, 1]))

        # Clamp box to image boundaries
        img_h, img_w = image.shape[:2]
        min_x = max(0, min_x)
        max_x = min(img_w - 1, max_x)
        min_y = max(0, min_y)
        max_y = min(img_h - 1, max_y)

        # Sanity check: ensure box has area
        if min_x >= max_x or min_y >= max_y:
            return

        cv2.rectangle(
            image,
            (min_x, min_y),
            (max_x, max_y),
            bb_color,
            self.config.BBOX_THICKNESS,
        )

    def _project_vertex_with_occlusion_handling(
        self,
        vertex : carla.Location,
        camera: CameraInterface,
        camera_forward_vec : carla.Vector3D,
        camera_location : carla.Location
    ) -> np.ndarray:
        """
        Project vertex to 2D with proper handling of vertices behind camera.

        Args:
            vertex: 3D vertex location
            camera: Camera interface object
            camera_forward_vec: Camera forward vector
            camera_location: Camera location

        Returns:
            2D projected point or None if not projectable
        """
        # Check if vertex is behind camera
        ray_to_vertex = vertex - camera_location
        is_behind_camera = camera_forward_vec.dot(ray_to_vertex) <= 0

        try:
            # Use camera's projection method with appropriate matrix
            return camera.project_3d_to_2d(vertex, is_behind_camera)
        except Exception:
            return None

    def _draw_actor_label(
        self,
        image: np.ndarray,
        camera: CameraInterface,
        actor: carla.Actor,
        label_bg_color: Tuple,
        label_text : str,
    ) -> None:
        actor_loc = actor.get_location()
        try:
            actor_2d = camera.project_3d_to_2d(actor_loc)

            if camera.is_point_in_canvas(actor_2d):
                center_x, center_y = int(actor_2d[0]), int(actor_2d[1])

                # Draw label background
                box_width, box_height = self.config.LABEL_BOX_SIZE
                top_left = (
                    center_x - box_width // 2,
                    center_y - box_height // 2
                )
                bottom_right = (
                    center_x + box_width // 2,
                    center_y + box_height // 2
                )

                cv2.rectangle(
                    image,
                    top_left,
                    bottom_right,
                    label_bg_color,
                    thickness=-1
                )

                # Draw text
                text = label_text
                text_size = cv2.getTextSize(
                    text,
                    self.config.LABEL_FONT,
                    self.config.LABEL_FONT_SCALE,
                    self.config.LABEL_FONT_THICKNESS
                )[0]

                text_x = center_x - text_size[0] // 2
                text_y = center_y + text_size[1] // 2

                cv2.putText(
                    image,
                    text,
                    (text_x, text_y),
                    self.config.LABEL_FONT,
                    self.config.LABEL_FONT_SCALE,
                    self.config.LABEL_COLOR,
                    self.config.LABEL_FONT_THICKNESS,
                    cv2.LINE_AA
                )

        except Exception as e:
            # Silently skip labels that can't be rendered
            pass
