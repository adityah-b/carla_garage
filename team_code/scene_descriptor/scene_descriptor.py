import carla
import numpy as np

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

from .camera_interface import CameraInterface
from .image_renderer import ImageRenderer

from privileged_route_planner import PlannerState

# Data extractors
from .data_extractors.scene_extractor import SceneData, SceneExtractor

# Formatters
from .formatters.scene_formatter import SceneFormatter

@dataclass(frozen=True, slots=True)
class SceneContext:
    scene_data : SceneData
    formatted_text : Optional[str]

class SceneDescriptor:
    def __init__(self, config, carla_map: carla.Map):
        self.config = config
        self.carla_map = carla_map

        # Component initialization
        self._cameras: Dict[str, CameraInterface] = {}
        self._camera_tags: List[str] = []

        # Initialize specialized processors
        self._bbox_renderer = ImageRenderer()

        # Extractors
        self._data_extractor = SceneExtractor(self.config, self.carla_map)

        # Formatters
        self._formatter = SceneFormatter()

    def setup_cameras(
        self,
        cameras: List[Tuple[str, Any]]
    ) -> None:
        """
        Initialize camera interfaces from CARLA camera actors.

        Args:
            cameras: List of tuples containing (tag, camera_actor_obj)
        """
        self._cameras.clear()
        self._camera_tags.clear()

        for tag, camera_obj in cameras:
            self._cameras[tag] = CameraInterface(camera_obj)
            self._camera_tags.append(tag)

    def set_camera_observations(
        self,
        images: List[Tuple[str, np.ndarray]]
    ) -> None:
        """
        Update camera observations with new image data.

        Args:
            images: List of tuples containing (camera_tag, image_data)
        """
        for tag, image in images:
            if tag in self._cameras:
                self._cameras[tag].set_image(image)
            else:
                print(f"Warning: Camera tag '{tag}' not found in registered cameras.")

    def draw_actor_bounding_boxes(
        self,
        ego_vehicle : carla.Vehicle,
        actors : carla.ActorList
    ) -> Dict[str, np.ndarray]:
        """
        Render vehicle bounding boxes on camera images.

        Args:
            ego_context: Dictionary containing ego vehicle context
            agent_context: Dictionary containing agent/NPC vehicle context

        Returns:
            Dictionary of camera tags mapped to rendered images
        """
        return self._bbox_renderer.render_actor_bounding_boxes(
            self._cameras, ego_vehicle, actors
        )

    def get_structured_scene_data(
        self,
        ego_vehicle : carla.Vehicle,
        actors : carla.ActorList,
        planner_state : PlannerState,
    ) -> SceneData:
        """
        Convert privileged simulator data into structured format.

        Args:
            traffic_context: Traffic control information
            ego_context: Ego vehicle information
            agent_context: Other vehicle information

        Returns:
            Dictionary containing structured scene data
        """
        return self._data_extractor.extract_scene(
            ego_vehicle=ego_vehicle,
            actors=actors,
            planner_state=planner_state
        )

    def format_scene_as_text(
        self,
        scene_data: SceneData,
    ) -> str:
        """
        Convert structured scene data to formatted text representation.

        Args:
            structured_data: Dictionary containing structured scene data
            compact: Whether to use compact formatting

        Returns:
            Formatted string representation of scene data
        """
        return self._formatter.format_scene(
            scene_data=scene_data
        )

    def process_complete_scene(
        self,
        ego_vehicle : carla.Vehicle,
        actors : carla.ActorList,
        planner_state : PlannerState,
        format_as_text: bool = True,
    ) -> SceneContext:
        """
        Complete scene processing pipeline.

        Args:
            traffic_context: Traffic control information
            ego_context: Ego vehicle information
            agent_context: Other vehicle information
            format_as_text: Whether to include formatted text representation
            compact_format: Whether to use compact text formatting

        Returns:
            Dictionary containing structured data and optional formatted text
        """
        # Get structured data
        scene_data = self.get_structured_scene_data(
            ego_vehicle, actors, planner_state
        )

        # Add formatted text if requested
        formatted_text = None
        if format_as_text:
            formatted_text = self.format_scene_as_text(scene_data)

        return SceneContext(
            scene_data=scene_data,
            formatted_text=formatted_text
        )

    # Properties for accessing internal components (useful for testing/debugging)

    @property
    def cameras(self) -> Dict[str, CameraInterface]:
        """Access to camera interfaces."""
        return self._cameras.copy()

    @property
    def camera_tags(self) -> List[str]:
        """Access to camera tags."""
        return self._camera_tags.copy()

    def get_camera_by_tag(self, tag: str) -> Optional[CameraInterface]:
        """Get specific camera interface by tag."""
        return self._cameras.get(tag)

    def has_camera(self, tag: str) -> bool:
        """Check if camera with given tag exists."""
        return tag in self._cameras
