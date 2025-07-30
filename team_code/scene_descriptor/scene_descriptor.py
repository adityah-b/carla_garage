from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from camera_interface import CameraInterface
from image_renderer import ImageRenderer
from data_extractors import TrafficDataExtractor, EgoVehicleDataExtractor, VehicleDataExtractor, VehicleGrouper
from formatters import SceneFormatter, CompactSceneFormatter

class SceneDescriptor:
    """
    Main interface for converting privileged CARLA simulator data into structured text.

    This refactored version follows single responsibility principle and separates concerns:
    - Camera management
    - Data extraction
    - Rendering
    - Formatting

    Responsibilities:
    - Coordinate between specialized components
    - Manage camera lifecycle
    - Provide unified API for scene data processing
    """

    def __init__(self, config):
        """
        Initialize SceneDescriptor with configuration.

        Args:
            config: Configuration object containing thresholds and parameters
        """
        self.config = config

        # Component initialization
        self._cameras: Dict[str, CameraInterface] = {}
        self._camera_tags: List[str] = []

        # Initialize specialized processors
        self._vehicle_extractor = VehicleDataExtractor()
        self._vehicle_grouper = VehicleGrouper()
        self._bbox_renderer = ImageRenderer(config)
        self._traffic_extractor = TrafficDataExtractor(config)
        self._ego_extractor = EgoVehicleDataExtractor()
        self._formatter = SceneFormatter(config)
        self._compact_formatter = CompactSceneFormatter(config)

    def setup_cameras(self, cameras: List[Tuple[str, Any]]) -> None:
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

    def set_camera_observations(self, images: List[Tuple[str, np.ndarray]]) -> None:
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
        ego_context: Dict[str, Any],
        agent_context: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Render vehicle bounding boxes on camera images.

        Args:
            ego_context: Dictionary containing ego vehicle context
            agent_context: Dictionary containing agent/NPC vehicle context

        Returns:
            Dictionary of camera tags mapped to rendered images
        """
        return self._bbox_renderer.render_vehicle_bounding_boxes(
            self._cameras, ego_context, agent_context
        )

    def get_structured_scene_data(
        self,
        traffic_context: Dict[str, Any],
        ego_context: Dict[str, Any],
        agent_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert privileged simulator data into structured format.

        Args:
            traffic_context: Traffic control information
            ego_context: Ego vehicle information
            agent_context: Other vehicle information

        Returns:
            Dictionary containing structured scene data
        """
        # Extract structured data using specialized extractors
        traffic_data = self._traffic_extractor.extract_traffic_data(traffic_context)
        ego_data = self._ego_extractor.extract_ego_data(ego_context)

        # Group vehicle data by lanes and directions
        grouped_vehicles = self._vehicle_grouper.group_vehicles_by_lane(
            agent_context, ego_context, self._vehicle_extractor
        )

        return {
            "traffic": traffic_data.__dict__ if traffic_data else None,
            "ego": ego_data.__dict__ if ego_data else None,
            "agent": grouped_vehicles
        }

    def format_scene_as_text(
        self,
        structured_data: Dict[str, Any],
        compact: bool = False
    ) -> str:
        """
        Convert structured scene data to formatted text representation.

        Args:
            structured_data: Dictionary containing structured scene data
            compact: Whether to use compact formatting

        Returns:
            Formatted string representation of scene data
        """
        formatter = self._compact_formatter if compact else self._formatter
        return formatter.format_scene_data(structured_data)

    def process_complete_scene(
        self,
        traffic_context: Dict[str, Any],
        ego_context: Dict[str, Any],
        agent_context: Dict[str, Any],
        format_as_text: bool = True,
        compact_format: bool = False
    ) -> Dict[str, Any]:
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
        structured_data = self.get_structured_scene_data(
            traffic_context, ego_context, agent_context
        )

        result = {"structured_data": structured_data}

        # Add formatted text if requested
        if format_as_text:
            result["formatted_text"] = self.format_scene_as_text(
                structured_data, compact=compact_format
            )

        return result

    # Convenience methods for backward compatibility and specific use cases

    def get_traffic_data(self, traffic_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract traffic data (backward compatibility method)."""
        traffic_data = self._traffic_extractor.extract_traffic_data(traffic_context)
        return traffic_data.__dict__ if traffic_data else {}

    def get_ego_data(self, ego_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract ego vehicle data (backward compatibility method)."""
        ego_data = self._ego_extractor.extract_ego_data(ego_context)
        return ego_data.__dict__ if ego_data else {}

    def get_agent_data(
        self,
        agent_context: Dict[str, Any],
        ego_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract and group agent vehicle data (backward compatibility method)."""
        return self._vehicle_grouper.group_vehicles_by_lane(
            agent_context, ego_context, self._vehicle_extractor
        )

    # Legacy method name support
    def to_formatted_string(self, structured_data: Dict[str, Any]) -> str:
        """Legacy method name for formatting (backward compatibility)."""
        return self.format_scene_as_text(structured_data, compact=False)

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
