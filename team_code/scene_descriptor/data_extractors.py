import numpy as np
import carla

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# -------------------------------------------------------------------------------------------------- #
#  DATA CLASSES
# -------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------- #
#  Agent Context
# -------------------------------------------------------------------- #

@dataclass
class VehicleData:
    """
    Structured vehicle data.
    """
    vehicle_id: int
    speed: float
    relative_orientation: float
    relative_position: List[float]
    relative_distance: float

# -------------------------------------------------------------------- #
#  Traffic Context
# -------------------------------------------------------------------- #

@dataclass
class TrafficLightData:
    """Traffic light information."""
    id: int
    distance_to_light: float
    state: str


@dataclass
class StopSignData:
    """Stop sign information."""
    distance_to_stop_sign: float

@dataclass
class TrafficData:
    """Traffic control information."""
    next_traffic_light: Optional[TrafficLightData]
    next_stop_sign: Optional[StopSignData]
    speed_limit: float

# -------------------------------------------------------------------- #
#  Ego Context
# -------------------------------------------------------------------- #

@dataclass
class LaneChangeData:
    """Lane change information."""
    has_upcoming_lane_change: bool
    lane_change_direction: Optional[str]
    can_change_lane: bool
    available_lane_change_distance: float


@dataclass
class EgoVehicleData:
    """Ego vehicle information."""
    speed: float
    orientation: float
    position: List[float]
    lane_change: LaneChangeData


# -------------------------------------------------------------------------------------------------- #
#  DATA EXTRACTORS
# -------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------- #
#  Agent Context
# -------------------------------------------------------------------- #

class VehicleDataExtractor:
    """
    Extracts and processes vehicle data from CARLA simulation.

    Responsibilities:
    - Convert vehicle actors to structured data
    - Calculate relative positions and orientations
    - Handle coordinate transformations
    """

    # -------------------------------------------------------------------- #
    #  Utility
    # -------------------------------------------------------------------- #

    def extract_vehicle_data(
        self,
        ego_context: Dict[str, Any],
        vehicles: List[carla.Vehicle],
        closest_only: bool = False
    ) -> List[VehicleData]:
        """
        Extract structured data from vehicle actors.

        Args:
            ego_context: Dictionary containing ego vehicle context
            vehicles: List of CARLA vehicle actors
            closest_only: Whether to return only the closest vehicle

        Returns:
            List of VehicleData objects
        """
        if not vehicles:
            return []

        ego_transform_matrix = self._get_ego_transform_matrix(ego_context)
        ego_yaw = self._get_ego_yaw(ego_context)

        # Vectorized data extraction
        vehicle_data = self._extract_vectorized_data(vehicles, ego_transform_matrix, ego_yaw)

        if closest_only and vehicle_data:
            # Return only the closest vehicle
            closest_idx = np.argmin([data.relative_distance for data in vehicle_data])
            return [vehicle_data[closest_idx]]

        return vehicle_data

    # -------------------------------------------------------------------- #
    #  Private
    # -------------------------------------------------------------------- #

    def _get_ego_transform_matrix(self, ego_context: Dict[str, Any]) -> np.ndarray:
        """
        Extract ego vehicle transformation matrix.
        """
        ego_wp = ego_context['waypoint']
        return np.array(ego_wp.transform.get_matrix())

    def _get_ego_yaw(self, ego_context: Dict[str, Any]) -> float:
        """
        Extract ego vehicle yaw angle in radians.
        """
        ego_wp = ego_context['waypoint']
        return np.deg2rad(ego_wp.transform.rotation.yaw)

    def _extract_vectorized_data(
        self,
        vehicles: List[carla.Vehicle],
        ego_matrix: np.ndarray,
        ego_yaw: float
    ) -> List[VehicleData]:
        """
        Extract vehicle data using vectorized operations for efficiency.

        Args:
            vehicles: List of CARLA vehicle actors
            ego_matrix: Ego vehicle transformation matrix
            ego_yaw: Ego vehicle yaw angle

        Returns:
            List of VehicleData objects
        """
        # Extract all data in vectorized form
        transform_matrices = np.array([v.get_transform().get_matrix() for v in vehicles])
        yaws = np.array([np.deg2rad(v.get_transform().rotation.yaw) for v in vehicles])
        speeds = np.array([v.get_velocity().length() for v in vehicles])
        vehicle_ids = np.array([v.id for v in vehicles])

        # Calculate relative positions
        relative_positions = self._calculate_relative_positions(transform_matrices, ego_matrix)

        # Calculate relative yaws
        relative_yaws = self._normalize_angles(yaws - ego_yaw)

        # Calculate distances
        relative_distances = np.linalg.norm(relative_positions, axis=1)

        # Create VehicleData objects
        vehicle_data = []
        for i in range(len(vehicles)):
            data = VehicleData(
                vehicle_id=int(vehicle_ids[i]),
                speed=round(float(speeds[i]), 2),
                relative_orientation=round(float(relative_yaws[i]), 2),
                relative_position=relative_positions[i][:2].round(2).tolist(),
                relative_distance=round(float(relative_distances[i]), 2)
            )
            vehicle_data.append(data)

        return vehicle_data

    def _calculate_relative_positions(
        self,
        vehicle_matrices: np.ndarray,
        ego_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate relative positions of vehicles with respect to ego vehicle.

        Args:
            vehicle_matrices: Nx4x4 array of vehicle transformation matrices
            ego_matrix: 4x4 ego vehicle transformation matrix

        Returns:
            Nx3 array of relative positions
        """
        # Get positions from transformation matrices
        vehicle_positions = vehicle_matrices[:, :3, 3]
        ego_position = ego_matrix[:3, 3]

        # Calculate relative positions in world coordinates
        relative_world = vehicle_positions - ego_position[np.newaxis, :]

        # Transform to ego vehicle coordinate system
        ego_rotation = ego_matrix[:3, :3]
        relative_ego = (ego_rotation.T @ relative_world.T).T

        return relative_ego

    def _normalize_angles(self, angles: np.ndarray) -> np.ndarray:
        """Normalize angles to [-π, π] range."""
        return (angles + np.pi) % (2 * np.pi) - np.pi

# -------------------------------------------------------------------- #
#  Traffic Context
# -------------------------------------------------------------------- #

class TrafficDataExtractor:
    """
    Extracts traffic control information from simulation context.

    Responsibilities:
    - Extract traffic light states and distances
    - Extract stop sign information
    - Handle distance thresholds for relevance
    """

    def __init__(self, config):
        """Initialize with configuration parameters."""
        self.config = config

    # -------------------------------------------------------------------- #
    #  Utility
    # -------------------------------------------------------------------- #

    def extract_traffic_data(self, traffic_context: Dict[str, Any]) -> TrafficData:
        """
        Extract structured traffic data from context.

        Args:
            traffic_context: Dictionary containing traffic information

        Returns:
            TrafficData object with structured information
        """
        return TrafficData(
            next_traffic_light=self._extract_traffic_light_data(
                traffic_context["next_traffic_light"],
                traffic_context["distance_to_next_traffic_light"]
            ),
            next_stop_sign=self._extract_stop_sign_data(
                traffic_context["next_stop_sign"],
                traffic_context["distance_to_next_stop_sign"]
            ),
            speed_limit=traffic_context["speed_limit"]
        )

    # -------------------------------------------------------------------- #
    #  Private
    # -------------------------------------------------------------------- #

    def _extract_traffic_light_data(
        self,
        traffic_light,
        distance_to_light: float
    ) -> Optional[TrafficLightData]:
        """
        Extract traffic light data if within relevant distance.

        Args:
            traffic_light: CARLA traffic light actor
            distance_to_light: Distance to the traffic light

        Returns:
            TrafficLightData object or None if not relevant
        """
        if not traffic_light or distance_to_light >= self.config.traffic_light_distance_threshold:
            return None

        state_mapping = {
            carla.TrafficLightState.Red: "RED",
            carla.TrafficLightState.Yellow: "YELLOW",
            carla.TrafficLightState.Green: "GREEN"
        }

        light_state = state_mapping.get(traffic_light.get_state(), "UNKNOWN")

        return TrafficLightData(
            id=traffic_light.id,
            distance_to_light=distance_to_light,
            state=light_state
        )

    def _extract_stop_sign_data(
        self,
        stop_sign,
        distance_to_stop_sign: float
    ) -> Optional[StopSignData]:
        """
        Extract stop sign data if within relevant distance.

        Args:
            stop_sign: CARLA stop sign actor
            distance_to_stop_sign: Distance to the stop sign

        Returns:
            StopSignData object or None if not relevant
        """
        if not stop_sign or distance_to_stop_sign >= self.config.stop_sign_distance_threshold:
            return None

        return StopSignData(distance_to_stop_sign=distance_to_stop_sign)

# -------------------------------------------------------------------- #
#  Ego Context
# -------------------------------------------------------------------- #

class EgoVehicleDataExtractor:
    """
    Extracts ego vehicle information from simulation context.

    Responsibilities:
    - Extract ego vehicle state (speed, position, orientation)
    - Process lane change information
    - Handle route and waypoint data
    """

    # -------------------------------------------------------------------- #
    #  Utility
    # -------------------------------------------------------------------- #

    def extract_ego_data(self, ego_context: Dict[str, Any]) -> EgoVehicleData:
        """
        Extract structured ego vehicle data from context.

        Args:
            ego_context: Dictionary containing ego vehicle information

        Returns:
            EgoVehicleData object with structured information
        """
        return EgoVehicleData(
            speed=ego_context["speed"],
            orientation=ego_context["compass"],
            position=ego_context["gps"][:2].tolist(),
            lane_change=self._extract_lane_change_data(ego_context)
        )

    # -------------------------------------------------------------------- #
    #  Private
    # -------------------------------------------------------------------- #

    def _extract_lane_change_data(self, ego_context: Dict[str, Any]) -> LaneChangeData:
        """
        Extract and process lane change information.

        Args:
            ego_context: Dictionary containing ego vehicle context

        Returns:
            LaneChangeData object with processed lane change information
        """
        lane_change_raw = ego_context.get("lane_change")

        if not lane_change_raw:
            return self._default_lane_change_data()

        has_lane_change = lane_change_raw.get("has_lane_change", False)

        if not has_lane_change:
            return self._default_lane_change_data()

        return self._process_active_lane_change(ego_context, lane_change_raw)

    def _default_lane_change_data(self) -> LaneChangeData:
        """Return default lane change data when no lane change is active."""
        return LaneChangeData(
            has_upcoming_lane_change=False,
            lane_change_direction=None,
            can_change_lane=False,
            available_lane_change_distance=-1.0
        )

    def _process_active_lane_change(
        self,
        ego_context: Dict[str, Any],
        lane_change_data: Dict[str, Any]
    ) -> LaneChangeData:
        """
        Process active lane change scenario.

        Args:
            ego_context: Ego vehicle context
            lane_change_data: Raw lane change data

        Returns:
            Processed LaneChangeData object
        """
        direction = lane_change_data["lane_change_direction"]
        early_start_point = lane_change_data["lane_change_early_start_point"]
        late_start_point = lane_change_data["lane_change_late_start_point"]

        ego_wp = ego_context["route"][0]
        ego_transform = ego_wp.transform

        # Check if ego has passed the early start point
        has_passed_start = self._has_passed_waypoint(ego_transform, early_start_point)

        # Determine if lane change is possible
        can_change, available_distance = self._evaluate_lane_change_feasibility(
            ego_wp, ego_transform, direction, late_start_point, has_passed_start
        )

        return LaneChangeData(
            has_upcoming_lane_change=True,
            lane_change_direction=direction,
            can_change_lane=can_change,
            available_lane_change_distance=available_distance
        )

    def _has_passed_waypoint(self, ego_transform, waypoint) -> bool:
        """
        Check if ego vehicle has passed a given waypoint.

        Args:
            ego_transform: Ego vehicle transform
            waypoint: Target waypoint to check

        Returns:
            True if ego has passed the waypoint
        """
        ego_heading = ego_transform.get_forward_vector()
        waypoint_vector = waypoint.transform.location - ego_transform.location
        return ego_heading.dot(waypoint_vector) < 0

    def _evaluate_lane_change_feasibility(
        self,
        ego_wp,
        ego_transform,
        direction: str,
        late_start_point,
        has_passed_start: bool
    ) -> tuple[bool, float]:
        """
        Evaluate if lane change is feasible and calculate available distance.

        Args:
            ego_wp: Ego waypoint
            ego_transform: Ego transform
            direction: Lane change direction ("left" or "right")
            late_start_point: Latest point to start lane change
            has_passed_start: Whether ego has passed early start point

        Returns:
            Tuple of (can_change_lane, available_distance)
        """
        if not has_passed_start:
            return False, -1.0

        # Get target lane
        target_lane = (ego_wp.get_left_lane() if direction == "left"
                      else ego_wp.get_right_lane())

        if not target_lane:
            return False, -1.0

        # Calculate available distance
        available_distance = late_start_point.transform.location.distance(
            ego_transform.location
        )

        # Lane change is feasible if there's sufficient distance
        can_change = available_distance > 0.5

        return can_change, available_distance


class VehicleGrouper:
    """
    Groups vehicles by lane and traffic direction.

    Responsibilities:
    - Categorize vehicles by lane relative to ego
    - Group by traffic direction (ongoing, oncoming, cross)
    - Generate lane labels
    """

    def group_vehicles_by_lane(
        self,
        agent_context: Dict[str, Any],
        ego_context: Dict[str, Any],
        extractor: VehicleDataExtractor
    ) -> Dict[str, Dict[str, Any]]:
        """
        Group vehicles by lane and direction relative to ego vehicle.

        Args:
            agent_context: Agent context data
            ego_context: Ego vehicle context data
            extractor: Vehicle data extractor instance

        Returns:
            Dictionary of grouped vehicle data
        """
        grouped_vehicles = {
            "Ongoing Traffic": {},
            "Oncoming Traffic": {},
            "Cross Traffic": {}
        }

        ego_wp = ego_context['waypoint']
        ego_lane_id = ego_wp.lane_id

        # Process ongoing traffic
        self._process_traffic_direction(
            grouped_vehicles["Ongoing Traffic"],
            agent_context["ongoing_leading_vehicles"],
            agent_context.get("ongoing_trailing_vehicles", {}),
            ego_context,
            ego_lane_id,
            extractor,
            closest_only=True
        )

        # Process oncoming traffic
        self._process_traffic_direction(
            grouped_vehicles["Oncoming Traffic"],
            agent_context["oncoming_leading_vehicles"],
            agent_context.get("oncoming_trailing_vehicles", {}),
            ego_context,
            ego_lane_id,
            extractor,
            closest_only=False
        )

        return grouped_vehicles

    def _process_traffic_direction(
        self,
        traffic_dict: Dict[str, Any],
        leading_vehicles: Dict[int, List],
        trailing_vehicles: Dict[int, List],
        ego_context: Dict[str, Any],
        ego_lane_id: int,
        extractor: VehicleDataExtractor,
        closest_only: bool
    ) -> None:
        """Process vehicles in a specific traffic direction."""
        ego_wp = ego_context['waypoint']

        for lane_id, lane_vehicles in leading_vehicles.items():
            lane_key = self._generate_lane_key(lane_id, ego_lane_id, ego_wp)

            leading_data = extractor.extract_vehicle_data(
                ego_context, lane_vehicles, closest_only
            )
            trailing_data = []

            if lane_id in trailing_vehicles:
                trailing_data = extractor.extract_vehicle_data(
                    ego_context, trailing_vehicles[lane_id], closest_only
                )

            if leading_data or trailing_data:
                traffic_dict[lane_key] = {
                    "leading_vehicles": [data.__dict__ for data in leading_data],
                    "trailing_vehicles": [data.__dict__ for data in trailing_data],
                }

    def _generate_lane_key(self, lane_id: int, ego_lane_id: int, ego_wp) -> str:
        """Generate descriptive key for lane relative to ego."""
        if lane_id == ego_lane_id:
            return "Ego Lane"

        # Check left lanes
        left_wp = ego_wp.get_left_lane()
        if left_wp and self._is_lane_in_direction(lane_id, ego_lane_id, left_wp.lane_id):
            offset = abs(lane_id - ego_lane_id)
            return f"Left-{offset} Lane"

        # Check right lanes
        right_wp = ego_wp.get_right_lane()
        if right_wp and self._is_lane_in_direction(lane_id, ego_lane_id, right_wp.lane_id):
            offset = abs(lane_id - ego_lane_id)
            return f"Right-{offset} Lane"

        # Other lanes
        offset = abs(lane_id - ego_lane_id)
        return f"Other-{offset}"

    def _is_lane_in_direction(self, lane_id: int, ego_lane_id: int, direction_lane_id: int) -> bool:
        """Check if lane_id is in the direction of direction_lane_id from ego_lane_id."""
        if direction_lane_id < ego_lane_id:
            return lane_id <= direction_lane_id
        else:
            return lane_id >= direction_lane_id
