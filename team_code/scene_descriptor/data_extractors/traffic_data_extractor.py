import numpy as np
import carla

from typing import Optional, Set
from dataclasses import dataclass

from privileged_route_planner import PlannerState

@dataclass(frozen=True, slots=True)
class TrafficLightData:
    """Traffic light information."""
    id: int
    distance_to_light: float
    state: str


@dataclass(frozen=True, slots=True)
class StopSignData:
    """Stop sign information."""
    id : int
    distance_to_stop_sign: float
    cleared : bool

@dataclass(frozen=True, slots=True)
class TrafficData:
    """Traffic control information."""
    next_traffic_light: Optional[TrafficLightData]
    next_stop_sign: Optional[StopSignData]
    speed_limit: float


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

    def extract_traffic_data(
        self,
        ego_transform : carla.Transform,
        planner_state: PlannerState
    ) -> TrafficData:
        """
        Extract structured traffic data from context.

        Returns:
            TrafficData object with structured information
        """
        # Unpack planner snapshot
        route_index = planner_state.route_index
        next_tl = planner_state.next_traffic_lights[route_index]
        dist_to_next_tl = planner_state.dist_to_next_traffic_lights[route_index]
        next_ss = planner_state.next_stop_signs[route_index]
        dist_to_next_ss = planner_state.dist_to_next_stop_signs[route_index]
        speed_limit = round(planner_state.speed_limits[route_index], 2)
        cleared_stop_sign_ids = planner_state.cleared_stop_sign_ids

        # Get ego location
        ego_loc = ego_transform.location

        return TrafficData(
            next_traffic_light=self._extract_traffic_light_data(
                traffic_light=next_tl,
                distance_to_light=dist_to_next_tl
            ),
            next_stop_sign=self._extract_stop_sign_data(
                stop_sign=next_ss,
                ego_loc=ego_loc,
                dist_to_next_ss=dist_to_next_ss,
                cleared_stop_sign_ids=cleared_stop_sign_ids
            ),
            speed_limit=speed_limit
        )

    # -------------------------------------------------------------------- #
    #  Private
    # -------------------------------------------------------------------- #

    def _extract_traffic_light_data(
        self,
        traffic_light : carla.TrafficLight,
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
        if not traffic_light or distance_to_light >= self.config.traffic_light_detection_radius_m:
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
        ego_loc: carla.Location,
        dist_to_next_ss : float,
        cleared_stop_sign_ids : Set[int],
    ) -> Optional[StopSignData]:
        """
        Extract stop sign data if within relevant distance.

        Args:
            stop_sign: CARLA stop sign actor
            distance_to_stop_sign: Distance to the stop sign

        Returns:
            StopSignData object or None if not relevant
        """
        if not stop_sign:
            return None

        distance_to_stop_sign = stop_sign.get_transform().transform(stop_sign.trigger_volume.location).distance(ego_loc)
        if distance_to_stop_sign <= self.config.clearing_distance_to_stop_sign:
            distance_to_stop_sign = dist_to_next_ss

        if distance_to_stop_sign >= self.config.stop_sign_detection_radius_m:
            return None

        ss_cleared = stop_sign.id in cleared_stop_sign_ids

        return StopSignData(
            id=stop_sign.id,
            distance_to_stop_sign=distance_to_stop_sign,
            cleared=ss_cleared
        )
