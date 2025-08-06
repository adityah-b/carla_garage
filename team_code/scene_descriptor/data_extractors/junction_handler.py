import numpy as np
import carla

from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

JunctionPair = Tuple[carla.Waypoint, carla.Waypoint]

@dataclass(frozen=True, slots=True)
class JunctionConnection:
    # Lane waypoint connecting to junction entry
    entry_connection : carla.Waypoint

    # Waypoint at junction entry
    entry_junction : carla.Waypoint

    # Waypoint at junction exit
    exit_junction : carla.Waypoint

    # Lane waypoint connecting to junction exit
    exit_connection : carla.Waypoint

class JunctionHandler:
    """
    Identifies all junction lanelets
    """

    # Step distance from junction entry/exit waypoints
    JUNCTION_HOP_DISTANCE = 0.1

    @staticmethod
    def _step_into_junction(
        wp : carla.Waypoint,
        max_travel_distance : float,
        step_func
    ) -> Optional[JunctionPair]:
        """
        Step into a junction given a lane waypoint.
        """
        traveled_distance = 0.0
        prev_wp = wp

        while wp and (not wp.is_junction) and (traveled_distance < max_travel_distance):
            next_wps = step_func(wp)
            if not next_wps:
                return None

            prev_wp = wp
            wp = next_wps[0]
            traveled_distance += wp.transform.location.distance(prev_wp.transform.location)

        if not wp.is_junction:
            return None

        return (prev_wp, wp)

    @staticmethod
    def _step_out_of_junction(
        wp : carla.Waypoint,
        step_func
    ) -> Optional[carla.Waypoint]:
        """
        Step through junction lanelet until a connecting lanelet is found or reached dead-end
        """

        while wp and wp.is_junction:
            next_wps = step_func(wp)
            if not next_wps:
                return None
            wp = next_wps[0]

        return wp

    @staticmethod
    def get_next_junction(
        wp : carla.Waypoint,
        max_travel_distance : float = 50.0,
    ) -> Optional[JunctionPair]:
        """
        Get the next junction waypoint within max_travel_distance.
        """
        return JunctionHandler._step_into_junction(
            wp,
            max_travel_distance,
            step_func = lambda w: w.next(JunctionHandler.JUNCTION_HOP_DISTANCE)
        )

    @staticmethod
    def get_previous_junction(
        wp : carla.Waypoint,
        max_travel_distance : float = 50.0,
    ) -> Optional[JunctionPair]:
        """
        Get the previous junction waypoint within max_travel_distance.
        """
        return JunctionHandler._step_into_junction(
            wp,
            max_travel_distance,
            step_func = lambda w: w.previous(JunctionHandler.JUNCTION_HOP_DISTANCE)
        )

    @staticmethod
    def get_junction_entry_connection(wp : carla.Waypoint) -> Optional[carla.Waypoint]:
        return JunctionHandler._step_out_of_junction(
            wp,
            step_func = lambda w : w.previous(JunctionHandler.JUNCTION_HOP_DISTANCE)
        )

    @staticmethod
    def get_junction_exit_connection(wp : carla.Waypoint) -> Optional[carla.Waypoint]:
        return JunctionHandler._step_out_of_junction(
            wp,
            step_func = lambda w : w.next(JunctionHandler.JUNCTION_HOP_DISTANCE)
        )

    @staticmethod
    def create_junction_map(junction_wp : carla.Waypoint) -> Dict[tuple[int, int], Set[JunctionConnection]]:
        """
        Given a junction waypoint, build a map connecting all junction lanelets to their corresponding outer lanes
        """
        junction_connection_map = None

        if junction_wp:
            junction_obj = junction_wp.get_junction()

            # Get the junction waypoints
            # NOTE: The junction entry/exit waypoints refer only to the lanelets WITHIN the junction
            junction_wps = junction_obj.get_waypoints(carla.LaneType.Driving)

            junction_connection_map = defaultdict(set)
            for entry_wp, exit_wp in junction_wps:
                # Identify lanelets connecting to junction entry and exit points
                entry_connection_wp = entry_wp
                entry_connection_wp = JunctionHandler.get_junction_entry_connection(entry_connection_wp)

                exit_connection_wp = exit_wp
                exit_connection_wp = JunctionHandler.get_junction_exit_connection(exit_connection_wp)

                if not entry_connection_wp or not exit_connection_wp:
                    continue

                lanelet_connection = JunctionConnection(
                    entry_connection=entry_connection_wp,
                    entry_junction=entry_wp,
                    exit_junction=exit_wp,
                    exit_connection=exit_connection_wp
                )

                for wp in (entry_connection_wp, entry_wp, exit_wp, exit_connection_wp):
                    junction_connection_map[(wp.lane_id, wp.road_id)].add(lanelet_connection)

        return junction_connection_map

    @staticmethod
    def get_junction_connections(
        junction_connection_map : Dict[tuple[int, int], Set[JunctionConnection]],
        lane_wp : carla.Waypoint
    ) -> List[JunctionConnection]:
      """
      Given a waypoint, return a list of all possible junction connection lanelets
      """
      return list(junction_connection_map.get((lane_wp.lane_id, lane_wp.road_id), []))
