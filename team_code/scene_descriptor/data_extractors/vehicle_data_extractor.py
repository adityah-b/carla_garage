import numpy as np
import carla

from typing import List, Dict, Any, Optional, Tuple, Literal, Set
from dataclasses import dataclass, field

from .base_actor_extractor import BaseActorExtractor
from .road_handler import RoadHandler, Lanelet, TrafficGroup
from privileged_route_planner import PlannerState

TrafficType = Literal["leading", "trailing", "oncoming", "crossing"]
VehicleType = Literal["vehicle", "cyclist", "emergency"]

@dataclass
class VehicleDataEntry:
    # Vehicle context
    vehicle : carla.Vehicle
    vehicle_type : VehicleType

    # Traffic context
    traffic_type : TrafficType
    lanelet_id : int
    lanelet : Lanelet
    lane_name : str
    lanelet_route_idx : int
    vehicle_route_points : np.ndarray

    # Vehicle data
    x : float
    y : float
    z : float
    heading : float
    speed: float
    relative_orientation: float
    relative_position: Tuple[float]
    relative_distance: float
    intrusion_idx : int
    intrudes_ego : bool
    emergency_sirens_active : bool

    # Vehicle control
    throttle : float
    steer : float
    brake : float

    @property
    def id(self) -> int:
        return self.vehicle.id

@dataclass
class VehicleData:
    all_vehicles_flat : List[VehicleDataEntry]
    all_vehicles_grouped : Dict[TrafficType, List[VehicleDataEntry]]
    _view : Optional[Dict[TrafficType, Dict[str, List[VehicleDataEntry]]]] = field(default=None, init=False)

    @property
    def view(self) -> Dict[TrafficType, Dict[str, List[VehicleDataEntry]]]:
        if self._view is None:
            self._view : Dict[TrafficType, Dict[str, List[VehicleDataEntry]]] = {}
            for traffic_type, all_vehicle_data in self.all_vehicles_grouped.items():
                lane_grouping : Dict[str, List[VehicleDataEntry]] = {}

                for v_entry in all_vehicle_data:
                    lane_name = v_entry.lane_name
                    if lane_name not in lane_grouping:
                        lane_grouping[lane_name] = []

                    lane_grouping[lane_name].append(v_entry)

                self._view[traffic_type] = lane_grouping

        return self._view

    def get(
        self,
        *,
        traffic_type : Optional[TrafficType] = None,
        lane_name : Optional[str] = None,
        vehicle_types : Optional[Set[VehicleType]] = None,
        vehicle_ids : Optional[Set[int]] = None,
        get_intruders : bool = False,
    ) -> List[VehicleDataEntry]:
        out = []
        if traffic_type is not None and lane_name is not None:
            out = self.view.get(traffic_type, {}).get(lane_name, [])
        elif traffic_type is not None:
            out = self.all_vehicles_grouped.get(traffic_type, [])
        elif lane_name is not None:
            for t_type in self.view.keys():
                out.extend(self.view[t_type].get(lane_name, []))
        else:
            out = self.all_vehicles_flat

        if vehicle_types is not None:
            out = [v for v in out if v.vehicle_type in vehicle_types]

        if vehicle_ids is not None:
            out = [v for v in out if v.id in vehicle_ids]

        if get_intruders:
            out = [v for v in out if v.intrudes_ego]

        return out

class VehicleDataExtractor(BaseActorExtractor):
    """
    Extracts and processes vehicle data from CARLA simulation.

    Responsibilities:
    - Convert vehicle actors to structured data
    - Calculate relative positions and orientations
    - Handle coordinate transformations
    """

    def __init__(self, config, carla_map : carla.Map):
        super().__init__(config)
        self.carla_map = carla_map
        self.road_handler = RoadHandler(config, carla_map)

    # def extract_vehicle_data(
    #     self,
    #     ego_transform : carla.Transform,
    #     ego_wp : carla.Waypoint,
    #     planner_state : PlannerState,
    #     vehicles : List[carla.Vehicle],
    # ) -> AllVehicleData:
    #     # Global route intruding vehicles list
    #     all_intruding_vehicles : Dict[int, int] = {}

    #     leading_vehicles_group = self.road_handler.get_leading_vehicles(ego_transform, ego_wp, planner_state, vehicles)
    #     trailing_vehicles_group = self.road_handler.get_trailing_vehicles(ego_transform, ego_wp, planner_state, vehicles)
    #     oncoming_vehicles_group = self.road_handler.get_oncoming_vehicles(ego_transform, ego_wp, planner_state, vehicles)
    #     cross_vehicles_group = self.road_handler.get_cross_vehicles(ego_transform, ego_wp, planner_state, vehicles)

    #     # Group all traffic types
    #     grouped_leading, _ = self._group_vehicles(
    #         ego_transform=ego_transform,
    #         vehicles_dict=leading_vehicles_group,
    #         traffic_type="leading",
    #         planner_state=planner_state,
    #     )
    #     grouped_trailing, _ = self._group_vehicles(
    #         ego_transform=ego_transform,
    #         vehicles_dict=trailing_vehicles_group,
    #         traffic_type="trailing",
    #         planner_state=planner_state,
    #     )
    #     grouped_oncoming, oncoming_intruders = self._group_vehicles(
    #         ego_transform=ego_transform,
    #         vehicles_dict=oncoming_vehicles_group,
    #         traffic_type="oncoming",
    #         planner_state=planner_state,
    #     )
    #     grouped_crossing, _ = self._group_vehicles(
    #         ego_transform=ego_transform,
    #         vehicles_dict=cross_vehicles_group,
    #         traffic_type="crossing",
    #         planner_state=planner_state,
    #     )

    #     # Accumulate all intruding vehicles
    #     print(f'\n\nONCOMING INTRUDERS: {len(oncoming_intruders)}')
    #     print(f'\tACTORS')
    #     for actor_id in oncoming_intruders.keys():
    #         print(f'\t\tID: {actor_id}')
    #     all_intruding_vehicles.update(oncoming_intruders)

    #     all_vehicles_grouped = {
    #         "leading" : grouped_leading,
    #         "trailing" : grouped_trailing,
    #         "oncoming" : grouped_oncoming,
    #         "cross" : grouped_crossing,
    #     }

    #     all_vehicle_data = AllVehicleData(
    #         all_vehicles_grouped=all_vehicles_grouped,
    #         all_intruding_vehicles=all_intruding_vehicles
    #     )

    #     return all_vehicle_data

    def extract_data(
        self,
        ego_transform : carla.Transform,
        ego_wp : carla.Waypoint,
        planner_state : PlannerState,
        vehicles : List[carla.Vehicle],
    ) -> VehicleData:
        leading_vehicles_group = self.road_handler.get_leading_vehicles(ego_transform, ego_wp, planner_state, vehicles)
        trailing_vehicles_group = self.road_handler.get_trailing_vehicles(ego_transform, ego_wp, planner_state, vehicles)
        oncoming_vehicles_group = self.road_handler.get_oncoming_vehicles(ego_transform, ego_wp, planner_state, vehicles)
        crossing_vehicles_group = self.road_handler.get_cross_vehicles(ego_transform, ego_wp, planner_state, vehicles)

        # Extract vehicle data for all groups
        # TODO: UPDATE API TO USE VEHICLE DICT INSTEAD OF LIST
        all_vehicles_dict = {v.id : v for v in vehicles}

        leading_group_data = self._extract_data(
            ego_transform=ego_transform,
            planner_state=planner_state,
            all_vehicles_dict=all_vehicles_dict,
            vehicles_group=leading_vehicles_group,
            check_intruders=True,
        )

        trailing_group_data = self._extract_data(
            ego_transform=ego_transform,
            planner_state=planner_state,
            all_vehicles_dict=all_vehicles_dict,
            vehicles_group=trailing_vehicles_group,
            check_intruders=False,
        )

        oncoming_group_data = self._extract_data(
            ego_transform=ego_transform,
            planner_state=planner_state,
            all_vehicles_dict=all_vehicles_dict,
            vehicles_group=oncoming_vehicles_group,
            check_intruders=True,
        )

        crossing_group_data = self._extract_data(
            ego_transform=ego_transform,
            planner_state=planner_state,
            all_vehicles_dict=all_vehicles_dict,
            vehicles_group=crossing_vehicles_group,
            check_intruders=False,
        )

        all_vehicles_flat = leading_group_data + trailing_group_data + oncoming_group_data + crossing_group_data
        all_vehicles_grouped = {
            "leading" : leading_group_data,
            "trailing" : trailing_group_data,
            "oncoming" : oncoming_group_data,
            "crossing" : crossing_group_data,
        }

        vehicle_data = VehicleData(
            all_vehicles_flat=all_vehicles_flat,
            all_vehicles_grouped=all_vehicles_grouped
        )

        return vehicle_data

    # -------------------------------------------------------------------- #
    #  Utility
    # -------------------------------------------------------------------- #

    def _extract_data(
        self,
        ego_transform : carla.Transform,
        planner_state : PlannerState,
        all_vehicles_dict : Dict[int, carla.Vehicle],
        vehicles_group : TrafficGroup,
        check_intruders : bool = False,
    ) -> List[VehicleDataEntry]:
        """
        Extract vehicle data and group them based on their lane info
        """
        vehicles = [all_vehicles_dict[v_id] for v_id in vehicles_group.vehicle_to_lanelet_id.keys()]
        if not vehicles:
            return []

        ego_transform_matrix = self._get_ego_transform_matrix(ego_transform)
        ego_yaw = self._get_ego_yaw(ego_transform)

        return self._extract_vehicle_data(
            ego_matrix=ego_transform_matrix,
            ego_yaw=ego_yaw,
            planner_state=planner_state,
            vehicles=vehicles,
            vehicles_group=vehicles_group,
            check_intruders=check_intruders,
        )

    # def _group_vehicles(
    #     self,
    #     ego_transform : carla.Transform,
    #     vehicles_dict : Dict[str, List[LaneVehicles]],
    #     traffic_type : str,
    #     planner_state : PlannerState,
    # ) -> Tuple[Dict[str, List[LaneVehicleData]], Dict[int, int]]:
    #     """
    #     Extract vehicle data and group them based on their lane info
    #     """
    #     all_intruding_vehicles = {}
    #     grouped_data = {}
    #     for lane_name, lane_vehicles_list in vehicles_dict.items():
    #         veh_data_list : List[LaneVehicleData] = []
    #         for lv in lane_vehicles_list:
    #             ll = lv.lanelet
    #             vehicles = lv.vehicles

    #             intruding_vehicles = {}
    #             if traffic_type == "oncoming":
    #                 intruding_vehicles = self._check_lane_intrusions(
    #                     vehicles=vehicles,
    #                     planner_state=planner_state,
    #                     traffic_type=traffic_type
    #                 )
    #                 all_intruding_vehicles.update(intruding_vehicles)

    #             veh_data, cyclist_data, emergency_vehicle_data = self._extract_vehicle_data(
    #                 ego_transform=ego_transform,
    #                 vehicles=vehicles,
    #                 planner_state=planner_state,
    #                 traffic_type=traffic_type,
    #                 intruding_vehicles=intruding_vehicles,
    #             )

    #             lv_data = LaneVehicleData(
    #                 lanelet=ll,
    #                 vehicle_data=veh_data,
    #                 cyclist_data=cyclist_data,
    #                 emergency_vehicle_data=emergency_vehicle_data
    #             )

    #             veh_data_list.append(lv_data)

    #         grouped_data[lane_name] = veh_data_list

    #     return grouped_data, all_intruding_vehicles

    # def _extract_vehicle_data(
    #     self,
    #     ego_transform: carla.Transform,
    #     vehicles: List[carla.Vehicle],
    #     traffic_type : str,
    #     planner_state : PlannerState,
    #     intruding_vehicles : Dict[int, int],
    #     closest_only: bool = False,
    # ) -> Tuple[List[VehicleData], List[VehicleData], List[VehicleData]]:
    #     if not vehicles:
    #         return []

    #     ego_transform_matrix = self._get_ego_transform_matrix(ego_transform)
    #     ego_yaw = self._get_ego_yaw(ego_transform)

    #     # Vectorized data extraction
    #     vehicle_data, cyclist_data, emergency_vehicle_data = self._extract_vectorized_data(
    #         vehicles=vehicles,
    #         ego_matrix=ego_transform_matrix,
    #         ego_yaw=ego_yaw,
    #         planner_state=planner_state,
    #         traffic_type=traffic_type,
    #         intruding_vehicles=intruding_vehicles,
    #     )

    #     return vehicle_data, cyclist_data, emergency_vehicle_data

    # -------------------------------------------------------------------- #
    #  Private
    # -------------------------------------------------------------------- #

    # def _extract_vectorized_data(
    #     self,
    #     vehicles: List[carla.Vehicle],
    #     ego_matrix: np.ndarray,
    #     ego_yaw: float,
    #     traffic_type : str,
    #     planner_state : PlannerState,
    #     intruding_vehicles : Dict[int, int],
    # ) -> Tuple[List[VehicleData], List[VehicleData], List[VehicleData]]:
    #     """
    #     Extract vehicle data using vectorized operations for efficiency.

    #     Args:
    #         vehicles: List of CARLA vehicle actors
    #         ego_matrix: Ego vehicle transformation matrix
    #         ego_yaw: Ego vehicle yaw angle

    #     Returns:
    #         List of VehicleData objects
    #     """
    #     # Extract all data in vectorized form
    #     transform_matrices = np.array([v.get_transform().get_matrix() for v in vehicles])
    #     yaws = np.array([np.deg2rad(v.get_transform().rotation.yaw) for v in vehicles])
    #     speeds = np.array([v.get_velocity().length() for v in vehicles])

    #     # Calculate relative positions
    #     relative_positions = self._calculate_relative_positions(transform_matrices, ego_matrix)

    #     # Calculate relative yaws
    #     relative_yaws = self._normalize_angles(yaws - ego_yaw)

    #     # Calculate distances
    #     relative_distances = np.linalg.norm(relative_positions, axis=1)

    #     # Create VehicleData objects
    #     vehicle_data = []
    #     cyclist_data = []
    #     emergency_vehicle_data = []

    #     for i in range(len(vehicles)):
    #         vehicle = vehicles[i]
    #         veh_id = vehicle.id

    #         rel_pos = relative_positions[i][:2].round(2)
    #         intrusion_index = intruding_vehicles.get(veh_id, -1)

    #         veh_type = vehicle.attributes.get("base_type", "")
    #         special_type = vehicle.attributes.get("special_type", "")

    #         data = VehicleData(
    #             vehicle=vehicle,
    #             id=veh_id,
    #             speed=round(float(speeds[i]), 2),
    #             relative_orientation=round(float(relative_yaws[i]), 2),
    #             relative_position=tuple(rel_pos),
    #             relative_distance=round(float(relative_distances[i]), 2),
    #             intrusion_index=intrusion_index,
    #             emergency_sirens_active=False,
    #         )

    #         if veh_type == "bicycle":
    #             cyclist_data.append(data)
    #         elif special_type == "emergency":
    #             if vehicle.get_light_state() == carla.VehicleLightState.Special1 | carla.VehicleLightState.Special2:
    #                 data.emergency_sirens_active = True
    #             emergency_vehicle_data.append(data)

    #         # TODO: FIX THIS, WE'RE GOING WITH SEPARATE ENTRIES FOR VEHICLE TYPES
    #         vehicle_data.append(data)

    #     vehicle_data.sort(key=lambda v: (v.relative_distance))
    #     cyclist_data.sort(key=lambda v: (v.relative_distance))
    #     emergency_vehicle_data.sort(key=lambda v: (v.relative_distance))

    #     return vehicle_data, cyclist_data, emergency_vehicle_data

    def _extract_vehicle_data(
        self,
        ego_matrix: np.ndarray,
        ego_yaw: float,
        planner_state : PlannerState,
        vehicles: List[carla.Vehicle],
        vehicles_group : TrafficGroup,
        check_intruders : bool = False,
    ) -> List[VehicleDataEntry]:
        """
        Extract vehicle data using vectorized operations for efficiency.

        Args:
            vehicles: List of CARLA vehicle actors
            ego_matrix: Ego vehicle transformation matrix
            ego_yaw: Ego vehicle yaw angle

        Returns:
            List of VehicleDataEntry objects
        """
        # Extract all data in vectorized form
        transform_matrices = np.array([v.get_transform().get_matrix() for v in vehicles])
        yaws = np.array([np.deg2rad(v.get_transform().rotation.yaw) for v in vehicles])
        speeds = np.array([v.get_velocity().length() for v in vehicles])

        # Calculate relative positions
        relative_positions = self._calculate_relative_positions(transform_matrices, ego_matrix)

        # Calculate relative yaws
        relative_yaws = self._normalize_angles(yaws - ego_yaw)

        # Calculate distances
        relative_distances = np.linalg.norm(relative_positions, axis=1)

        # Find intruders if requested
        intruders = {}
        if check_intruders:
            intruders = self._check_lane_intrusions(vehicles=vehicles, planner_state=planner_state)

        # Create VehicleDataEntry objects
        all_vehicle_data = []

        for i in range(len(vehicles)):
            vehicle = vehicles[i]
            veh_id = vehicle.id

            vehicle_loc = vehicle.get_location()
            vehicle_wp = self.carla_map.get_waypoint(vehicle_loc)
            vehicle_control = vehicle.get_control()
            vehicle_heading = yaws[i]

            rel_pos = relative_positions[i][:2].round(2)

            is_cyclist = vehicle.attributes.get("base_type", "") == "bicycle"
            is_emergency_vehicle = vehicle.attributes.get("special_type", "") == "emergency"
            emergency_sirens_active = vehicle.get_light_state() == carla.VehicleLightState.Special1 | carla.VehicleLightState.Special2

            vehicle_type = "vehicle"
            if is_cyclist:
                vehicle_type = "cyclist"
            elif is_emergency_vehicle:
                vehicle_type = "emergency"

            lanelet_id = vehicles_group.vehicle_to_lanelet_id[veh_id]
            lanelet = vehicles_group.lanelet_by_id[lanelet_id]
            lanelet_route_idx = lanelet.find(vehicle_wp)

            intrusion_idx = intruders.get(veh_id)
            intrudes_ego = intrusion_idx is not None

            data = VehicleDataEntry(
                # Vehicle context
                vehicle=vehicle,
                vehicle_type=vehicle_type,

                # Traffic context
                traffic_type=vehicles_group.traffic_type,
                lanelet_id=lanelet_id,
                lanelet=lanelet,
                lane_name=vehicles_group.lane_name_by_lanelet_id[lanelet_id],
                lanelet_route_idx=lanelet_route_idx,
                vehicle_route_points=lanelet.dense_points[lanelet_route_idx:].copy(),

                # Vehicle data
                x=vehicle_loc.x,
                y=vehicle_loc.y,
                z=vehicle_loc.z,
                heading=vehicle_heading,
                speed=round(float(speeds[i]), 2),
                relative_orientation=round(float(relative_yaws[i]), 2),
                relative_position=tuple(rel_pos),
                relative_distance=round(float(relative_distances[i]), 2),
                intrudes_ego=intrudes_ego,
                intrusion_idx=intrusion_idx,
                emergency_sirens_active=emergency_sirens_active,

                # Vehicle control
                throttle=vehicle_control.throttle,
                steer=vehicle_control.steer,
                brake=vehicle_control.brake,
            )
            all_vehicle_data.append(data)

        all_vehicle_data.sort(key=lambda v: (v.relative_distance))

        return all_vehicle_data
