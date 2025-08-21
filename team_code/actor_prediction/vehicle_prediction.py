import carla
import numpy as np

from typing import List, Dict
from dataclasses import dataclass
from agents.navigation.local_planner import RoadOption

from privileged_route_planner import PrivilegedRoutePlanner
from scene_descriptor.data_extractors.vehicle_data_extractor import LaneVehicleData
from .trajectory_forecasting import VehicleForecaster

@dataclass(frozen=True, slots=True)
class VehicleIntent:
    INTENT_MAPPING = {
        "follow_lane" : RoadOption.LANEFOLLOW,
        "change_lane_left" : RoadOption.CHANGELANELEFT,
        "change_lane_right" : RoadOption.CHANGELANERIGHT,
        "turn_left" : RoadOption.LEFT,
        "turn_right" : RoadOption.RIGHT,
    }

class VehiclePrediction:
    def __init__(self, config, carla_map : carla.Map):
        self.config = config
        self.world_map = carla_map

        self.forecaster = VehicleForecaster(self.config)

        # Dummy waypoint planner
        self.waypoint_planner = PrivilegedRoutePlanner(self.config)

    def _flatten_grouped_vehicles(
        self,
        grouped_vehicles: Dict[str, Dict[str, List[LaneVehicleData]]]
    ) -> List[LaneVehicleData]:
        return [lv
                for lanes in grouped_vehicles.values()
                for lv_list in lanes.values()
                for lv in lv_list]


    def _predict_vehicle_waypoints(
        self,
        vehicle_traffic : Dict[str, Dict[str, List[LaneVehicleData]]]
    ):
        # Flatten grouped vehicle traffic
        lane_vehicles = self._flatten_grouped_vehicles(vehicle_traffic)

        predicted_waypoints = {}

        for lv_data in lane_vehicles:
            ll = lv_data.lanelet

            for v_data in lv_data.vehicle_data:
                vehicle = v_data.vehicle

                vehicle_wp = self.world_map.get_waypoint(vehicle.get_location())
                start_idx = ll.find(vehicle_wp)

                predicted_waypoints[vehicle] = [vehicle_wp] + ll.dense_points[start_idx:]

        return predicted_waypoints

    def predict_vehicle_motion(
        self,
        vehicle_traffic : Dict[str, Dict[str, List[LaneVehicleData]]]
    ):
        forecasted_vehicle_bbs = {}
        forecast_length = self.config.forecast_length_lane_change
        num_future_frames = int(self.config.bicycle_frame_rate * forecast_length)


        predicted_waypoints = self._predict_vehicle_waypoints(vehicle_traffic)
        for vehicle, wps in predicted_waypoints.items():
            # Supersample path
            veh_route_pts = [wp.transform.location for wp in wps]
            veh_route_pts = np.array([[loc.x, loc.y, loc.z] for loc in veh_route_pts])
            veh_route_pts, _ = self.waypoint_planner.smooth_and_supersample(veh_route_pts)

            forecasted_vehicle_bbs[vehicle] = self.forecaster.forecast_vehicle_bbs(
                vehicle=vehicle,
                vehicle_route_points=veh_route_pts,
                veh_speed=vehicle.get_velocity().length(),
                veh_target_speed=vehicle.get_velocity().length(),
                num_future_frames=num_future_frames
            )

        return forecasted_vehicle_bbs

