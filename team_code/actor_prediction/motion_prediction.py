import carla
import numpy as np

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from agents.navigation.local_planner import RoadOption

from privileged_route_planner import PrivilegedRoutePlanner
from scene_descriptor.data_extractors.vehicle_data_extractor import VehicleData, VehicleDataEntry
from scene_descriptor.data_extractors.ped_data_extractor import PedestrianData

from team_code.actor_prediction.trajectory_forecasting import MotionForecaster
from team_code.actor_prediction.collision_checker import CollisionInterval, LaneOverlapInterval

# TODO: MOVE COLLISION AND OVERLAP CHECKS TO THIS CLASS

@dataclass
class PredictionData:
    ego_forecasted_bbs : List
    veh_forecasted_bbs : Dict[int, List]
    veh_dilated_forecasted_bbs : Dict[int, List]
    ped_forecasted_bbs : Dict[int, List]
    all_actor_collisions : Dict[int, List[CollisionInterval]]
    all_actor_overlaps : Dict[int, LaneOverlapInterval]

class MotionPrediction:
    def __init__(self, config, carla_map : carla.Map):
        self.config = config
        self.world_map = carla_map

        self.forecaster = MotionForecaster(self.config)

        # Dummy waypoint planner
        self.waypoint_planner = PrivilegedRoutePlanner(self.config)

    # TODO: NEED TO EXTEND THIS SO TRAJECTORYPLANNER CAN CONTROL WHICH VEHICLES TO PREDICT
    def _prepare_vehicle_predictions(
        self,
        vehicle_data : VehicleData,
        speed_limit : float,
        default_forecast_length : float,
    ) -> Tuple[List[VehicleDataEntry], Dict[int, int]]:
        default_future_frames = int(self.config.bicycle_frame_rate * default_forecast_length)
        veh_forecast_frames : Dict[int, int] = {} # K = vehicle_id, V = num_forecast_frames

        # Get all relevant vehicles
        all_vehicle_data : List[VehicleDataEntry] = []
        # all_vehicle_data.extend(vehicle_data.get(traffic_type="leading"))
        all_vehicle_data.extend(vehicle_data.get(traffic_type="leading", lane_name="left"))
        all_vehicle_data.extend(vehicle_data.get(traffic_type="leading", lane_name="right"))
        all_vehicle_data.extend(vehicle_data.get(traffic_type="leading", lane_name="ego"))

        all_vehicle_data.extend(vehicle_data.get(traffic_type="trailing", lane_name="left"))
        all_vehicle_data.extend(vehicle_data.get(traffic_type="trailing", lane_name="right"))
        all_vehicle_data.extend(vehicle_data.get(traffic_type="trailing", lane_name="ego", vehicle_types={"cyclist", "emergency_vehicle"}))

        all_vehicle_data.extend(vehicle_data.get(traffic_type="crossing"))
        all_vehicle_data.extend(vehicle_data.get(traffic_type="oncoming"))


        for v_data in all_vehicle_data:
            is_cyclist = v_data.vehicle_type == "cyclist"
            is_leading = v_data.traffic_type == "leading"
            is_trailing = v_data.traffic_type == "trailing"

            veh_speed = v_data.speed

            is_stopped_vehicle = veh_speed <= 0.5
            is_slow_vehicle = veh_speed > 0.5 and veh_speed < 3.0

            if is_cyclist:
                veh_forecast_frames[v_data.id] = default_future_frames
            elif is_stopped_vehicle:
                if is_leading:
                    v_data.throttle = 1.0
                    veh_forecast_frames[v_data.id] = default_future_frames
                elif is_trailing:
                    veh_forecast_frames[v_data.id] = 0
                else:
                    veh_forecast_frames[v_data.id] = 1
            elif is_slow_vehicle:
                if is_leading or is_trailing:
                    veh_forecast_frames[v_data.id] = default_future_frames
                else:
                    forecast_length = 0.5
                    forecast_frames = int(self.config.bicycle_frame_rate * forecast_length)
                    veh_forecast_frames[v_data.id] = forecast_frames
            else:
                veh_forecast_frames[v_data.id] = default_future_frames

            # Shift route points laterally based on vehicle distance to lanelet centerpoint
            lanelet_wp = v_data.lanelet.dense_waypoints[v_data.lanelet_route_idx]
            lane_width = lanelet_wp.lane_width

            lanelet_right_vec = lanelet_wp.transform.get_right_vector()
            lanelet_right_vec = np.array([lanelet_right_vec.x, lanelet_right_vec.y, lanelet_right_vec.z])

            lanelet_loc = lanelet_wp.transform.location
            lanelet_loc = np.array([lanelet_loc.x, lanelet_loc.y, lanelet_loc.z])

            vehicle_loc = np.array([v_data.x, v_data.y, v_data.z])

            route_to_vehicle_vec = vehicle_loc - lanelet_loc
            lateral_disp = np.dot(route_to_vehicle_vec[:2], lanelet_right_vec[:2])
            # print(f'\n\nID: {v_data.id}, LATERAL DISPLACEMENT: {lateral_disp}')

            lateral_disp = np.clip(lateral_disp, -lane_width / 2, lane_width / 2)
            # print(f'\n\nID: {v_data.id}, LATERAL DISPLACEMENT AFTER CLIPPING: {lateral_disp}')

            if np.abs(lateral_disp) >= 0.5:
                v_data.vehicle_route_points += lateral_disp * lanelet_right_vec

        return all_vehicle_data, veh_forecast_frames

    def predict_vehicle_motion(
        self,
        vehicle_data : VehicleData,
        speed_limit : float,
    ) -> Tuple[Dict[int, List[carla.BoundingBox]], Dict[int, List[carla.BoundingBox]]]:
        default_forecast_length = self.config.default_forecast_length
        default_future_frames = int(self.config.bicycle_frame_rate * default_forecast_length)

        all_vehicle_data, prediction_horizons = self._prepare_vehicle_predictions(
            vehicle_data=vehicle_data,
            speed_limit=speed_limit,
            default_forecast_length=default_forecast_length
        )

        if not all_vehicle_data:
            return {}, {}

        return self.forecaster.forecast_vehicle_bbs_array(
            all_vehicle_data=all_vehicle_data,
            prediction_horizons=prediction_horizons,
            default_future_frames=default_future_frames
        )

    def predict_ped_motion(
        self,
        peds : List[PedestrianData]
    ) -> Dict[int, List[carla.BoundingBox]]:
        if not peds:
            return {}

        forecast_length = self.config.default_forecast_length
        num_future_frames = int(self.config.bicycle_frame_rate * forecast_length)

        peds_objs = [ped.pedestrian for ped in peds]
        return self.forecaster.forecast_ped_bbs(peds_objs, num_future_frames)

    def predict_ego_motion(
        self,
        ego_vehicle : carla.Vehicle,
        ego_route_pts : np.ndarray,
        target_speed : float = None,
        velocity_profile : Optional[np.ndarray] = None,
    ) -> List[carla.BoundingBox]:
        forecast_length = self.config.default_forecast_length
        num_future_frames = int(self.config.bicycle_frame_rate * forecast_length)

        return self.forecaster.forecast_ego_vehicle_bbs(
            ego_vehicle,
            ego_route_pts,
            num_future_frames,
            target_speed,
            velocity_profile
        )
