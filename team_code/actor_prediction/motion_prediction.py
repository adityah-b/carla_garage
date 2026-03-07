import carla
import numpy as np

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from agents.navigation.local_planner import RoadOption

from privileged_route_planner import PrivilegedRoutePlanner
from scene_descriptor.data_extractors.vehicle_data_extractor import VehicleData, VehicleDataEntry
from scene_descriptor.data_extractors.ped_data_extractor import PedestrianData

from .trajectory_forecasting import MotionForecaster

@dataclass
class PredictionData:
    ego_forecasted_bbs : List
    veh_forecasted_bbs : Dict[int, List]
    ped_forecasted_bbs : Dict[int, List]
    all_actor_collisions : Dict[int, List]
    all_actor_overlaps : Dict[int, List]

class MotionPrediction:
    def __init__(self, config, carla_map : carla.Map):
        self.config = config
        self.world_map = carla_map

        self.forecaster = MotionForecaster(self.config)

        # Dummy waypoint planner
        self.waypoint_planner = PrivilegedRoutePlanner(self.config)

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
        all_vehicle_data.extend(vehicle_data.get(traffic_type="leading"))

        all_vehicle_data.extend(vehicle_data.get(traffic_type="trailing", lane_name="left"))
        all_vehicle_data.extend(vehicle_data.get(traffic_type="trailing", lane_name="right"))
        all_vehicle_data.extend(vehicle_data.get(traffic_type="trailing", lane_name="ego", vehicle_types={"cyclist", "emergency"}))

        all_vehicle_data.extend(vehicle_data.get(traffic_type="crossing"))
        all_vehicle_data.extend(vehicle_data.get(traffic_type="oncoming"))


        for v_data in all_vehicle_data:
            is_cyclist = v_data.vehicle_type == "cyclist"
            is_leading = v_data.traffic_type == "leading"

            veh_speed = v_data.speed

            is_stopped_vehicle = veh_speed <= 0.5
            is_slow_vehicle = veh_speed > 0.5 and veh_speed < 0.3 * speed_limit

            if is_cyclist:
                veh_forecast_frames[v_data.id] = default_future_frames
            elif is_stopped_vehicle:
                if is_leading:
                    v_data.throttle = 0.3
                    veh_forecast_frames[v_data.id] = default_future_frames
                else:
                    veh_forecast_frames[v_data.id] = 1
            elif is_slow_vehicle:
                forecast_length = 2.0
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
            print(f'\n\nID: {v_data.id}, LATERAL DISPLACEMENT: {lateral_disp}')

            lateral_disp = np.clip(lateral_disp, -lane_width / 2, lane_width / 2)
            print(f'\n\nID: {v_data.id}, LATERAL DISPLACEMENT AFTER CLIPPING: {lateral_disp}')


            if np.abs(lateral_disp) >= 0.5:
                v_data.vehicle_route_points += lateral_disp * lanelet_right_vec

        return all_vehicle_data, veh_forecast_frames

        # predicted_waypoints = {}

        # for lv_data in lane_vehicles:
        #     ll = lv_data.lanelet

        #     for v_data in lv_data.vehicle_data:
        #         vehicle = v_data.vehicle

        #         # TODO: APPLYING SPEED BASED PREDICTION FILTER, REVISIT
        #         is_cyclist = v_data.vehicle.attributes.get("base_type", "") == "bicycle"
        #         # if v_data.speed < 0.4 * (0.7 * speed_limit) and not is_cyclist:
        #         #     continue
        #         # if v_data.speed <= 1.0 and not is_cyclist:
        #         #     continue

        #         vehicle_wp = self.world_map.get_waypoint(vehicle.get_location())
        #         start_idx = ll.find(vehicle_wp)

        #         predicted_waypoints[vehicle] = [vehicle_wp] + ll.dense_waypoints[start_idx:]

        # return predicted_waypoints

    def predict_vehicle_motion(
        self,
        vehicle_data : VehicleData,
        speed_limit : float,
    ) -> Dict[int, List[carla.BoundingBox]]:
        default_forecast_length = self.config.default_forecast_length
        default_future_frames = int(self.config.bicycle_frame_rate * default_forecast_length)

        all_vehicle_data, prediction_horizons = self._prepare_vehicle_predictions(
            vehicle_data=vehicle_data,
            speed_limit=speed_limit,
            default_forecast_length=default_forecast_length
        )

        if not all_vehicle_data:
            return {}

        return self.forecaster.forecast_vehicle_bbs_array(
            all_vehicle_data=all_vehicle_data,
            prediction_horizons=prediction_horizons,
            default_future_frames=default_future_frames
        )

    # def predict_vehicle_motion(
    #     self,
    #     vehicle_traffic : Dict[str, Dict[str, List[LaneVehicleData]]]
    # ) -> Dict[int, List[carla.BoundingBox]]:
    #     forecasted_vehicle_bbs = {}
    #     forecast_length = self.config.default_forecast_length
    #     num_future_frames = int(self.config.bicycle_frame_rate * forecast_length)

    #     predicted_waypoints = self._predict_vehicle_waypoints(vehicle_traffic)
    #     for vehicle, wps in predicted_waypoints.items():
    #         # Supersample path
    #         veh_route_pts = [wp.transform.location for wp in wps]
    #         veh_route_pts = np.array([[loc.x, loc.y, loc.z] for loc in veh_route_pts])
    #         veh_route_pts, _ = self.waypoint_planner.smooth_and_supersample(veh_route_pts)

    #         forecasted_vehicle_bbs[vehicle.id] = self.forecaster.forecast_vehicle_bbs(
    #             vehicle=vehicle,
    #             vehicle_route_points=veh_route_pts,
    #             num_future_frames=num_future_frames
    #         )

    #     return forecasted_vehicle_bbs

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

        return self.forecaster.forecast_vehicle_bbs(
            ego_vehicle,
            ego_route_pts,
            num_future_frames,
            target_speed,
            velocity_profile
        )
