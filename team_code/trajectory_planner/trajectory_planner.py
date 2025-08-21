import carla
import numpy as np

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Literal, Optional

from config import GlobalConfig

from agents.navigation.local_planner import RoadOption
from team_code.scene_descriptor.data_extractors.vehicle_data_extractor import LaneVehicleData, VehicleData
from team_code.scene_descriptor.scene_descriptor import SceneContext, SceneData
from team_code.scene_descriptor.data_extractors.route_extractor import LaneChangeData
from privileged_route_planner import PrivilegedRoutePlanner, PlannerState

from team_code.scene_analyzer.parsers.ego_plan_parser import EgoPlan

from .idm import IDM
from .command_translator import CommandTranslator, Plan, Action, Params

class TrajectoryPlanner:
    def __init__(self, config : GlobalConfig):
        self.config = config

        self.scene_data : SceneData = None

        self.waypoint_planner = PrivilegedRoutePlanner(self.config)
        self.idm = IDM(self.config)

        self._dispatch_map = {
            Action.ACCELERATE: self._accelerate,
            Action.DECELERATE: self._decelerate,
            Action.MAINTAIN_SPEED: self._maintain_speed,
            Action.CHANGE_LANE_LEFT: self._change_lane_left,
            Action.CHANGE_LANE_RIGHT: self._change_lane_right,
            Action.STOP: self._stop,
        }

        # State vars
        self.active_lc : Optional[LaneChangeData] = None

    def setup_route(
        self,
        global_plan : List[Tuple[carla.Transform, Any]],
        carla_world : carla.World,
        carla_map : carla.Map,
        starts_with_parking_exit : bool,
        vehicle_location : carla.Location
    ) -> None:
        self.waypoint_planner.setup_route(
            global_plan=global_plan,
            carla_world=carla_world,
            carla_map=carla_map,
            starts_with_parking_exit=starts_with_parking_exit,
            vehicle_loc=vehicle_location
        )

    def get_planner_state(self) -> PlannerState:
        return self.waypoint_planner.get_planner_state()

    def update_planner(
        self,
        vehicle_location : np.ndarray
    ) -> None:
        self.waypoint_planner.run_step(
            agent_position=vehicle_location
        )

    def update_scene_data(
        self,
        scene_context : SceneContext
    ) -> None:
        self.scene_data = scene_context.scene_data

    def translate_plan(self, ego_plan: EgoPlan) -> Plan:
        """Parse and validate plan text from the LLM into a structured Plan."""
        return CommandTranslator.translate_plan(ego_plan)

    def execute_plan(self, plan: Plan) -> List[Any]:
        """
        Execute the sequence of actions in the plan.
        Returns a list of results from each low-level call.
        """
        results: List[Any] = []
        for action, param in zip(plan.actions, plan.params):
            fn = self._dispatch_map.get(action)
            if fn is None:
                raise NotImplementedError(f"No handler for action '{action.value}'")
            results.append(fn(param))
        return results

    def __get_target_speed(
        self,
        target_speed_initial : float,
        params : Params,
        target_lanes : Optional[List[str]] = None,
    ) -> float:
        ego_speed = self.scene_data.ego_data.speed

        print(f'Ego Speed: {ego_speed}\n')
        print(f'Target Speed Initial: {target_speed_initial}\n')

        target_speeds = [target_speed_initial]
        leading_vehicles : Dict[str, List[LaneVehicleData]] = self.scene_data.vehicle_data['leading']
        if leading_vehicles:
            lane_vehicle_data : List[LaneVehicleData] = []
            if target_lanes:
                for target_lane in target_lanes:
                    if target_lane in leading_vehicles:
                        lane_vehicle_data.extend(leading_vehicles[target_lane])
            else:
                lane_vehicle_data = leading_vehicles['ego']
            lane_leading_vehicles : List[VehicleData] = []

            # Check for leading vehicles in the ego lane
            for lane_veh_data in lane_vehicle_data:
                lane_leading_vehicles.extend(lane_veh_data.vehicle_data)

            for lv in lane_leading_vehicles:
                lv_speed = lv.speed
                lv_length = lv.vehicle.bounding_box.extent.x * 2

                dist_to_lv = lv.relative_distance

                desired_following_distance = self.config.idm_leading_vehicle_minimum_distance if not params.f_dist else params.f_dist
                desired_time_headway = self.config.idm_leading_vehicle_time_headway if not params.t_head else params.t_head

                target_speed_vehicle = self.idm.compute_target_speed_idm(
                    desired_speed = target_speed_initial,
                    leading_actor_length = lv_length,
                    ego_speed = ego_speed,
                    leading_actor_speed = lv_speed,
                    distance_to_leading_actor = dist_to_lv,
                    s0 = desired_following_distance,
                    T = desired_time_headway
                )
                print(f'Leading Vehicle ID: {lv.id}, IDM Speed: {target_speed_vehicle}\n')
                target_speeds.append(target_speed_vehicle)

        return min(target_speeds)

    def __change_lane(
        self,
        lc_data : LaneChangeData,
        direction_cmd : Literal[RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]
    ):
        dist_func = (lambda wp1, wp2 : wp1.transform.location.distance(wp2.transform.location))
        route_idx = self.waypoint_planner.route_index
        route_wps = self.waypoint_planner.route_waypoints

        ego_wp = route_wps[route_idx]
        direction = "left" if direction_cmd == RoadOption.CHANGELANELEFT else "right"
        _, _, lc_idx, end_idx = lc_data.entry

        lc_start_wp = route_wps[lc_idx]
        lc_end_wp = route_wps[end_idx]

        avail_lc_dist = dist_func(ego_wp, lc_end_wp)
        transition_len = max(self.config.transition_smoothness_distance, avail_lc_dist * self.config.points_per_meter)
        lane_transition_factor = max(1.0, np.abs(lc_start_wp.lane_id - lc_end_wp.lane_id))

        self.waypoint_planner.change_lane(route_idx, lc_idx, end_idx, direction, transition_len, lane_transition_factor)

    def _accelerate(self, params : Optional[Params] = None):
        ego_speed = self.scene_data.ego_data.speed
        speed_limit = self.scene_data.traffic_data.speed_limit
        if params and params.spd:
            target_speed_initial = params.spd
        else:
            target_speed_initial = min(ego_speed * 1.1, speed_limit)

        target_speed = self.__get_target_speed(target_speed_initial, params)
        start_index = self.waypoint_planner.route_index
        return target_speed, self.waypoint_planner.route_points[start_index:], self.waypoint_planner.route_waypoints[start_index:]


    def _decelerate(self, params : Optional[Params] = None):
        ego_speed = self.scene_data.ego_data.speed
        if params and params.spd:
            target_speed_initial = params.spd
        else:
            target_speed_initial = max(0.0, min(ego_speed * 0.9, 0.25))

        target_speed = self.__get_target_speed(target_speed_initial, params)
        start_index = self.waypoint_planner.route_index
        return target_speed, self.waypoint_planner.route_points[start_index:], self.waypoint_planner.route_waypoints[start_index:]

    def _maintain_speed(self, params : Optional[Params] = None):
        target_speed_initial = self.scene_data.ego_data.speed
        target_speed = self.__get_target_speed(target_speed_initial, params)
        start_index = self.waypoint_planner.route_index
        return target_speed, self.waypoint_planner.route_points[start_index:], self.waypoint_planner.route_waypoints[start_index:]

    def _change_lane_left(self, params : Optional[Params] = None):
        ego_speed = self.scene_data.ego_data.speed
        speed_limit = self.config.max_speed_in_junction
        target_speed_initial = min(ego_speed, speed_limit)

        lc_data = self.scene_data.route_data.lane_change_data
        if not lc_data:
            return self._maintain_speed(params)

        if self.active_lc is None or self.active_lc != lc_data:
            self.active_lc = lc_data
            self.__change_lane(lc_data, RoadOption.CHANGELANELEFT)

        target_speed = self.__get_target_speed(target_speed_initial, params, target_lanes=['ego', 'left'])
        start_index = self.waypoint_planner.route_index
        return target_speed, self.waypoint_planner.route_points[start_index:], self.waypoint_planner.route_waypoints[start_index:]

    def _change_lane_right(self, params : Optional[Params] = None):
        ego_speed = self.scene_data.ego_data.speed
        speed_limit = self.config.max_speed_in_junction
        target_speed_initial = min(ego_speed, speed_limit)

        lc_data = self.scene_data.route_data.lane_change_data
        if not lc_data:
            return self._maintain_speed(params)

        if self.active_lc is None or self.active_lc != lc_data:
            self.active_lc = lc_data
            self.__change_lane(lc_data, RoadOption.CHANGELANERIGHT)

        target_speed = self.__get_target_speed(target_speed_initial, params, target_lanes=['ego', 'right'])
        start_index = self.waypoint_planner.route_index
        return target_speed, self.waypoint_planner.route_points[start_index:], self.waypoint_planner.route_waypoints[start_index:]

    def _stop(self):
        target_speed = self.__get_target_speed(0.0)
        start_index = self.waypoint_planner.route_index
        return target_speed, self.waypoint_planner.route_points[start_index:], self.waypoint_planner.route_waypoints[start_index:]
