import carla
import numpy as np
import time

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Literal, Optional

from config import GlobalConfig

from agents.navigation.local_planner import RoadOption
from team_code.scene_descriptor.data_extractors.vehicle_data_extractor import LaneVehicleData, VehicleData
from team_code.scene_descriptor.scene_descriptor import SceneContext, SceneData
from team_code.scene_descriptor.data_extractors.route_extractor import LaneChangeData
from privileged_route_planner import PrivilegedRoutePlanner, PlannerState

from team_code.scene_analyzer.parsers.hl_beh_pydantic_models import KeyActor
from team_code.scene_analyzer.parsers.ego_plan_pydantic_models import (
    EgoPlan, LowLevelAction, Action, LongitudinalParams, LateralParams, ConditionalParams
)

from .idm import IDM
from team_code.actor_prediction.motion_prediction import MotionPrediction

class TrajectoryPlanner:
    def __init__(self, config : GlobalConfig, carla_map : carla.Map, ego_vehicle : carla.Vehicle):
        self.config = config
        self.carla_map = carla_map
        self.ego_vehicle = ego_vehicle

        self.scene_data : SceneData = None
        self.all_actors : Dict = None

        self.waypoint_planner = PrivilegedRoutePlanner(self.config)
        self.motion_forecaster = MotionPrediction(self.config, self.carla_map)

        self.idm = IDM(self.config)

        self._dispatch_map = {
            Action.FOLLOW_ROUTE: self._follow_route,
            Action.STOP_FOR: self._stop_for,
            Action.YIELD_FOR: self._yield_for,
            Action.CHANGE_LANE_LEFT: self._change_lane_left,
            Action.CHANGE_LANE_RIGHT: self._change_lane_right,
        }

        # State vars
        self.active_lc : Optional[LaneChangeData] = None
        self.wf_state : Dict[str, Any] = None

        self.cur_plan : EgoPlan = None
        self.needs_replan : bool = True

        # State transitions

        # Route following
        self.route_follow_ticks : int = 0
        self.route_follow_cleared : bool = False

        # Stop signs
        self.ss_cleared : bool = False
        self.ss_wait_ticks : int = 0

        # Traffic lights
        self.tl_prev_state : str = None
        self.tl_cleared : bool = False

        # Pedestrians
        self.ped_cleared : bool = False

        # Vehicles
        self.veh_cleared : bool = False

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

    def update_planner(
        self,
        vehicle_location : np.ndarray
    ) -> None:
        self.waypoint_planner.run_step(
            agent_position=vehicle_location
        )

    def get_planner_state(self) -> PlannerState:
        return self.waypoint_planner.get_planner_state()

    def _update_all_actors(self):
        all_actors = {
            "left" : {},
            "ego" : {},
            "right" : {},
            "oncoming" : {},
            "cross" : {},
            "peds" : {},
        }

        peds = self.scene_data.ped_data
        if peds:
            for ped in peds:
                all_actors["peds"][ped.id] = ped

        vehicles_dict = self.scene_data.vehicle_data
        if vehicles_dict:
            for lv_type, lv_dict in vehicles_dict.items(): # Traffic type
                for lv_lane, lv_list in lv_dict.items(): # Traffic lanes per traffic type
                    for lv_data in lv_list: # Lanelets per lane
                        for v_data in lv_data.vehicle_data: # Vehicles per lanelet
                            if lv_type in ["leading", "trailing"]:
                                all_actors[lv_lane][v_data.id] = v_data
                            else:
                                all_actors[lv_type][v_data.id] = v_data

        return all_actors

    def update_scene_data(
        self,
        scene_context : SceneContext
    ) -> None:
        self.scene_data = scene_context.scene_data
        self.all_actors = self._update_all_actors()

    def update_key_actors(
        self,
        key_actors : List[KeyActor] = []
    ) -> None:
        self.key_actors = key_actors

    def plan_with_reasoning(self) -> bool:
        needs_reasoning = False
        if self.scene_data.traffic_data:
            needs_reasoning = needs_reasoning or self.scene_data.traffic_data.next_stop_sign != None
            needs_reasoning = needs_reasoning or self.scene_data.traffic_data.next_traffic_light != None

        if self.scene_data.ped_data:
            for ped in self.scene_data.ped_data:
                needs_reasoning = needs_reasoning or ped.relative_distance < self.config.detection_radius

        if self.scene_data.route_data:
            needs_reasoning = needs_reasoning or self.scene_data.route_data.intersection_data != None
            needs_reasoning = needs_reasoning or self.scene_data.route_data.lane_change_data != None

        return needs_reasoning

    def set_plan(self, ego_plan : EgoPlan) -> None:
        self.cur_plan = ego_plan
        self.cur_plan_idx = 0
        self.needs_replan = False

    def execute_plan(self) -> Tuple:
        plan = self.cur_plan.plan

        # Forecast ego vehicle
        self.forecasted_ego_bbs = self.motion_forecaster.predict_ego_motion(
            ego_vehicle=self.ego_vehicle,
            ego_route_pts=self.waypoint_planner.route_points[self.waypoint_planner.route_index:],
            target_speed=self.scene_data.ego_data.speed
        )

        # Forecast all vehicles
        vehicles_dict = self.scene_data.vehicle_data
        self.forecasted_veh_bbs = self.motion_forecaster.predict_vehicle_motion(vehicles_dict) if vehicles_dict else {}

        # Forecast all pedestrians
        peds = self.scene_data.ped_data
        self.forecasted_ped_bbs = self.motion_forecaster.predict_ped_motion(peds) if peds else {}

        relevant_veh_ids : List[carla.Actor] = []
        relevant_veh_ids.extend(list(self.all_actors["oncoming"].keys()))
        relevant_veh_ids.extend(list(self.all_actors["cross"].keys()))

        has_collision : bool = False
        next_tl = self.scene_data.traffic_data.next_traffic_light
        next_ss = self.scene_data.traffic_data.next_stop_sign
        ego_loc = self.waypoint_planner.route_waypoints[self.waypoint_planner.route_index].transform.location
        # Check vehicle collisions
        world = self.ego_vehicle.get_world()
        for veh_id in relevant_veh_ids:
            veh_bbs = self.forecasted_veh_bbs[veh_id]
            collides, bb_ego, bb_veh = self.motion_forecaster.check_collision_point(self.forecasted_ego_bbs, veh_bbs)
            if collides:
                world.debug.draw_box(
                    box=bb_ego,
                    rotation=bb_ego.rotation,
                    thickness=0.1,
                    color=self.config.ego_vehicle_forecasted_bbs_hazard_color,
                    life_time=self.config.draw_life_time * 2
                )
                world.debug.draw_box(
                    box=bb_veh,
                    rotation=bb_veh.rotation,
                    thickness=0.1,
                    color=self.config.trailing_vehicle_color,
                    life_time=self.config.draw_life_time * 2
                )
                if next_tl:
                    dist_to_collision = ego_loc.distance(bb_ego.location)
                    if dist_to_collision > next_tl.distance_to_light and next_tl.distance_to_light > 3.0:
                        continue
                if next_ss:
                    dist_to_collision = ego_loc.distance(bb_ego.location)
                    if dist_to_collision > next_ss.distance_to_stop_sign and next_ss.distance_to_stop_sign > 3.0:
                        continue

            has_collision = has_collision or collides
            if has_collision:
                break

        if not has_collision:
            # Check pedestrian collisions
            for _, ped_bbs in self.forecasted_ped_bbs.items():
                collides, _, _ = self.motion_forecaster.check_collision_point(self.forecasted_ego_bbs, ped_bbs)
                has_collision = has_collision or collides
                if has_collision:
                    break

        # # If we finished last tick, ask for replan
        # if self.cur_plan_idx is None or self.cur_plan_idx >= len(plan):
        #     print(f'Finished Plan')
        #     self._reset_plan(replan=True)
        #     return self.follow_route()

        cur_action = plan[self.cur_plan_idx]
        print(f'Running Action: {cur_action.action.value}')

        # if next_action and self._should_preempt(action, next_action, next_param):
        #     # advance to next action
        #     self.cur_plan_idx += 1
        #     i = self.cur_plan_idx
        #     action = actions[i]
        #     param  = params[i] if i < len(params) else None

        if cur_action.action != Action.YIELD_FOR:
            if has_collision:
                print(f'PLAN HAS COLLISION DURING EXECUTION, REPLANNING')
                self._reset_plan(replan=True)

                target_speed, route_pts, route_wps = self.follow_route(0.0)
                brake = target_speed < 0.1
                return (target_speed, brake, route_pts, route_wps)

        fn = self._dispatch_map.get(cur_action.action)
        # if fn is None:
        #     # Unknown action → skip
        #     self.cur_plan_idx += 1
        #     return (self.target_speed, self.route_pts, self.route_wps)

        if cur_action.action in (Action.STOP_FOR, Action.YIELD_FOR):
            params = cur_action.conditional_params
        elif cur_action.action in (Action.FOLLOW_ROUTE):
            params = cur_action.longitudinal_params
        else:
            params = cur_action.lateral_params

        target_speed, brake, move_to_next_state, route_pts, route_wps = fn(params)

        if move_to_next_state:
            self.cur_plan_idx += 1

        if self.cur_plan_idx >= len(plan):
            # if not has_collision and plan[-1].action == Action.FOLLOW_ROUTE:
            #     ll_action = plan[-1]
            #     ego_plan = EgoPlan(
            #         plan=LowLevelAction(
            #             action=ll_action.action,
            #             longitudinal_params=LongitudinalParams(spd=self.scene_data.traffic_data.speed_limit),
            #             lateral_params=None,
            #             conditional_params=None
            #         ),
            #         reasoning=[""]
            #     )
            #     self.set_plan(ego_plan)
            # else:
            #     self._reset_plan(replan=True)
            self._reset_plan(replan=True)

        return (target_speed, brake, route_pts, route_wps)

        # # Longitudinal primitives
        # if action in (Action.ACCELERATE, Action.DECELERATE, Action.MAINTAIN_SPEED):
        #     self.target_speed, self.route_pts, self.route_wps = fn(param)
        #     ego_spd = self.scene_data.ego_data.speed
        #     tgt     = getattr(param, "spd", None)

        #     # done conditions with small hysteresis
        #     if action == Action.ACCELERATE and (tgt is not None) and (ego_spd >= tgt - 0.2):
        #         self.cur_plan_idx += 1
        #     elif action == Action.DECELERATE and (tgt is not None) and (ego_spd <= tgt + 0.2):
        #         self.cur_plan_idx += 1
        #     elif action == Action.MAINTAIN_SPEED:
        #         # default: one-tick step unless next guard wants to keep maintaining
        #         # if there is no next action, finish; otherwise advance to next
        #         self.cur_plan_idx += 1

        #     # If finished the sequence, mark for replan next tick
        #     if self.cur_plan_idx >= len(actions):
        #         self._reset_plan(replan=True)

        #     return (self.target_speed, self.route_pts, self.route_wps)

        # # Lane changes / brake (single-step or until they naturally complete)
        # self.target_speed, self.route_pts, self.route_wps = fn(param)
        # # For CHANGE_LANE_* you might want to hold index until lane change completes.
        # # Here we advance immediately; if you need to block until completion, add a condition.
        # self.cur_plan_idx += 1

        # if self.cur_plan_idx >= len(actions):
        #     self._reset_plan(replan=True)

        # return (self.target_speed, self.route_pts, self.route_wps)


    # ---------------- helpers ----------------

    def _reset_plan(self, replan: bool = False):
        """Reset plan bookkeeping."""
        self.cur_plan = None
        self.cur_plan_idx = None
        self.needs_replan = replan
        # Keep last target_speed/route so the controller has something to follow this tick


    def _should_preempt(self, cur_action: Action, nxt_action: Action, nxt_param) -> bool:
        """
        Decide whether to preempt current action and enter the next one.
        Useful to jump from maintain→decelerate near a stop/red or when next spd < current spd.
        """
        ego_spd = self.scene_data.ego_data.speed
        nxt_spd = getattr(nxt_param, "spd", None)

        # 1) If next wants significantly lower speed than we have, preempt
        if nxt_action in (Action.DECELERATE, Action.BRAKE) and (nxt_spd is not None):
            if ego_spd > (nxt_spd + 0.5):
                return True

        # 2) Approaching stop sign
        ss = self.scene_data.traffic_data.next_stop_sign
        if ss is not None:
            # preempt if close to stop line (tunable)
            if ss.distance_to_stop_sign <= max(12.0, ego_spd * 1.2):
                # If the next step is decelerate/brake/wait, jump to it
                if nxt_action in (Action.DECELERATE, Action.BRAKE, Action.WAIT_FOR):
                    return True

        # 3) Red traffic light
        tl = self.scene_data.traffic_data.next_traffic_light
        if tl is not None and tl.state in ("RED", "YELLOW"):
            if tl.distance_to_light <= max(15.0, ego_spd * 1.5):
                if nxt_action in (Action.DECELERATE, Action.BRAKE, Action.WAIT_FOR):
                    return True

        # 4) If current is MAINTAIN_SPEED and next has a valid lower target, it’s safe to preempt
        if cur_action == Action.MAINTAIN_SPEED and nxt_action in (Action.DECELERATE, Action.BRAKE):
            if (nxt_spd is not None) and (ego_spd > nxt_spd + 0.5):
                return True

        return False

    def __get_target_speed(
        self,
        target_speed_initial : float,
        params : LongitudinalParams,
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

    def follow_route(
        self,
        target_speed_initial : float
    ) -> Tuple:
        target_speed_veh = self.__get_target_speed(target_speed_initial, LongitudinalParams())
        target_speed = min(target_speed_initial, target_speed_veh)
        start_index = self.waypoint_planner.route_index
        return target_speed, self.waypoint_planner.route_points[start_index:], self.waypoint_planner.route_waypoints[start_index:]

    def _follow_route(
        self,
        params : LongitudinalParams
    ) -> Tuple:
        ego_speed = self.scene_data.ego_data.speed
        target_speed_initial = params.spd

        self.route_follow_ticks += 1
        if self.route_follow_ticks > 25:
            self.route_follow_ticks = 0
            move_to_next_state = True
        else:
            move_to_next_state = False

        target_speed = self.__get_target_speed(target_speed_initial, params)
        print(f'_follow_route Target Speed: {target_speed}')
        brake = target_speed < 1e-2

        route_index = self.waypoint_planner.route_index
        route_pts = self.waypoint_planner.route_points
        route_wps = self.waypoint_planner.route_waypoints


        return target_speed, brake, move_to_next_state, route_pts[route_index:], route_wps[route_index:]

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

    def _accelerate(self, params : Optional[LongitudinalParams] = None):
        ego_speed = self.scene_data.ego_data.speed
        speed_limit = self.scene_data.traffic_data.speed_limit
        if params and params.spd:
            target_speed_initial = params.spd
        else:
            target_speed_initial = min(ego_speed * 1.1, speed_limit)

        target_speed = self.__get_target_speed(target_speed_initial, params)
        start_index = self.waypoint_planner.route_index
        return target_speed, self.waypoint_planner.route_points[start_index:], self.waypoint_planner.route_waypoints[start_index:]

    def _decelerate(self, params : Optional[LongitudinalParams] = None):
        ego_speed = self.scene_data.ego_data.speed
        if params and params.spd:
            target_speed_initial = params.spd
        else:
            target_speed_initial = max(0.0, min(ego_speed * 0.9, 0.25))

        target_speed = self.__get_target_speed(target_speed_initial, params)
        start_index = self.waypoint_planner.route_index
        return target_speed, self.waypoint_planner.route_points[start_index:], self.waypoint_planner.route_waypoints[start_index:]

    def _maintain_speed(self, params : Optional[LongitudinalParams] = None):
        target_speed_initial = self.scene_data.ego_data.speed
        target_speed = self.__get_target_speed(target_speed_initial, params)
        start_index = self.waypoint_planner.route_index
        return target_speed, self.waypoint_planner.route_points[start_index:], self.waypoint_planner.route_waypoints[start_index:]

    def _change_lane_left(self, params : Optional[LateralParams] = None):
        ego_speed = self.scene_data.ego_data.speed
        speed_limit = self.config.max_speed_in_junction
        target_speed_initial = min(ego_speed, speed_limit)

        lc_data = self.scene_data.route_data.lane_change_data
        if not lc_data:
            return self._maintain_speed(LongitudinalParams())

        if self.active_lc is None or self.active_lc != lc_data:
            self.active_lc = lc_data
            self.__change_lane(lc_data, RoadOption.CHANGELANELEFT)

        target_speed = self.__get_target_speed(target_speed_initial, LongitudinalParams(), target_lanes=['ego', 'left'])
        start_index = self.waypoint_planner.route_index
        return target_speed, self.waypoint_planner.route_points[start_index:], self.waypoint_planner.route_waypoints[start_index:]

    def _change_lane_right(self, params : Optional[LateralParams] = None):
        ego_speed = self.scene_data.ego_data.speed
        speed_limit = self.config.max_speed_in_junction
        target_speed_initial = min(ego_speed, speed_limit)

        lc_data = self.scene_data.route_data.lane_change_data
        if not lc_data:
            return self._maintain_speed(LongitudinalParams())

        if self.active_lc is None or self.active_lc != lc_data:
            self.active_lc = lc_data
            self.__change_lane(lc_data, RoadOption.CHANGELANERIGHT)

        target_speed = self.__get_target_speed(target_speed_initial, LongitudinalParams(), target_lanes=['ego', 'right'])
        start_index = self.waypoint_planner.route_index
        return target_speed, self.waypoint_planner.route_points[start_index:], self.waypoint_planner.route_waypoints[start_index:]

    def _brake(self, param = None):
        target_speed = self.__get_target_speed(0.0, LongitudinalParams())
        start_index = self.waypoint_planner.route_index
        return target_speed, self.waypoint_planner.route_points[start_index:], self.waypoint_planner.route_waypoints[start_index:]

    def _process_stop_sign(self) -> Tuple[float, bool]:
        ego_speed = self.scene_data.ego_data.speed
        target_speed_initial = self.scene_data.traffic_data.speed_limit
        next_ss = self.scene_data.traffic_data.next_stop_sign
        if next_ss is None:
            return target_speed_initial, True

        if ego_speed < 0.1 and next_ss.distance_to_stop_sign < self.config.clearing_distance_to_stop_sign:
            self.ss_wait_ticks += 1
            if self.ss_wait_ticks > 25:
                self.ss_cleared = True
        else:
            self.ss_wait_ticks = 0
            self.ss_cleared = False

        if self.ss_cleared:
            target_speed = target_speed_initial
        else:
            target_speed = self.idm.compute_target_speed_idm(
                desired_speed=target_speed_initial,
                leading_actor_length=0.0,
                ego_speed=ego_speed,
                leading_actor_speed=0.0,
                distance_to_leading_actor=next_ss.distance_to_stop_sign,
                s0=self.config.idm_stop_sign_minimum_distance,
                T=self.config.idm_stop_sign_desired_time_headway
            )

        return target_speed, self.ss_cleared

    def _process_traffic_light(self) -> Tuple[float, bool]:
        ego_speed = self.scene_data.ego_data.speed
        target_speed_initial = self.scene_data.traffic_data.speed_limit
        next_tl = self.scene_data.traffic_data.next_traffic_light
        if next_tl is None:
            return target_speed_initial, True

        if next_tl.state == "GREEN":
            target_speed = target_speed_initial
            if self.tl_prev_state and self.tl_prev_state == "RED":
                self.tl_cleared = True
        else:
            self.tl_prev_state = next_tl.state
            self.tl_cleared = False

            target_speed = self.idm.compute_target_speed_idm(
                desired_speed=target_speed_initial,
                leading_actor_length=0.0,
                ego_speed=ego_speed,
                leading_actor_speed=0.0,
                distance_to_leading_actor=next_tl.distance_to_light,
                s0=self.config.idm_red_light_minimum_distance,
                T=self.config.idm_red_light_desired_time_headway,
            )

        return target_speed, self.tl_cleared

    def _stop_for(
        self,
        target_obj : ConditionalParams
    ) -> Tuple:
        move_to_next_state = True

        if target_obj.target == "stop_sign":
            target_speed, move_to_next_state = self._process_stop_sign()
        elif target_obj.target == "traffic_light":
            target_speed, move_to_next_state = self._process_traffic_light()
        # TODO: Add obstacle handling
        elif target_obj.target == "obstacle":
            pass
        else:
            target_speed = self.scene_data.traffic_data.speed_limit

        target_speed_lead = self.__get_target_speed(self.scene_data.traffic_data.speed_limit, LongitudinalParams())
        target_speed = min(target_speed, target_speed_lead)

        brake = target_speed < 1e-2

        route_index = self.waypoint_planner.route_index
        route_pts = self.waypoint_planner.route_points
        route_wps = self.waypoint_planner.route_waypoints

        return target_speed, brake, move_to_next_state, route_pts[route_index:], route_wps[route_index:]

    def _process_ped(
        self,
        ped_id : int
    ) -> Tuple[float, bool]:
        ego_speed = self.scene_data.ego_data.speed
        target_speed_initial = self.scene_data.traffic_data.speed_limit

        peds = self.scene_data.ped_data
        if not peds:
            return target_speed_initial, True

        # Check if target pedestrian exists in actors list
        if ped_id in self.all_actors["peds"]:
            ped_data = self.all_actors["peds"][ped_id]

        # Otherwise find the nearest closest pedestrian
        else:
            ped_data = None
            for p_data in self.all_actors["peds"].values():
                if p_data.relative_distance < self.config.detection_radius:
                    if ped_data is None:
                        ped_data = p_data
                    else:
                        ped_data = ped_data if ped_data.relative_distance < p_data.relative_distance else p_data

        if ped_data is None:
            return target_speed_initial, True

        target_speed = self.idm.compute_target_speed_idm(
            desired_speed=target_speed_initial,
            leading_actor_length=0.5 + self.scene_data.ego_data.ego_vehicle.bounding_box.extent.x,
            ego_speed=ego_speed,
            leading_actor_speed=0.0,
            distance_to_leading_actor=ped_data.relative_distance,
            s0=self.config.idm_pedestrian_minimum_distance,
            T=self.config.idm_pedestrian_desired_time_headway,
        )

        # Check for collisions
        forecasted_ego_bbs = self.motion_forecaster.predict_ego_motion(
            ego_vehicle=self.ego_vehicle,
            ego_route_pts=self.waypoint_planner.route_points[self.waypoint_planner.route_index:],
            target_speed=target_speed_initial
        )
        forecasted_ped_bbs = self.motion_forecaster.predict_ped_motion([ped_data])

        ped_bbs = forecasted_ped_bbs[ped_data.id]
        has_collision, bb_a, bb_b = self.motion_forecaster.check_collision_point(forecasted_ego_bbs, ped_bbs)
        print(f'Ped has collision: {has_collision}')

        world = self.ego_vehicle.get_world()
        # for bb_ped, bb_ego in zip(ped_bbs, forecasted_ego_bbs):
        #     world.debug.draw_box(
        #         box=bb_ped,
        #         rotation=bb_ped.rotation,
        #         thickness=0.1,
        #         color=self.config.other_vehicles_forecasted_bbs_color,
        #         life_time=self.config.draw_life_time
        #     )
        #     world.debug.draw_box(
        #         box=bb_ego,
        #         rotation=bb_ego.rotation,
        #         thickness=0.1,
        #         color=self.config.ego_vehicle_forecasted_bbs_normal_color,
        #         life_time=self.config.draw_life_time
        #     )

        if has_collision:
            world.debug.draw_box(
                box=bb_a,
                rotation=bb_a.rotation,
                thickness=0.1,
                color=self.config.ego_vehicle_forecasted_bbs_hazard_color,
                life_time=self.config.draw_life_time
            )
            world.debug.draw_box(
                box=bb_b,
                rotation=bb_b.rotation,
                thickness=0.1,
                color=self.config.trailing_vehicle_color,
                life_time=self.config.draw_life_time
            )

        self.ped_cleared = (not has_collision) and (ped_data.relative_distance > 5.0)

        return target_speed, self.ped_cleared

    def _process_vehicle(
        self,
        veh_id : int
    ) -> Tuple[float, bool]:
        ego_speed = self.scene_data.ego_data.speed
        target_speed_initial = self.scene_data.traffic_data.speed_limit

        vehicles_dict = self.scene_data.vehicle_data
        if not vehicles_dict:
            return target_speed_initial, True

        # Find target vehicle in key actors list
        veh_traffic_type : str = None
        for key_actor in self.key_actors:
            if key_actor.id == veh_id and key_actor.obj_type == "vehicle":
                veh_traffic_type = key_actor.traffic_type

        if not veh_traffic_type:
            return target_speed_initial, True

        veh_data = None
        for actor_key in self.all_actors.keys():
            if actor_key == "peds":
                continue

            # Check if target vehicle exists in actors dict
            if veh_id in self.all_actors[actor_key]:
                veh_data = self.all_actors[actor_key][veh_id]
                break

        if veh_data is None:
            return target_speed_initial, True

        if veh_traffic_type in ["oncoming", "cross"]:
            # Check for collisions
            forecasted_ego_bbs = self.motion_forecaster.predict_ego_motion(
                ego_vehicle=self.ego_vehicle,
                ego_route_pts=self.waypoint_planner.route_points[self.waypoint_planner.route_index:],
                target_speed=veh_data.speed * 1.2
            )
            forecasted_veh_bbs = self.motion_forecaster.predict_vehicle_motion(self.scene_data.vehicle_data)
            veh_bbs = forecasted_veh_bbs[veh_data.id]

            has_collision, bb_ego, bb_veh = self.motion_forecaster.check_collision_point(forecasted_ego_bbs, veh_bbs)

            if has_collision:
                print(f'EGO COLLIDES WITH VEHICLE: {veh_data.id}')
                world = self.ego_vehicle.get_world()
                world.debug.draw_box(
                    box=bb_ego,
                    rotation=bb_ego.rotation,
                    thickness=0.1,
                    color=self.config.ego_vehicle_forecasted_bbs_hazard_color,
                    life_time=self.config.draw_life_time * 2
                )
                world.debug.draw_box(
                    box=bb_veh,
                    rotation=bb_veh.rotation,
                    thickness=0.1,
                    color=self.config.trailing_vehicle_color,
                    life_time=self.config.draw_life_time * 2
                )

            elif not has_collision and veh_data.relative_distance > 5.0:
                print(f'NO COLLISION WITH VEHICLE: {veh_data.id}')

            self.veh_cleared = (not has_collision) and veh_data.relative_distance > 5.0
            target_speed = ego_speed if self.veh_cleared else 0.0

        else:
            # target_speed = veh_data.speed * 0.8
            target_speed = target_speed_initial

        # if veh_traffic_type in ["oncoming", "cross"]:
        #     # Check for collisions
        #     forecasted_ego_bbs = self.motion_forecaster.predict_ego_motion(
        #         ego_vehicle=self.ego_vehicle,
        #         ego_route_pts=self.waypoint_planner.route_points[self.waypoint_planner.route_index:],
        #         target_speed=target_speed_initial
        #     )
        #     forecasted_veh_bbs = self.motion_forecaster.predict_vehicle_motion(self.scene_data.vehicle_data)
        #     veh_bbs = forecasted_veh_bbs[veh_data.id]

        #     has_collision, bb_a, bb_b = self.motion_forecaster.check_collision_point(forecasted_ego_bbs, veh_bbs)
        #     if not has_collision and veh_data.relative_distance > 5.0:
        #         self.veh_cleared = True
        #     else:
        #         self.veh_cleared = False

        return target_speed, self.veh_cleared

    def _yield_for(
        self,
        target_obj : ConditionalParams
    ) -> Tuple:
        move_to_next_state = True

        if target_obj.target == "ped":
            target_speed, move_to_next_state = self._process_ped(target_obj.id)
        elif target_obj.target == "vehicle":
            target_speed, move_to_next_state = self._process_vehicle(target_obj.id)
        else:
            target_speed = self.scene_data.traffic_data.speed_limit

        target_speed_lead = self.__get_target_speed(self.scene_data.traffic_data.speed_limit, LongitudinalParams())
        target_speed = min(target_speed, target_speed_lead)

        brake = target_speed < 1e-2

        route_index = self.waypoint_planner.route_index
        route_pts = self.waypoint_planner.route_points
        route_wps = self.waypoint_planner.route_waypoints

        return target_speed, brake, move_to_next_state, route_pts[route_index:], route_wps[route_index:]
