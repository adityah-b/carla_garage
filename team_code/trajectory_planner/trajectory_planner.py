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
from team_code.scene_analyzer.parsers.ego_plan_pydantic_models import *

from .idm import IDM
from team_code.actor_prediction.motion_prediction import MotionPrediction

# Collision checker
from team_code.actor_prediction.collision_checker import CollisionChecker

# Longitudinal planner
from local_planner.longitudinal.long_planner import LongPlanner

# Lateral planner
from local_planner.lateral.lat_planner import LatPlanner

@dataclass
class PredictionData:
    ego_forecasted_bbs : List
    veh_forecasted_bbs : Dict[int, List]
    ped_forecasted_bbs : Dict[int, List]
    all_actor_collisions : Dict[int, List]

class TrajectoryPlanner:
    def __init__(self, config : GlobalConfig, carla_map : carla.Map, ego_vehicle : carla.Vehicle):
        self.config = config
        self.carla_map = carla_map
        self.ego_vehicle = ego_vehicle

        self.scene_data : SceneData = None
        self.all_actors : Dict = None
        self.lidar_pts : Dict = None

        self.waypoint_planner = PrivilegedRoutePlanner(self.config)
        self.motion_forecaster = MotionPrediction(self.config, self.carla_map)

        self.idm = IDM(self.config)

        # Longitudinal planner
        self.long_planner = LongPlanner(st_grid_spec=self.config.st_grid_spec, st_algo_spec=self.config.st_algo_spec, algo_name='dijkstra', sim_freq=self.config.fps)

        # Lateral planner
        self.lat_planner = LatPlanner(self.ego_vehicle, lat_grid_spec=self.config.lat_grid_spec, lat_algo_spec=self.config.lat_algo_spec)

        self._dispatch_map = {
            Action.FOLLOW_ROUTE: self._follow_route,

            Action.TURN_LEFT: self._turn_left,
            Action.TURN_RIGHT: self._turn_right,

            Action.CHANGE_LANE_LEFT: self._change_lane_left,
            Action.CHANGE_LANE_RIGHT: self._change_lane_right,

            Action.OVERTAKE_LEFT: self._overtake_left,
            Action.OVERTAKE_RIGHT: self._overtake_right,

            Action.PULL_OVER_LEFT: self._pull_over_left,
            Action.PULL_OVER_RIGHT: self._pull_over_right,
        }

        # State vars
        self.active_lc : Optional[LaneChangeData] = None
        self.wf_state : Dict[str, Any] = None

        self.cur_plan : EgoPlan = None
        self.cur_plan_status: Optional[PlanStatus] = None
        self.cur_plan_reason: Optional[str] = None
        self.prev_plan_execution: Optional[PlanExecution] = None
        self.needs_replan : bool = True

        # State transitions

        # Route following
        self.route_follow_ticks : int = 0

        # Turns
        self.intersection_prev_end_idx : int = None

        # Lane changes
        self.lane_change_prev_end_idx : int = None

        # Stop signs
        self.ss_cleared : bool = False
        self.ss_wait_ticks : int = 0

        # Traffic lights
        self.tl_prev_state : str = None
        self.tl_cleared : bool = False

        # Lateral planner route changes
        self.cur_route_changes : Tuple[int, int] = None

        # Pull over state
        self.pull_over_state : Optional[Dict[str, Any]] = None

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
                                if lv_lane not in all_actors:
                                    all_actors[lv_lane] = {}
                                all_actors[lv_lane][v_data.id] = v_data
                            else:
                                all_actors[lv_type][v_data.id] = v_data

        return all_actors

    def update_scene_data(
        self,
        scene_context : SceneContext,
        lidar_pts : Dict,
    ) -> None:
        self.scene_data = scene_context.scene_data
        self.all_actors = self._update_all_actors()
        self.lidar_pts = lidar_pts

    def update_key_actors(
        self,
        key_actors : List[KeyActor] = []
    ) -> None:
        self.key_actors = key_actors

    def plan_with_reasoning(self) -> bool:
        needs_reasoning = False
        # if self.scene_data.traffic_data:
        #     needs_reasoning = needs_reasoning or self.scene_data.traffic_data.next_stop_sign != None
        #     needs_reasoning = needs_reasoning or self.scene_data.traffic_data.next_traffic_light != None

        if self.scene_data.ped_data:
            for ped in self.scene_data.ped_data:
                # TODO: START SETTING REASONING DISTANCE TOLERANCES
                needs_reasoning = needs_reasoning or ped.relative_distance <= 15.0

        if self.scene_data.route_data:
            needs_reasoning = needs_reasoning or self.scene_data.route_data.intersection_data != None
            # TODO: START SETTING REASONING DISTANCE TOLERANCES
            needs_reasoning = needs_reasoning or (self.scene_data.route_data.lane_change_data != None and self.scene_data.route_data.lane_change_data.distance_to_lane_change <= 20.0)

        if self.scene_data.obstacle_data:
            # TODO: START SETTING REASONING DISTANCE TOLERANCES
            needs_reasoning = needs_reasoning or self.scene_data.nearest_obstacle.relative_distance <= 15.0
        return needs_reasoning

    def predict_collisions(self) -> PredictionData:
        ########################################
        # Actor motion prediction
        ########################################
        ego_velocity_profile = None
        if self.long_planner.has_active_plan:
            remaining_profile = self.long_planner.current_vel_profile[
                self.long_planner.profile_step_idx :
            ]
            if remaining_profile.size > 0:
                ego_velocity_profile = remaining_profile

        # TODO: Pass conditions into motion prediction
        # Forecast ego vehicle
        ego_forecasted_bbs = self.motion_forecaster.predict_ego_motion(
            ego_vehicle=self.ego_vehicle,
            ego_route_pts=self.waypoint_planner.route_points[self.waypoint_planner.route_index:],
            target_speed=None if ego_velocity_profile is not None else self.target_speed_initial,
            velocity_profile=ego_velocity_profile,
        )

        # Forecast all vehicles
        veh_forecasted_bbs = self.motion_forecaster.predict_vehicle_motion(self.scene_data.vehicle_data)

        # Forecast all pedestrians
        ped_forecasted_bbs = self.motion_forecaster.predict_ped_motion(self.scene_data.ped_data)

        # Compute all collisions
        actor_predictions = veh_forecasted_bbs | ped_forecasted_bbs
        all_actor_collisions = CollisionChecker.predict_actor_collisions(
            ego_bb_preds=ego_forecasted_bbs,
            actor_predictions=actor_predictions
        )

        # NOTE: Debugging
        world = self.ego_vehicle.get_world()
        for actor_id, collision_intervals in all_actor_collisions.items():
            collision_interval = collision_intervals[0]
            collision_bbs = collision_interval.collision_bboxes_b[collision_interval.start_idx:collision_interval.end_idx]

            world.debug.draw_string(
                location=collision_bbs[0].location,
                text=str(actor_id),
                color=self.config.other_vehicles_forecasted_bbs_color,
                life_time=self.config.draw_life_time
            )

            for bb in collision_bbs:
                world.debug.draw_box(
                    box=bb,
                    rotation=bb.rotation,
                    thickness=0.1,
                    color=self.config.ego_vehicle_forecasted_bbs_hazard_color,
                    life_time=self.config.draw_life_time
                )

        return PredictionData(
            ego_forecasted_bbs=ego_forecasted_bbs,
            veh_forecasted_bbs=veh_forecasted_bbs,
            ped_forecasted_bbs=ped_forecasted_bbs,
            all_actor_collisions=all_actor_collisions
        )

    def set_plan(self, ego_plan : EgoPlan) -> None:
        self.cur_plan = ego_plan
        self.cur_plan_status = PlanStatus.EXECUTING
        self.cur_plan_reason = None
        self.needs_replan = False

    def _archive_current_plan(self, status: PlanStatus, reason: Optional[str] = None) -> None:
        if self.cur_plan is not None:
            self.prev_plan_execution = PlanExecution(
                plan=self.cur_plan,
                status=status,
                reason=reason,
            )
        self.cur_plan = None
        self.cur_plan_status = None
        self.cur_plan_reason = None
        self.needs_replan = True

    def reset_plan(self):
        """Reset plan bookkeeping."""
        if self.cur_plan is not None:
            self._archive_current_plan(
                status=self.cur_plan_status or PlanStatus.FINISHED,
                reason=self.cur_plan_reason,
            )
        else:
            self.needs_replan = True

    def fail_plan(self, reason: str) -> None:
        self.cur_plan_status = PlanStatus.FAILED
        self.cur_plan_reason = reason
        self._archive_current_plan(status=PlanStatus.FAILED, reason=reason)

    def execute_plan(self, cur_tick) -> Tuple:
        cur_action = self.cur_plan.action
        print(f'Running Action: {cur_action.value}')

        conditions = self.cur_plan.conditions

        ########################################
        # Set initial target speed
        ########################################

        speed_limit = self.scene_data.traffic_data.speed_limit
        target_speed_initial = min(speed_limit * self.config.ratio_target_speed_limit, 72. / 3.6)  # merge the two last speed bins

        # Reduce target speed if near junction
        if self.scene_data.route_data and self.scene_data.route_data.intersection_data:
            target_speed_initial = min(target_speed_initial, self.config.max_speed_in_junction)

        # # Update target speed if provided in EgoPlan
        # if self.cur_plan.target_speed is not None:
        #     target_speed_initial = min(target_speed_initial, self.cur_plan.target_speed)

        self.target_speed_initial = target_speed_initial

        ########################################
        # Categorize all conditions
        ########################################

        # Stop conditions
        stop_conditions = {}

        # Yield and monitor conditions
        other_conditions = {}

        # TODO: Fix stop_for state transition logic
        for condition in conditions:
            cond_entry = (condition.condition_action, condition.obj_type, condition.importance)
            if condition.condition_action == ConditionAction.STOP_FOR:
                if condition.obj_type in ['stop_sign', 'traffic_light', 'obstacle']:
                    stop_conditions[condition.id] = cond_entry
            else:
                other_conditions[condition.id] = cond_entry

        ########################################
        # Command execution
        ########################################

        # Get API call
        fn = self._dispatch_map.get(cur_action)

        # Call function
        target_speed, brake, action_complete, completion_reason, route_pts, route_wps = fn(self.cur_plan, stop_conditions, other_conditions, cur_tick)

        if action_complete:
            self._archive_current_plan(
                status=PlanStatus.FINISHED,
                reason=completion_reason or "Plan conditions satisfied",
            )

        return (target_speed, brake, route_pts, route_wps)

    ########################################
    # Low-level command API
    ########################################

    # TODO: Remove stop signs/traffic lights once they've been cleared so as to avoid getting stuck at them
    def _follow_route(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        other_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        cur_tick : float
    ) -> Tuple:
        completion_reason = None

        speed_limit = self.scene_data.traffic_data.speed_limit
        ego_speed = self.scene_data.ego_data.speed

        route_index = self.waypoint_planner.route_index
        route_pts = self.waypoint_planner.route_points[route_index:]
        route_wps = self.waypoint_planner.route_waypoints[route_index:]

        # Get IDM leading vehicle target speed
        idm_lead_speed = self._idm_get_lead_speed(self.target_speed_initial)
        print(f'Leading vehicle IDM target speed: {idm_lead_speed}')

        # Get IDM stop_for object target speeds
        # NOTE: TEMP COMMENTED OUT (FUCK THESE STOP SIGNS)
        # stop_for_idm_target_speed, action_complete, completion_reason = self._stop_for(stop_conditions)


        # print(f'stop_for IDM target speed: {stop_for_idm_target_speed}, action_complete: {action_complete}')

        # TODO: Fix decision-tree for deciding when to rely on ST planner
        # Run longitudinal planner if conditions exist
        st_planner_target_speed = float('inf')

        # NOTE: TEMP - TURNING OFF ST-PLANNER
        # NOTE: ONLY ADDING PEDESTRIAN HANDLING FOR NOW
        # if len(other_conditions) > 0:
        #     prediction_data = self.predict_collisions()
        #     if self.scene_data.ped_data:
        #         ped_obj = self.scene_data.ped_data[0]
        #         # If pedestrian found in other conditions, run the ST planner
        #         if ped_obj.id in other_conditions:
        #             dist_to_ped = ped_obj.relative_distance
        #             # TODO: MANUAL DISTANCE TO PEDESTRIAN CHECK FOR ST PLANNER
        #             goal_idx = min(route_pts.shape[0] - 1, route_index + int((dist_to_ped + 10.0) * self.config.points_per_meter))
        #             print(f'goal_idx: {goal_idx}')
        #             print(f'FOUND PEDESTRIAN')
        #             st_planner_target_speed = self.long_planner.run_step(
        #                 ego_route_points_3d=route_pts,
        #                 ego_speed=ego_speed,
        #                 ego_max_speed=speed_limit,
        #                 actor_collisions=prediction_data.all_actor_collisions,
        #                 plan_tick_counter=cur_tick,
        #                 all_conditions=other_conditions,
        #                 plan_s_goal_m=dist_to_ped + 10.0
        #             )
        if len(other_conditions) > 0:
            prediction_data = self.predict_collisions()

            st_planner_target_speed = self.long_planner.run_step(
                ego_route_points_3d=route_pts,
                ego_speed=ego_speed,
                ego_max_speed=speed_limit,
                actor_collisions=prediction_data.all_actor_collisions,
                plan_tick_counter=cur_tick,
                all_conditions=other_conditions,
                plan_s_goal_m=30.0
            )
        print(f'ST Planner target speed: {st_planner_target_speed}')

        # NOTE: TEMP FUCK THIS STOP SIGN BULLSHIT
        # target_speed = min(idm_target_speed, stop_for_idm_target_speed, st_planner_target_speed)
        # brake = target_speed < 1e-2

        target_speed = min(idm_lead_speed, st_planner_target_speed)
        brake = target_speed < 1e-2

        # TODO: Fix stop_for state transition logic
        # NOTE: TEMP FUCK THIS STOP SIGN BULLSHIT
        self.route_follow_ticks += 1
        if self.route_follow_ticks > 25:
            self.route_follow_ticks = 0
            action_complete = True
            completion_reason = completion_reason or "Followed route for 1s"
        else:
            action_complete = False
            completion_reason = None
        # else:
        #     # Reset the counter so we require a fresh stretch of clear roadway
        #     # before declaring the route-follow action complete.
        #     self.route_follow_ticks = 0

        # if len(stop_conditions) == 0:
        #     self.route_follow_ticks += 1
        #     if self.route_follow_ticks > 25:
        #         self.route_follow_ticks = 0
        #         action_complete = True
        #         completion_reason = completion_reason or "Maintained route without stop conditions for 25 ticks"
        #     else:
        #         action_complete = False
        #         completion_reason = None
        # else:
        #     # Reset the counter so we require a fresh stretch of clear roadway
        #     # before declaring the route-follow action complete.
        #     self.route_follow_ticks = 0

        return target_speed, brake, action_complete, completion_reason, route_pts, route_wps

    def _turn(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        other_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        cur_tick : float
    ) -> Tuple:
        action_complete = False
        completion_reason = None
        speed_limit = self.scene_data.traffic_data.speed_limit
        ego_speed = self.scene_data.ego_data.speed

        route_index = self.waypoint_planner.route_index
        route_pts = self.waypoint_planner.route_points[route_index:]
        route_wps = self.waypoint_planner.route_waypoints[route_index:]

        # Get IDM leading vehicle target speed
        idm_target_speed = self._get_target_speed(
            self.target_speed_initial
        )
        print(f'(turn) Leading IDM target speed: {idm_target_speed}')

        # Run longitudinal planner if conditions exist
        # NOTE: TEMP - TURNING OFF ST PLANNER
        prediction_data = self.predict_collisions()

        st_planner_target_speed = idm_target_speed
        # NOTE: ONLY ADDING PEDESTRIAN HANDLING FOR NOW
        if len(other_conditions) > 0:
            prediction_data = self.predict_collisions()
            if self.scene_data.ped_data:
                ped_obj = self.scene_data.ped_data[0]
                # If pedestrian found in other conditions, run the ST planner
                if ped_obj.id in other_conditions:
                    dist_to_ped = ped_obj.relative_distance
                    # TODO: MANUAL DISTANCE TO PEDESTRIAN CHECK FOR ST PLANNER
                    goal_idx = min(route_pts.shape[0] - 1, route_index + int((dist_to_ped + 10.0) * self.config.points_per_meter))
                    print(f'goal_idx: {goal_idx}')
                    print(f'FOUND PEDESTRIAN')
                    st_planner_target_speed = self.long_planner.run_step(
                        ego_route_points_3d=route_pts,
                        ego_speed=ego_speed,
                        ego_max_speed=speed_limit,
                        actor_collisions=prediction_data.all_actor_collisions,
                        plan_tick_counter=cur_tick,
                        all_conditions=other_conditions,
                        plan_s_goal_m=dist_to_ped + 10.0
                    )

        # st_planner_target_speed = self.long_planner.run_step(
        #     ego_route_points_3d=route_pts,
        #     ego_speed=ego_speed,
        #     ego_max_speed=speed_limit,
        #     actor_collisions=prediction_data.all_actor_collisions,
        #     plan_tick_counter=cur_tick,
        #     all_conditions=other_conditions
        # )

        print(f'(turn) ST Planner target speed: {st_planner_target_speed}')

        target_speed = min(idm_target_speed, st_planner_target_speed)
        brake = target_speed < 1e-2

        if self.intersection_prev_end_idx is None:
            _, start_idx, end_idx, _ = self.scene_data.route_data.intersection_data.entry
            self.intersection_prev_end_idx = end_idx
        else:
            if route_index >= self.intersection_prev_end_idx:
                self.intersection_prev_end_idx = None
                action_complete = True
                completion_reason = "Cleared intersection segment"

        return target_speed, brake, action_complete, completion_reason, route_pts, route_wps

    def _turn_left(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        other_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        cur_tick : float
    ) -> Tuple:
        return self._turn(cur_plan, stop_conditions, other_conditions, cur_tick)

    def _turn_right(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        other_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        cur_tick : float
    ) -> Tuple:
        return self._turn(cur_plan, stop_conditions, other_conditions, cur_tick)

    def _change_lane(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        other_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        cur_tick : float
    ) -> Tuple:
        action_complete = False
        completion_reason = None
        speed_limit = self.scene_data.traffic_data.speed_limit
        ego_speed = self.scene_data.ego_data.speed

        route_index = self.waypoint_planner.route_index
        route_pts = self.waypoint_planner.route_points[route_index:]
        route_wps = self.waypoint_planner.route_waypoints[route_index:]

        # Run longitudinal planner if conditions exist
        prediction_data = self.predict_collisions()

        plan_s_goal_m = None
        if self.lane_change_prev_end_idx is not None:
            end_wp = self.waypoint_planner.route_waypoints[self.lane_change_prev_end_idx]
            cur_wp = self.waypoint_planner.route_waypoints[route_index]
            # TODO: FIX PLAN GOAL SETPOINT
            plan_s_goal_m = cur_wp.transform.location.distance(end_wp.transform.location) + 15.0
            print(f'lane change plan_s_goal_m: {plan_s_goal_m}')

        st_planner_target_speed = 0.0
        st_planner_target_speed = self.long_planner.run_step(
            ego_route_points_3d=route_pts,
            ego_speed=ego_speed,
            ego_max_speed=speed_limit,
            actor_collisions=prediction_data.all_actor_collisions,
            plan_tick_counter=cur_tick,
            all_conditions=other_conditions,
            plan_s_goal_m=plan_s_goal_m
        )

        print(f'(lane_change) ST Planner target speed: {st_planner_target_speed}')

        idm_target_speed = self._idm_get_lead_speed(self.target_speed_initial)
        print(f'(lane_change) IDM target speed: {idm_target_speed}')

        target_speed = min(idm_target_speed, st_planner_target_speed)
        brake = target_speed < 1e-2

        if self.lane_change_prev_end_idx is None:
            _, start_idx, end_idx = self.scene_data.route_data.lane_change_data.entry
            self.lane_change_prev_end_idx = end_idx
        else:
            end_wp = self.waypoint_planner.route_waypoints[self.lane_change_prev_end_idx]
            cur_wp = self.waypoint_planner.route_waypoints[route_index]

            lc_dist = cur_wp.transform.location.distance(end_wp.transform.location)
            print(f'(lane_change) route_index: {route_index}, end_idx: {self.lane_change_prev_end_idx}, lc_dist: {lc_dist}')
            # TODO: FIX LANE CHANGE END CONDITION
            if route_index >= self.lane_change_prev_end_idx and lc_dist >= 5.0:
                self.lane_change_prev_end_idx = None
                action_complete = True
                completion_reason = "Completed lane change segment"

        return target_speed, brake, action_complete, completion_reason, route_pts, route_wps

    def _change_lane_left(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        other_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        cur_tick : float
    ) -> Tuple:
        return self._change_lane(cur_plan, stop_conditions, other_conditions, cur_tick)

    def _change_lane_right(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        other_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        cur_tick : float
    ) -> Tuple:
        return self._change_lane(cur_plan, stop_conditions, other_conditions, cur_tick)

    def _overtake(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        other_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        cur_tick : float
    ) -> Tuple:
        def cumulative_arclength(points: np.ndarray) -> np.ndarray:
            """
            points: (N, 3)
            returns: (N,) cumulative distance along the polyline starting at 0
            """
            diffs = np.diff(points, axis=0)
            seg_lengths = np.linalg.norm(diffs, axis=1)
            s = np.concatenate([[0.0], np.cumsum(seg_lengths)])
            return s

        action_complete = False
        completion_reason = None
        adjust_route = False

        speed_limit = self.scene_data.traffic_data.speed_limit
        ego_speed = self.scene_data.ego_data.speed

        route_index = self.waypoint_planner.route_index
        route_pts = self.waypoint_planner.route_points
        # TODO: Adjust route waypoints after local route modifications
        route_wps = self.waypoint_planner.route_waypoints
        plan_goal_idx = None

        # Check if we need to run the lateral planner
        if self.cur_route_changes is None or route_index > self.cur_route_changes[1]:
            adjust_route = True

        if adjust_route:
            ego_loc_carla = self.ego_vehicle.get_location()
            start_point_world = np.array([ego_loc_carla.x, ego_loc_carla.y, ego_loc_carla.z])

            # TODO: NEED TO FIX HOW WE CHOOSE GOAL POINT (IDEALLY SHOULD BE PROVIDED BY LLM)
            goal_idx = min(route_pts.shape[0] - 1, route_index + int(self.config.detection_radius * self.config.points_per_meter))
            plan_goal_idx = goal_idx
            goal_point_world = route_pts[goal_idx]

            # Run lateral planner
            path_world_3d = self.lat_planner.run_step(
                route_points_world_3d=route_pts[route_index:goal_idx + 1], # goal_idx inclusive
                lidar_data=self.lidar_pts,
                start_point_world_3d=start_point_world,
                goal_point_world_3d=goal_point_world,
            )
            if path_world_3d.size > 0:
                # Find planner indices closest to start and goal in world space
                start_point = route_pts[route_index]
                goal_point  = route_pts[goal_idx]

                dist_to_start = np.linalg.norm(path_world_3d - start_point, axis=1)
                dist_to_goal  = np.linalg.norm(path_world_3d - goal_point,  axis=1)

                planner_start_idx = int(np.argmin(dist_to_start))
                planner_goal_idx  = int(np.argmin(dist_to_goal))

                # Ensure the indices are ordered correctly
                if planner_goal_idx <= planner_start_idx:
                    # Fall back to sorted order if something weird happens
                    planner_start_idx, planner_goal_idx = sorted([planner_start_idx, planner_goal_idx])

                # Clip the planner path: this removes any points "ahead" of the goal
                path_segment = path_world_3d[planner_start_idx:planner_goal_idx + 1]

                s_path = cumulative_arclength(path_segment)
                s_route = cumulative_arclength(route_pts[route_index:goal_idx])

                # Normalized param along route_subset: 0..1
                t_route = s_route / s_route[-1]

                # Target arc-lengths along the planner segment
                s_target = t_route * s_path[-1]

                # Interpolate x, y, z by arc length
                x_interp = np.interp(s_target, s_path, path_segment[:, 0])
                y_interp = np.interp(s_target, s_path, path_segment[:, 1])
                z_interp = np.interp(s_target, s_path, path_segment[:, 2])

                route_interp = np.column_stack([x_interp, y_interp, z_interp])

                # Overwrite only [start_idx : goal_idx), leave goal_idx as-is
                route_pts[route_index:goal_idx] = route_interp

                self.waypoint_planner.route_points[route_index : goal_idx] = route_interp

                # TODO: PASS IN ACTUAL DRIVING COMMAND BASED ON FUNCTION TYPE CALLED
                # mid_idx = int((route_index + goal_idx) / 2)
                # self.waypoint_planner.commands[route_index : mid_idx] = RoadOption.CHANGELANELEFT
                # self.waypoint_planner.commands[mid_idx : goal_idx] = RoadOption.CHANGELANERIGHT

                # Save current route modification start and end points
                self.cur_route_changes = (route_index, goal_idx)
        else:
            if self.cur_route_changes is not None:
                _, plan_goal_idx = self.cur_route_changes

        plan_s_goal_m = None
        if plan_goal_idx is not None and plan_goal_idx > route_index:
            route_segment = route_pts[route_index : plan_goal_idx + 1]
            if route_segment.shape[0] > 1:
                plan_s_goal_m = cumulative_arclength(route_segment)[-1]
                print(f'plan_s_goal_m: {plan_s_goal_m}')

        # Run longitudinal planner
        prediction_data = self.predict_collisions()

        st_planner_target_speed = 0.0
        st_planner_target_speed = self.long_planner.run_step(
            ego_route_points_3d=route_pts[route_index:],
            ego_speed=ego_speed,
            ego_max_speed=speed_limit,
            actor_collisions=prediction_data.all_actor_collisions,
            plan_tick_counter=cur_tick,
            all_conditions=other_conditions,
            plan_s_goal_m=plan_s_goal_m
        )

        target_speed = st_planner_target_speed
        brake = target_speed < 1e-2

        if self.cur_route_changes is None:
            self.cur_route_changes = (route_index, goal_idx)
        else:
            _, end_idx = self.cur_route_changes
            if route_index >= end_idx:
                self.cur_route_changes = None
                action_complete = True
                completion_reason = "Completed overtake path"

        return target_speed, brake, action_complete, completion_reason, route_pts[route_index:], route_wps[route_index:]

    def _overtake_left(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        other_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        cur_tick : float
    ) -> Tuple:
        return self._overtake(cur_plan, stop_conditions, other_conditions, cur_tick)

    def _overtake_right(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        other_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        cur_tick : float
    ) -> Tuple:
        return self._overtake(cur_plan, stop_conditions, other_conditions, cur_tick)

    ########################################
    # Low-level condition API
    ########################################

    def _stop_for(
        self,
        stop_conditions : Dict[int, Tuple[ConditionAction, str, float]],
    ) -> Tuple[float, bool, Optional[str]]:
        action_complete = False
        completion_reason = None
        target_speed = self.target_speed_initial

        for actor_id, condition in stop_conditions.items():
            print(f'stop_for actor_id: {actor_id}, type: {condition[1]}')
            if condition[1] == 'stop_sign':
                target_speed_ss, action_complete, completion_reason = self._process_stop_sign()
                target_speed = min(target_speed, target_speed_ss)
            elif condition[1] == 'traffic_light':
                target_speed_tl, action_complete, completion_reason = self._process_traffic_light()
                target_speed = min(target_speed, target_speed_tl)
            elif condition[1] == 'obstacle':
                # TODO: IMPLEMENT BEHAVIOUR
                action_complete = True
                completion_reason = "Encountered obstacle stop condition"

        return target_speed, action_complete, completion_reason

    ########################################
    # IDM Helpers
    ########################################
    def _idm_get_lead_speed(
        self,
        target_speed_initial : float,
    ) -> float:
        ego_speed = self.scene_data.ego_data.speed
        idm_target_speed = target_speed_initial
        leading_vehicles : Dict[str, List[LaneVehicleData]] = self.scene_data.vehicle_data['leading']
        if leading_vehicles:
            lane_vehicle_data : List[LaneVehicleData] = []
            lane_vehicle_data = leading_vehicles['ego']
            lane_leading_vehicles : List[VehicleData] = []

            # Check for leading vehicles in the ego lane
            for lane_veh_data in lane_vehicle_data:
                lane_leading_vehicles.extend(lane_veh_data.vehicle_data)

            if lane_leading_vehicles:
                lv = lane_leading_vehicles[0]
                lv_speed = lv.speed
                lv_length = lv.vehicle.bounding_box.extent.x * 2

                dist_to_lv = lv.relative_distance

                desired_following_distance = self.config.idm_leading_vehicle_minimum_distance
                desired_time_headway = self.config.idm_leading_vehicle_time_headway

                idm_target_speed = self.idm.compute_target_speed_idm(
                    desired_speed = target_speed_initial,
                    leading_actor_length = lv_length,
                    ego_speed = ego_speed,
                    leading_actor_speed = lv_speed,
                    distance_to_leading_actor = dist_to_lv,
                    s0 = desired_following_distance,
                    T = desired_time_headway
                )
                print(f'Lead Vehicle ID: {lv.id}, IDM Speed: {idm_target_speed}\n')
        return idm_target_speed

    def _idm_get_obstacle_speed(
        self,
        target_speed_initial : float,
    ) -> float:
        # TODO: REFACTOR THIS SHIT
        ego_speed = self.scene_data.ego_data.speed
        idm_target_speed = target_speed_initial
        if self.scene_data.nearest_obstacle:
            obstacle_obj = self.scene_data.nearest_obstacle.obstacle

            idm_target_speed = self.idm.compute_target_speed_idm(
                desired_speed=target_speed_initial,
                leading_actor_length=obstacle_obj.bounding_box.extent.x * 2 + self.scene_data.ego_data.ego_vehicle.bounding_box.extent.x,
                ego_speed=ego_speed,
                leading_actor_speed=0.0,
                distance_to_leading_actor=self.scene_data.nearest_obstacle.relative_distance,
                s0=self.config.idm_leading_vehicle_minimum_distance,
                T=self.config.idm_leading_vehicle_time_headway
            )
            print(f'Obstacle ID: {self.scene_data.nearest_obstacle.id}, IDM Speed: {idm_target_speed}\n')
        return idm_target_speed

    def _idm_get_ped_speed(
        self,
        target_speed_initial : float,
    ) -> float:
        ego_speed = self.scene_data.ego_data.speed
        idm_target_speed = target_speed_initial
        if self.scene_data.ped_data:
            ped_obj = self.scene_data.ped_data[0]

            idm_target_speed = self.idm.compute_target_speed_idm(
                desired_speed=target_speed_initial,
                leading_actor_length=0.5 + self.scene_data.ego_data.ego_vehicle.bounding_box.extent.x,
                ego_speed=ego_speed,
                leading_actor_speed=0.0,
                distance_to_leading_actor=ped_obj.relative_distance,
                s0=self.config.idm_pedestrian_minimum_distance,
                T=self.config.idm_pedestrian_desired_time_headway,
            )
            print(f'Pedestrian ID: {ped_obj.id}, IDM Speed: {idm_target_speed}\n')
        return idm_target_speed

    def _get_target_speed(
        self,
        target_speed_initial : float,
    ) -> float:
        ego_speed = self.scene_data.ego_data.speed

        print(f'Ego Speed: {ego_speed}\n')
        print(f'Target Speed Initial: {target_speed_initial}\n')

        idm_lead_speed = self._idm_get_lead_speed(target_speed_initial)
        idm_obstacle_speed = self._idm_get_obstacle_speed(target_speed_initial)
        idm_ped_speed = self._idm_get_ped_speed(target_speed_initial)

        return min(target_speed_initial, idm_lead_speed, idm_obstacle_speed, idm_ped_speed)

    def follow_route(
        self,
        target_speed_initial : float
    ) -> Tuple:
        target_speed = self._get_target_speed(target_speed_initial)
        start_index = self.waypoint_planner.route_index
        return target_speed, self.waypoint_planner.route_points[start_index:], self.waypoint_planner.route_waypoints[start_index:]

    # def _brake(self, param = None):
    #     target_speed = self.__get_target_speed(0.0, LongitudinalParams())
    #     start_index = self.waypoint_planner.route_index
    #     return target_speed, self.waypoint_planner.route_points[start_index:], self.waypoint_planner.route_waypoints[start_index:]

    def _process_stop_sign(self) -> Tuple[float, bool, Optional[str]]:
        ego_speed = self.scene_data.ego_data.speed
        target_speed_initial = self.scene_data.traffic_data.speed_limit
        next_ss = self.scene_data.traffic_data.next_stop_sign
        if next_ss is None:
            return target_speed_initial, True, "No stop sign detected"

        if ego_speed < 0.1 and next_ss.distance_to_stop_sign < self.config.clearing_distance_to_stop_sign:
            self.ss_wait_ticks += 1
            if self.ss_wait_ticks > 25:
                self.ss_cleared = True
        else:
            self.ss_wait_ticks = 0
            self.ss_cleared = False

        if self.ss_cleared:
            target_speed = target_speed_initial
            reason = "Finished waiting at stop sign. Can ignore."
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
            reason = None

        return target_speed, self.ss_cleared, reason

    def _process_traffic_light(self) -> Tuple[float, bool, Optional[str]]:
        ego_speed = self.scene_data.ego_data.speed
        target_speed_initial = self.scene_data.traffic_data.speed_limit
        next_tl = self.scene_data.traffic_data.next_traffic_light
        if next_tl is None:
            return target_speed_initial, True, "No traffic light detected"

        if next_tl.state == "GREEN":
            target_speed = target_speed_initial
            self.tl_cleared = True
            reason = "Traffic light turned green"
        else:
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
            reason = None

        return target_speed, self.tl_cleared, reason
