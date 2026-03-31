import carla
import numpy as np
import time

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Literal, Optional, Union, Deque
from collections import deque

from config import GlobalConfig

from agents.navigation.local_planner import RoadOption
from team_code.scene_descriptor.data_extractors.vehicle_data_extractor import VehicleDataEntry, VehicleData
from team_code.scene_descriptor.data_extractors.collision_data_extractor import CollisionData
from team_code.scene_descriptor.scene_descriptor import SceneContext, SceneData
from team_code.scene_descriptor.data_extractors.route_extractor import LaneChangeData
from privileged_route_planner import PrivilegedRoutePlanner, PlannerState

from team_code.scene_analyzer.parsers.base_pydantic_models import *
from team_code.scene_analyzer.parsers.ego_plan_pydantic_models import *

from .idm import IDM
from .state_machines import *

from team_code.actor_prediction.motion_prediction import MotionPrediction, PredictionData
from team_code.actor_prediction.geometric_utils import GeometricUtils

# Collision checker
from team_code.actor_prediction.collision_checker import CollisionChecker

# Longitudinal planner
from local_planner.longitudinal.long_planner import LongPlanner

# Lateral planner
from local_planner.lateral.lat_planner import LatPlanner, LatPlannerResult


class TrajectoryPlanner:
    def __init__(self, config : GlobalConfig, carla_map : carla.Map, ego_vehicle : carla.Vehicle):
        self.config = config
        self.carla_map = carla_map
        self.ego_vehicle = ego_vehicle

        self.scene_data : SceneData = None
        self.lidar_pts : Dict = None

        self.waypoint_planner = PrivilegedRoutePlanner(self.config)
        self.motion_predictor = MotionPrediction(self.config, self.carla_map)

        self.idm = IDM(self.config)

        # Longitudinal planner
        self.long_planner = LongPlanner(
            config=self.config,
        )

        # Lateral planner
        self.lat_planner = LatPlanner(
            config=self.config,
        )

        self.dispatch_map = {
            Action.FOLLOW_ROUTE: self._follow_route,

            Action.TURN_LEFT: self._turn_left,
            Action.TURN_RIGHT: self._turn_right,
            Action.TURN_STRAIGHT : self._turn_straight,

            Action.CHANGE_LANE_LEFT: self._change_lane_left,
            Action.CHANGE_LANE_RIGHT: self._change_lane_right,

            Action.OVERTAKE_LEFT: self._overtake_left,
            Action.OVERTAKE_RIGHT: self._overtake_right,

            Action.PULL_OVER_LEFT: self._pull_over_left,
            Action.PULL_OVER_RIGHT: self._pull_over_right,
            Action.PULL_OVER_IN_LANE: self._pull_over_in_lane,

            Action.SHARE_LANE: self._share_lane,
        }

        # State machines
        self.follow_route_sm = FollowRouteSM(config)
        self.turn_sm = TurnSM(config)
        self.lane_change_sm = LaneChangeSM(config)
        self.overtake_sm = OvertakeSM(config)
        self.pull_over_sm = PullOverSM(config)
        self.share_lane_sm = ShareLaneSM(config)

        # Maps each Action to its owning SM — used to distinguish CLEARED vs FAILED completions
        self.action_sm_map = {
            Action.FOLLOW_ROUTE:    self.follow_route_sm,
            Action.TURN_LEFT:       self.turn_sm,
            Action.TURN_RIGHT:      self.turn_sm,
            Action.TURN_STRAIGHT:   self.turn_sm,
            Action.CHANGE_LANE_LEFT:  self.lane_change_sm,
            Action.CHANGE_LANE_RIGHT: self.lane_change_sm,
            Action.OVERTAKE_LEFT:   self.overtake_sm,
            Action.OVERTAKE_RIGHT:  self.overtake_sm,
            Action.PULL_OVER_LEFT:  self.pull_over_sm,
            Action.PULL_OVER_RIGHT: self.pull_over_sm,
            Action.PULL_OVER_IN_LANE: self.pull_over_sm,
            Action.SHARE_LANE:      self.share_lane_sm,
        }

        # State vars
        self.cur_plan_state: Optional[PlanState] = None
        self.prev_plan_state: Optional[PlanState] = None
        self.needs_replan : bool = True

        # Last prediction result — exposed for external consumers (e.g. loggers)
        self.prediction_data : Optional[PredictionData] = None

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

    @staticmethod
    def _cumulative_arclength(points: np.ndarray) -> np.ndarray:
        if points.shape[0] < 2:
            return np.zeros(points.shape[0])

        diffs = np.diff(points, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        return np.concatenate([[0.0], np.cumsum(seg_lengths)])

    def _compute_ego_station_from_index(self, route_points: np.ndarray, start_index: int) -> float:
        route_index = self.waypoint_planner.route_index
        print(f'Start_index: {start_index}, Route index: {route_index}')
        print(f'route_pts shape: {route_points.shape}')
        if start_index >= route_points.shape[0]:
            return 0.0

        ego_loc = self.ego_vehicle.get_location()
        ego_xy = np.array([[ego_loc.x, ego_loc.y, ego_loc.z]], dtype=np.float32)
        print(f'ego_xy: {ego_xy}, route_index point: {route_points[route_index]}')

        route_subset = route_points[start_index:route_index]
        # print(f'route_subset shape: {route_subset.shape}')
        if route_subset.size == 0:
            return 0.0

        route_subset[-1] = ego_xy

        s_route = self._cumulative_arclength(route_subset)
        s_ego = s_route[-1]
        print(f's_ego: {s_ego}')

        return s_ego
    # TODO: PROPERLY IMPLEMENT THE EGO STATION CHECK, MAYBE PRECOMPUTE ROUTE DURING INITIALIZATION OR AFTER ROUTE CHANGES

    def update_scene_data(
        self,
        scene_context : SceneContext,
        lidar_pts : Dict,
    ) -> None:
        self.scene_data = scene_context.scene_data
        self.lidar_pts = lidar_pts

    def plan_with_reasoning(self) -> bool:
        needs_reasoning = False

        # Traffic context
        traffic_data = self.scene_data.traffic_data

        process_ss = traffic_data.next_stop_sign != None and traffic_data.next_stop_sign.distance_to_stop_sign <= self.config.stop_sign_processing_distance_m
        process_tl = traffic_data.next_traffic_light != None and traffic_data.next_traffic_light.distance_to_light <= self.config.traffic_light_processing_distance_m

        needs_reasoning = needs_reasoning or process_ss
        needs_reasoning = needs_reasoning or process_tl

        # Intersection approach / execution — covers cases where the stop sign or
        # traffic light is no longer the "next" one but we're still inside the junction.
        intersection_data = self.scene_data.route_data.intersection_data
        if intersection_data is not None:
            needs_reasoning = needs_reasoning or intersection_data.inside_intersection
            needs_reasoning = needs_reasoning or intersection_data.distance_to_intersection <= self.config.intersection_processing_distance_m

        if self.scene_data.ped_data:
            for ped in self.scene_data.ped_data:
                needs_reasoning = needs_reasoning or ped.relative_distance <= self.config.pedestrian_processing_distance_m

        needs_reasoning = needs_reasoning or (self.scene_data.route_data.lane_change_data != None and self.scene_data.route_data.lane_change_data.distance_to_lane_change <= self.config.lane_change_processing_distance_m)

        ego_obstacles = self.scene_data.obstacle_data.ego_obstacles
        if ego_obstacles:
            needs_reasoning = needs_reasoning or ego_obstacles[0].relative_distance <= self.config.obstacle_processing_distance_m

        # TODO: TEMP INTRUDING VEHICLES DETECTION
        # NOTE: BETTER ARCHITECTURE/DESIGN PREFERRED
        # needs_reasoning = needs_reasoning or self.scene_data.all_intruding_vehicles

        # NOTE/TODO: TEMP EMERGENCY VEHICLE DETECTION
        # TODO: TEMP CYCLIST DETECTION

        all_special_vehicles = self.scene_data.vehicle_data.get(vehicle_types={'cyclist', 'emergency_vehicle'})
        for special_vehicle in all_special_vehicles:
            if special_vehicle.vehicle_type == 'cyclist':
                needs_reasoning = needs_reasoning or special_vehicle.relative_distance <= self.config.cyclist_processing_distance_m
            elif special_vehicle.vehicle_type == 'emergency_vehicle':
                needs_reasoning = needs_reasoning or special_vehicle.emergency_sirens_active

        return needs_reasoning

    def predict_collisions(self, overlap_len : float = 80.0) -> PredictionData:
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
        ego_forecasted_bbs = self.motion_predictor.predict_ego_motion(
            ego_vehicle=self.ego_vehicle,
            ego_route_pts=self.waypoint_planner.route_points[self.waypoint_planner.route_index:],
            target_speed=None if ego_velocity_profile is not None else self.target_speed_initial,
            velocity_profile=ego_velocity_profile,
        )

        # Forecast all vehicles
        veh_forecasted_bbs, veh_dilated_forecasted_bbs = self.motion_predictor.predict_vehicle_motion(self.scene_data.vehicle_data, self.scene_data.traffic_data.speed_limit)

        # world = self.ego_vehicle.get_world()
        # crossing_vehs = self.scene_data.vehicle_data.get(traffic_type="oncoming")
        # for c_data in crossing_vehs:
        #     if c_data.id in veh_forecasted_bbs:
        #         for bb in veh_forecasted_bbs[c_data.id]:
        #             world.debug.draw_box(
        #                 box=bb,
        #                 rotation=bb.rotation,
        #                 thickness=0.1,
        #                 color=self.config.ego_vehicle_forecasted_bbs_hazard_color,
        #                 life_time=self.config.draw_life_time
        #             )

        # Forecast all pedestrians
        ped_forecasted_bbs = self.motion_predictor.predict_ped_motion(self.scene_data.ped_data)

        # Compute all collisions
        actor_predictions = veh_forecasted_bbs | ped_forecasted_bbs
        all_actor_collisions = CollisionChecker.predict_actor_collisions(
            ego_bb_preds=ego_forecasted_bbs,
            actor_predictions=actor_predictions
        )

        # NOTE: Debugging

        print(f'\n\nCOLLISIONS')
        print(f'\t\tNum collisions: {len(all_actor_collisions.keys())}')
        for actor_id, collision_intervals in all_actor_collisions.items():
            print(f'\t\tactor id: {actor_id}')
            collision_interval = collision_intervals[0]
            collision_bbs = collision_interval.collision_bboxes_b[collision_interval.start_idx:collision_interval.end_idx + 1]

            ego_bbs = collision_interval.collision_bboxes_a[collision_interval.start_idx:collision_interval.end_idx + 1]

            # world.debug.draw_string(
            #     location=collision_bbs[0].location,
            #     text=str(actor_id),
            #     color=self.config.other_vehicles_forecasted_bbs_color,
            #     life_time=self.config.draw_life_time
            # )

            # bb = collision_bbs[0]
            # world.debug.draw_box(
            #     box=bb,
            #     rotation=bb.rotation,
            #     thickness=0.1,
            #     color=self.config.ego_vehicle_forecasted_bbs_hazard_color,
            #     life_time=self.config.draw_life_time
            # )
            # for bb in collision_bbs:
            #     world.debug.draw_box(
            #         box=bb,
            #         rotation=bb.rotation,
            #         thickness=0.1,
            #         color=self.config.ego_vehicle_forecasted_bbs_hazard_color,
            #         life_time=self.config.draw_life_time
            #     )

            # bb = ego_bbs[0]
            # world.debug.draw_box(
            #     box=bb,
            #     rotation=bb.rotation,
            #     thickness=0.1,
            #     color=self.config.other_vehicles_forecasted_bbs_color,
            #     life_time=self.config.draw_life_time
            # )

            # for bb in ego_bbs:
            #     world.debug.draw_box(
            #         box=bb,
            #         rotation=bb.rotation,
            #         thickness=0.1,
            #         color=self.config.other_vehicles_forecasted_bbs_color,
            #         life_time=self.config.draw_life_time
            #     )

        # NOTE TEMP FIX
        route_index = self.waypoint_planner.route_index

        bb_from_index = self.config.dense_route_idx_to_bb_route_idx(route_index)

        max_route_bbs = len(self.waypoint_planner.route_bbs)
        lookahead_distance = self.config.meters_to_bb_route_idx(overlap_len)
        bb_to_index = min(max_route_bbs, bb_from_index + lookahead_distance + 1)

        all_actor_overlaps = CollisionChecker.get_lane_overlaps(
            veh_tbb_predictions=veh_forecasted_bbs,
            veh_ebb_predictions=veh_dilated_forecasted_bbs,
            ped_predictions=ped_forecasted_bbs,
            route_subset_bbs=self.waypoint_planner.route_bbs[bb_from_index:bb_to_index]
        )
        print(f'\n\nOVERLAPS')
        print(f'\tNum overlaps: {len(all_actor_overlaps.keys())}')
        for actor_id, overlap_interval in all_actor_overlaps.items():
            print(f'\t\tactor id: {actor_id}')

            space_start = overlap_interval.space_start_idx
            space_end = overlap_interval.space_end_idx

            time_start = overlap_interval.time_start_idx
            time_end = overlap_interval.time_end_idx

            print(f'\t\ttime_start: {time_start}, time_end: {time_end}')
            print(f'\t\tspace_start: {space_start}, space_end: {space_end}')

            overlap_route_bbs = overlap_interval.route_subset_bboxes[space_start:space_end + 1]
            overlap_actor_bbs = overlap_interval.actor_ebb_bboxes[time_start:time_end + 1]
            # overlap_actor_bbs = overlap_interval.overlap_actor_bboxes[:30]

            # bb = overlap_actor_bbs[0]
            # world.debug.draw_box(
            #     box=bb,
            #     rotation=bb.rotation,
            #     thickness=0.1,
            #     color=self.config.ego_vehicle_forecasted_bbs_hazard_color,
            #     life_time=self.config.draw_life_time
            # )

            # bb = overlap_route_bbs[0]
            # world.debug.draw_box(
            #     box=bb,
            #     rotation=bb.rotation,
            #     thickness=0.1,
            #     color=self.config.pedestrian_forecasted_bbs_color,
            #     life_time=self.config.draw_life_time
            # )

        # ego_obs = self.scene_data.obstacle_data.ego_obstacles
        # print(f'\n\nOBSTACLE INTRUSIONS')
        # print(f'\tNum ego obstacles: {len(ego_obs)}')
        # for obs in ego_obs:
        #     intrusion_index_dense = obs.intrusion_idx
        #     intrusion_idx_bb = self.config.dense_route_idx_to_bb_route_idx(intrusion_index_dense)
        #     bb = self.waypoint_planner.route_bbs[intrusion_idx_bb]
        #     world.debug.draw_box(
        #         box=bb,
        #         rotation=bb.rotation,
        #         thickness=0.1,
        #         color=self.config.ego_vehicle_forecasted_bbs_hazard_color,
        #         life_time=self.config.draw_life_time
        #     )

        #     bb = GeometricUtils.get_actor_bbox(obs.obstacle)
        #     world.debug.draw_box(
        #         box=bb,
        #         rotation=bb.rotation,
        #         thickness=0.1,
        #         color=self.config.pedestrian_forecasted_bbs_color,
        #         life_time=self.config.draw_life_time
        #     )

        self.prediction_data = PredictionData(
            ego_forecasted_bbs=ego_forecasted_bbs,
            veh_forecasted_bbs=veh_forecasted_bbs,
            veh_dilated_forecasted_bbs=veh_dilated_forecasted_bbs,
            ped_forecasted_bbs=ped_forecasted_bbs,
            all_actor_collisions=all_actor_collisions,
            all_actor_overlaps=all_actor_overlaps
        )
        return self.prediction_data

    @property
    def cur_plan(self) -> Optional[EgoPlan]:
        return self.cur_plan_state.plan if self.cur_plan_state else None

    def set_plan(self, ego_plan: EgoPlan, hl_beh, scene_text=None, scene_image=None) -> None:
        self.cur_plan_state = PlanState(
            plan=ego_plan,
            hl_beh=hl_beh,
            status=PlanStatus.EXECUTING,
            scene_text=scene_text,
            scene_image=scene_image,
        )
        self.needs_replan = False

    def _archive_current_plan(
        self,
        status: PlanStatus,
        reason: Optional[str] = None,
        collision_events: Optional[Deque[CollisionData]] = None,
    ) -> None:
        if self.cur_plan_state is not None:
            print(f'\n\nARCHIVING CURRENT PLAN, COLLISION EVENTS: {collision_events}\n\n')
            self.prev_plan_state = PlanState(
                plan=self.cur_plan_state.plan,
                hl_beh=self.cur_plan_state.hl_beh,
                status=status,
                reason=reason,
                collision_events=collision_events,
                scene_text=self.cur_plan_state.scene_text,
                scene_image=self.cur_plan_state.scene_image,
            )
        self.cur_plan_state = None
        self.needs_replan = True

    def reset_plan(self):
        """Reset plan bookkeeping."""
        if self.cur_plan_state is not None:
            self._archive_current_plan(
                status=self.cur_plan_state.status or PlanStatus.FINISHED,
                reason=self.cur_plan_state.reason,
            )
        else:
            self.needs_replan = True

    def fail_plan(self, reason: str) -> None:
        self._archive_current_plan(status=PlanStatus.FAILED, reason=reason)

    def execute_plan(self, cur_tick) -> Tuple:
        ########################################
        # Set initial target speed
        ########################################

        speed_limit = self.scene_data.traffic_data.speed_limit
        target_speed_initial = min(speed_limit * self.config.ratio_target_speed_limit, 72. / 3.6)  # merge the two last speed bins

        # Reduce target speed if near junction
        if self.scene_data.route_data.intersection_data:
            target_speed_initial = min(target_speed_initial, self.config.max_speed_in_junction)

        # Reduce target speed to be slightly faster than trailing vehicle during lane changes
        if self.scene_data.route_data.lane_change_data:
            lc_dir = self.scene_data.route_data.lane_change_data.target_maneuver
            target_lane = "left" if lc_dir == RoadOption.CHANGELANELEFT else "right"

            trailing_vehicles = self.scene_data.vehicle_data.get(traffic_type="trailing", lane_name=target_lane)
            leading_vehicles = self.scene_data.vehicle_data.get(traffic_type="leading", lane_name=target_lane)

            v_target = target_speed_initial

            ego_speed = self.scene_data.ego_data.speed

            if trailing_vehicles:
                target_vehicle = trailing_vehicles[0]
                dist_to_trail = max(target_vehicle.relative_distance, 0.1)

                v_delta = np.clip(10.0 / dist_to_trail, 1.0, 5.0)
                v_pull_away = target_vehicle.speed + v_delta

                if ego_speed > target_vehicle.speed:
                    # Ego is already pulling away.
                    # Cap the speed to match traffic flow (+ delta) to prevent rocketing
                    # to the speed limit and requiring hard braking later.
                    v_target = min(v_target, v_pull_away)
                else:
                    # Ego is slower than the trailer and losing the gap.
                    # Do not cap; keep the target at the speed limit to encourage
                    # acceleration and secure the lane change safely.
                    pass

            if leading_vehicles:
                target_vehicle = leading_vehicles[0]

                v_lead = self._idm_get_lead_speed(target_speed_initial=v_target, target_vehicle=target_vehicle, desired_following_distance=1.0)
                v_lead = max(target_vehicle.speed, v_lead)

                v_target = min(v_target, v_lead)

            target_speed_initial = min(target_speed_initial, v_target)

        # # Update target speed if provided in EgoPlan
        # if self.cur_plan.target_speed is not None:
        #     target_speed_initial = min(target_speed_initial, self.cur_plan.target_speed)

        self.target_speed_initial = target_speed_initial

        if self.cur_plan is not None:
            cur_action = self.cur_plan.action
            print(f'Running Action: {cur_action.value}')

            conditions = self.cur_plan.conditions

            ########################################
            # Categorize all conditions
            ########################################

            # Stop conditions
            stop_conditions = {}

            # Yield and monitor conditions
            other_conditions = {}

            for condition in conditions:
                cond_key = (condition.condition_action.value, condition.target.actor_type.value)
                if condition.condition_action == ConditionAction.STOP_FOR:
                    if condition.target.actor_type in {EntityType.STOP_SIGN, EntityType.STOP_LIGHT, EntityType.OBSTACLE, EntityType.PEDESTRIAN}:
                        stop_conditions[cond_key] = condition
                        continue
                other_conditions[cond_key] = condition

            ########################################
            # Resolve abstract conditions to actor IDs
            ########################################
            resolved_conditions = self._resolve_conditions_to_actors(other_conditions)

            ########################################
            # Command execution
            ########################################

            # Get API call
            fn = self.dispatch_map.get(cur_action)

            # Call function
            target_speed, brake, action_complete, completion_reason, route_pts, route_wps, collision_events = fn(self.cur_plan, stop_conditions, resolved_conditions, cur_tick)

            if action_complete:
                sm = self.action_sm_map.get(cur_action)
                plan_status = PlanStatus.FINISHED if (sm is not None and sm.is_cleared) else PlanStatus.FAILED
                self._archive_current_plan(
                    status=plan_status,
                    reason=completion_reason or "Plan conditions satisfied",
                    collision_events=collision_events,
                )
                self.long_planner.reset_plan()
                self.lat_planner.reset_plan()
        else:
            print(f'Running Action: default follow_route')
            target_speed, brake, route_pts, route_wps = self.follow_route(self.target_speed_initial)

        return (target_speed, brake, route_pts, route_wps)

    ########################################
    # Low-level command API
    ########################################

    # TODO: Remove stop signs/traffic lights once they've been cleared so as to avoid getting stuck at them
    def _follow_route(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[Tuple[str, str], ConditionCommand],
        resolved_conditions : Dict[int, Tuple[str, str, str, str]],
        cur_tick : float
    ) -> Tuple:
        # Reset state machine if cleared
        if self.follow_route_sm.is_cleared or self.follow_route_sm.is_failed:
            self.follow_route_sm.reset()
            self.long_planner.reset_plan()

        # Update state machine conditions
        self.follow_route_sm.update_conditions(self.cur_plan_state)

        # Get planner state
        planner_state = self.get_planner_state()
        route_index = planner_state.route_index
        route_pts = planner_state.route_points[route_index:]
        route_wps = planner_state.route_waypoints[route_index:]
        s_route = planner_state.s_route

        # Execute one step in state machine. Downstream functions generate target speeds based on current state
        cmd_status = self.follow_route_sm.update_state(
            planner_state=planner_state,
            scene_data=self.scene_data,
        )

        speed_limit = self.scene_data.traffic_data.speed_limit

        # Get IDM leading vehicle target speed
        idm_lead_speed = self._idm_get_lead_speed(self.target_speed_initial)

        idm_obs_speed = self._idm_get_obstacle_speed(self.target_speed_initial)

        # Get IDM stop_for object target speeds
        stop_for_idm_target_speed = self._stop_for(stop_conditions, self.follow_route_sm.conditions_registry)

        # TODO: Fix decision-tree for deciding when to rely on ST planner
        # Run longitudinal planner if conditions exist
        st_planner_target_speed = float('inf')

        if len(resolved_conditions) > 0:
            prediction_data = self.predict_collisions()
            plan_route_start_idx = self.long_planner.plan_route_start_idx
            if plan_route_start_idx < 0 or not self.long_planner.has_active_plan:
                print(f'SETTING TO ROUTE INDEX, PLAN ROUTE IDX IS UNSET')
                plan_route_start_idx = route_index

            s_ego_m = s_route[route_index] - s_route[plan_route_start_idx]
            print(f'EGO STATION: {s_ego_m}')

            # TODO: THIS IS ACTUAL BS, FIX THIS. WE NEED A BETTER WAY TO SELECT TARGET S DISTANCE
            def get_s_target(stop_conditions):
                if not stop_conditions:
                    return 30.0, speed_limit

                plan_s_goal_m = 10.0
                for cond_key, condition in stop_conditions.items():
                    actor_type = condition.target.actor_type
                    if actor_type == EntityType.STOP_SIGN:
                        if self.scene_data.traffic_data and self.scene_data.traffic_data.next_stop_sign:
                            plan_s_goal_m = max(plan_s_goal_m, self.scene_data.traffic_data.next_stop_sign.distance_to_stop_sign)
                    elif actor_type == EntityType.STOP_LIGHT:
                        if self.scene_data.traffic_data and self.scene_data.traffic_data.next_traffic_light:
                            plan_s_goal_m = max(plan_s_goal_m, self.scene_data.traffic_data.next_traffic_light.distance_to_light)
                return plan_s_goal_m, 3.0

            goal_s, goal_speed = get_s_target(stop_conditions)
            plan_s_goal_m = min(goal_s, 30.0)
            print(f'\n\nFOLLOW ROUTE S TARGET')
            print(f'\tS: {plan_s_goal_m}')

            long_plan_result = self.long_planner.run_step(
                plan_tick_counter=cur_tick,
                planner_state=planner_state,
                scene_data=self.scene_data,
                prediction_data=prediction_data,
                all_conditions=resolved_conditions,
                plan_s_goal_m=plan_s_goal_m,
                s_ego_m=s_ego_m,
                ego_cruise_speed=self.target_speed_initial,
                ego_goal_speed=goal_speed,
                s_bounds=(0.0, plan_s_goal_m),
            )
            st_planner_target_speed = long_plan_result.target_speed

        print(f'follow_route speeds -> speed_limit: {speed_limit}, idm: {idm_lead_speed}, stop_for: {stop_for_idm_target_speed}, st_plan: {st_planner_target_speed}, obs: {idm_obs_speed}')
        target_speed = min(idm_lead_speed, stop_for_idm_target_speed, st_planner_target_speed, idm_obs_speed)
        brake = target_speed < 1e-2

        print(f'\n\nCommand Status:')
        print(f'{cmd_status.to_string()}')

        action_complete = self.follow_route_sm.is_cleared or self.follow_route_sm.is_failed
        collision_events = None if not action_complete else self.follow_route_sm.collision_events
        if collision_events:
            print(f'\n\n\nSTORED COLLISION EVENTS LOGGING')
        completion_reason = cmd_status.reasons[0]

        return target_speed, brake, action_complete, completion_reason, route_pts, route_wps, collision_events

    def _turn(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[Tuple[str, str], ConditionCommand],
        resolved_conditions : Dict[int, Tuple[str, str, str, str]],
        turn_cmd : Action,
        cur_tick : float,
    ) -> Tuple:
        # If fresh command, clear previous state
        if self.turn_sm.is_cleared or self.turn_sm.is_failed:
            self.turn_sm.reset()
            self.long_planner.reset_plan()

        # Activate the command state
        self.turn_sm.activate()

        # Update state machine conditions
        self.turn_sm.update_conditions(self.cur_plan_state)

        # Get current planner state
        planner_state = self.get_planner_state()

        route_index = planner_state.route_index
        max_route_len = planner_state.route_points.shape[0]
        route_pts = planner_state.route_points[route_index:]
        route_wps = planner_state.route_waypoints[route_index:]
        s_route = planner_state.s_route

        speed_limit = self.scene_data.traffic_data.speed_limit

        # Execute one step in state machine. Downstream functions generate target speeds based on current state
        cmd_status = self.turn_sm.update_state(
            scene_data=self.scene_data,
            planner_state=planner_state,
            turn_cmd=turn_cmd,
        )

        # Get predictions
        prediction_data = self.predict_collisions()

        # Get IDM leading vehicle target speed
        idm_lead_speed = self._idm_get_lead_speed(self.target_speed_initial)

        idm_obs_speed = self._idm_get_obstacle_speed(self.target_speed_initial)

        # Get IDM stop_for object target speeds
        stop_for_idm_target_speed = self._stop_for(stop_conditions, self.turn_sm.conditions_registry)

        st_planner_target_speed = 0.0

        plan_route_start_idx = self.long_planner.plan_route_start_idx
        if plan_route_start_idx < 0 or not self.long_planner.has_active_plan:
            print(f'SETTING TO ROUTE INDEX, PLAN ROUTE IDX IS UNSET')
            plan_route_start_idx = route_index

        s_ego_m = s_route[route_index] - s_route[plan_route_start_idx]
        print(f'EGO STATION: {s_ego_m}')

        plan_s_goal_m = 30.0
        s_bounds = None
        ego_cruise_speed = None
        ego_goal_speed = None

        if self.turn_sm.has_active_plan:
            turn_start_idx, turn_end_idx = self.turn_sm.cur_plan_idxs
            print(f'\n\nTURN INDICES: START: {turn_start_idx}, END: {turn_end_idx}')

            s_to_start = s_route[turn_start_idx] - s_route[route_index]
            s_to_end = s_route[turn_end_idx] - s_route[route_index]

            plan_s_goal_m = max(10.0, s_to_end)
            s_bounds = (0.0, max(0.0, plan_s_goal_m))

            # idm_lead_speed = self.target_speed_initial
            idm_lead_speed = speed_limit
            ego_cruise_speed = self.target_speed_initial
            # ego_goal_speed = self.target_speed_initial
            ego_goal_speed = speed_limit

        long_plan_result = self.long_planner.run_step(
            plan_tick_counter=cur_tick,
            planner_state=planner_state,
            scene_data=self.scene_data,
            prediction_data=prediction_data,
            all_conditions=resolved_conditions,
            plan_s_goal_m=plan_s_goal_m,
            s_ego_m=s_ego_m,
            ego_cruise_speed=ego_cruise_speed,
            ego_goal_speed=ego_goal_speed,
            s_bounds=s_bounds,
        )
        st_planner_target_speed = long_plan_result.target_speed

        print(f'turn speeds -> speed_limit: {speed_limit}, idm: {idm_lead_speed}, stop_for: {stop_for_idm_target_speed}, st_plan: {st_planner_target_speed}, obs: {idm_obs_speed}')
        target_speed = min(idm_lead_speed, stop_for_idm_target_speed, st_planner_target_speed, idm_obs_speed)
        brake = target_speed < 1e-2

        print(f'\n\nCommand Status:')
        print(f'{cmd_status.to_string()}')

        action_complete = self.turn_sm.is_cleared or self.turn_sm.is_failed
        collision_events = None if not action_complete else self.turn_sm.collision_events
        completion_reason = cmd_status.reasons[0]

        return target_speed, brake, action_complete, completion_reason, route_pts, route_wps, collision_events

    def _turn_left(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[Tuple[str, str], ConditionCommand],
        resolved_conditions : Dict[int, Tuple[str, str, str, str]],
        cur_tick : float
    ) -> Tuple:
        return self._turn(cur_plan, stop_conditions, resolved_conditions, Action.TURN_LEFT, cur_tick)

    def _turn_right(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[Tuple[str, str], ConditionCommand],
        resolved_conditions : Dict[int, Tuple[str, str, str, str]],
        cur_tick : float
    ) -> Tuple:
        return self._turn(cur_plan, stop_conditions, resolved_conditions, Action.TURN_RIGHT, cur_tick)

    def _turn_straight(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[Tuple[str, str], ConditionCommand],
        resolved_conditions : Dict[int, Tuple[str, str, str, str]],
        cur_tick : float
    ) -> Tuple:
        return self._turn(cur_plan, stop_conditions, resolved_conditions, Action.TURN_STRAIGHT, cur_tick)

    def _change_lane(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[Tuple[str, str], ConditionCommand],
        resolved_conditions : Dict[int, Tuple[str, str, str, str]],
        lc_cmd : Action,
        cur_tick : float
    ) -> Tuple:
        # If fresh command, clear previous state
        if self.lane_change_sm.is_cleared or self.lane_change_sm.is_failed:
            self.lane_change_sm.reset()
            self.long_planner.reset_plan()

        # Activate the command state
        self.lane_change_sm.activate()

        # Update state machine conditions
        self.lane_change_sm.update_conditions(self.cur_plan_state)

        planner_state = self.get_planner_state()
        max_route_len = planner_state.route_len

        speed_limit = self.scene_data.traffic_data.speed_limit
        ego_speed = self.scene_data.ego_data.speed

        # Execute one step in state machine. Downstream functions generate target speeds based on current state
        cmd_status = self.lane_change_sm.update_state(
            scene_data=self.scene_data,
            planner_state=planner_state,
            lc_cmd=lc_cmd
        )

        # TODO: KINDA UNNECESARY TO PERFORM HAS ACTIVE PLAN CHECKS, WE CAN EXIT EARLY BASED ON COMMAND STATUS
        is_highway_lane_change = speed_limit >= self.config.lane_change_highway_speeds

        # Smoothen lane change if driving on highways
        if self.lane_change_sm.has_active_plan and not self.lane_change_sm.route_shifted:
            if is_highway_lane_change:
                lc_start_idx, lc_end_idx = self.lane_change_sm.cur_plan_idxs

                lc_duration = self.config.lane_change_highway_duration
                desired_lc_dist_m = speed_limit * lc_duration

                lc_buffer = self.config.meters_to_dense_route_idx(desired_lc_dist_m / 2.0)

                lc_start_idx = max(0, lc_start_idx - lc_buffer)
                lc_end_idx = min(max_route_len - 1, lc_end_idx + lc_buffer)
                lc_len = lc_end_idx - lc_start_idx

                self.waypoint_planner.shift_route_single_direction(
                    start_index=lc_start_idx,
                    end_index=lc_end_idx,
                    shift_to_left_lane=lc_cmd == Action.CHANGE_LANE_LEFT,
                    transition_length=lc_len,
                )
            else:
                lc_start_idx, lc_end_idx = self.lane_change_sm.cur_plan_idxs

                lc_end_buffer_m = self.config.long_planning_lc_buffer_m
                lc_buffer = self.config.meters_to_dense_route_idx(lc_end_buffer_m)
                lc_end_idx = min(max_route_len - 1, lc_end_idx + lc_buffer)
                print(f'\n\nBUFFER: {lc_buffer}, NEW LC END: {lc_end_idx}')

            self.lane_change_sm.cur_plan_idxs = (lc_start_idx, lc_end_idx)
            self.lane_change_sm.route_shifted = True

        # Get planner state
        planner_state = self.get_planner_state()
        route_index = planner_state.route_index
        route_pts = planner_state.route_points[route_index:]
        route_wps = planner_state.route_waypoints[route_index:]
        s_route = planner_state.s_route

        # Get predictions
        prediction_data = self.predict_collisions()

        print(f'(change_lane) original target speed: {self.target_speed_initial}')

        # Get IDM leading vehicle target speed
        idm_lead_speed = self._idm_get_lead_speed(self.target_speed_initial)
        print(f'(change_lane) Leading vehicle IDM target speed: {idm_lead_speed}')

        # Get IDM stop_for object target speeds
        stop_for_idm_target_speed = self._stop_for(stop_conditions, self.lane_change_sm.conditions_registry)
        print(f'(change_lane) Leading vehicle stop for target speed: {stop_for_idm_target_speed}')

        st_planner_target_speed = 0.0

        plan_route_start_idx = self.long_planner.plan_route_start_idx
        if plan_route_start_idx < 0 or not self.long_planner.has_active_plan:
            print(f'SETTING TO ROUTE INDEX, PLAN ROUTE IDX IS UNSET')
            plan_route_start_idx = route_index

        s_ego_m = s_route[route_index] - s_route[plan_route_start_idx]
        print(f'EGO STATION: {s_ego_m}')

        plan_s_goal_m = 40.0
        ego_cruise_speed = None
        ego_goal_speed = None
        ego_max_speed = speed_limit
        s_bounds = None

        if self.lane_change_sm.has_active_plan:
            plan_start_idx, plan_goal_idx = self.lane_change_sm.cur_plan_idxs
            print(f'\n\nLANE CHANGE INDICES: START: {plan_start_idx}, END: {plan_goal_idx}')

            s_to_start = s_route[plan_start_idx] - s_route[route_index]
            s_to_lane_boundary = s_to_start
            s_to_end   = s_route[plan_goal_idx]  - s_route[route_index]

            lc_data = self.scene_data.route_data.lane_change_data if self.scene_data.route_data else None
            if lc_data:
                s_to_lane_boundary = s_route[lc_data.lane_boundary_idx] - s_route[route_index]

            plan_s_goal_m = max(10.0, s_to_end)

            # s_bounds = (max(0.0, s_to_start), s_to_end)
            s_bounds = (0.0, s_to_end)
            print(f'\n\nS BOUNDS: {s_bounds}')

            #TODO: fix lane change target speed selection
            idm_lead_speed = speed_limit
            stop_for_idm_target_speed = speed_limit
            ego_cruise_speed = idm_lead_speed
            ego_goal_speed = ego_max_speed

            print(f"plan_s_goal_m: {plan_s_goal_m}")

        long_plan_result = self.long_planner.run_step(
            plan_tick_counter=cur_tick,
            planner_state=planner_state,
            scene_data=self.scene_data,
            prediction_data=prediction_data,
            all_conditions=resolved_conditions,
            plan_s_goal_m=plan_s_goal_m,
            s_ego_m=s_ego_m,
            ego_cruise_speed=ego_cruise_speed,
            ego_goal_speed=ego_goal_speed,
            s_bounds=s_bounds
        )
        st_planner_target_speed = long_plan_result.target_speed

        print(f'(lane_change) ST Planner target speed: {st_planner_target_speed}')

        target_speed = min(idm_lead_speed, stop_for_idm_target_speed, st_planner_target_speed)
        brake = target_speed < 1e-2

        print(f'\n\nCommand Status:')
        print(f'{cmd_status.to_string()}')

        action_complete = self.lane_change_sm.is_cleared or self.lane_change_sm.is_failed
        collision_events = None if not action_complete else self.lane_change_sm.collision_events
        completion_reason = cmd_status.reasons[0]

        return target_speed, brake, action_complete, completion_reason, route_pts, route_wps, collision_events

    def _change_lane_left(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[Tuple[str, str], ConditionCommand],
        resolved_conditions : Dict[int, Tuple[str, str, str, str]],
        cur_tick : float
    ) -> Tuple:
        return self._change_lane(cur_plan, stop_conditions, resolved_conditions, Action.CHANGE_LANE_LEFT, cur_tick)

    def _change_lane_right(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[Tuple[str, str], ConditionCommand],
        resolved_conditions : Dict[int, Tuple[str, str, str, str]],
        cur_tick : float
    ) -> Tuple:
        return self._change_lane(cur_plan, stop_conditions, resolved_conditions, Action.CHANGE_LANE_RIGHT, cur_tick)

    def _overtake(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[Tuple[str, str], ConditionCommand],
        resolved_conditions : Dict[int, Tuple[str, str, str, str]],
        overtake_cmd : Action,
        cur_tick : float,
    ) -> Tuple:
        # Reset on fresh command
        if self.overtake_sm.is_cleared or self.overtake_sm.is_failed:
            self.overtake_sm.reset()
            self.lat_planner.reset_plan()
            self.long_planner.reset_plan()

        self.overtake_sm.activate()
        self.overtake_sm.update_conditions(self.cur_plan_state)

        speed_limit = self.scene_data.traffic_data.speed_limit

        planner_state = self.get_planner_state()
        prediction_data = self.predict_collisions()

        # ── LatPlanner: replan only when route intrusions are detected ───────────
        needs_replan = self.overtake_sm.compute_replan_need(
            scene_data=self.scene_data,
            prediction_data=prediction_data,
        )

        lat_plan_result = None
        if needs_replan:
            lat_plan_result = self.lat_planner.run_step(
                plan_tick_counter=cur_tick,
                planner_state=planner_state,
                lidar_data=self.lidar_pts,
                scene_data=self.scene_data,
                prediction_data=prediction_data,
                ego_plan=cur_plan,
                all_conditions={},
            )

        # ── Splice new route into global plan ────────────────────────────────────
        if lat_plan_result is not None and lat_plan_result.is_new_plan:
            self._interpolate_lateral_plan(
                planner_state=planner_state,
                lat_planner_result=lat_plan_result,
                update_wps=self.overtake_sm.overtake_type != OvertakeType.ONCOMING
            )

            # Refresh planner state
            planner_state = self.get_planner_state()

            # Get new predictions
            prediction_data = self.predict_collisions()

            # Reset ST plan for new route
            self.long_planner.reset_plan()

        route_index = planner_state.route_index
        route_pts = planner_state.route_points[route_index:]
        route_wps = planner_state.route_waypoints[route_index:]
        s_route = planner_state.s_route

        # ── Advance state machine (type detection + segment ingestion inside) ────
        cmd_status = self.overtake_sm.update_state(
            scene_data=self.scene_data,
            planner_state=planner_state,
            overtake_cmd=overtake_cmd,
            lat_planner_result=lat_plan_result,
        )

        # ── Longitudinal speed target ────────────────────────────────────────────
        idm_lead_speed = self._idm_get_lead_speed(self.target_speed_initial, desired_following_distance=1.0)
        stop_for_idm_target_speed = self._stop_for(stop_conditions, self.overtake_sm.conditions_registry)

        plan_route_start_idx = self.long_planner.plan_route_start_idx
        if plan_route_start_idx < 0 or not self.long_planner.has_active_plan:
            plan_route_start_idx = route_index

        s_ego_m = s_route[route_index] - s_route[plan_route_start_idx]

        # Default: no ST constraint.
        #   PENDING_PLAN   → hold position (0 m/s)
        #   IN_TARGET_LANE → IDM governs, no explicit ST bounds
        #   LC_OUTBOUND / LC_RETURN / ONCOMING → constrained ST plan below
        st_planner_target_speed = self.target_speed_initial

        if self.overtake_sm.sub_phase == OvertakeSubPhase.PENDING_PLAN:
            st_planner_target_speed = 0.0

        elif self.overtake_sm.has_active_plan:
            sub_phase = self.overtake_sm.sub_phase
            segs = self.overtake_sm.segments

            if self.overtake_sm.overtake_type == OvertakeType.ONCOMING:
                s_to_end = max(1.0, s_route[self.overtake_sm.maneuver_end_idx] - s_route[route_index])
            elif sub_phase == OvertakeSubPhase.LC_OUTBOUND:
                s_to_end = max(1.0, s_route[segs.lc_out_end] - s_route[route_index])
            else:  # LC_RETURN
                s_to_end = max(1.0, s_route[segs.lc_return_end] - s_route[route_index])

            # suppress IDM during active LC
            stop_for_idm_target_speed = speed_limit
            idm_lead_speed = speed_limit

            long_plan_result = self.long_planner.run_step(
                plan_tick_counter=cur_tick,
                planner_state=planner_state,
                scene_data=self.scene_data,
                prediction_data=prediction_data,
                all_conditions=resolved_conditions,
                plan_s_goal_m=s_to_end,
                s_ego_m=s_ego_m,
                ego_cruise_speed=self.target_speed_initial,
                ego_goal_speed=self.target_speed_initial,
                s_bounds=(0.0, s_to_end),
            )
            st_planner_target_speed = long_plan_result.target_speed

        print(
            f'overtake speeds -> idm: {idm_lead_speed:.2f}  '
            f'stop_for: {stop_for_idm_target_speed:.2f}  '
            f'st: {st_planner_target_speed:.2f}  '
            f'sub_phase: {self.overtake_sm.sub_phase.name}'
        )

        target_speed = min(idm_lead_speed, stop_for_idm_target_speed, st_planner_target_speed)
        brake = target_speed < 1e-2

        print(f'Command Status:\n{cmd_status.to_string()}')

        action_complete = self.overtake_sm.is_cleared or self.overtake_sm.is_failed
        collision_events = None if not action_complete else self.overtake_sm.collision_events
        completion_reason = cmd_status.reasons[0]

        return target_speed, brake, action_complete, completion_reason, route_pts, route_wps, collision_events

    def _overtake_left(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[Tuple[str, str], ConditionCommand],
        resolved_conditions : Dict[int, Tuple[str, str, str, str]],
        cur_tick : float
    ) -> Tuple:
        return self._overtake(cur_plan, stop_conditions, resolved_conditions, Action.OVERTAKE_LEFT, cur_tick)

    def _overtake_right(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[Tuple[str, str], ConditionCommand],
        resolved_conditions : Dict[int, Tuple[str, str, str, str]],
        cur_tick : float
    ) -> Tuple:
        return self._overtake(cur_plan, stop_conditions, resolved_conditions, Action.OVERTAKE_RIGHT, cur_tick)

    def _pull_over(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[Tuple[str, str], ConditionCommand],
        resolved_conditions : Dict[int, Tuple[str, str, str, str]],
        pull_over_cmd : Optional[Action],
        cur_tick : float,
    ) -> Tuple:
        # Reset on fresh command
        if self.pull_over_sm.is_cleared or self.pull_over_sm.is_failed:
            self.pull_over_sm.reset()
            self.long_planner.reset_plan()

        self.pull_over_sm.activate()
        self.pull_over_sm.update_conditions(self.cur_plan_state)

        speed_limit = self.scene_data.traffic_data.speed_limit

        planner_state   = self.get_planner_state()
        prediction_data = self.predict_collisions()
        route_index     = planner_state.route_index
        s_route         = planner_state.s_route

        # ── Route shift (PULL_OVER_L/R only) ─────────────────────────────────
        route_shifted_idxs = None
        needs_replan = self.pull_over_sm.compute_replan_need(self.scene_data, prediction_data)

        if needs_replan and pull_over_cmd is not None:
            ego_speed = self.scene_data.ego_data.speed

            prep_distance    = ego_speed * self.config.pull_over_preparation_time
            pre_shift_points = int(prep_distance * self.config.points_per_meter)
            route_last_idx   = len(planner_state.route_points) - 1

            pull_over_start_idx = min(route_last_idx, route_index + pre_shift_points)

            braking_lookahead = int(
                self.config.points_per_meter * (((ego_speed * 3.6) / 10.0) ** 2 / 2.0)
                + self.config.braking_distance_calculation_safety_distance
            )
            pull_over_yield_idx = min(route_last_idx, pull_over_start_idx + braking_lookahead)
            pull_over_end_idx   = min(route_last_idx, pull_over_yield_idx + self.config.pull_over_route_lookahead)

            self.waypoint_planner.shift_route_smoothly(
                pull_over_start_idx,
                pull_over_end_idx,
                pull_over_cmd == Action.PULL_OVER_LEFT,
                self.config.transition_smoothness_distance,
            )

            planner_state   = self.get_planner_state()
            prediction_data = self.predict_collisions()
            route_index     = planner_state.route_index
            s_route         = planner_state.s_route

            route_shifted_idxs = (pull_over_start_idx, pull_over_yield_idx, pull_over_end_idx)
            self.long_planner.reset_plan()

        # ── Advance state machine ─────────────────────────────────────────────
        prev_sub_phase = self.pull_over_sm.sub_phase

        cmd_status = self.pull_over_sm.update_state(
            plan_state=self.cur_plan_state,
            scene_data=self.scene_data,
            planner_state=planner_state,
            prediction_data=prediction_data,
            route_shifted_idxs=route_shifted_idxs,
        )

        # Reset long planner on sub-phase transition
        if self.pull_over_sm.sub_phase != prev_sub_phase:
            self.long_planner.reset_plan()

        # ── Longitudinal speed target ─────────────────────────────────────────
        idm_lead_speed            = self._idm_get_lead_speed(self.target_speed_initial)
        stop_for_idm_target_speed = self._stop_for(stop_conditions, self.pull_over_sm.conditions_registry)

        plan_route_start_idx = self.long_planner.plan_route_start_idx
        if plan_route_start_idx < 0 or not self.long_planner.has_active_plan:
            plan_route_start_idx = route_index
        s_ego_m = s_route[route_index] - s_route[plan_route_start_idx]

        st_planner_target_speed = 0.0   # safe default: hold

        sub = self.pull_over_sm.sub_phase

        if sub == PullOverPhase.YIELDING:
            if self.pull_over_sm.pull_action == Action.PULL_OVER_IN_LANE:
                st_planner_target_speed = 0.0
            elif self.pull_over_sm.has_active_plan:
                wait_idx  = self.pull_over_sm.maneuver_wait_idx
                s_to_end  = max(1.0, s_route[wait_idx] - s_route[route_index])

                long_plan_result = self.long_planner.run_step(
                    plan_tick_counter=cur_tick,
                    planner_state=planner_state,
                    scene_data=self.scene_data,
                    prediction_data=prediction_data,
                    all_conditions=resolved_conditions,
                    plan_s_goal_m=s_to_end,
                    s_ego_m=s_ego_m,
                    ego_cruise_speed=self.target_speed_initial,
                    ego_goal_speed=3.0,
                    s_bounds=(0.0, s_to_end),
                )
                st_planner_target_speed = long_plan_result.target_speed

        elif sub == PullOverPhase.WAITING:
            st_planner_target_speed = 0.0

        elif sub == PullOverPhase.RETURNING:
            end_idx  = self.pull_over_sm.maneuver_end_idx
            s_to_end = max(1.0, s_route[end_idx] - s_route[route_index])

            long_plan_result = self.long_planner.run_step(
                plan_tick_counter=cur_tick,
                planner_state=planner_state,
                scene_data=self.scene_data,
                prediction_data=prediction_data,
                all_conditions=resolved_conditions,
                plan_s_goal_m=s_to_end,
                s_ego_m=s_ego_m,
                ego_cruise_speed=speed_limit,
                ego_goal_speed=speed_limit,
                s_bounds=(0.0, s_to_end),
            )
            st_planner_target_speed = long_plan_result.target_speed

        print(
            f'pull_over speeds -> idm: {idm_lead_speed:.2f}  '
            f'stop_for: {stop_for_idm_target_speed:.2f}  '
            f'st: {st_planner_target_speed:.2f}  '
            f'sub_phase: {sub.name}'
        )

        target_speed = min(idm_lead_speed, stop_for_idm_target_speed, st_planner_target_speed)
        brake        = target_speed < 1e-2

        print(f'Command Status:\n{cmd_status.to_string()}')

        action_complete   = self.pull_over_sm.is_cleared or self.pull_over_sm.is_failed
        collision_events  = None if not action_complete else self.pull_over_sm.collision_events
        completion_reason = cmd_status.reasons[0]

        route_pts_view = planner_state.route_points[route_index:]
        route_wps_view = self.waypoint_planner.route_waypoints[route_index:]

        return target_speed, brake, action_complete, completion_reason, route_pts_view, route_wps_view, collision_events

    def _pull_over_left(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[Tuple[str, str], ConditionCommand],
        resolved_conditions : Dict[int, Tuple[str, str, str, str]],
        cur_tick : float
    ) -> Tuple:
        return self._pull_over(cur_plan, stop_conditions, resolved_conditions, Action.PULL_OVER_LEFT, cur_tick)

    def _pull_over_right(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[Tuple[str, str], ConditionCommand],
        resolved_conditions : Dict[int, Tuple[str, str, str, str]],
        cur_tick : float
    ) -> Tuple:
        return self._pull_over(cur_plan, stop_conditions, resolved_conditions, Action.PULL_OVER_RIGHT, cur_tick)

    def _pull_over_in_lane(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[Tuple[str, str], ConditionCommand],
        resolved_conditions : Dict[int, Tuple[str, str, str, str]],
        cur_tick : float
    ) -> Tuple:
        return self._pull_over(cur_plan, stop_conditions, resolved_conditions, None, cur_tick)

    def _share_lane(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[Tuple[str, str], ConditionCommand],
        resolved_conditions : Dict[int, Tuple[str, str, str, str]],
        cur_tick : float
    ) -> Tuple:
        # Reset on fresh command
        if self.share_lane_sm.is_cleared or self.share_lane_sm.is_failed:
            self.share_lane_sm.reset()
            self.lat_planner.reset_plan()
            self.long_planner.reset_plan()

        self.share_lane_sm.activate()
        self.share_lane_sm.update_conditions(self.cur_plan_state)

        speed_limit = self.scene_data.traffic_data.speed_limit

        planner_state   = self.get_planner_state()
        prediction_data = self.predict_collisions()

        # ── LatPlanner: replan only when oncoming vehicles intrude ───────────────
        maneuver_end_idx = self.share_lane_sm.maneuver_end_idx
        if maneuver_end_idx is not None:
            plan_lat_s_goal_m = max(10.0, planner_state.s_route[maneuver_end_idx] - planner_state.s_route[planner_state.route_index])
        else:
            plan_lat_s_goal_m = 50.0

        needs_replan = self.share_lane_sm.compute_replan_need(
            scene_data=self.scene_data,
            prediction_data=prediction_data,
        )

        lat_plan_result = None
        if needs_replan:
            lat_plan_result = self.lat_planner.run_step(
                plan_tick_counter=cur_tick,
                planner_state=planner_state,
                lidar_data=self.lidar_pts,
                scene_data=self.scene_data,
                prediction_data=prediction_data,
                ego_plan=cur_plan,
                plan_s_goal_m=plan_lat_s_goal_m,
                all_conditions={},
            )

        # ── Splice new route into global plan ────────────────────────────────────
        if lat_plan_result is not None and lat_plan_result.is_new_plan:
            self._interpolate_lateral_plan(
                planner_state=planner_state,
                lat_planner_result=lat_plan_result,
            )
            self.long_planner.reset_plan()
            planner_state   = self.get_planner_state()
            prediction_data = self.predict_collisions()

        route_index = planner_state.route_index
        route_pts   = planner_state.route_points
        route_wps   = self.waypoint_planner.route_waypoints
        s_route     = planner_state.s_route

        # ── Advance state machine ─────────────────────────────────────────────────
        cmd_status = self.share_lane_sm.update_state(
            scene_data=self.scene_data,
            planner_state=planner_state,
            lat_planner_result=lat_plan_result,
        )

        # ── Longitudinal speed target ─────────────────────────────────────────────
        idm_lead_speed            = self._idm_get_lead_speed(self.target_speed_initial, desired_following_distance=1.0)
        stop_for_idm_target_speed = self._stop_for(stop_conditions, self.share_lane_sm.conditions_registry)

        plan_route_start_idx = self.long_planner.plan_route_start_idx
        if plan_route_start_idx < 0 or not self.long_planner.has_active_plan:
            plan_route_start_idx = route_index

        s_ego_m = s_route[route_index] - s_route[plan_route_start_idx]

        st_planner_target_speed = self.target_speed_initial

        if self.share_lane_sm.lat_plan_status == LateralPlanStatus.NONE:
            st_planner_target_speed = 0.0

        elif self.share_lane_sm.has_active_plan:
            s_to_end = max(1.0, s_route[self.share_lane_sm.maneuver_end_idx] - s_route[route_index])

            long_plan_result = self.long_planner.run_step(
                plan_tick_counter=cur_tick,
                planner_state=planner_state,
                scene_data=self.scene_data,
                prediction_data=prediction_data,
                all_conditions=resolved_conditions,
                plan_s_goal_m=s_to_end,
                s_ego_m=s_ego_m,
                ego_cruise_speed=self.target_speed_initial,
                ego_goal_speed=speed_limit,
                s_bounds=(0.0, s_to_end),
            )
            st_planner_target_speed = long_plan_result.target_speed

        print(
            f'share_lane speeds -> idm: {idm_lead_speed:.2f}  '
            f'stop_for: {stop_for_idm_target_speed:.2f}  '
            f'st: {st_planner_target_speed:.2f}'
        )

        target_speed = min(idm_lead_speed, stop_for_idm_target_speed, st_planner_target_speed)
        brake        = target_speed < 1e-2

        print(f'Command Status:\n{cmd_status.to_string()}')

        action_complete   = self.share_lane_sm.is_cleared or self.share_lane_sm.is_failed
        collision_events  = None if not action_complete else self.share_lane_sm.collision_events
        completion_reason = cmd_status.reasons[0]

        return target_speed, brake, action_complete, completion_reason, route_pts[route_index:], route_wps[route_index:], collision_events

    def _interpolate_lateral_plan(
        self,
        planner_state: PlannerState,
        lat_planner_result: LatPlannerResult,
        update_wps : bool = False,
    ) -> None:
        # 1. Extract new lateral plan data
        new_pts  = lat_planner_result.route_points   # (N, 3)
        new_yaws = lat_planner_result.route_yaws     # (N,)
        new_cmds = lat_planner_result.route_commands

        start_idx = lat_planner_result.start_idx
        splice_end = start_idx + len(new_pts)

        # 2. Splice positions, yaws, and commands in-place into the global route
        self.waypoint_planner.route_points[start_idx:splice_end] = new_pts
        self.waypoint_planner.rotation_angles[start_idx:splice_end] = new_yaws
        self.waypoint_planner.commands[start_idx:splice_end] = new_cmds

        if update_wps:
            new_wps = [self.carla_map.get_waypoint(carla.Location(x=pt[0], y=pt[1], z=pt[2])) for pt in new_pts]
            self.waypoint_planner.route_waypoints[start_idx:splice_end] = new_wps


        # 3. Recompute and update arc-length in-place
        self.waypoint_planner.s_route[start_idx:] = (
            self.waypoint_planner.cumulative_arclength(
                self.waypoint_planner.route_points[start_idx:]
            )
            + self.waypoint_planner.s_route[start_idx]
        )

        # 4. Recompute bounding boxes for the modified segment
        # (Assuming route_waypoints are strictly read-only here, we can pull
        # directly from planner_state without copying)
        route_bbs_new = self.waypoint_planner.generate_route_bbs(
            route_points=new_pts,
            route_yaws=new_yaws,
            route_waypoints=planner_state.route_waypoints[start_idx:splice_end],
        )

        # 5. Splice bounding boxes in-place
        bb_start_idx = self.config.dense_route_idx_to_bb_route_idx(start_idx)
        bb_goal_idx  = self.config.dense_route_idx_to_bb_route_idx(splice_end)
        self.waypoint_planner.route_bbs[bb_start_idx:bb_goal_idx] = route_bbs_new

        # 6. Rebuild precomputed intersection/lane-change segment lists so that
        #    PlannerState is immediately consistent after the route splice.
        self.waypoint_planner.recompute_route_segments()

    ########################################
    # Condition resolution
    ########################################

    def _resolve_conditions_to_actors(
        self,
        conditions: Dict[Tuple[str, str], ConditionCommand],
    ) -> Dict[int, Tuple[str, str, str, str]]:
        """
        Resolve abstract ConditionCommand targets to concrete actor IDs.

        Returns dict mapping actor_id → (directive, actor_type, traffic_type, priority)
        consumed by st_occupancy and sl_occupancy.
        """
        resolved: Dict[int, Tuple[str, str, str, str]] = {}

        for _cond_key, condition in conditions.items():
            target = condition.target
            directive = condition.condition_action.value
            priority = condition.priority

            region = target.region
            lane_name = region if region != "any" else None
            traffic_type = target.traffic_type if target.traffic_type != "any" else None

            if target.actor_type in {EntityType.VEHICLE, EntityType.CYCLIST, EntityType.EMERGENCY_VEHICLE}:
                vehicles = self.scene_data.vehicle_data.get(
                    traffic_type=traffic_type,
                    lane_name=lane_name,
                    vehicle_types={target.actor_type},
                )
                for v in vehicles:
                    resolved[v.id] = (directive, target.actor_type, v.traffic_type, priority)

            elif target.actor_type == EntityType.PEDESTRIAN:
                if self.scene_data.ped_data:
                    for ped in self.scene_data.ped_data:
                        resolved[ped.id] = (directive, EntityType.PEDESTRIAN, target.traffic_type, priority)

            elif target.actor_type == EntityType.OBSTACLE:
                obstacles = self.scene_data.obstacle_data.ego_obstacles if region == "ego" else self.scene_data.obstacle_data.all_obstacles
                for obs in obstacles:
                    resolved[obs.id] = (directive, EntityType.OBSTACLE, target.traffic_type, priority)

        return resolved

    ########################################
    # Low-level condition API
    ########################################

    def _stop_for(
        self,
        stop_conditions : Dict[Tuple[str, str], ConditionCommand],
        conditions_registry : Dict[Tuple[str, str], StopForSM],
    ) -> float:

        target_speed = self.target_speed_initial
        for cond_key, condition in stop_conditions.items():
            actor_type = condition.target.actor_type
            print(f'stop_for key: {cond_key}, type: {actor_type}')
            if actor_type == EntityType.STOP_SIGN:
                target_speed_ss = self._stop_for_stop_sign(conditions_registry)
                target_speed = min(target_speed, target_speed_ss)
            elif actor_type == EntityType.STOP_LIGHT:
                target_speed_tl = self._stop_for_traffic_light(conditions_registry)
                target_speed = min(target_speed, target_speed_tl)
            elif actor_type == EntityType.OBSTACLE:
                target_speed_obstacle = self._stop_for_obstacle(conditions_registry)
                target_speed = min(target_speed, target_speed_obstacle)
            elif actor_type == EntityType.PEDESTRIAN:
                target_speed_pedestrian = self._stop_for_pedestrian(conditions_registry)
                target_speed = min(target_speed, target_speed_pedestrian)

        return target_speed

    ########################################
    # IDM Helpers
    ########################################
    def _idm_get_lead_speed(
        self,
        target_speed_initial : float,
        *,
        target_vehicle : Optional[VehicleDataEntry] = None,
        desired_following_distance : Optional[float] = None,
        desired_time_headway : Optional[float] = None,
    ) -> float:
        lv = target_vehicle
        if lv is None:
            ego_leading_vehicles = self.scene_data.vehicle_data.get(traffic_type="leading", lane_name="ego")
            if ego_leading_vehicles:
                lv = ego_leading_vehicles[0]

        if not lv:
            return target_speed_initial

        ego_speed = self.scene_data.ego_data.speed
        idm_target_speed = target_speed_initial

        lv_speed = lv.speed
        lv_length = lv.vehicle.bounding_box.extent.x * 2

        dist_to_lv = lv.relative_distance

        print(f'EGO LV DIST: {dist_to_lv}, speed: {lv_speed}, id: {lv.id}')

        s0 = desired_following_distance if desired_following_distance is not None else self.config.idm_leading_vehicle_minimum_distance
        t_headway = desired_time_headway if desired_time_headway is not None else self.config.idm_leading_vehicle_time_headway

        idm_target_speed = self.idm.compute_target_speed_idm(
            desired_speed=target_speed_initial,
            leading_actor_length=lv_length,
            ego_speed=ego_speed,
            leading_actor_speed=lv_speed,
            distance_to_leading_actor=dist_to_lv,
            s0=s0,
            T=t_headway
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

        ego_obstacles = self.scene_data.obstacle_data.ego_obstacles
        if ego_obstacles:
            nearest_obs_data = ego_obstacles[0]
            obstacle_obj = nearest_obs_data.obstacle

            idm_target_speed = self.idm.compute_target_speed_idm(
                desired_speed=target_speed_initial,
                leading_actor_length=obstacle_obj.bounding_box.extent.x * 2 + self.scene_data.ego_data.ego_vehicle.bounding_box.extent.x,
                ego_speed=ego_speed,
                leading_actor_speed=0.0,
                distance_to_leading_actor=nearest_obs_data.relative_distance,
                s0=self.config.idm_leading_vehicle_minimum_distance,
                T=self.config.idm_leading_vehicle_time_headway
            )
            print(f'Obstacle ID: {nearest_obs_data.id}, IDM Speed: {idm_target_speed}\n')
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
        planner_state = self.get_planner_state()
        route_index = planner_state.route_index
        route_pts = planner_state.route_points[route_index:]
        route_wps = planner_state.route_waypoints[route_index:]

        target_speed = self._get_target_speed(target_speed_initial)
        brake = target_speed < 1e-2

        return target_speed, brake, route_pts, route_wps

    def _stop_for_stop_sign(
        self,
        conditions_registry : Dict[Tuple[str, str], StopForSM],
    ):
        ego_speed = self.scene_data.ego_data.speed
        target_speed_initial = self.target_speed_initial
        next_ss = self.scene_data.traffic_data.next_stop_sign

        if next_ss is None:
            return target_speed_initial

        cond_key = (ConditionAction.STOP_FOR.value, EntityType.STOP_SIGN.value)
        if cond_key not in conditions_registry:
            return target_speed_initial

        cond_sm = conditions_registry[cond_key]

        if cond_sm.phase == ActionPhase.EXECUTING:
            target_speed = self.idm.compute_target_speed_idm(
                desired_speed=target_speed_initial,
                leading_actor_length=0.0,
                ego_speed=ego_speed,
                leading_actor_speed=0.0,
                distance_to_leading_actor=next_ss.distance_to_stop_sign,
                s0=self.config.idm_stop_sign_minimum_distance,
                T=self.config.idm_stop_sign_desired_time_headway
            )
            return target_speed

        if cond_sm.phase == ActionPhase.CLEARED:
            self.waypoint_planner.update_cleared_stop_signs(next_ss.id)

        return target_speed_initial

    def _stop_for_traffic_light(
        self,
        conditions_registry : Dict[Tuple[str, str], StopForSM],
    ):
        ego_speed = self.scene_data.ego_data.speed
        target_speed_initial = self.target_speed_initial
        next_tl = self.scene_data.traffic_data.next_traffic_light

        if next_tl is None:
            return target_speed_initial

        cond_key = (ConditionAction.STOP_FOR.value, EntityType.STOP_LIGHT.value)
        if cond_key not in conditions_registry:
            return target_speed_initial

        cond_sm = conditions_registry[cond_key]

        if cond_sm.phase == ActionPhase.EXECUTING:
            target_speed = self.idm.compute_target_speed_idm(
                desired_speed=target_speed_initial,
                leading_actor_length=0.0,
                ego_speed=ego_speed,
                leading_actor_speed=0.0,
                distance_to_leading_actor=next_tl.distance_to_light,
                s0=self.config.idm_red_light_minimum_distance,
                T=self.config.idm_red_light_desired_time_headway,
            )
            return target_speed

        # NOTE: NOT TRACKING CLEARED TRAFFIC LIGHTS, MIGHT NOT BE IMPORTANT

        return target_speed_initial

    def _stop_for_obstacle(
        self,
        conditions_registry : Dict[Tuple[str, str], StopForSM],
    ):
        target_speed_initial = self.target_speed_initial

        ego_obstacles = self.scene_data.obstacle_data.ego_obstacles
        if not ego_obstacles:
            return target_speed_initial

        cond_key = (ConditionAction.STOP_FOR.value, EntityType.OBSTACLE.value)
        if cond_key not in conditions_registry:
            return target_speed_initial

        cond_sm = conditions_registry[cond_key]

        if cond_sm.phase == ActionPhase.EXECUTING:
            target_speed = self._idm_get_obstacle_speed(target_speed_initial)
            return target_speed

        # NOTE: NOT TRACKING CLEARED OBSTACLES, COULD BE IMPORTANT TO AVOID INDEFINITELY WAITING FOR THEM

        return target_speed_initial

    def _stop_for_pedestrian(
        self,
        conditions_registry : Dict[Tuple[str, str], StopForSM],
    ):
        target_speed_initial = self.target_speed_initial

        if not self.scene_data.ped_data:
            return target_speed_initial

        cond_key = (ConditionAction.STOP_FOR.value, EntityType.PEDESTRIAN.value)
        if cond_key not in conditions_registry:
            return target_speed_initial

        cond_sm = conditions_registry[cond_key]

        if cond_sm.phase == ActionPhase.EXECUTING:
            target_speed = self._idm_get_ped_speed(target_speed_initial)
            return target_speed

        # NOTE: NOT TRACKING CLEARED PEDESTRIANS, COULD BE IMPORTANT TO AVOID INDEFINITELY WAITING FOR THEM

        return target_speed_initial
