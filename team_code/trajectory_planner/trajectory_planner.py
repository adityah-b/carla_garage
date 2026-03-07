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

from team_code.scene_analyzer.parsers.hl_beh_pydantic_models import *
from team_code.scene_analyzer.parsers.ego_plan_pydantic_models import *

from .idm import IDM
from .state_machines import *

from team_code.actor_prediction.motion_prediction import MotionPrediction, PredictionData

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
        self.all_actors : Dict = None
        self.lidar_pts : Dict = None

        self.waypoint_planner = PrivilegedRoutePlanner(self.config)
        self.motion_forecaster = MotionPrediction(self.config, self.carla_map)

        self.idm = IDM(self.config)

        # Longitudinal planner
        self.long_planner = LongPlanner(
            config=self.config,
            algo_name='dijkstra',
        )

        # Lateral planner
        self.lat_planner = LatPlanner(
            config=self.config,
            ego_vehicle=self.ego_vehicle
        )

        self._dispatch_map = {
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

            Action.SHARE_LANE: self._share_lane,
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

        # State machines
        self.follow_route_sm = FollowRouteSM(config)
        self.turn_sm = TurnSM(config)
        self.lane_change_sm = LaneChangeSM(config)
        self.overtake_sm = OvertakeSM(config)
        self.pull_over_sm = PullOverSM(config)
        self.share_lane_sm = ShareLaneSM(config)

        # Actor registry
        self.key_actor_registry : Dict[str, Set[int]] = {
            "vehicle" : set(),
            "pedestrian" : set(),
            "cyclist" : set(),
            "emergency_vehicle" : set(),
            "traffic_light" : set(),
            "stop_sign" : set(),
            "obstacles" : set(),
        }

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

    # TODO: PROPERLY IMPLEMENT THE EGO STATION CHECK, MAYBE PRECOMPUTE ROUTE DURING INITIALIZATION OR AFTER ROUTE CHANGES
    def _compute_ego_station(self, route_points: np.ndarray) -> float:
        ego_loc = self.ego_vehicle.get_location()
        ego_xy = np.array([[ego_loc.x, ego_loc.y]], dtype=np.float32)

        if route_points.shape[0] < 2:
            return 0.0

        route_geom = self.long_planner.grid_mapper.precompute_route_geometry(route_points)
        s_ego = self.long_planner.grid_mapper.project_points_to_route_s(
            points_xy=ego_xy,
            geom=route_geom,
            window=2,
        )
        return float(s_ego[0])

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

    # def _update_all_actors(self):
    #     all_actors = {
    #         "left" : {},
    #         "ego" : {},
    #         "right" : {},
    #         "oncoming" : {},
    #         "cross" : {},
    #         "peds" : {},
    #     }

    #     peds = self.scene_data.ped_data
    #     if peds:
    #         for ped in peds:
    #             all_actors["peds"][ped.id] = ped

    #     vehicles_dict = self.scene_data.vehicle_data
    #     if vehicles_dict:
    #         for lv_type, lv_dict in vehicles_dict.items(): # Traffic type
    #             for lv_lane, lv_list in lv_dict.items(): # Traffic lanes per traffic type
    #                 for lv_data in lv_list: # Lanelets per lane
    #                     for v_data in lv_data.vehicle_data: # Vehicles per lanelet
    #                         if lv_type in ["leading", "trailing"]:
    #                             if lv_lane not in all_actors:
    #                                 all_actors[lv_lane] = {}
    #                             all_actors[lv_lane][v_data.id] = v_data
    #                         else:
    #                             all_actors[lv_type][v_data.id] = v_data

    #     return all_actors

    # NOTE: TEMPORARY EMERGENCY VEHICLE HANDLING
    def _get_vehicle_by_id(self, vehicle_id: int) -> Optional[VehicleData]:
        if self.all_actors is None:
            return None

        for lane_dict in self.all_actors.values():
            if vehicle_id in lane_dict:
                return lane_dict[vehicle_id]

        return None

    # NOTE: TEMPORARY EMERGENCY VEHICLE HANDLING
    @staticmethod
    def _is_emergency_vehicle(vehicle: Optional[carla.Vehicle]) -> bool:
        if vehicle is None:
            return False

        type_id = vehicle.type_id.lower() if hasattr(vehicle, "type_id") and vehicle.type_id else ""
        role_name = vehicle.attributes.get("role_name", "").lower()

        emergency_keywords = (
            "ambulance",
            "firetruck",
            "fire_truck",
            "police",
            "emergency",
        )

        return any(keyword in type_id for keyword in emergency_keywords) or any(
            keyword in role_name for keyword in emergency_keywords
        )

    def update_scene_data(
        self,
        scene_context : SceneContext,
        lidar_pts : Dict,
    ) -> None:
        self.scene_data = scene_context.scene_data
        # self.all_actors = self._update_all_actors()
        self.lidar_pts = lidar_pts

    def update_key_actor_registry(
        self,
        hl_beh : HighLevelBehaviour
    ) -> None:
        # Clear all previous entries from registry
        for key in self.key_actor_registry.keys():
            self.key_actor_registry[key].clear()

        key_actors = hl_beh.key_actors
        traffic_objects = hl_beh.traffic_objects
        obstacles = hl_beh.obstacles

        for obs in obstacles:
            self.key_actor_registry["obstacles"].add(obs.id)

    def plan_with_reasoning(self) -> bool:
        needs_reasoning = False
        # TODO: Start setting reasoning distance tolerances
        if self.scene_data.traffic_data:
            needs_reasoning = needs_reasoning or (self.scene_data.traffic_data.next_stop_sign != None and self.scene_data.traffic_data.next_stop_sign.distance_to_stop_sign <= 50.0)
            needs_reasoning = needs_reasoning or self.scene_data.traffic_data.next_traffic_light != None

        if self.scene_data.ped_data:
            for ped in self.scene_data.ped_data:
                # TODO: START SETTING REASONING DISTANCE TOLERANCES
                needs_reasoning = needs_reasoning or ped.relative_distance <= 15.0

        if self.scene_data.route_data:
            needs_reasoning = needs_reasoning or (self.scene_data.route_data.intersection_data != None and self.scene_data.route_data.intersection_data.distance_to_intersection <= 20.0)
            # TODO: START SETTING REASONING DISTANCE TOLERANCES
            needs_reasoning = needs_reasoning or (self.scene_data.route_data.lane_change_data != None and self.scene_data.route_data.lane_change_data.distance_to_lane_change <= 20.0)

        ego_obstacles = self.scene_data.obstacle_data.ego_obstacles
        if ego_obstacles:
            needs_reasoning = needs_reasoning or ego_obstacles[0] <= self.config.obstacle_processing_distance

        # TODO: TEMP INTRUDING VEHICLES DETECTION
        # NOTE: BETTER ARCHITECTURE/DESIGN PREFERRED
        # needs_reasoning = needs_reasoning or self.scene_data.all_intruding_vehicles

        # NOTE/TODO: TEMP EMERGENCY VEHICLE DETECTION
        # TODO: TEMP CYCLIST DETECTION
        # emergency_reasoning_radius = self.config.detection_radius

        all_special_vehicles = self.scene_data.vehicle_data.get(vehicle_types={'cyclist', 'emergency'})
        print(f'ALL SPECIAL VEHICLES: {len(all_special_vehicles)}')
        for special_vehicle in all_special_vehicles:
            if special_vehicle.vehicle_type == 'cyclist':
                needs_reasoning = needs_reasoning or special_vehicle.relative_distance <= 15.0
            elif special_vehicle.vehicle_type == 'emergency':
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
        ego_forecasted_bbs = self.motion_forecaster.predict_ego_motion(
            ego_vehicle=self.ego_vehicle,
            ego_route_pts=self.waypoint_planner.route_points[self.waypoint_planner.route_index:],
            target_speed=None if ego_velocity_profile is not None else self.target_speed_initial,
            velocity_profile=ego_velocity_profile,
        )

        # Forecast all vehicles
        veh_forecasted_bbs = self.motion_forecaster.predict_vehicle_motion(self.scene_data.vehicle_data, self.scene_data.traffic_data.speed_limit)

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
        # print(f'\n\nCOLLISIONS')
        # print(f'\t\tNum collisions: {len(all_actor_collisions.keys())}')
        for actor_id, collision_intervals in all_actor_collisions.items():
            # print(f'\t\tactor id: {actor_id}')
            collision_interval = collision_intervals[0]
            collision_bbs = collision_interval.collision_bboxes_b[collision_interval.start_idx:collision_interval.end_idx]

            ego_bbs = collision_interval.collision_bboxes_a[collision_interval.start_idx:collision_interval.end_idx]

            # world.debug.draw_string(
            #     location=collision_bbs[0].location,
            #     text=str(actor_id),
            #     color=self.config.other_vehicles_forecasted_bbs_color,
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

        # TODO: UPDATE INDEX TRANSORMATION WITH CONFIG NUMBERS
        route_index_spaced = route_index // self.config.points_per_meter
        route_index_spaced = route_index_spaced // 2

        max_route_bbs = len(self.waypoint_planner.route_bbs)
        lookahead_distance = int(overlap_len // 2)
        to_index_spaced = min(max_route_bbs, route_index_spaced + lookahead_distance + 1)

        all_actor_overlaps = CollisionChecker.get_lane_overlaps(
            actor_predictions=actor_predictions,
            route_bbs=self.waypoint_planner.route_bbs[route_index_spaced:to_index_spaced]
        )
        # print(f'\n\nOVERLAPS')
        # print(f'\tNum overlaps: {len(all_actor_overlaps.keys())}')
        for actor_id, overlaps in all_actor_overlaps.items():
            # print(f'\t\tactor id: {actor_id}')
            overlap_interval = overlaps[0]

            space_start = overlap_interval.space_start_idx
            space_end = overlap_interval.space_end_idx

            time_start = overlap_interval.time_start_idx
            time_end = overlap_interval.time_end_idx

            # print(f'\t\ttime_start: {time_start}, time_end: {time_end}')
            # print(f'\t\tspace_start: {space_start}, space_end: {space_end}')

            overlap_route_bbs = overlap_interval.overlap_route_bboxes[space_start:space_end]
            overlap_actor_bbs = overlap_interval.overlap_actor_bboxes[time_start:time_end]

            # for bb in overlap_actor_bbs:
            #     world.debug.draw_box(
            #         box=bb,
            #         rotation=bb.rotation,
            #         thickness=0.1,
            #         color=self.config.ego_vehicle_forecasted_bbs_hazard_color,
            #         life_time=self.config.draw_life_time
            #     )

            # for bb in overlap_route_bbs:
            #     world.debug.draw_box(
            #         box=bb,
            #         rotation=bb.rotation,
            #         thickness=0.1,
            #         color=self.config.pedestrian_forecasted_bbs_color,
            #         life_time=self.config.draw_life_time
            #     )

        # return PredictionData(
        #     ego_forecasted_bbs=ego_forecasted_bbs,
        #     veh_forecasted_bbs=veh_forecasted_bbs,
        #     ped_forecasted_bbs=ped_forecasted_bbs,
        #     all_actor_collisions=all_actor_collisions
        # )
        return PredictionData(
            ego_forecasted_bbs=ego_forecasted_bbs,
            veh_forecasted_bbs=veh_forecasted_bbs,
            ped_forecasted_bbs=ped_forecasted_bbs,
            all_actor_collisions=all_actor_collisions,
            all_actor_overlaps=all_actor_overlaps
        )

    def set_plan(self, ego_plan : EgoPlan) -> None:
        self.cur_plan = ego_plan
        self.cur_plan_status = PlanStatus.EXECUTING
        self.cur_plan_reason = None
        self.needs_replan = False

    # TODO: UPDATE THIS API, WE CAN LIKELY CREATE A CUR PLANEXECUTION AND PREV PLANEXECUTION OBJECT AND USE THOSE DIRECTLY
    def _archive_current_plan(
        self,
        status: PlanStatus,
        reason: Optional[str] = None,
        collision_events : Optional[Deque[CollisionData]] = None,
    ) -> None:
        if self.cur_plan is not None:
            print(f'\n\nARCHIVING CURRENT PLAN, COLLISION EVENTS: {collision_events}\n\n')
            self.prev_plan_execution = PlanExecution(
                plan=self.cur_plan,
                status=status,
                reason=reason,
                collision_events=collision_events,
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
        # print(f'\n\nTARGET SPEED INITIAL')
        # print(f'\t\tspeed: {self.target_speed_initial}')

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
        target_speed, brake, action_complete, completion_reason, route_pts, route_wps, collision_events = fn(self.cur_plan, stop_conditions, other_conditions, cur_tick)

        if action_complete:
            self._archive_current_plan(
                status=PlanStatus.FINISHED,
                reason=completion_reason or "Plan conditions satisfied",
                collision_events=collision_events,
            )
            self.long_planner.reset_plan()
            # TODO ADD PLAN RESET FOR LAT PLANNER

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
        # Reset state machine if cleared
        if self.follow_route_sm.phase == ActionPhase.CLEARED:
            self.follow_route_sm.reset()

        # Update state machine conditions
        self.follow_route_sm.update_conditions(cur_plan.conditions)

        # Execute one step in state machine. Downstream functions generate target speeds based on current state
        cmd_status = self.follow_route_sm.update_state(
            config=self.config,
            scene_data=self.scene_data,
            planner_state=self.waypoint_planner.get_planner_state()
        )

        speed_limit = self.scene_data.traffic_data.speed_limit
        ego_speed = self.scene_data.ego_data.speed

        route_index = self.waypoint_planner.route_index
        route_pts = self.waypoint_planner.route_points[route_index:]
        route_wps = self.waypoint_planner.route_waypoints[route_index:]

        # Get IDM leading vehicle target speed
        idm_lead_speed = self._idm_get_lead_speed(self.target_speed_initial)

        idm_obs_speed = self._idm_get_obstacle_speed(self.target_speed_initial)

        # Get IDM stop_for object target speeds
        stop_for_idm_target_speed = self._stop_for(stop_conditions, self.follow_route_sm.conditions_registry)

        # TODO: Fix decision-tree for deciding when to rely on ST planner
        # Run longitudinal planner if conditions exist
        st_planner_target_speed = float('inf')

        if len(other_conditions) > 0:
            prediction_data = self.predict_collisions()
            # TODO: PROPERLY IMPLEMENT THE EGO STATION CHECK, MAYBE PRECOMPUTE ROUTE DURING INITIALIZATION OR AFTER ROUTE CHANGES
            plan_route_start_idx = self.long_planner.plan_route_start_idx
            if plan_route_start_idx < 0 or not self.long_planner.has_active_plan:
                print(f'SETTING TO ROUTE INDEX, PLAN ROUTE IDX IS UNSET')
                plan_route_start_idx = route_index

            s_ego_m = self._compute_ego_station_from_index(
                self.waypoint_planner.route_points,
                plan_route_start_idx,
            )
            print(f'EGO STATION: {s_ego_m}')

            # TODO: THIS IS ACTUAL BS, FIX THIS. WE NEED A BETTER WAY TO SELECT TARGET S DISTANCE
            def get_s_target(stop_conditions):
                if not stop_conditions:
                    return 30.0

                plan_s_goal_m = 2.0
                for actor_id, condition in stop_conditions.items():
                    if condition[1] == 'stop_sign':
                        if self.scene_data.traffic_data and self.scene_data.traffic_data.next_stop_sign:
                            plan_s_goal_m = max(plan_s_goal_m, self.scene_data.traffic_data.next_stop_sign.distance_to_stop_sign)
                    elif condition[1] == 'traffic_light':
                        if self.scene_data.traffic_data and self.scene_data.traffic_data.next_traffic_light:
                            plan_s_goal_m = max(plan_s_goal_m, self.scene_data.traffic_data.next_traffic_light.distance_to_light)
                return plan_s_goal_m

            plan_s_goal_m = min(get_s_target(stop_conditions), 30.0)
            print(f'\n\nFOLLOW ROUTE S TARGET')
            print(f'\tS: {plan_s_goal_m}')

            # st_planner_target_speed = self.long_planner.run_step(
            #     ego_route_points_3d=route_pts,
            #     ego_speed=ego_speed,
            #     ego_max_speed=speed_limit,
            #     actor_collisions=prediction_data.all_actor_collisions,
            #     plan_tick_counter=cur_tick,
            #     all_conditions=other_conditions,
            #     plan_s_goal_m=30.0,
            #     s_ego_m=s_ego_m,
            #     plan_route_start_idx=route_index
            # )
            st_planner_target_speed = self.long_planner.run_step(
                ego_route_points_3d=route_pts,
                ego_speed=ego_speed,
                ego_max_speed=speed_limit,
                actor_collisions=prediction_data.all_actor_collisions,
                actor_overlaps=prediction_data.all_actor_overlaps,
                plan_tick_counter=cur_tick,
                all_conditions=other_conditions,
                plan_s_goal_m=plan_s_goal_m,
                s_ego_m=s_ego_m,
                plan_route_start_idx=plan_route_start_idx,
                route_index=route_index,
            )

        print(f'follow_route speeds -> speed_limit: {speed_limit}, idm: {idm_lead_speed}, stop_for: {stop_for_idm_target_speed}, st_plan: {st_planner_target_speed}, obs: {idm_obs_speed}')
        # target_speed = min(idm_lead_speed, stop_for_idm_target_speed, st_planner_target_speed)
        target_speed = min(idm_lead_speed, stop_for_idm_target_speed, st_planner_target_speed, idm_obs_speed)
        # brake = target_speed < 1e-2
        brake = False

        print(f'\n\nCommand Status:')
        print(f'{cmd_status.to_string()}')

        action_complete = cmd_status.phase == ActionPhase.CLEARED
        collision_events = None if not action_complete else self.follow_route_sm.collision_events
        if collision_events:
            print(f'\n\n\nSTORED COLLISION EVENTS LOGGING')
        completion_reason = cmd_status.reasons[0]

        return target_speed, brake, action_complete, completion_reason, route_pts, route_wps, collision_events

    def _turn(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        other_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        turn_dir : RoadOption,
        cur_tick : float,
    ) -> Tuple:
        # Reset state machine if cleared
        if self.turn_sm.phase == ActionPhase.CLEARED:
            self.turn_sm.reset()

        # Get current planner state
        planner_state = self.get_planner_state()
        route_index = planner_state.route_index
        max_route_len = planner_state.route_points.shape[0]
        route_pts = planner_state.route_points[route_index:]
        route_wps = planner_state.route_waypoints[route_index:]
        s_route = planner_state.s_route[route_index:]

        # Update state machine conditions
        self.turn_sm.update_conditions(cur_plan.conditions)

        # Execute one step in state machine. Downstream functions generate target speeds based on current state
        cmd_status = self.turn_sm.update_state(
            config=self.config,
            scene_data=self.scene_data,
            planner_state=planner_state,
            turn_dir=turn_dir
        )

        speed_limit = self.scene_data.traffic_data.speed_limit
        ego_speed = self.scene_data.ego_data.speed

        # Get IDM leading vehicle target speed
        idm_lead_speed = self._idm_get_lead_speed(self.target_speed_initial)

        idm_obs_speed = self._idm_get_obstacle_speed(self.target_speed_initial)

        # Get IDM stop_for object target speeds
        stop_for_idm_target_speed = self._stop_for(stop_conditions, self.turn_sm.conditions_registry)

        # Run longitudinal planner if conditions exist
        prediction_data = self.predict_collisions()

        st_planner_target_speed = 0.0

        # NOTE: ONLY ADDING PEDESTRIAN HANDLING FOR NOW
        # TODO: FIX THE PLAN_S_GOAL_M DISTANCE
        # TODO: PROPERLY IMPLEMENT THE EGO STATION CHECK, MAYBE PRECOMPUTE ROUTE DURING INITIALIZATION OR AFTER ROUTE CHANGES
        plan_route_start_idx = self.long_planner.plan_route_start_idx
        if plan_route_start_idx < 0 or not self.long_planner.has_active_plan:
            print(f'SETTING TO ROUTE INDEX, PLAN ROUTE IDX IS UNSET')
            plan_route_start_idx = route_index

        s_ego_m = self._compute_ego_station_from_index(
            self.waypoint_planner.route_points,
            plan_route_start_idx,
        )

        plan_s_goal_m = 30.0
        turn_end_idx = self.turn_sm.prev_end_idx
        if turn_end_idx != -1:
            ego_s = s_route[0]

            turn_buffer_pts = self.config.long_planning_turn_buffer_m * self.config.points_per_meter
            turn_end_idx = int(min(max_route_len - 1, turn_end_idx + turn_buffer_pts))
            turn_end_s = s_route[turn_end_idx - route_index]

            plan_s_goal_m = min(plan_s_goal_m, max(1.0, turn_end_s - ego_s))

            print(f'\n\nS ROUTE MEASUREMENTS')
            print(f'\tEGO: {ego_s}, TURN_END: {turn_end_s}')
            print(f'\t\tREMAINING S: {plan_s_goal_m}')

        print(f'EGO STATION: {s_ego_m}')

        # st_planner_target_speed = self.long_planner.run_step(
        #     ego_route_points_3d=route_pts[:to_index],
        #     ego_speed=ego_speed,
        #     ego_max_speed=speed_limit,
        #     actor_collisions=prediction_data.all_actor_collisions,
        #     plan_tick_counter=cur_tick,
        #     all_conditions=other_conditions,
        #     plan_s_goal_m=30.0,
        #     s_ego_m=s_ego_m,
        #     plan_route_start_idx=plan_route_start_idx
        # )
        st_planner_target_speed = self.long_planner.run_step(
            ego_route_points_3d=route_pts,
            ego_speed=ego_speed,
            ego_max_speed=speed_limit,
            actor_collisions=prediction_data.all_actor_collisions,
            actor_overlaps=prediction_data.all_actor_overlaps,
            plan_tick_counter=cur_tick,
            all_conditions=other_conditions,
            plan_s_goal_m=plan_s_goal_m,
            s_ego_m=s_ego_m,
            plan_route_start_idx=plan_route_start_idx,
            route_index=route_index,
        )

        print(f'turn speeds -> speed_limit: {speed_limit}, idm: {idm_lead_speed}, stop_for: {stop_for_idm_target_speed}, st_plan: {st_planner_target_speed}, obs: {idm_obs_speed}')
        target_speed = min(idm_lead_speed, stop_for_idm_target_speed, st_planner_target_speed, idm_obs_speed)
        # brake = target_speed < 1e-2
        brake = False

        print(f'\n\nCommand Status:')
        print(f'{cmd_status.to_string()}')

        action_complete = cmd_status.phase == ActionPhase.CLEARED
        collision_events = None if not action_complete else self.turn_sm.collision_events
        completion_reason = cmd_status.reasons[0]

        return target_speed, brake, action_complete, completion_reason, route_pts, route_wps, collision_events

    def _turn_left(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        other_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        cur_tick : float
    ) -> Tuple:
        return self._turn(cur_plan, stop_conditions, other_conditions, RoadOption.LEFT, cur_tick)

    def _turn_right(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        other_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        cur_tick : float
    ) -> Tuple:
        return self._turn(cur_plan, stop_conditions, other_conditions, RoadOption.RIGHT, cur_tick)

    def _turn_straight(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        other_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        cur_tick : float
    ) -> Tuple:
        return self._turn(cur_plan, stop_conditions, other_conditions, RoadOption.STRAIGHT, cur_tick)

    def _change_lane(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        other_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        lc_dir : RoadOption,
        cur_tick : float
    ) -> Tuple:
        # Reset state machine if cleared
        if self.lane_change_sm.phase == ActionPhase.CLEARED:
            self.lane_change_sm.reset()

        # Update state machine conditions
        self.lane_change_sm.update_conditions(cur_plan.conditions)

        # Execute one step in state machine. Downstream functions generate target speeds based on current state
        cmd_status = self.lane_change_sm.update_state(
            config=self.config,
            scene_data=self.scene_data,
            planner_state=self.waypoint_planner.get_planner_state(),
            lc_dir=lc_dir
        )

        speed_limit = self.scene_data.traffic_data.speed_limit
        ego_speed = self.scene_data.ego_data.speed

        route_index = self.waypoint_planner.route_index
        max_route_len = self.waypoint_planner.route_points.shape[0]
        route_pts = self.waypoint_planner.route_points[route_index:]
        route_wps = self.waypoint_planner.route_waypoints[route_index:]
        s_route = self.waypoint_planner.s_route[route_index:]

        # Get IDM leading vehicle target speed
        idm_lead_speed = self._idm_get_lead_speed(self.target_speed_initial)
        print(f'(change_lane) Leading vehicle IDM target speed: {idm_lead_speed}')

        # Get IDM stop_for object target speeds
        stop_for_idm_target_speed = self._stop_for(stop_conditions, self.lane_change_sm.conditions_registry)

        # Run longitudinal planner if conditions exist
        prediction_data = self.predict_collisions()

        st_planner_target_speed = 0.0

        # TODO: FIX THE PLAN_S_GOAL_M DISTANCE
        # TODO: PROPERLY IMPLEMENT THE EGO STATION CHECK, MAYBE PRECOMPUTE ROUTE DURING INITIALIZATION OR AFTER ROUTE CHANGES
        plan_route_start_idx = self.long_planner.plan_route_start_idx
        if plan_route_start_idx < 0 or not self.long_planner.has_active_plan:
            print(f'SETTING TO ROUTE INDEX, PLAN ROUTE IDX IS UNSET')
            plan_route_start_idx = route_index

        s_ego_m = self._compute_ego_station_from_index(
            self.waypoint_planner.route_points,
            plan_route_start_idx,
        )

        plan_s_goal_m = 40.0
        lc_end_idx = self.lane_change_sm.prev_end_idx
        if lc_end_idx != -1:
            ego_s = s_route[0]

            lc_buffer_pts = self.config.long_planning_turn_buffer_m * self.config.points_per_meter
            lc_end_idx = int(min(max_route_len - 1, lc_end_idx + lc_buffer_pts))
            lc_end_s = s_route[lc_end_idx - route_index]

            plan_s_goal_m = min(plan_s_goal_m, max(1.0, lc_end_s - ego_s))

            print(f'\n\nS ROUTE MEASUREMENTS')
            print(f'\tEGO: {ego_s}, LC_END: {lc_end_s}')
            print(f'\t\tREMAINING S: {plan_s_goal_m}')

        print(f'EGO STATION: {s_ego_m}')

        # st_planner_target_speed = self.long_planner.run_step(
        #     ego_route_points_3d=route_pts,
        #     ego_speed=ego_speed,
        #     ego_max_speed=speed_limit,
        #     actor_collisions=prediction_data.all_actor_collisions,
        #     plan_tick_counter=cur_tick,
        #     all_conditions=other_conditions,
        #     plan_s_goal_m=30.0,
        #     s_ego_m=s_ego_m,
        #     plan_route_start_idx=plan_route_start_idx
        # )
        st_planner_target_speed = self.long_planner.run_step(
            ego_route_points_3d=route_pts,
            ego_speed=ego_speed,
            ego_max_speed=speed_limit,
            actor_collisions=prediction_data.all_actor_collisions,
            actor_overlaps=prediction_data.all_actor_overlaps,
            plan_tick_counter=cur_tick,
            all_conditions=other_conditions,
            plan_s_goal_m=plan_s_goal_m,
            s_ego_m=s_ego_m,
            plan_route_start_idx=plan_route_start_idx,
            route_index=route_index,
        )

        print(f'(lane_change) ST Planner target speed: {st_planner_target_speed}')

        target_speed = min(idm_lead_speed, stop_for_idm_target_speed, st_planner_target_speed)
        # brake = target_speed < 1e-2
        brake = False

        print(f'\n\nCommand Status:')
        print(f'{cmd_status.to_string()}')

        action_complete = cmd_status.phase == ActionPhase.CLEARED
        collision_events = None if not action_complete else self.lane_change_sm.collision_events
        completion_reason = cmd_status.reasons[0]

        return target_speed, brake, action_complete, completion_reason, route_pts, route_wps, collision_events

    def _change_lane_left(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        other_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        cur_tick : float
    ) -> Tuple:
        return self._change_lane(cur_plan, stop_conditions, other_conditions, RoadOption.CHANGELANELEFT, cur_tick)

    def _change_lane_right(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        other_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        cur_tick : float
    ) -> Tuple:
        return self._change_lane(cur_plan, stop_conditions, other_conditions, RoadOption.CHANGELANERIGHT, cur_tick)

    def _overtake(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        other_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        overtake_dir : RoadOption,
        cur_tick : float,
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

        # Reset state machine if cleared
        if self.overtake_sm.phase == ActionPhase.CLEARED:
            self.overtake_sm.reset()
            self.lat_planner.reset_plan()
            self.long_planner.reset_plan()

        # Update state machine conditions
        self.overtake_sm.update_conditions(cur_plan.conditions)

        speed_limit = self.scene_data.traffic_data.speed_limit
        ego_speed = self.scene_data.ego_data.speed

        route_index = self.waypoint_planner.route_index
        route_pts = self.waypoint_planner.route_points
        route_yaws = self.waypoint_planner.rotation_angles
        # TODO: Adjust route waypoints after local route modifications
        route_wps = self.waypoint_planner.route_waypoints
        s_route = self.waypoint_planner.s_route

        # Get planner state
        planner_state = self.get_planner_state()
        prediction_data = self.predict_collisions()

        # Get start and end points of overtake segment
        # NOTE: USING EGO LOCATION CAN RESULT IN SCENARIOS WHERE THE ACTUAL TARGET DISTANCE IS MUCH LARGER THAN THE
        # DESIRED AMOUNT BECAUSE THE ROUTE INDEX COULD BE FURTHER AHEAD OF THE EGO. THEREFORE RELATIVE TO THE ROUTE INDEX,
        # THE DISTANCE IS CAPPED AT THE TARGET DISTANCE, BUT WRT EGO, THE DISTANCE IS SLIGHTLY LARGER. THIS CAUSES THE
        # GRID TRANSFORMATION METHODS TO NOT EXACTLY OUTPUT THE CORRECT END POINT TRANSFORMATIONS DUE TO THE CLIPPING THAT'S DONE

        lat_plan_result = self.lat_planner.run_step(
            state_machine=self.overtake_sm,
            plan_tick_counter=cur_tick,
            planner_state=planner_state,
            lidar_data=self.lidar_pts,
            scene_data=self.scene_data,
            prediction_data=prediction_data,
            ego_plan=cur_plan,
            key_actor_registry=self.key_actor_registry,
            buffer_distance=self.ego_vehicle.bounding_box.extent.x * 2 + 5.0,
            all_conditions={} # TODO: UPDATE THIS
        )

        if lat_plan_result.is_empty_plan:
            self.overtake_sm.reset()
        elif lat_plan_result.is_new_plan:
            # Interpolate and reassign route with new modifications
            route_pts, route_yaws, s_route, route_bbs = self._interpolate_lateral_plan(
                planner_state=planner_state,
                lat_planner_result=lat_plan_result
            )

            self.waypoint_planner.route_points = route_pts
            self.waypoint_planner.rotation_angles = route_yaws
            self.waypoint_planner.s_route = s_route
            self.waypoint_planner.route_bbs = route_bbs

            # TODO: PASS IN ACTUAL DRIVING COMMAND BASED ON FUNCTION TYPE CALLED
            # mid_idx = int((route_index + goal_idx) / 2)
            # self.waypoint_planner.commands[route_index : mid_idx] = RoadOption.CHANGELANELEFT
            # self.waypoint_planner.commands[mid_idx : goal_idx] = RoadOption.CHANGELANERIGHT

            # Save current route modification start and end points
            self.overtake_sm.set_route_changes(
                route_change_idxs=(lat_plan_result.start_idx, lat_plan_result.goal_idx)
            )

            # TODO: TESTING TO SEE IF OVERTAKE ISSUE OF ST PLANS THAT COLLIDE PERSIST
            self.long_planner.reset_plan()

        # Execute one step in state machine. Downstream functions generate target speeds based on current state
        cmd_status = self.overtake_sm.update_state(
            config=self.config,
            scene_data=self.scene_data,
            planner_state=planner_state,
            overtake_dir=overtake_dir
        )

        # Get IDM leading vehicle target speed
        # idm_lead_speed = self._idm_get_lead_speed(self.target_speed_initial)
        # print(f'(overtake) Leading vehicle IDM target speed: {idm_lead_speed}')

        # Get IDM stop_for object target speeds
        stop_for_idm_target_speed = self._stop_for(stop_conditions, self.overtake_sm.conditions_registry)

        plan_s_goal_m = 50.0
        if self.overtake_sm.phase == ActionPhase.EXECUTING:
            plan_start_idx, plan_goal_idx = self.overtake_sm.cur_route_changes
            route_segment = route_pts[route_index : plan_goal_idx + 1]
            if route_segment.shape[0] > 1:
                # # TODO TEMP SHORTENING OF OVERTAKE PORTION FOR ST PLANNING
                # end_loc = carla.Location(x=route_segment[-1, 0], y=route_segment[-1, 1], z=route_segment[-1, 2])
                # end_wp = self.carla_map.get_waypoint(end_loc)

                # n = int(route_segment.shape[0])
                # plan_goal_idx_short = plan_goal_idx
                # for i in range(n - 1, -1, -1):
                #     cur_loc = carla.Location(x=route_segment[i, 0], y=route_segment[i, 1], z=route_segment[i, 2])
                #     cur_wp = self.carla_map.get_waypoint(cur_loc)
                #     if cur_wp.lane_id != end_wp.lane_id:
                #         plan_goal_idx_short = i
                #         break

                # print(f'\n\nPLAN GOAL IDXs')
                # print(f'\tOG: {plan_goal_idx - plan_start_idx}, NEW: {plan_goal_idx_short}')
                # plan_goal_idx_short = int(min(plan_goal_idx + 1, plan_goal_idx_short + 5.0 * self.config.points_per_meter))
                # route_segment = route_segment[:plan_goal_idx_short]
                # print(f'\tNEW + BUFFER: {plan_goal_idx_short}')

                plan_s_goal_m = cumulative_arclength(route_segment)[-1]
                print(f'plan_s_goal_m: {plan_s_goal_m}')

        # Run longitudinal planner
        st_planner_target_speed = 0.0
        # TODO: FIX THE PLAN_S_GOAL_M DISTANCE
        # TODO: PROPERLY IMPLEMENT THE EGO STATION CHECK, MAYBE PRECOMPUTE ROUTE DURING INITIALIZATION OR AFTER ROUTE CHANGES
        # TODO: ADJUST WHETHER WE SHOULD AGGRESSIVELY SWAP WITH BETTER ST PLANS DURING EXECUTIONS OF CERTAIN CRITICAL COMMANDS LIKE OVERTAKE
        plan_route_start_idx = self.long_planner.plan_route_start_idx
        if plan_route_start_idx < 0 or not self.long_planner.has_active_plan:
            # print(f'SETTING TO ROUTE INDEX, PLAN ROUTE IDX IS UNSET')
            plan_route_start_idx = route_index

        s_ego_m = self._compute_ego_station_from_index(
            self.waypoint_planner.route_points,
            plan_route_start_idx,
        )
        print(f'EGO STATION: {s_ego_m}')

        # st_planner_target_speed = self.long_planner.run_step(
        #     ego_route_points_3d=route_pts[route_index:],
        #     ego_speed=ego_speed,
        #     ego_max_speed=speed_limit,
        #     actor_collisions=prediction_data.all_actor_collisions,
        #     plan_tick_counter=cur_tick,
        #     all_conditions=other_conditions,
        #     plan_s_goal_m=plan_s_goal_m,
        #     s_ego_m=s_ego_m,
        #     plan_route_start_idx=plan_route_start_idx
        # )
        # st_planner_target_speed = self.long_planner.run_step(
        #     ego_route_points_3d=route_pts[route_index:],
        #     ego_speed=ego_speed,
        #     ego_max_speed=speed_limit,
        #     actor_collisions=prediction_data.all_actor_collisions,
        #     actor_overlaps=prediction_data.all_actor_overlaps,
        #     plan_tick_counter=cur_tick,
        #     all_conditions=other_conditions,
        #     plan_s_goal_m=plan_s_goal_m,
        #     s_ego_m=s_ego_m,
        #     plan_route_start_idx=plan_route_start_idx,
        #     route_index=route_index,
        # )

        # print(f'(overtake) ST Planner target speed: {st_planner_target_speed}')
        # target_speed = min(idm_lead_speed, stop_for_idm_target_speed, st_planner_target_speed)
        target_speed = min(0.0, stop_for_idm_target_speed, st_planner_target_speed)
        # brake = target_speed < 1e-2
        brake = False

        # print(f'Command Status:')
        # print(f'{cmd_status.to_string()}')

        action_complete = cmd_status.phase == ActionPhase.CLEARED
        collision_events = None if not action_complete else self.overtake_sm.collision_events
        completion_reason = cmd_status.reasons[0]

        return target_speed, brake, action_complete, completion_reason, route_pts[route_index:], route_wps[route_index:], collision_events

    def _overtake_left(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        other_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        cur_tick : float
    ) -> Tuple:
        return self._overtake(cur_plan, stop_conditions, other_conditions, RoadOption.CHANGELANELEFT, cur_tick)

    def _overtake_right(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        other_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        cur_tick : float
    ) -> Tuple:
        return self._overtake(cur_plan, stop_conditions, other_conditions, RoadOption.CHANGELANERIGHT, cur_tick)

    def _pull_over(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        other_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        pull_over_dir : RoadOption,
        cur_tick : float,
    ) -> Tuple:
        # Reset state machine if cleared
        if self.pull_over_sm.phase == ActionPhase.CLEARED:
            self.pull_over_sm.reset()

        # Update state machine conditions
        self.pull_over_sm.update_conditions(cur_plan.conditions)

        speed_limit = self.scene_data.traffic_data.speed_limit
        ego_speed = self.scene_data.ego_data.speed

        route_index = self.waypoint_planner.route_index
        route_pts = self.waypoint_planner.route_points
        route_wps = self.waypoint_planner.route_waypoints

        # TODO: VEHICLE ID MAY NOT BE RELEVANT, INVESTIGATE
        target_vehicle_id = 0
        for cond in cur_plan.conditions:
            if cond.condition_action == ConditionAction.YIELD_FOR and cond.obj_type == "vehicle":
                target_vehicle_id = cond.id
                break

        # Check if we need to update the route, modify route to change to target lane
        if self.pull_over_sm.needs_route_adjustment:
            prep_distance = ego_speed * self.config.pull_over_preparation_time
            pre_shift_points = int(prep_distance * self.config.points_per_meter)

            pull_over_start_idx = min(route_pts.shape[0], route_index + pre_shift_points)

            braking_lookahead = int(self.config.points_per_meter * ((
                (ego_speed * 3.6) / 10.0)**2 / 2.0) + self.config.braking_distance_calculation_safety_distance)

            print(f'braking_distance: {braking_lookahead / self.config.points_per_meter}')

            pull_over_yield_idx = min(route_pts.shape[0], pull_over_start_idx + braking_lookahead)

            pull_over_end_idx = min(route_pts.shape[0], pull_over_yield_idx + self.config.pull_over_route_lookahead)

            self.waypoint_planner.shift_route_smoothly(pull_over_start_idx, pull_over_end_idx, pull_over_dir == RoadOption.CHANGELANELEFT, self.config.transition_smoothness_distance)

            route_change_idxs = (pull_over_start_idx, pull_over_yield_idx, pull_over_end_idx)
            self.pull_over_sm.set_route_changes(route_change_idxs)

        # Localize emergency vehicle
        vehicle_passed = False
        has_vehicle = False
        vehicle_data = self._get_vehicle_by_id(target_vehicle_id)
        if vehicle_data:
            ego_tf = self.ego_vehicle.get_transform()
            diff = vehicle_data.vehicle.get_location() - ego_tf.location
            ahead = ego_tf.get_forward_vector().dot(diff) > 0

            if ahead and diff.length() > self.config.emergency_vehicle_clear_distance:
                vehicle_passed = True

        # Execute one step in state machine. Downstream functions generate target speeds based on current state
        cmd_status = self.pull_over_sm.update_state(
            config=self.config,
            scene_data=self.scene_data,
            planner_state=self.waypoint_planner.get_planner_state(),
            pull_over_dir=pull_over_dir,
            vehicle_id=target_vehicle_id,
            vehicle_passed=vehicle_passed,
            has_vehicle=vehicle_data is not None
        )

        # Get IDM leading vehicle target speed
        idm_lead_speed = self._idm_get_lead_speed(self.target_speed_initial)
        print(f'(overtake) Leading vehicle IDM target speed: {idm_lead_speed}')

        # Get IDM stop_for object target speeds
        stop_for_idm_target_speed = self._stop_for(stop_conditions, self.pull_over_sm.conditions_registry)

        prediction_data = self.predict_collisions()

        st_planner_target_speed = float("inf")

        if self.pull_over_sm.sub_phase == PullOverPhase.YIELDING:
            yield_idx = self.pull_over_sm.cur_route_changes[1]
            ego_loc = self.ego_vehicle.get_location()
            end_point = self.waypoint_planner.route_points[yield_idx]
            dist_to_end = np.linalg.norm(end_point - np.array([ego_loc.x, ego_loc.y, ego_loc.z]))
            at_terminal_endpoint = dist_to_end < 2.0 or route_index >= yield_idx

            # TODO: NEED TO UPDATE LONG PLANNER TO NOT CRASH WHEN ROUTE_PTS IS EMPTY OR HAS 1 ELEMENT
            if not at_terminal_endpoint:
                # st_planner_target_speed = self.long_planner.run_step(
                #     ego_route_points_3d=route_pts[route_index:],
                #     ego_speed=ego_speed,
                #     ego_max_speed=self.target_speed_initial,
                #     actor_collisions=prediction_data.all_actor_collisions,
                #     plan_tick_counter=cur_tick,
                #     all_conditions=other_conditions,
                #     plan_s_goal_m=dist_to_end
                # )
                plan_route_start_idx = self.long_planner.plan_route_start_idx
                if plan_route_start_idx < 0 or not self.long_planner.has_active_plan:
                    print(f'SETTING TO ROUTE INDEX, PLAN ROUTE IDX IS UNSET')
                    plan_route_start_idx = route_index

                s_ego_m = self._compute_ego_station_from_index(
                    self.waypoint_planner.route_points,
                    plan_route_start_idx,
                )
                print(f'EGO STATION: {s_ego_m}')

                st_planner_target_speed = self.long_planner.run_step(
                    ego_route_points_3d=route_pts[route_index:],
                    ego_speed=ego_speed,
                    ego_max_speed=self.target_speed_initial,
                    actor_collisions=prediction_data.all_actor_collisions,
                    actor_overlaps=prediction_data.all_actor_overlaps,
                    plan_tick_counter=cur_tick,
                    plan_route_start_idx=plan_route_start_idx,
                    route_index=route_index,
                    all_conditions=other_conditions,
                    plan_s_goal_m=dist_to_end,
                    s_ego_m=s_ego_m,
                )
                print(f'YIELD PHASE ST_SPEED: {st_planner_target_speed}')
        elif self.pull_over_sm.sub_phase is PullOverPhase.WAITING:
            st_planner_target_speed = 0.0
        elif self.pull_over_sm.sub_phase is PullOverPhase.RETURNING:
            end_idx = self.pull_over_sm.cur_route_changes[2]
            ego_loc = self.ego_vehicle.get_location()
            end_point = self.waypoint_planner.route_points[end_idx]
            dist_to_end = np.linalg.norm(end_point - np.array([ego_loc.x, ego_loc.y, ego_loc.z]))

            # st_planner_target_speed = self.long_planner.run_step(
            #     ego_route_points_3d=route_pts[route_index:],
            #     ego_speed=ego_speed,
            #     ego_max_speed=speed_limit,
            #     actor_collisions=prediction_data.all_actor_collisions,
            #     plan_tick_counter=cur_tick,
            #     all_conditions=other_conditions,
            #     plan_s_goal_m=dist_to_end
            # )
            plan_route_start_idx = self.long_planner.plan_route_start_idx
            if plan_route_start_idx < 0 or not self.long_planner.has_active_plan:
                print(f'SETTING TO ROUTE INDEX, PLAN ROUTE IDX IS UNSET')
                plan_route_start_idx = route_index

            s_ego_m = self._compute_ego_station_from_index(
                self.waypoint_planner.route_points,
                plan_route_start_idx,
            )
            print(f'EGO STATION: {s_ego_m}')

            st_planner_target_speed = self.long_planner.run_step(
                ego_route_points_3d=route_pts[route_index:],
                ego_speed=ego_speed,
                ego_max_speed=speed_limit,
                actor_collisions=prediction_data.all_actor_collisions,
                actor_overlaps=prediction_data.all_actor_overlaps,
                plan_tick_counter=cur_tick,
                plan_route_start_idx=plan_route_start_idx,
                route_index=route_index,
                all_conditions=other_conditions,
                plan_s_goal_m=dist_to_end,
                s_ego_m=s_ego_m,
            )
            print(f'RETURN PHASE ST_SPEED: {st_planner_target_speed}')

        target_speed = min(idm_lead_speed, stop_for_idm_target_speed, st_planner_target_speed, speed_limit)
        # brake = target_speed < 1e-2
        brake = False

        print(f'\n\nCommand Status:')
        print(f'{cmd_status.to_string()}')

        action_complete = cmd_status.phase == ActionPhase.CLEARED
        collision_events = None if not action_complete else self.pull_over_sm.collision_events
        completion_reason = cmd_status.reasons[0]

        route_pts_view = self.waypoint_planner.route_points[route_index:]
        route_wps_view = self.waypoint_planner.route_waypoints[route_index:]

        return target_speed, brake, action_complete, completion_reason, route_pts_view, route_wps_view, collision_events

    def _pull_over_left(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        other_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        cur_tick : float
    ) -> Tuple:
        return self._pull_over(cur_plan, stop_conditions, other_conditions, RoadOption.CHANGELANELEFT, cur_tick)

    def _pull_over_right(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        other_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        cur_tick : float
    ) -> Tuple:
        return self._pull_over(cur_plan, stop_conditions, other_conditions, RoadOption.CHANGELANERIGHT, cur_tick)

    def _share_lane(
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

        # Reset state machine if cleared
        if self.share_lane_sm.phase == ActionPhase.CLEARED:
            self.share_lane_sm.reset()
            self.lat_planner.reset_plan()
            self.long_planner.reset_plan()

        # Update state machine conditions
        self.share_lane_sm.update_conditions(cur_plan.conditions)

        speed_limit = self.scene_data.traffic_data.speed_limit
        ego_speed = self.scene_data.ego_data.speed

        route_index = self.waypoint_planner.route_index
        route_pts = self.waypoint_planner.route_points
        route_yaws = self.waypoint_planner.rotation_angles
        # TODO: Adjust route waypoints after local route modifications
        route_wps = self.waypoint_planner.route_waypoints
        s_route = self.waypoint_planner.s_route

        # Get planner state
        planner_state = self.get_planner_state()
        prediction_data = self.predict_collisions()

        # Get start and end points of lane share segment
        # NOTE: USING EGO LOCATION FOR THE PLANNER START POINT CAN RESULT IN SCENARIOS WHERE THE ACTUAL TARGET DISTANCE IS MUCH LARGER THAN THE
        # DESIRED AMOUNT BECAUSE THE ROUTE INDEX COULD BE FURTHER AHEAD OF THE EGO. THEREFORE RELATIVE TO THE ROUTE INDEX,
        # THE DISTANCE IS CAPPED AT THE TARGET DISTANCE, BUT WRT EGO, THE DISTANCE IS SLIGHTLY LARGER. THIS CAUSES THE
        # GRID TRANSFORMATION METHODS TO NOT EXACTLY OUTPUT THE CORRECT END POINT TRANSFORMATIONS DUE TO THE CLIPPING THAT'S DONE

        all_intruder_data = self.scene_data.vehicle_data.get(traffic_type="oncoming", get_intruders=True)
        # if all_intruder_data:
        #     goal_wp = route_wps[goal_idx]
        #     wp_right_vec = goal_wp.transform.get_right_vector()
        #     lateral_shift = (self.ego_vehicle.bounding_box.extent.y) * wp_right_vec
        #     lateral_shift_arr = np.array([lateral_shift.x, lateral_shift.y, lateral_shift.z])

        #     goal_point_world = planner_state.original_route_points[goal_idx] + lateral_shift_arr
        # else:
        #     goal_point_world = route_pts[goal_idx]

        lat_plan_result = self.lat_planner.run_step(
            state_machine=self.share_lane_sm,
            plan_tick_counter=cur_tick,
            planner_state=planner_state,
            lidar_data=self.lidar_pts,
            scene_data=self.scene_data,
            prediction_data=prediction_data,
            ego_plan=cur_plan,
            key_actor_registry=self.key_actor_registry,
            buffer_distance=self.ego_vehicle.bounding_box.extent.x * 2,
            all_conditions={} # TODO: UPDATE THIS
        )

        if lat_plan_result.is_empty_plan:
            self.share_lane_sm.reset()
        elif lat_plan_result.is_new_plan:
            # Interpolate and reassign route with new modifications
            route_pts, route_yaws, s_route, route_bbs = self._interpolate_lateral_plan(
                planner_state=planner_state,
                lat_planner_result=lat_plan_result
            )

            self.waypoint_planner.route_points = route_pts
            self.waypoint_planner.rotation_angles = route_yaws
            self.waypoint_planner.s_route = s_route
            self.waypoint_planner.route_bbs = route_bbs

            # TODO: PASS IN ACTUAL DRIVING COMMAND BASED ON FUNCTION TYPE CALLED
            # mid_idx = int((route_index + goal_idx) / 2)
            # self.waypoint_planner.commands[route_index : mid_idx] = RoadOption.CHANGELANELEFT
            # self.waypoint_planner.commands[mid_idx : goal_idx] = RoadOption.CHANGELANERIGHT

            # Save current route modification start and end points
            self.share_lane_sm.set_route_changes(
                route_change_idxs=(lat_plan_result.start_idx, lat_plan_result.goal_idx)
            )

            # TODO: TESTING TO SEE IF SHARE LANE ISSUE OF ST PLANS THAT COLLIDE PERSIST
            self.long_planner.reset_plan()

        # Execute one step in state machine. Downstream functions generate target speeds based on current state
        cmd_status = self.share_lane_sm.update_state(
            config=self.config,
            scene_data=self.scene_data,
            planner_state=planner_state,
        )

        # Get IDM leading vehicle target speed
        idm_lead_speed = self._idm_get_lead_speed(self.target_speed_initial)
        print(f'(share_lane) Leading vehicle IDM target speed: {idm_lead_speed}')

        # Get IDM stop_for object target speeds
        stop_for_idm_target_speed = self._stop_for(stop_conditions, self.share_lane_sm.conditions_registry)

        plan_s_goal_m = 50.0
        if self.share_lane_sm.phase == ActionPhase.EXECUTING:
            plan_start_idx, plan_goal_idx = self.share_lane_sm.cur_route_changes
            route_segment = route_pts[route_index : plan_goal_idx + 1]
            if route_segment.shape[0] > 1:
                plan_s_goal_m = cumulative_arclength(route_segment)[-1]
                print(f'plan_s_goal_m: {plan_s_goal_m}')

        # Run longitudinal planner
        st_planner_target_speed = 0.0
        # TODO: FIX THE PLAN_S_GOAL_M DISTANCE
        # TODO: PROPERLY IMPLEMENT THE EGO STATION CHECK, MAYBE PRECOMPUTE ROUTE DURING INITIALIZATION OR AFTER ROUTE CHANGES
        # TODO: ADJUST WHETHER WE SHOULD AGGRESSIVELY SWAP WITH BETTER ST PLANS DURING EXECUTIONS OF CERTAIN CRITICAL COMMANDS LIKE OVERTAKE
        plan_route_start_idx = self.long_planner.plan_route_start_idx
        if plan_route_start_idx < 0 or not self.long_planner.has_active_plan:
            print(f'SETTING TO ROUTE INDEX, PLAN ROUTE IDX IS UNSET')
            plan_route_start_idx = route_index

        s_ego_m = self._compute_ego_station_from_index(
            self.waypoint_planner.route_points,
            plan_route_start_idx,
        )
        print(f'EGO STATION: {s_ego_m}')

        # st_planner_target_speed = self.long_planner.run_step(
        #     ego_route_points_3d=route_pts[route_index:],
        #     ego_speed=ego_speed,
        #     ego_max_speed=speed_limit,
        #     actor_collisions=prediction_data.all_actor_collisions,
        #     plan_tick_counter=cur_tick,
        #     all_conditions=other_conditions,
        #     plan_s_goal_m=plan_s_goal_m,
        #     s_ego_m=s_ego_m,
        #     plan_route_start_idx=plan_route_start_idx
        # )
        st_planner_target_speed = self.long_planner.run_step(
            ego_route_points_3d=route_pts[route_index:],
            ego_speed=ego_speed,
            ego_max_speed=speed_limit,
            actor_collisions=prediction_data.all_actor_collisions,
            actor_overlaps=prediction_data.all_actor_overlaps,
            plan_tick_counter=cur_tick,
            all_conditions=other_conditions,
            plan_s_goal_m=plan_s_goal_m,
            s_ego_m=s_ego_m,
            plan_route_start_idx=plan_route_start_idx,
            route_index=route_index,
        )

        print(f'(share_lane) ST Planner target speed: {st_planner_target_speed}')
        target_speed = min(idm_lead_speed, stop_for_idm_target_speed, st_planner_target_speed)
        # brake = target_speed < 1e-2
        brake = False

        # print(f'Command Status:')
        # print(f'{cmd_status.to_string()}')

        action_complete = cmd_status.phase == ActionPhase.CLEARED
        collision_events = None if not action_complete else self.share_lane_sm.collision_events
        completion_reason = cmd_status.reasons[0]

        return target_speed, brake, action_complete, completion_reason, route_pts[route_index:], route_wps[route_index:], collision_events

    def _interpolate_lateral_plan(
        self,
        planner_state : PlannerState,
        lat_planner_result : LatPlannerResult
    ) -> Tuple:
        # Get relevant planner state variables
        route_pts = planner_state.route_points.copy()
        route_wps = planner_state.route_waypoints.copy()
        route_yaws = planner_state.rotation_angles.copy()
        route_bbs = planner_state.route_bbs.copy()
        s_route = planner_state.s_route.copy()

        # Setup start and goal indices and points
        start_idx = lat_planner_result.start_idx
        start_point_world = lat_planner_result.start_point_world

        goal_idx = lat_planner_result.goal_idx
        goal_point_world = lat_planner_result.goal_point_world

        planned_path = lat_planner_result.planned_path

        print(f'\n\n(share_lane) ASTAR PATH')
        print(f'\tpath_len: {planned_path.shape[0]}, og route segment len: {route_pts[start_idx:goal_idx + 1].shape[0]}')

        print(f'\tOG PLANNING POINTS')
        print(f'\t\tROUTE START: {start_point_world}, GOAL: {goal_point_world}')
        print(f'\tASTAR POINTS')
        print(f'\t\tSTART: {planned_path[0]}, GOAL: {planned_path[-1]}')


        print(f'\n\nSTART AND END POINTS')
        print(f'\tOG POINTS')
        print(f'\t\tSTART: {route_pts[start_idx]}, END - 2: {route_pts[goal_idx - 2]}, END - 1: {route_pts[goal_idx - 1]}, END: {route_pts[goal_idx]}')
        print(f'\tNEW POINTS')
        print(f'\t\tSTART: {planned_path[0]}, END - 2: {planned_path[-3]}, END - 1: {planned_path[-2]}, END: {planned_path[-1]}')

        # Slice path segment
        path_segment = planned_path[:-1] # Get all points exclusive of goal point

        s_path = self.waypoint_planner.cumulative_arclength(path_segment)
        s_route_segment = self.waypoint_planner.cumulative_arclength(route_pts[start_idx:goal_idx])

        # Normalized param along route_subset: 0..1
        t_route = s_route_segment / s_route_segment[-1]

        # Target arc-lengths along the planner segment
        s_target = t_route * s_path[-1]

        # Interpolate x, y, z by arc length
        x_interp = np.interp(s_target, s_path, path_segment[:, 0])
        y_interp = np.interp(s_target, s_path, path_segment[:, 1])
        z_pts = route_pts[start_idx:goal_idx, 2]

        route_pts_new = np.column_stack([x_interp, y_interp, z_pts])
        print(f'\tinterp route len: {route_pts_new.shape[0]}')

        # Recompute route yaws
        indices = np.arange(1, route_pts_new.shape[0] - 1)
        diffs = route_pts_new[indices + 1] - route_pts_new[indices - 1]
        route_yaws_new = np.arctan2(diffs[:, 1], diffs[:, 0]) * 180. / np.pi

        # Start point yaw using yaw of original start point
        route_yaw_new_start = route_yaws[start_idx]

        # Goal point yaw using last interpolated point and true goal point
        dx = goal_point_world[0] - route_pts_new[-1, 0]
        dy = goal_point_world[1] - route_pts_new[-1, 1]
        route_yaw_new_goal = np.degrees(np.arctan2(dy, dx))

        route_yaws_new = np.concatenate([[route_yaw_new_start], route_yaws_new, [route_yaw_new_goal]])

        # Recompute lane bounding boxes
        route_bbs_new = self.waypoint_planner.generate_route_bbs(
            route_points=route_pts_new,
            route_yaws=route_yaws_new,
            route_waypoints=route_wps[start_idx:goal_idx],
            stride=self.config.points_per_meter,
        )

        # Recompute route cumulative distances
        s_route_new = self.waypoint_planner.cumulative_arclength(
            route_points=route_pts[start_idx:],
        ) + s_route[start_idx]

        # TODO: UPDATE INDEX TRANSORMATION CODE WITH CONFIG NUMBERS
        start_idx_spaced = start_idx // self.config.points_per_meter
        start_idx_spaced = start_idx_spaced // 2

        goal_idx_spaced = goal_idx // self.config.points_per_meter
        goal_idx_spaced = goal_idx_spaced // 2

        # Overwrite only [start_idx : goal_idx), leave goal_idx as-is
        route_pts[start_idx:goal_idx] = route_pts_new

        print(f'\tNEW POINTS INTERP')
        print(f'\t\tSTART: {route_pts[start_idx]}, END - 2: {route_pts[goal_idx - 2]}, END - 1: {route_pts[goal_idx - 1]}, END: {route_pts[goal_idx]}')

        route_yaws[start_idx:goal_idx] = route_yaws_new

        print(f'\n\nS ROUTE')
        print(f'\tOG')
        print(f'\t\tSTART: {s_route[start_idx]}, END - 1: {s_route[goal_idx - 1]} END: {s_route[goal_idx]}')
        s_route[start_idx:] = s_route_new

        print(f'\tINTERP')
        print(f'\t\tSTART: {s_route[start_idx]}, END - 1: {s_route[goal_idx - 1]} END: {s_route[goal_idx]}')

        route_bbs[start_idx_spaced : goal_idx_spaced] = route_bbs_new

        return (route_pts, route_yaws, s_route, route_bbs)

    ########################################
    # Low-level condition API
    ########################################

    # TODO: TEMP METHOD DESIGN, FIX SOON
    def _stop_for(
        self,
        stop_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        conditions_registry : Dict[int, Union[StopForSM, YieldForSM]],
    ) -> Tuple[float, bool, Optional[str]]:

        target_speed = self.target_speed_initial
        for actor_id, condition in stop_conditions.items():
            print(f'stop_for actor_id: {actor_id}, type: {condition[1]}')
            if condition[1] == 'stop_sign':
                target_speed_ss = self._stop_for_stop_sign(conditions_registry)
                target_speed = min(target_speed, target_speed_ss)
            elif condition[1] == 'traffic_light':
                target_speed_tl = self._stop_for_traffic_light(conditions_registry)
                target_speed = min(target_speed, target_speed_tl)
            elif condition[1] == 'obstacle':
                target_speed_obstacle = self._stop_for_obstacle(conditions_registry)
                target_speed = min(target_speed, target_speed_obstacle)
            elif condition[1] == 'pedestrian':
                target_speed_pedestrian = self._stop_for_pedestrian(conditions_registry)
                target_speed = min(target_speed, target_speed_pedestrian)

        return target_speed

    ########################################
    # IDM Helpers
    ########################################
    # def _idm_get_lead_speed(
    #     self,
    #     target_speed_initial : float,
    # ) -> float:
    #     ego_speed = self.scene_data.ego_data.speed
    #     idm_target_speed = target_speed_initial
    #     leading_vehicles : Dict[str, List[LaneVehicleData]] = self.scene_data.vehicle_data['leading']
    #     if leading_vehicles:
    #         lane_vehicle_data : List[LaneVehicleData] = []
    #         if 'ego' in leading_vehicles:
    #             lane_vehicle_data = leading_vehicles['ego']
    #         lane_leading_vehicles : List[VehicleData] = []

    #         # Check for leading vehicles in the ego lane
    #         for lane_veh_data in lane_vehicle_data:
    #             lane_leading_vehicles.extend(lane_veh_data.vehicle_data)

    #         if lane_leading_vehicles:
    #             lv = lane_leading_vehicles[0]
    #             lv_speed = lv.speed
    #             lv_length = lv.vehicle.bounding_box.extent.x * 2

    #             dist_to_lv = lv.relative_distance

    #             desired_following_distance = self.config.idm_leading_vehicle_minimum_distance
    #             desired_time_headway = self.config.idm_leading_vehicle_time_headway

    #             idm_target_speed = self.idm.compute_target_speed_idm(
    #                 desired_speed = target_speed_initial,
    #                 leading_actor_length = lv_length,
    #                 ego_speed = ego_speed,
    #                 leading_actor_speed = lv_speed,
    #                 distance_to_leading_actor = dist_to_lv,
    #                 s0 = desired_following_distance,
    #                 T = desired_time_headway
    #             )
    #             print(f'Lead Vehicle ID: {lv.id}, IDM Speed: {idm_target_speed}\n')
    #     return idm_target_speed
    def _idm_get_lead_speed(
        self,
        target_speed_initial : float,
    ) -> float:
        ego_speed = self.scene_data.ego_data.speed
        idm_target_speed = target_speed_initial
        ego_leading_vehicles : List[VehicleDataEntry] = self.scene_data.vehicle_data.get(traffic_type="leading", lane_name="ego")
        if ego_leading_vehicles:
            lv = ego_leading_vehicles[0]
            lv_speed = lv.speed
            lv_length = lv.vehicle.bounding_box.extent.x * 2

            dist_to_lv = lv.relative_distance

            print(f'EGO LV DIST: {dist_to_lv}, speed: {lv_speed}, id: {lv.id}')

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
        target_speed = self._get_target_speed(target_speed_initial)
        start_index = self.waypoint_planner.route_index
        return target_speed, self.waypoint_planner.route_points[start_index:], self.waypoint_planner.route_waypoints[start_index:]

    def _stop_for_stop_sign(
        self,
        conditions_registry : Dict[int, Union[StopForSM, YieldForSM]],
    ):
        ego_speed = self.scene_data.ego_data.speed
        target_speed_initial = self.target_speed_initial
        next_ss = self.scene_data.traffic_data.next_stop_sign

        if next_ss is None:
            return target_speed_initial

        if next_ss.id not in conditions_registry:
            return target_speed_initial

        cond_sm = conditions_registry[next_ss.id]

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
        conditions_registry : Dict[int, Union[StopForSM, YieldForSM]],
    ):
        ego_speed = self.scene_data.ego_data.speed
        target_speed_initial = self.target_speed_initial
        next_tl = self.scene_data.traffic_data.next_traffic_light

        if next_tl is None:
            return target_speed_initial

        if next_tl.id not in conditions_registry:
            return target_speed_initial

        cond_sm = conditions_registry[next_tl.id]

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
        conditions_registry : Dict[int, Union[StopForSM, YieldForSM]],
    ):
        target_speed_initial = self.target_speed_initial

        ego_obstacles = self.scene_data.obstacle_data.ego_obstacles
        if not ego_obstacles:
            return target_speed_initial

        next_obstacle = ego_obstacles[0]
        if next_obstacle.id not in conditions_registry:
            return target_speed_initial

        cond_sm = conditions_registry[next_obstacle.id]

        if cond_sm.phase == ActionPhase.EXECUTING:
            target_speed = self._idm_get_obstacle_speed(target_speed_initial)
            return target_speed

        # NOTE: NOT TRACKING CLEARED OBSTACLES, COULD BE IMPORTANT TO AVOID INDEFINITELY WAITING FOR THEM

        return target_speed_initial

    def _stop_for_pedestrian(
        self,
        conditions_registry : Dict[int, Union[StopForSM, YieldForSM]],
    ):
        target_speed_initial = self.target_speed_initial

        if not self.scene_data.ped_data:
            return target_speed_initial

        next_ped = self.scene_data.ped_data[0]
        if next_ped.id not in conditions_registry:
            print(f'\n\nSTOP FOR PED NOT IN CONDITIONS REGISTRY')
            return target_speed_initial

        cond_sm = conditions_registry[next_ped.id]

        if cond_sm.phase == ActionPhase.EXECUTING:
            target_speed = self._idm_get_ped_speed(target_speed_initial)
            return target_speed

        # NOTE: NOT TRACKING CLEARED PEDESTRIANS, COULD BE IMPORTANT TO AVOID INDEFINITELY WAITING FOR THEM

        return target_speed_initial
