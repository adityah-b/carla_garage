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
from local_planner.longitudinal.config_specs import *
from local_planner.longitudinal.long_planner import LongPlanner

# Lateral planner
from local_planner.lateral.config_specs import *
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
        st_algo_spec = STAlgoSpec(
            A_max=self.config.idm_maximum_acceleration,
            J_max=50.0,
            W_vel=1.0,
            W_acc=1.0,
            W_jerk=1.0,
            ds_grid=0.25,
            dt_grid=0.05
        )
        st_grid_spec = STGridSpec(
            S_max=10.0,
            ds=0.25,
            T_max=2.0,
            dt=0.05
        )
        self.long_planner = LongPlanner(st_grid_spec=st_grid_spec, st_algo_spec=st_algo_spec, algo_name='dijkstra', sim_freq=self.config.fps)

        # Lateral planner
        lat_algo_spec = LatAlgoSpec()
        lat_grid_spec = LatGridSpec()
        self.lat_planner = LatPlanner(self.ego_vehicle, lat_grid_spec, lat_algo_spec)

        self._dispatch_map = {
            Action.FOLLOW_ROUTE: self._follow_route,

            Action.TURN_LEFT: self._turn_left,
            Action.TURN_RIGHT: self._turn_right,

            Action.CHANGE_LANE_LEFT: self._change_lane_left,
            Action.CHANGE_LANE_RIGHT: self._change_lane_right,

            Action.OVERTAKE_LEFT: self._overtake_left,
            Action.OVERTAKE_RIGHT: self._overtake_right,
        }

        # State vars
        self.active_lc : Optional[LaneChangeData] = None
        self.wf_state : Dict[str, Any] = None

        self.cur_plan : EgoPlan = None
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

    def predict_collisions(self) -> PredictionData:
        ########################################
        # Actor motion prediction
        ########################################

        # TODO: Pass conditions into motion prediction
        # Forecast ego vehicle
        ego_forecasted_bbs = self.motion_forecaster.predict_ego_motion(
            ego_vehicle=self.ego_vehicle,
            ego_route_pts=self.waypoint_planner.route_points[self.waypoint_planner.route_index:],
            target_speed=self.target_speed_initial
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

        # Debugging
        # world = self.ego_vehicle.get_world()
        # for actor_id, collision_intervals in all_actor_collisions.items():
        #     collision_interval = collision_intervals[0]
        #     collision_bbs = collision_interval.collision_bboxes_b

        #     world.debug.draw_string(
        #         location=collision_bbs[0].location,
        #         text=str(actor_id),
        #         color=self.config.other_vehicles_forecasted_bbs_color,
        #         life_time=self.config.draw_life_time
        #     )

        #     for bb in collision_bbs:
        #         world.debug.draw_box(
        #             box=bb,
        #             rotation=bb.rotation,
        #             thickness=0.1,
        #             color=self.config.ego_vehicle_forecasted_bbs_hazard_color,
        #             life_time=self.config.draw_life_time
        #         )

        return PredictionData(
            ego_forecasted_bbs=ego_forecasted_bbs,
            veh_forecasted_bbs=veh_forecasted_bbs,
            ped_forecasted_bbs=ped_forecasted_bbs,
            all_actor_collisions=all_actor_collisions
        )

    def set_plan(self, ego_plan : EgoPlan) -> None:
        self.cur_plan = ego_plan
        self.needs_replan = False

    def reset_plan(self):
        """Reset plan bookkeeping."""
        self.cur_plan = None
        self.needs_replan = True

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
        target_speed, brake, action_complete, route_pts, route_wps = fn(self.cur_plan, stop_conditions, other_conditions, cur_tick)

        if action_complete:
            self.reset_plan()

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
        speed_limit = self.scene_data.traffic_data.speed_limit
        ego_speed = self.scene_data.ego_data.speed

        route_index = self.waypoint_planner.route_index
        route_pts = self.waypoint_planner.route_points[route_index:]
        route_wps = self.waypoint_planner.route_waypoints[route_index:]

        # Get IDM leading vehicle target speed
        idm_target_speed = self._get_target_speed(
            self.target_speed_initial
        )
        print(f'Leading IDM target speed: {idm_target_speed}')

        # Get IDM stop_for object target speeds
        stop_for_idm_target_speed, action_complete = self._stop_for(stop_conditions)
        print(f'stop_for IDM target speed: {stop_for_idm_target_speed}, action_complete: {action_complete}')

        # TODO: Fix decision-tree for deciding when to rely on ST planner
        # Run longitudinal planner if conditions exist
        st_planner_target_speed = float('inf')
        if len(other_conditions) > 0:
            prediction_data = self.predict_collisions()

            st_planner_target_speed = self.long_planner.run_step(
                ego_route_points_3d=route_pts,
                ego_speed=ego_speed,
                ego_max_speed=speed_limit,
                actor_collisions=prediction_data.all_actor_collisions,
                plan_tick_counter=cur_tick,
                all_conditions=other_conditions
            )
            if self.long_planner.current_vel_profile.size == 0:
                st_planner_target_speed = 0.0

        print(f'ST Planner target speed: {st_planner_target_speed}')

        target_speed = min(idm_target_speed, stop_for_idm_target_speed, st_planner_target_speed)
        brake = target_speed < 1e-2

        # TODO: Fix stop_for state transition logic
        self.route_follow_ticks += 1
        if len(stop_conditions) == 0:
            if self.route_follow_ticks > 25:
                self.route_follow_ticks = 0
                action_complete = True
            else:
                action_complete = False

        return target_speed, brake, action_complete, route_pts, route_wps

    def _turn(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        other_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        cur_tick : float
    ) -> Tuple:
        action_complete = False
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
        st_planner_target_speed = 0.0
        if len(other_conditions) > 0:
            prediction_data = self.predict_collisions()

            st_planner_target_speed = self.long_planner.run_step(
                ego_route_points_3d=route_pts,
                ego_speed=ego_speed,
                ego_max_speed=speed_limit,
                actor_collisions=prediction_data.all_actor_collisions,
                plan_tick_counter=cur_tick,
                all_conditions=other_conditions
            )

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

        return target_speed, brake, action_complete, route_pts, route_wps

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
        speed_limit = self.scene_data.traffic_data.speed_limit
        ego_speed = self.scene_data.ego_data.speed

        route_index = self.waypoint_planner.route_index
        route_pts = self.waypoint_planner.route_points[route_index:]
        route_wps = self.waypoint_planner.route_waypoints[route_index:]

        # Run longitudinal planner if conditions exist
        prediction_data = self.predict_collisions()

        st_planner_target_speed = 0.0
        st_planner_target_speed = self.long_planner.run_step(
            ego_route_points_3d=route_pts,
            ego_speed=ego_speed,
            ego_max_speed=speed_limit,
            actor_collisions=prediction_data.all_actor_collisions,
            plan_tick_counter=cur_tick,
            all_conditions=other_conditions
        )

        print(f'(lane_change) ST Planner target speed: {st_planner_target_speed}')

        target_speed = st_planner_target_speed
        brake = target_speed < 1e-2

        if self.lane_change_prev_end_idx is None:
            _, _, start_idx, end_idx = self.scene_data.route_data.lane_change_data.entry
            self.lane_change_prev_end_idx = end_idx
        else:
            end_wp = self.waypoint_planner.route_waypoints[self.lane_change_prev_end_idx]
            cur_wp = self.waypoint_planner.route_waypoints[route_index]

            lc_dist = cur_wp.transform.location.distance(end_wp.transform.location)
            print(f'(lane_change) route_index: {route_index}, end_idx: {self.lane_change_prev_end_idx}, lc_dist: {lc_dist}')
            # TODO: FIX LANE CHANGE END CONDITION
            if route_index >= self.lane_change_prev_end_idx and lc_dist >= 10.0:
                self.lane_change_prev_end_idx = None
                action_complete = True

        return target_speed, brake, action_complete, route_pts, route_wps

    def _change_lane_left(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        other_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        cur_tick : float
    ) -> Tuple:
        return self._change_lane(cur_plan, stop_conditions, other_conditions, cur_tick)

        # ego_speed = self.scene_data.ego_data.speed
        # speed_limit = self.config.max_speed_in_junction
        # # target_speed_initial = min(ego_speed, speed_limit)
        # target_speed_initial = speed_limit

        # lc_data = self.scene_data.route_data.lane_change_data
        # if not lc_data:
        #     return self._maintain_speed(LongitudinalParams())

        # if self.active_lc is None or self.active_lc != lc_data:
        #     self.active_lc = lc_data
        #     # self.__change_lane(lc_data, RoadOption.CHANGELANELEFT)

        # # target_speed = self.__get_target_speed(target_speed_initial, LongitudinalParams(), target_lanes=['ego', 'left'])
        # target_speed = self.__get_target_speed(target_speed_initial, LongitudinalParams(), target_lanes=['left'])
        # brake = target_speed < 1e-2

        # end_idx = lc_data.entry[3]
        # route_wps = self.waypoint_planner.route_waypoints
        # end_wp = route_wps[end_idx]

        # start_index = self.waypoint_planner.route_index
        # route_points, route_waypoints = self._get_lateral_route(start_index)

        # cur_wp = route_wps[start_index]
        # if cur_wp.lane_id == end_wp.lane_id:
        #     move_to_next_state = True
        # else:
        #     move_to_next_state = False

        # return target_speed, brake, move_to_next_state, route_points, route_waypoints

    def _change_lane_right(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        other_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        cur_tick : float
    ) -> Tuple:
        return self._change_lane(cur_plan, stop_conditions, other_conditions, cur_tick)

        # ego_speed = self.scene_data.ego_data.speed
        # speed_limit = self.config.max_speed_in_junction
        # # target_speed_initial = min(ego_speed, speed_limit)
        # target_speed_initial = speed_limit

        # lc_data = self.scene_data.route_data.lane_change_data
        # if not lc_data:
        #     return self._maintain_speed(LongitudinalParams())

        # if self.active_lc is None or self.active_lc != lc_data:
        #     self.active_lc = lc_data
        #     # self.__change_lane(lc_data, RoadOption.CHANGELANERIGHT)

        # # target_speed = self.__get_target_speed(target_speed_initial, LongitudinalParams(), target_lanes=['ego', 'right'])
        # target_speed = self.__get_target_speed(target_speed_initial, LongitudinalParams(), target_lanes=['right'])
        # brake = target_speed < 1e-2

        # end_idx = lc_data.entry[3]
        # route_wps = self.waypoint_planner.route_waypoints
        # end_wp = route_wps[end_idx]

        # start_index = self.waypoint_planner.route_index
        # route_points, route_waypoints = self._get_lateral_route(start_index)

        # cur_wp = route_wps[start_index]
        # if cur_wp.lane_id == end_wp.lane_id:
        #     move_to_next_state = True
        # else:
        #     move_to_next_state = False

        # return target_speed, brake, move_to_next_state, route_points, route_waypoints

    def _overtake(
        self,
        cur_plan : EgoPlan,
        stop_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        other_conditions : Dict[int, Tuple[ConditionAction, str, float]],
        cur_tick : float
    ) -> Tuple:
        action_complete = False
        adjust_route = False

        speed_limit = self.scene_data.traffic_data.speed_limit
        ego_speed = self.scene_data.ego_data.speed

        route_index = self.waypoint_planner.route_index
        route_pts = self.waypoint_planner.route_points
        route_wps = self.waypoint_planner.route_waypoints

        # Check if we need to run the lateral planner
        if (self.cur_route_changes is None) or (self.cur_route_changes and route_index > self.cur_route_changes[1]):
            adjust_route = True

        if adjust_route:
            ego_loc_carla = self.ego_vehicle.get_location()
            start_point_world = np.array([ego_loc_carla.x, ego_loc_carla.y, ego_loc_carla.z])

            # TODO: NEED TO FIX HOW WE CHOOSE GOAL POINT (IDEALLY SHOULD BE PROVIDED BY LLM)
            goal_idx = min(route_pts.shape[0], route_index + int(self.config.detection_radius * self.config.points_per_meter))
            goal_point_world = route_pts[goal_idx]

            # Run lateral planner
            path_world_3d = self.lat_planner.run_step(
                route_points_world_3d=route_pts[route_index:goal_idx + 1], # goal_idx inclusive
                lidar_data=self.lidar_pts,
                start_point_world_3d=start_point_world,
                goal_point_world_3d=goal_point_world
            )
            if path_world_3d.size > 0:
                route_len = goal_idx - route_index + 1

                desired = np.arange(route_len)
                orig = np.arange(path_world_3d.shape[0])

                x_interp = np.interp(desired, orig, path_world_3d[:, 0])
                y_interp = np.interp(desired, orig, path_world_3d[:, 1])
                z_interp = np.interp(desired, orig, path_world_3d[:, 2])

                route_interp = np.column_stack([x_interp, y_interp, z_interp])

                print(f'path_world_3d shape: {path_world_3d.shape}')
                print(f'route_interp shape: {route_interp.shape}')
                print(f'route_pts start to goal shape: {self.waypoint_planner.route_points[route_index : goal_idx + 1].shape}')

                self.waypoint_planner.route_points[route_index : goal_idx + 1] = route_interp

                # TODO: PASS IN ACTUAL DRIVING COMMAND BASED ON FUNCTION TYPE CALLED
                # mid_idx = int((route_index + goal_idx) / 2)
                # self.waypoint_planner.commands[route_index : mid_idx] = RoadOption.CHANGELANELEFT
                # self.waypoint_planner.commands[mid_idx : goal_idx] = RoadOption.CHANGELANERIGHT

                # Save current route modification start and end points
                self.cur_route_changes = (route_index, goal_idx)

        # Run longitudinal planner
        prediction_data = self.predict_collisions()

        st_planner_target_speed = 0.0
        st_planner_target_speed = self.long_planner.run_step(
            ego_route_points_3d=route_pts,
            ego_speed=ego_speed,
            ego_max_speed=speed_limit,
            actor_collisions=prediction_data.all_actor_collisions,
            plan_tick_counter=cur_tick,
            all_conditions=other_conditions
        )

        target_speed = st_planner_target_speed
        brake = target_speed < 1e-2

        if self.cur_route_changes is None:
            self.cur_route_changes = (route_index, min(route_pts.shape[0], route_index + 1))
        else:
            _, end_idx = self.cur_route_changes
            if route_index > end_idx:
                self.cur_route_changes = None
                action_complete = True

        return target_speed, brake, action_complete, route_pts[route_index:], route_wps[route_index:]

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
    ) -> float:
        action_complete = False
        target_speed = self.target_speed_initial

        for actor_id, condition in stop_conditions.items():
            print(f'stop_for actor_id: {actor_id}, type: {condition[1]}')
            if condition[1] == 'stop_sign':
                target_speed_ss, action_complete = self._process_stop_sign()
                target_speed = min(target_speed, target_speed_ss)
            elif condition[1] == 'traffic_light':
                target_speed_tl, action_complete = self._process_traffic_light()
                target_speed = min(target_speed, target_speed_tl)
            elif condition[1] == 'obstacle':
                # TODO: IMPLEMENT BEHAVIOUR
                action_complete = True

        return target_speed, action_complete

    def _get_target_speed(
        self,
        target_speed_initial : float,
    ) -> float:
        ego_speed = self.scene_data.ego_data.speed

        print(f'Ego Speed: {ego_speed}\n')
        print(f'Target Speed Initial: {target_speed_initial}\n')

        target_speeds = [target_speed_initial]
        leading_vehicles : Dict[str, List[LaneVehicleData]] = self.scene_data.vehicle_data['leading']
        if leading_vehicles:
            lane_vehicle_data : List[LaneVehicleData] = []
            lane_vehicle_data = leading_vehicles['ego']
            lane_leading_vehicles : List[VehicleData] = []

            # Check for leading vehicles in the ego lane
            for lane_veh_data in lane_vehicle_data:
                lane_leading_vehicles.extend(lane_veh_data.vehicle_data)

            for lv in lane_leading_vehicles:
                lv_speed = lv.speed
                lv_length = lv.vehicle.bounding_box.extent.x * 2

                dist_to_lv = lv.relative_distance

                desired_following_distance = self.config.idm_leading_vehicle_minimum_distance
                desired_time_headway = self.config.idm_leading_vehicle_time_headway

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
        target_speed_veh = self._get_target_speed(target_speed_initial)
        target_speed = min(target_speed_initial, target_speed_veh)
        start_index = self.waypoint_planner.route_index
        return target_speed, self.waypoint_planner.route_points[start_index:], self.waypoint_planner.route_waypoints[start_index:]

    # def _brake(self, param = None):
    #     target_speed = self.__get_target_speed(0.0, LongitudinalParams())
    #     start_index = self.waypoint_planner.route_index
    #     return target_speed, self.waypoint_planner.route_points[start_index:], self.waypoint_planner.route_waypoints[start_index:]

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
            self.tl_cleared = True
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

        return target_speed, self.tl_cleared

    # def _process_ped(
    #     self,
    #     ped_id : int
    # ) -> Tuple[float, bool]:
    #     ego_speed = self.scene_data.ego_data.speed
    #     target_speed_initial = self.scene_data.traffic_data.speed_limit

    #     peds = self.scene_data.ped_data
    #     if not peds:
    #         return target_speed_initial, True

    #     # Check if target pedestrian exists in actors list
    #     if ped_id in self.all_actors["peds"]:
    #         ped_data = self.all_actors["peds"][ped_id]

    #     # Otherwise find the nearest closest pedestrian
    #     else:
    #         ped_data = None
    #         for p_data in self.all_actors["peds"].values():
    #             if p_data.relative_distance < self.config.detection_radius:
    #                 if ped_data is None:
    #                     ped_data = p_data
    #                 else:
    #                     ped_data = ped_data if ped_data.relative_distance < p_data.relative_distance else p_data

    #     if ped_data is None:
    #         return target_speed_initial, True

    #     target_speed = self.idm.compute_target_speed_idm(
    #         desired_speed=target_speed_initial,
    #         leading_actor_length=0.5 + self.scene_data.ego_data.ego_vehicle.bounding_box.extent.x,
    #         ego_speed=ego_speed,
    #         leading_actor_speed=0.0,
    #         distance_to_leading_actor=ped_data.relative_distance,
    #         s0=self.config.idm_pedestrian_minimum_distance,
    #         T=self.config.idm_pedestrian_desired_time_headway,
    #     )

    #     # Check for collisions
    #     forecasted_ego_bbs = self.motion_forecaster.predict_ego_motion(
    #         ego_vehicle=self.ego_vehicle,
    #         ego_route_pts=self.waypoint_planner.route_points[self.waypoint_planner.route_index:],
    #         target_speed=target_speed_initial
    #     )
    #     forecasted_ped_bbs = self.motion_forecaster.predict_ped_motion([ped_data])

    #     ped_bbs = forecasted_ped_bbs[ped_data.id]
    #     has_collision, bb_a, bb_b = self.motion_forecaster.check_collision_point(forecasted_ego_bbs, ped_bbs)
    #     print(f'Ped has collision: {has_collision}')

    #     world = self.ego_vehicle.get_world()
    #     # for bb_ped, bb_ego in zip(ped_bbs, forecasted_ego_bbs):
    #     #     world.debug.draw_box(
    #     #         box=bb_ped,
    #     #         rotation=bb_ped.rotation,
    #     #         thickness=0.1,
    #     #         color=self.config.other_vehicles_forecasted_bbs_color,
    #     #         life_time=self.config.draw_life_time
    #     #     )
    #     #     world.debug.draw_box(
    #     #         box=bb_ego,
    #     #         rotation=bb_ego.rotation,
    #     #         thickness=0.1,
    #     #         color=self.config.ego_vehicle_forecasted_bbs_normal_color,
    #     #         life_time=self.config.draw_life_time
    #     #     )

    #     if has_collision:
    #         world.debug.draw_box(
    #             box=bb_a,
    #             rotation=bb_a.rotation,
    #             thickness=0.1,
    #             color=self.config.ego_vehicle_forecasted_bbs_hazard_color,
    #             life_time=self.config.draw_life_time
    #         )
    #         world.debug.draw_box(
    #             box=bb_b,
    #             rotation=bb_b.rotation,
    #             thickness=0.1,
    #             color=self.config.trailing_vehicle_color,
    #             life_time=self.config.draw_life_time
    #         )

    #     self.ped_cleared = (not has_collision) and (ped_data.relative_distance > 5.0)

    #     return target_speed, self.ped_cleared

    # def _process_vehicle(
    #     self,
    #     veh_id : int
    # ) -> Tuple[float, bool]:
    #     ego_speed = self.scene_data.ego_data.speed
    #     target_speed_initial = self.scene_data.traffic_data.speed_limit

    #     vehicles_dict = self.scene_data.vehicle_data
    #     if not vehicles_dict:
    #         return target_speed_initial, True

    #     # Find target vehicle in key actors list
    #     veh_traffic_type : str = None
    #     for key_actor in self.key_actors:
    #         if key_actor.id == veh_id and key_actor.obj_type == "vehicle":
    #             veh_traffic_type = key_actor.traffic_type

    #     if not veh_traffic_type:
    #         return target_speed_initial, True

    #     veh_data = None
    #     for actor_key in self.all_actors.keys():
    #         if actor_key == "peds":
    #             continue

    #         # Check if target vehicle exists in actors dict
    #         if veh_id in self.all_actors[actor_key]:
    #             veh_data = self.all_actors[actor_key][veh_id]
    #             break

    #     if veh_data is None:
    #         return target_speed_initial, True

    #     if veh_traffic_type in ["oncoming", "cross"]:
    #         # Check for collisions
    #         forecasted_ego_bbs = self.motion_forecaster.predict_ego_motion(
    #             ego_vehicle=self.ego_vehicle,
    #             ego_route_pts=self.waypoint_planner.route_points[self.waypoint_planner.route_index:],
    #             target_speed=veh_data.speed * 1.2
    #         )
    #         forecasted_veh_bbs = self.motion_forecaster.predict_vehicle_motion(self.scene_data.vehicle_data)
    #         veh_bbs = forecasted_veh_bbs[veh_data.id]

    #         has_collision, bb_ego, bb_veh = self.motion_forecaster.check_collision_point(forecasted_ego_bbs, veh_bbs)

    #         if has_collision:
    #             print(f'EGO COLLIDES WITH VEHICLE: {veh_data.id}')
    #             world = self.ego_vehicle.get_world()
    #             world.debug.draw_box(
    #                 box=bb_ego,
    #                 rotation=bb_ego.rotation,
    #                 thickness=0.1,
    #                 color=self.config.ego_vehicle_forecasted_bbs_hazard_color,
    #                 life_time=self.config.draw_life_time * 2
    #             )
    #             world.debug.draw_box(
    #                 box=bb_veh,
    #                 rotation=bb_veh.rotation,
    #                 thickness=0.1,
    #                 color=self.config.trailing_vehicle_color,
    #                 life_time=self.config.draw_life_time * 2
    #             )

    #         elif not has_collision and veh_data.relative_distance > 5.0:
    #             print(f'NO COLLISION WITH VEHICLE: {veh_data.id}')

    #         self.veh_cleared = (not has_collision) and veh_data.relative_distance > 5.0
    #         target_speed = ego_speed if self.veh_cleared else 0.0

    #     else:
    #         # target_speed = veh_data.speed * 0.8
    #         target_speed = target_speed_initial

    #     # if veh_traffic_type in ["oncoming", "cross"]:
    #     #     # Check for collisions
    #     #     forecasted_ego_bbs = self.motion_forecaster.predict_ego_motion(
    #     #         ego_vehicle=self.ego_vehicle,
    #     #         ego_route_pts=self.waypoint_planner.route_points[self.waypoint_planner.route_index:],
    #     #         target_speed=target_speed_initial
    #     #     )
    #     #     forecasted_veh_bbs = self.motion_forecaster.predict_vehicle_motion(self.scene_data.vehicle_data)
    #     #     veh_bbs = forecasted_veh_bbs[veh_data.id]

    #     #     has_collision, bb_a, bb_b = self.motion_forecaster.check_collision_point(forecasted_ego_bbs, veh_bbs)
    #     #     if not has_collision and veh_data.relative_distance > 5.0:
    #     #         self.veh_cleared = True
    #     #     else:
    #     #         self.veh_cleared = False

    #     return target_speed, self.veh_cleared

    # def _yield_for(
    #     self,
    #     target_obj : ConditionalParams
    # ) -> Tuple:
    #     move_to_next_state = True

    #     if target_obj.target == "ped":
    #         target_speed, move_to_next_state = self._process_ped(target_obj.id)
    #     elif target_obj.target == "vehicle":
    #         target_speed, move_to_next_state = self._process_vehicle(target_obj.id)
    #     else:
    #         target_speed = self.scene_data.traffic_data.speed_limit

    #     target_speed_lead = self.__get_target_speed(self.scene_data.traffic_data.speed_limit, LongitudinalParams())
    #     target_speed = min(target_speed, target_speed_lead)

    #     brake = target_speed < 1e-2

    #     route_index = self.waypoint_planner.route_index
    #     route_pts = self.waypoint_planner.route_points
    #     route_wps = self.waypoint_planner.route_waypoints

    #     return target_speed, brake, move_to_next_state, route_pts[route_index:], route_wps[route_index:]
