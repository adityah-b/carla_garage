"""State machine helpers for high-level trajectory planner commands."""

import carla
import numpy as np

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Set, Tuple, Dict, Union, Literal
from collections import deque

from config import GlobalConfig
from agents.navigation.local_planner import RoadOption
from privileged_route_planner import PlannerState
from team_code.scene_descriptor.scene_descriptor import SceneData
from team_code.actor_prediction.motion_prediction import PredictionData
from team_code.actor_prediction.collision_checker import LaneOverlapInterval
from team_code.scene_analyzer.parsers.ego_plan_pydantic_models import ConditionAction, ConditionCommand, PlanState, EntityType
from team_code.local_planner.lateral.lat_planner import LatPlannerResult

from .command_status import *

class BaseSM:
    """Base class for all state machines."""
    def __init__(self):
        self.phase = ActionPhase.IDLE

    def reset(self) -> None:
        """Return to the IDLE state and clear any derived-machine bookkeeping."""
        self.phase = ActionPhase.IDLE

    @property
    def is_idle(self) -> bool:
        return self.phase is ActionPhase.IDLE

    @property
    def is_executing(self) -> bool:
        return self.phase is ActionPhase.EXECUTING

    @property
    def is_cleared(self) -> bool:
        return self.phase is ActionPhase.CLEARED

    @property
    def is_failed(self) -> bool:
        return self.phase is ActionPhase.FAILED

class StopForSM(BaseSM):
    def __init__(self, condition : ConditionCommand):
        super().__init__()
        self.condition = condition
        self.wait_ticks = 0

        # Traffic light initial state
        self.tl_initial_state : str = ""

    def update_condition(
        self,
        new_condition : ConditionCommand
    ):
        self.reset()
        self.wait_ticks = 0
        self.tl_initial_state = ""
        self.condition = new_condition

    def step(
        self,
        *,
        config,
        scene_data : SceneData,
        planner_state : PlannerState,
    ) -> ConditionStatus:
        actor_type = self.condition.target.actor_type

        if actor_type == EntityType.STOP_SIGN:
            is_blocking, reason = self._step_stop_sign(config=config, scene_data=scene_data, planner_state=planner_state)
        elif actor_type == EntityType.STOP_LIGHT:
            is_blocking, reason = self._step_traffic_light(config=config, scene_data=scene_data, planner_state=planner_state)
        elif actor_type == EntityType.OBSTACLE:
            is_blocking, reason = self._step_obstacle(config=config, scene_data=scene_data)
        elif actor_type == EntityType.PEDESTRIAN:
            is_blocking, reason = self._step_pedestrian(config=config, scene_data=scene_data)
        else:
            return None

        return ConditionStatus(
            condition=self.condition,
            phase=self.phase,
            is_blocking=is_blocking,
            reason=reason,
            needs_replan=self.phase == ActionPhase.FAILED,
        )

    def _step_stop_sign(
        self,
        *,
        config,
        scene_data : SceneData,
        planner_state : PlannerState,
    ) -> Tuple[bool, str]:
        ego_speed = scene_data.ego_data.speed

        # Get the nearest stop sign
        next_ss = scene_data.traffic_data.next_stop_sign

        if next_ss is None:
            self.phase = ActionPhase.FAILED
            return False, "No stop sign detected"

        if next_ss.id in planner_state.cleared_stop_sign_ids:
            self.phase = ActionPhase.FAILED
            return False, "Stop sign already cleared. Proceed to next command"

        reason = ""
        # Stopped and waiting at the stop sign
        if (
            ego_speed < config.stopped_speed_threshold
            and next_ss.distance_to_stop_sign < config.clearing_distance_to_stop_sign
        ):
            self.phase = ActionPhase.EXECUTING
            self.wait_ticks += 1
            reason = f"Waiting at stop sign {next_ss.id}"
        else:
            self.phase = ActionPhase.EXECUTING
            self.wait_ticks = 0
            reason = f"Approaching stop sign {next_ss.id}"

        if self.phase == ActionPhase.EXECUTING and self.wait_ticks >= config.stop_sign_min_wait_ticks:
            self.phase = ActionPhase.CLEARED
            return False, "Finished waiting at stop sign. Proceed to next command"

        return True, reason

    def _step_traffic_light(
        self,
        *,
        config,
        scene_data : SceneData,
        planner_state : PlannerState,
    ) -> Tuple[bool, str]:
        ego_speed = scene_data.ego_data.speed

        # Get the nearest traffic light
        next_tl = scene_data.traffic_data.next_traffic_light

        if next_tl is None:
            self.phase = ActionPhase.FAILED
            return False, "No traffic light detected"

        # NOTE: Not tracking cleared traffic lights, may not be necessary

        # Set initial traffic light state if uninitialized
        if not self.tl_initial_state:
            self.tl_initial_state = next_tl.state

        reason = ""

        # Set condition state to executing
        self.phase = ActionPhase.EXECUTING

        # Traffic light is GREEN
        if next_tl.state == "GREEN":
            self.phase = ActionPhase.CLEARED
            if self.tl_initial_state and self.tl_initial_state == "RED":
                return False, f"Traffic light {next_tl.id} turned green"
            else:
                return False, f"Traffic light {next_tl.id} is green"

        # Traffic light is RED
        if ego_speed > config.stopped_speed_threshold:
            reason = f"Approaching red traffic light {next_tl.id}"
        else:
            reason = f"Waiting at red traffic light {next_tl.id}"

        return True, reason

    def _step_obstacle(
        self,
        *,
        config,
        scene_data : SceneData,
    ) -> Tuple[bool, str]:
        ego_speed = scene_data.ego_data.speed
        ego_obstacles = scene_data.obstacle_data.ego_obstacles

        lane_info = scene_data.route_data.lane_info if scene_data.route_data else None
        can_change_lane = lane_info is not None and lane_info.same_direction_lane_change_available

        # No obstacle in scene
        if not ego_obstacles:
            if not can_change_lane:
                # No same-direction lane available: obstacle must have cleared itself
                self.phase = ActionPhase.CLEARED
                return False, "Obstacle no longer present, maneuver complete"
            self.phase = ActionPhase.FAILED
            return False, "No obstacle detected"

        next_obstacle = ego_obstacles[0]

        reason = ""
        # Stopped and waiting near obstacle
        if (
            ego_speed < config.stopped_speed_threshold
            and next_obstacle.relative_distance < config.clearing_distance_to_obstacle
        ):
            self.phase = ActionPhase.EXECUTING
            self.wait_ticks += 1
            reason = f"Waiting near obstacle {next_obstacle.id}"
        else:
            self.phase = ActionPhase.EXECUTING
            self.wait_ticks = 0
            reason = f"Approaching obstacle {next_obstacle.id}"

        if self.phase == ActionPhase.EXECUTING and self.wait_ticks >= config.obstacle_min_wait_ticks:
            if next_obstacle.is_near_junction and not can_change_lane:
                # At intersection with no lane change: timeout does not clear, wait for obstacle to leave
                return True, f"Waiting near obstacle {next_obstacle.id}, no lane change available at junction"
            self.phase = ActionPhase.CLEARED
            return False, f"Waited for {self.wait_ticks} ticks near obstacle {next_obstacle.id}, obstacle is blocking route"

        return True, reason

    def _step_pedestrian(
        self,
        *,
        config,
        scene_data : SceneData,
    ) -> Tuple[bool, str]:
        ego_speed = scene_data.ego_data.speed

        # Get the nearest pedestrian
        if not scene_data.ped_data:
            self.phase = ActionPhase.FAILED
            return False, f"No pedestrian detected"

        next_ped = scene_data.ped_data[0]

        # NOTE: Not tracking cleared pedestrians, may be important

        reason = ""
        # Stopped and waiting near pedestrian
        if (
            ego_speed < config.stopped_speed_threshold
            and next_ped.relative_distance < config.clearing_distance_to_stop_sign
        ):
            self.phase = ActionPhase.EXECUTING
            self.wait_ticks += 1
            reason = f"Stopped and waiting for pedestrian {next_ped.id}"
        else:
            self.phase = ActionPhase.EXECUTING
            self.wait_ticks = 0
            reason = f"Approaching pedestrian {next_ped.id}"

        if self.phase == ActionPhase.EXECUTING and self.wait_ticks >= config.ped_min_wait_ticks:
            self.phase = ActionPhase.CLEARED
            return False, f"Waited for {self.wait_ticks} ticks near pedestrian {next_ped.id}"

        return True, reason

class BaseActionSM(BaseSM):
    """Base class for simple action state machines."""

    # Condition key: (condition_action.value, actor_type)
    ConditionKey = Tuple[str, str]

    def __init__(self, config : GlobalConfig):
        super().__init__()
        self.config = config

        self.conditions_registry : Dict[BaseActionSM.ConditionKey, StopForSM] = {}
        self.cleared_condition_keys : Set[BaseActionSM.ConditionKey] = set()
        self.collision_events : deque = deque(maxlen=self.config.collision_data_max_entries)

    def reset(self) -> None:
        """Return to the IDLE state and clear any derived-machine bookkeeping."""
        super().reset()

        self.conditions_registry.clear()
        self.cleared_condition_keys.clear()

        self.collision_events.clear()

    @staticmethod
    def _cond_key(cond: ConditionCommand) -> 'BaseActionSM.ConditionKey':
        return (cond.condition_action.value, cond.target.actor_type.value)

    def update_conditions(self, plan_state: PlanState) -> None:
        """
        Register StopForSM state machines for STOP_FOR conditions only.

        All other condition types (YIELD_FOR, WATCH_FOR, etc.) are treated as
        costmap/speed modifiers and are handled inline by subclass implementations
        rather than via persistent state machines.
        """
        def is_same_cond(a: ConditionCommand, b: ConditionCommand) -> bool:
            return (
                a.condition_action == b.condition_action and
                a.target.actor_type  == b.target.actor_type and
                a.target.traffic_type == b.target.traffic_type and
                a.priority           == b.priority
            )

        for cond in plan_state.plan.conditions:
            if cond.condition_action != ConditionAction.STOP_FOR:
                continue
            key = self._cond_key(cond)
            if key in self.cleared_condition_keys:
                continue
            if key not in self.conditions_registry:
                self.conditions_registry[key] = StopForSM(condition=cond)
            else:
                cond_sm = self.conditions_registry[key]
                if not is_same_cond(cond_sm.condition, cond):
                    cond_sm.update_condition(cond)

    def _step_conditions(
        self,
        scene_data: SceneData,
        planner_state: PlannerState,
    ) -> Tuple[List[ConditionStatus], List[str], List[str], bool]:
        """Step all registered StopFor SMs and collect results.

        Returns (condition_statuses, reasons, replan_reasons, has_blocking).
        replan_reasons is populated whenever a condition has needs_replan set,
        regardless of the condition's current phase.
        """
        condition_statuses: List[ConditionStatus] = []
        reasons: List[str] = []
        replan_reasons: List[str] = []
        has_blocking = False

        for cond_key, cond_sm in list(self.conditions_registry.items()):
            cond_status: ConditionStatus = cond_sm.step(
                config=self.config,
                scene_data=scene_data,
                planner_state=planner_state,
            )
            if cond_status is None:
                continue

            has_blocking = has_blocking or cond_status.is_blocking
            condition_statuses.append(cond_status)

            if cond_status.phase == ActionPhase.CLEARED:
                self.cleared_condition_keys.add(cond_key)
                del self.conditions_registry[cond_key]

            if cond_status.needs_replan:
                replan_reasons.append(cond_status.reason)

            reasons.append(cond_status.reason)

        if scene_data.collision_data:
            self.collision_events.append(scene_data.collision_data[0])

        return condition_statuses, reasons, replan_reasons, has_blocking

# ---------------------------------------------------------------------------
# Route Following
# ---------------------------------------------------------------------------

class FollowRouteSM(BaseActionSM):
    """Tracks progress while following the route without interruptions."""
    def __init__(self, config : GlobalConfig):
        super().__init__(config)
        self.clear_ticks = 0
        # TODO: PUT THIS AS CONFIG CONSTANT
        self.completion_threshold = 25

    def reset(self) -> None:
        super().reset()
        self.clear_ticks = 0

    def step(self, has_blocking_conditions: bool) -> Tuple[bool, str]:
        """
        :param has_blocking_conditions: True if current plan has any stop/yield/etc. conditions.
        :returns: (completed, reason)
        """
        # Set state to executing if idle
        if self.is_idle:
            self.phase = ActionPhase.EXECUTING

        if has_blocking_conditions:
            # Any blocking condition means we are no longer in a clean, uninterrupted follow.
            return False, "Command follow_route waiting on conditions to complete"

        self.clear_ticks += 1

        if self.clear_ticks > self.completion_threshold:
            # One-shot completion; caller can optionally reset afterwards.
            self.phase = ActionPhase.CLEARED
            self.clear_ticks = 0
            return True, f"Ran follow_route for {self.completion_threshold} ticks"

        return False, "Running follow_route"

    def update_state(
        self,
        *,
        scene_data : SceneData,
        planner_state : PlannerState,
    ) -> CommandStatus:
        condition_statuses, command_reasons, replan_reasons, has_blocking_conditions = \
            self._step_conditions(scene_data, planner_state)

        if scene_data.collision_data:
            print(f'\n\nFOLLOW ROUTE SM ADDING COLLISION EVENT\n\n')

        if replan_reasons:
            self.phase = ActionPhase.FAILED
            return CommandStatus(
                cur_cmd=Action.FOLLOW_ROUTE,
                phase=self.phase,
                is_blocked=False,
                conditions_status=condition_statuses,
                reasons=replan_reasons,
            )

        completed, base_reason = self.step(has_blocking_conditions)
        command_reasons.append(base_reason)

        return CommandStatus(
            cur_cmd=Action.FOLLOW_ROUTE,
            phase=self.phase,
            is_blocked=has_blocking_conditions,
            conditions_status=condition_statuses,
            reasons=command_reasons,
        )

# ---------------------------------------------------------------------------
# Intersection / Turn Handling
# ---------------------------------------------------------------------------

class TurnSM(BaseActionSM):
    """Tracks intersection traversal for turn actions."""

    # Used only when converting route_cmd (RoadOption) from route data to Action
    _RO_TO_ACTION: Dict[RoadOption, 'Action'] = {
        RoadOption.LEFT:     Action.TURN_LEFT,
        RoadOption.RIGHT:    Action.TURN_RIGHT,
        RoadOption.STRAIGHT: Action.TURN_STRAIGHT,
    }

    def __init__(self, config : GlobalConfig):
        super().__init__(config)

        self.cur_plan_idxs : Optional[Tuple[int, int]] = None # (start_idx, end_idx)
        self.cur_turn_cmd  : Optional[Action] = None

    def reset(self) -> None:
        super().reset()

        self.cur_plan_idxs = None
        self.cur_turn_cmd = None

    def activate(self) -> None:
        self.phase = ActionPhase.EXECUTING

    @property
    def has_active_plan(self) -> bool:
        return self.is_executing and self.cur_plan_idxs is not None and self.cur_turn_cmd is not None

    def step(self, turn_cmd: Action, route_index: int, scene_data: SceneData) -> Tuple[bool, str]:
        """
        :param route_index: Current index along ego's route.
        :returns: (completed, reason)
        """
        cmd_name = turn_cmd.value

        if not self.is_executing:
            self.phase = ActionPhase.FAILED
            return True, f"Failed to activate {cmd_name}"

        if not self.has_active_plan:
            self.phase = ActionPhase.FAILED
            return True, f"Cannot run {cmd_name}. No turns in upcoming route, reevaluate"

        if self.cur_turn_cmd != turn_cmd:
            self.phase = ActionPhase.FAILED
            return True, f"Mismatched turn commands, called {cmd_name} when route requires {self.cur_turn_cmd.value} maneuver"

        _, end_idx = self.cur_plan_idxs
        if route_index >= end_idx:
            self.phase = ActionPhase.CLEARED
            return True, f"Completed {cmd_name} maneuver and cleared intersection segment"

        return False, f"Running {cmd_name}"

    def update_state(
        self,
        *,
        scene_data    : SceneData,
        planner_state : PlannerState,
        turn_cmd      : Action,
    ) -> CommandStatus:
        condition_statuses, command_reasons, replan_reasons, has_blocking_conditions = \
            self._step_conditions(scene_data, planner_state)

        if replan_reasons:
            self.phase = ActionPhase.FAILED
            return CommandStatus(
                cur_cmd=turn_cmd,
                phase=self.phase,
                is_blocked=False,
                conditions_status=condition_statuses,
                reasons=replan_reasons,
            )

        turn_data = scene_data.route_data.intersection_data
        if not self.has_active_plan and turn_data is not None:
            route_cmd, turn_start_idx, turn_end_idx, _ = turn_data.entry
            turn_buffer = self.config.meters_to_dense_route_idx(self.config.long_planning_turn_buffer_m)
            turn_end_idx = min(planner_state.route_len - 1, turn_end_idx + turn_buffer)
            self.cur_plan_idxs = (turn_start_idx, turn_end_idx)
            self.cur_turn_cmd = self._RO_TO_ACTION[route_cmd]

        completed, base_reason = self.step(turn_cmd, planner_state.route_index, scene_data)
        command_reasons.append(base_reason)
        has_blocking_conditions = False if completed else has_blocking_conditions

        return CommandStatus(
            cur_cmd=turn_cmd,
            phase=self.phase,
            is_blocked=has_blocking_conditions,
            conditions_status=condition_statuses,
            reasons=command_reasons,
        )
# ---------------------------------------------------------------------------
# Lane Change Handling
# ---------------------------------------------------------------------------

class LaneChangeSM(BaseActionSM):
    """Tracks lane change progression and planning horizons."""

    # Used only when converting route_cmd (RoadOption) from route data to Action
    _RO_TO_ACTION: Dict[RoadOption, 'Action'] = {
        RoadOption.CHANGELANELEFT:  Action.CHANGE_LANE_LEFT,
        RoadOption.CHANGELANERIGHT: Action.CHANGE_LANE_RIGHT,
    }

    def __init__(self, config : GlobalConfig):
        super().__init__(config)

        self.cur_plan_idxs : Optional[Tuple[int, int]] = None # (start_idx, end_idx)
        self.cur_lc_cmd    : Optional[Action] = None
        self.route_shifted : bool = False

    def reset(self) -> None:
        super().reset()

        self.cur_plan_idxs = None
        self.cur_lc_cmd = None
        self.route_shifted = False

    def activate(self) -> None:
        self.phase = ActionPhase.EXECUTING

    @property
    def has_active_plan(self) -> bool:
        return self.is_executing and self.cur_plan_idxs is not None and self.cur_lc_cmd is not None

    def step(self, lc_cmd: Action, route_index: int, scene_data: SceneData) -> Tuple[bool, str]:
        cmd_name = lc_cmd.value

        if not self.is_executing:
            self.phase = ActionPhase.FAILED
            return True, f"Failed to activate {cmd_name}"

        if not self.has_active_plan:
            self.phase = ActionPhase.FAILED
            return True, f"Cannot run {cmd_name}. No lane changes in upcoming route, reevaluate"

        if self.cur_lc_cmd != lc_cmd:
            self.phase = ActionPhase.FAILED
            return True, f"Mismatched lane change commands, called {cmd_name} when route requires {self.cur_lc_cmd.value} maneuver"

        _, end_idx = self.cur_plan_idxs
        if route_index >= end_idx:
            self.phase = ActionPhase.CLEARED
            return True, f"Completed {cmd_name} maneuver and cleared lane change segment"

        return False, f"Running {cmd_name}"

    def update_state(
        self,
        *,
        scene_data    : SceneData,
        planner_state : PlannerState,
        lc_cmd        : Action,
    ) -> CommandStatus:
        condition_statuses, command_reasons, replan_reasons, has_blocking_conditions = \
            self._step_conditions(scene_data, planner_state)

        if replan_reasons:
            self.phase = ActionPhase.FAILED
            return CommandStatus(
                cur_cmd=lc_cmd,
                phase=self.phase,
                is_blocked=False,
                conditions_status=condition_statuses,
                reasons=replan_reasons,
            )

        lane_change_data = scene_data.route_data.lane_change_data
        if not self.has_active_plan and lane_change_data is not None:
            route_cmd, lc_start_idx, lc_end_idx = lane_change_data.entry
            self.cur_plan_idxs = (lc_start_idx, lc_end_idx)
            self.cur_lc_cmd = self._RO_TO_ACTION[route_cmd]

        completed, base_reason = self.step(lc_cmd, planner_state.route_index, scene_data)
        command_reasons.append(base_reason)
        has_blocking_conditions = False if completed else has_blocking_conditions

        return CommandStatus(
            cur_cmd=lc_cmd,
            phase=self.phase,
            is_blocked=has_blocking_conditions,
            conditions_status=condition_statuses,
            reasons=command_reasons,
        )

# ---------------------------------------------------------------------------
# Overtake Handling
# ---------------------------------------------------------------------------

@dataclass
class OvertakeSegments:
    """Route-index boundaries for each sub-phase of a lane-change overtake."""
    lc_out_start  : int   # route_index at plan time (current ego position)
    lc_out_end    : int   # outbound LC complete — ego is fully in the target lane
    follow_end    : int   # IDM-follow phase ends, return LC begins
    lc_return_end : int   # return LC complete — ego is back in the source lane


class OvertakeSM(BaseActionSM):
    """
    Tracks overtake maneuvers.

    Two overtake types are supported:
      LANE_CHANGE — overtake via an adjacent parallel driving lane.
                    Sub-phases: LC_OUTBOUND → IN_TARGET_LANE → LC_RETURN
      ONCOMING    — overtake by temporarily invading the oncoming lane.
                    Sub-phases: LC_OUTBOUND (executing) only; completion when
                    route_index reaches maneuver_end_idx.
    """
    def __init__(self, config : GlobalConfig):
        super().__init__(config)

        self.overtake_type  : Optional[OvertakeType]    = None
        self.sub_phase      : OvertakeSubPhase           = OvertakeSubPhase.PENDING_PLAN

        # Overtake segment boundaries
        self.segments       : Optional[OvertakeSegments] = None

        # Full maneuver end index
        self.maneuver_end_idx : Optional[int]            = None

        self.lat_plan_status : LateralPlanStatus = LateralPlanStatus.NONE
        self.plan_fail_count : int = 0
        self.max_plan_failures : int = 5  # TODO: move to config

    def reset(self) -> None:
        super().reset()

        self.overtake_type    = None
        self.sub_phase        = OvertakeSubPhase.PENDING_PLAN
        self.segments         = None
        self.maneuver_end_idx = None

        self.lat_plan_status  = LateralPlanStatus.NONE
        self.plan_fail_count  = 0

    def activate(self) -> None:
        self.phase = ActionPhase.EXECUTING

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_active_plan(self) -> bool:
        """
        True when the ST planner should run with explicit s-bounds.
        For LANE_CHANGE: only during LC_OUTBOUND and LC_RETURN.
        For ONCOMING:    whenever we have a valid segment (sub_phase != PENDING_PLAN).
        IN_TARGET_LANE uses IDM only — no ST bounds needed.
        """
        if not self.is_executing or self.segments is None:
            return False
        if self.overtake_type == OvertakeType.ONCOMING:
            return self.sub_phase != OvertakeSubPhase.PENDING_PLAN
        return self.sub_phase in (OvertakeSubPhase.LC_OUTBOUND, OvertakeSubPhase.LC_RETURN)

    # ------------------------------------------------------------------
    # Replanning gate
    # ------------------------------------------------------------------

    def compute_replan_need(
        self,
        scene_data    : SceneData,
        prediction_data : PredictionData,
    ) -> bool:
        """
        Returns True if lat_planner.run_step should be called this tick.

        Triggers:
          1. No lateral plan has been generated yet.
          2. A static obstacle is currently intruding the route.
          3. A cyclist is currently intruding the route.
        """
        print(f'\n\nOVERTAKE REPLANNING')
        # 1. No plan yet
        if self.lat_plan_status == LateralPlanStatus.NONE:
            print(f'\tNO LAT PLAN')
            return True

        # 2. Static obstacle intruding the route
        if scene_data.obstacle_data and scene_data.obstacle_data.ego_obstacles:
            print(f'\tHAS EGO OBSTACLES')
            return True

        # 3. Cyclist intruding the route
        cyclists = scene_data.vehicle_data.get(vehicle_types={'cyclist'}, get_intruders=True)
        if cyclists:
            print(f'\tHAS CYCLISTS')
            return True

        return False

    # ------------------------------------------------------------------
    # Helpers (computed from data already available in update_state)
    # ------------------------------------------------------------------

    def _determine_overtake_type(
        self,
        scene_data : SceneData,
    ) -> OvertakeType:
        """
        Returns LANE_CHANGE if there is a proper parallel driving lane in the
        overtake direction, otherwise ONCOMING.
        """
        lane_info = scene_data.route_data.lane_info

        if lane_info.same_direction_lane_change_available:
            return OvertakeType.LANE_CHANGE

        return OvertakeType.ONCOMING

    def _extract_segments(
        self,
        planner_state   : PlannerState,
        lat_plan_result : LatPlannerResult,
    ) -> Optional[OvertakeSegments]:
        """
        Convert the local (route-slice-relative) segment indices computed by
        lat_planner into global route indices and return an OvertakeSegments.
        Returns None if the LatPlanner result contains no segment data.
        """
        if lat_plan_result.lc_out_end_local is None:
            return None

        route_index = planner_state.route_index
        route_len   = planner_state.route_len

        lc_out_end    = min(route_index + lat_plan_result.lc_out_end_local,    route_len - 1)
        follow_end    = min(route_index + lat_plan_result.follow_end_local,    route_len - 1)
        lc_return_end = min(route_index + lat_plan_result.lc_return_end_local, route_len - 1)

        # Enforce strict monotonic ordering
        lc_out_end    = max(route_index + 1, lc_out_end)
        follow_end    = max(lc_out_end  + 1, follow_end)
        lc_return_end = max(follow_end  + 1, lc_return_end)

        return OvertakeSegments(
            lc_out_start  = route_index,
            lc_out_end    = lc_out_end,
            follow_end    = follow_end,
            lc_return_end = lc_return_end,
        )

    def _apply_segments(self, segments: OvertakeSegments) -> None:
        self.segments         = segments
        self.maneuver_end_idx = segments.lc_return_end
        if self.sub_phase == OvertakeSubPhase.PENDING_PLAN:
            self.sub_phase = OvertakeSubPhase.LC_OUTBOUND

    def _apply_oncoming_end(self, route_index: int, end_idx: int) -> None:
        self.segments = OvertakeSegments(
            lc_out_start  = route_index,
            lc_out_end    = end_idx,
            follow_end    = end_idx,
            lc_return_end = end_idx,
        )
        self.maneuver_end_idx = end_idx
        if self.sub_phase == OvertakeSubPhase.PENDING_PLAN:
            self.sub_phase = OvertakeSubPhase.LC_OUTBOUND

    # ------------------------------------------------------------------
    # State transition
    # ------------------------------------------------------------------

    def step(self, overtake_cmd: Action, route_index: int, scene_data: SceneData) -> Tuple[bool, str]:
        cmd_name = overtake_cmd.value

        if not self.is_executing:
            self.phase = ActionPhase.FAILED
            return True, f"Failed to activate {cmd_name}"

        if self.lat_plan_status == LateralPlanStatus.INVALID_FATAL:
            self.phase = ActionPhase.FAILED
            return True, f"Failed to generate a feasible plan for {cmd_name}"

        lane_info = scene_data.route_data.lane_info
        can_overtake = (
            (lane_info.has_left_lane and overtake_cmd == Action.OVERTAKE_LEFT) or
            (lane_info.has_right_lane and overtake_cmd == Action.OVERTAKE_RIGHT)
        )
        if self.sub_phase == OvertakeSubPhase.PENDING_PLAN and not can_overtake:
            self.phase = ActionPhase.FAILED
            return True, f"Cannot run {cmd_name}, no lanes available"

        if self.sub_phase == OvertakeSubPhase.PENDING_PLAN:
            return False, f"Waiting for feasible {cmd_name} plan"

        if self.segments is None:
            return False, f"Waiting for segment data for {cmd_name}"

        # ── ONCOMING: single-phase execution ──────────────────────────
        if self.overtake_type == OvertakeType.ONCOMING:
            if route_index >= self.maneuver_end_idx:
                self.phase = ActionPhase.CLEARED
                return True, f"Completed {cmd_name} (oncoming invasion) maneuver"
            return False, f"Running {cmd_name} (oncoming), sub_phase={self.sub_phase.name}"

        # ── LANE_CHANGE: advance through sub-phases ───────────────────
        segs = self.segments

        if self.sub_phase == OvertakeSubPhase.LC_OUTBOUND:
            if route_index >= segs.lc_out_end:
                self.sub_phase = OvertakeSubPhase.IN_TARGET_LANE
                return False, f"{cmd_name}: outbound LC complete — now in target lane"

        if self.sub_phase == OvertakeSubPhase.IN_TARGET_LANE:
            if route_index >= segs.follow_end:
                self.sub_phase = OvertakeSubPhase.LC_RETURN
                return False, f"{cmd_name}: beginning return LC"

        if self.sub_phase == OvertakeSubPhase.LC_RETURN:
            if route_index >= segs.lc_return_end:
                self.phase = ActionPhase.CLEARED
                return True, f"Completed {cmd_name} maneuver — back in source lane"

        return False, f"Running {cmd_name}, sub_phase={self.sub_phase.name}"

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update_state(
        self,
        *,
        scene_data         : SceneData,
        planner_state      : PlannerState,
        overtake_cmd       : Action,
        lat_planner_result : Optional[LatPlannerResult] = None,
    ) -> CommandStatus:

        # ── 1. Latch overtake type once ───────────────────────────────
        if self.overtake_type is None:
            self.overtake_type = self._determine_overtake_type(scene_data)

        # ── 2. Step condition state machines ─────────────────────────
        condition_statuses, command_reasons, replan_reasons, has_blocking_conditions = \
            self._step_conditions(scene_data, planner_state)

        # ── 3. Early-exit: condition-driven replan ────────────────────
        if replan_reasons:
            self.phase = ActionPhase.FAILED
            return CommandStatus(
                cur_cmd=overtake_cmd,
                phase=self.phase,
                is_blocked=False,
                conditions_status=condition_statuses,
                reasons=replan_reasons,
            )

        # ── 4. Update lateral plan status ────────────────────────────
        if lat_planner_result is not None:
            if lat_planner_result.is_empty_plan:
                self.plan_fail_count += 1
                self.lat_plan_status = (
                    LateralPlanStatus.INVALID_FATAL
                    if self.plan_fail_count >= self.max_plan_failures
                    else LateralPlanStatus.INVALID_TRANSIENT
                )
            else:
                self.lat_plan_status = LateralPlanStatus.VALID
                self.plan_fail_count = 0

        # ── 5. Ingest new segment data from a fresh LatPlanner result ─
        if lat_planner_result is not None and lat_planner_result.is_new_plan:
            segs = self._extract_segments(planner_state, lat_planner_result)
            if self.overtake_type == OvertakeType.LANE_CHANGE:
                if segs is not None:
                    self._apply_segments(segs)
            else:
                target_idx = lat_planner_result.goal_idx
                if segs is not None:
                    target_idx = segs.lc_return_end
                self._apply_oncoming_end(
                    planner_state.route_index,
                    target_idx,
                )

        # ── 6. Advance the state machine ─────────────────────────────
        completed, base_reason = self.step(
            overtake_cmd, planner_state.route_index, scene_data
        )
        command_reasons.append(base_reason)

        has_blocking_conditions = False if completed else has_blocking_conditions

        return CommandStatus(
            cur_cmd=overtake_cmd,
            phase=self.phase,
            is_blocked=has_blocking_conditions,
            conditions_status=condition_statuses,
            reasons=command_reasons,
        )

# ---------------------------------------------------------------------------
# Pull-Over Handling
# ---------------------------------------------------------------------------

class PullOverPhase(Enum):
    IDLE = auto()
    YIELDING = auto()
    WAITING = auto()
    RETURNING = auto()

class PullOverSM(BaseActionSM):
    """
    Tracks pull-over maneuvers for emergency vehicle yielding.

    Two action types (read from PlanState):
      PULL_OVER_LEFT / PULL_OVER_RIGHT — shift route to shoulder, stop, merge back.
      PULL_OVER_IN_LANE               — stop in place, no lateral deviation.

    Three sub-phases:
      YIELDING  — lateral shift + decelerate  (or just decelerate for IN_LANE)
      WAITING   — hold at stop, monitor YIELD_FOR emergency_vehicle conditions inline
      RETURNING — follow pre-shifted return route (L/R) or skip to CLEARED (IN_LANE)

    STOP_FOR conditions are tracked via the inherited conditions_registry (StopForSM).
    YIELD_FOR conditions for regular traffic are costmap modifiers handled by the
    longitudinal planner; they do not gate sub-phase transitions.
    YIELD_FOR emergency_vehicle conditions are checked inline to gate WAITING→RETURNING.
    """

    def __init__(self, config: GlobalConfig):
        super().__init__(config)

        self.sub_phase   : PullOverPhase    = PullOverPhase.IDLE
        self.pull_action : Optional[Action] = None   # latched once on first activate

        # YIELD_FOR emergency_vehicle conditions extracted from plan (refreshed each tick)
        self.ev_yield_conditions : List[ConditionCommand] = []

        # For PULL_OVER_LEFT / PULL_OVER_RIGHT route tracking
        self.route_shifted     : bool          = False
        self.maneuver_wait_idx : Optional[int] = None  # global route idx where ego parks
        self.maneuver_end_idx  : Optional[int] = None  # global route idx where return ends

        self.preconditions_checked : bool = False

    def reset(self) -> None:
        super().reset()
        self.sub_phase           = PullOverPhase.IDLE
        self.pull_action         = None
        self.ev_yield_conditions = []
        self.route_shifted       = False
        self.maneuver_wait_idx   = None
        self.maneuver_end_idx    = None
        self.preconditions_checked = False # Reset the flag

    def activate(self) -> None:
        self.phase = ActionPhase.EXECUTING
        if self.sub_phase == PullOverPhase.IDLE:
            self.sub_phase = PullOverPhase.YIELDING

    def update_conditions(self, plan_state: PlanState) -> None:
        """
        Delegate STOP_FOR tracking to base class (StopForSM per actor).
        Refresh EV-specific YIELD_FOR list for inline WAITING checks.
        """
        super().update_conditions(plan_state)
        self.ev_yield_conditions = [
            c for c in plan_state.plan.conditions
            if c.condition_action == ConditionAction.YIELD_FOR
            and c.target.actor_type == "emergency_vehicle"
        ]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_active_plan(self) -> bool:
        return self.is_executing and self.route_shifted

    # ------------------------------------------------------------------
    # Replanning gate
    # ------------------------------------------------------------------

    def compute_replan_need(
        self,
        scene_data      : SceneData,
        prediction_data : PredictionData,
    ) -> bool:
        """
        True when shift_route_smoothly should be called this tick.

        PULL_OVER_IN_LANE: never (no lateral shift needed).
        PULL_OVER_L/R:     only until the route has been shifted once.
        """
        if self.pull_action == Action.PULL_OVER_IN_LANE:
            return False
        return not self.route_shifted

    # ------------------------------------------------------------------
    # Inline EV condition check (no state machine needed)
    # ------------------------------------------------------------------

    def _ev_condition_cleared(
        self,
        condition       : ConditionCommand,
        scene_data      : SceneData,
        prediction_data : PredictionData,
    ) -> bool:
        """
        Returns True when the emergency vehicle described by *condition* has passed.

        Resolves the abstract ConditionTarget against live perception:
        - If no EVs with active sirens exist in scene → condition cleared.
        - crossing  → no active-siren EV overlaps our route bounding boxes.
        - trailing / oncoming → no active-siren EV matches the target traffic_type.
        """
        evs = scene_data.vehicle_data.get(vehicle_types={'emergency_vehicle'})
        # Filter to only EVs with active sirens
        active_evs = [ev for ev in evs if ev.emergency_sirens_active] if evs else []

        if not active_evs:
            return True  # no active-siren EVs → treat as passed

        target_traffic_type = condition.target.traffic_type

        if target_traffic_type == "crossing":
            for ev in active_evs:
                if ev.id in prediction_data.all_actor_overlaps:
                    return False
            return True

        # For trailing/oncoming: check if any active-siren EV still has matching traffic_type
        for ev in active_evs:
            if ev.traffic_type == target_traffic_type:
                return False
        return True

    # ------------------------------------------------------------------
    # State transition
    # ------------------------------------------------------------------

    def step(
        self,
        pull_over_cmd : Action,
        route_index     : int,
        ego_speed       : float,
        scene_data      : SceneData,
        prediction_data : PredictionData,
    ) -> Tuple[bool, str]:
        cmd_name = pull_over_cmd.value

        if not self.is_executing:
            self.phase = ActionPhase.FAILED
            return True, "Failed to activate pull_over"

        if not self.preconditions_checked:
            emergency_vehicles = scene_data.vehicle_data.get(vehicle_types={"emergency_vehicle"})
            evs_siren = [ev for ev in emergency_vehicles if ev.emergency_sirens_active]

            if not evs_siren:
                self.phase = ActionPhase.FAILED
                return True, f"Cannot run {cmd_name}, no emergency vehicles present"

            lane_info = scene_data.route_data.lane_info
            can_pull_over = (
                (lane_info.left_same_dir and pull_over_cmd == Action.PULL_OVER_LEFT) or
                (lane_info.right_same_dir and pull_over_cmd == Action.PULL_OVER_RIGHT) or
                (pull_over_cmd == Action.PULL_OVER_IN_LANE)
            )
            if not can_pull_over:
                self.phase = ActionPhase.FAILED
                return True, f"Cannot run {cmd_name}, no lanes available"

            # Checks passed; latch the boolean so we don't evaluate this again
            self.preconditions_checked = True

        # ── YIELDING → WAITING ────────────────────────────────────────
        if self.sub_phase == PullOverPhase.YIELDING:
            if self.pull_action == Action.PULL_OVER_IN_LANE:
                if ego_speed < 0.5:
                    self.sub_phase = PullOverPhase.WAITING
                    return False, "pull_over_in_lane: stopped, waiting for EV to pass"
            else:
                if self.route_shifted and self.maneuver_wait_idx is not None:
                    if route_index >= self.maneuver_wait_idx:
                        self.sub_phase = PullOverPhase.WAITING
                        return False, "pull_over: at shoulder, waiting for EV to pass"

        # ── WAITING → RETURNING ───────────────────────────────────────
        if self.sub_phase == PullOverPhase.WAITING:
            if self.ev_yield_conditions:
                all_cleared = all(
                    self._ev_condition_cleared(c, scene_data, prediction_data)
                    for c in self.ev_yield_conditions
                )
            else:
                # FIX: If there are no EV conditions dictating we should wait, the coast is clear.
                all_cleared = True

            if all_cleared:
                if self.pull_action == Action.PULL_OVER_IN_LANE:
                    self.phase = ActionPhase.CLEARED
                    return True, "pull_over_in_lane: EV passed, command complete"
                self.sub_phase = PullOverPhase.RETURNING
                return False, "pull_over: EV passed, merging back to lane"

        # ── RETURNING → CLEARED ───────────────────────────────────────
        if self.sub_phase == PullOverPhase.RETURNING:
            if self.maneuver_end_idx is not None and route_index >= self.maneuver_end_idx:
                self.phase = ActionPhase.CLEARED
                return True, "pull_over: return complete, back in lane"

        return False, f"pull_over sub_phase={self.sub_phase.name}"

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update_state(
        self,
        *,
        plan_state         : PlanState,
        scene_data         : SceneData,
        planner_state      : PlannerState,
        prediction_data    : PredictionData,
        route_shifted_idxs : Optional[Tuple[int, int, int]] = None,
    ) -> CommandStatus:
        pull_over_cmd = plan_state.plan.action

        # ── 1. Latch pull action from plan ────────────────────────────
        if self.pull_action is None:
            self.pull_action = plan_state.plan.action

        # ── 2. Step StopFor condition state machines ──────────────────
        condition_statuses, command_reasons, replan_reasons, has_blocking_conditions = \
            self._step_conditions(scene_data, planner_state)

        # ── 3. Early-exit: condition-driven replan ────────────────────
        if replan_reasons:
            self.phase = ActionPhase.FAILED
            return CommandStatus(
                cur_cmd=pull_over_cmd,
                phase=self.phase,
                is_blocked=False,
                conditions_status=condition_statuses,
                reasons=replan_reasons,
            )

        # ── 4. Ingest route-shifted indices (first time only) ─────────
        if route_shifted_idxs is not None and not self.route_shifted:
            _, wait_idx, end_idx = route_shifted_idxs
            self.maneuver_wait_idx = wait_idx
            self.maneuver_end_idx  = end_idx
            self.route_shifted     = True

        # ── 5. Advance the state machine ─────────────────────────────
        completed, base_reason = self.step(
            self.pull_action,
            planner_state.route_index,
            scene_data.ego_data.speed,
            scene_data,
            prediction_data,
        )
        command_reasons.append(base_reason)
        has_blocking_conditions = False if completed else has_blocking_conditions

        return CommandStatus(
            cur_cmd=pull_over_cmd,
            phase=self.phase,
            is_blocked=has_blocking_conditions,
            conditions_status=condition_statuses,
            reasons=command_reasons,
        )

class ShareLaneSM(BaseActionSM):
    """
    Tracks temporary lateral route adjustments during lane shares.

    Triggered when oncoming vehicles intrude into the ego lane.
    The lat planner shifts the path within the source lane to create clearance.
    Replanning triggers whenever oncoming intruders are detected on the route.
    The command clears once the ego has passed the maneuver end index and no
    oncoming intruders remain.
    """

    def __init__(self, config : GlobalConfig):
        super().__init__(config)

        self.maneuver_end_idx : Optional[int] = None

        self.lat_plan_status  : LateralPlanStatus = LateralPlanStatus.NONE
        self.plan_fail_count  : int = 0
        self.max_plan_failures: int = 5  # TODO: move to config

    def reset(self) -> None:
        super().reset()

        self.maneuver_end_idx = None
        self.lat_plan_status  = LateralPlanStatus.NONE
        self.plan_fail_count  = 0

    def activate(self) -> None:
        self.phase = ActionPhase.EXECUTING

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_active_plan(self) -> bool:
        return self.is_executing and self.maneuver_end_idx is not None

    # ------------------------------------------------------------------
    # Replanning gate
    # ------------------------------------------------------------------

    def compute_replan_need(
        self,
        scene_data      : SceneData,
        prediction_data : PredictionData,
    ) -> bool:
        """
        Returns True if lat_planner.run_step should be called this tick.

        Triggers:
          1. No lateral plan has been generated yet.
          2. An oncoming vehicle is currently intruding the route.
        """
        if self.lat_plan_status == LateralPlanStatus.NONE:
            return True

        if scene_data.vehicle_data.get(traffic_type="oncoming", get_intruders=True):
            return True

        return False

    # ------------------------------------------------------------------
    # State transition
    # ------------------------------------------------------------------

    def step(self, route_index: int, scene_data: SceneData) -> Tuple[bool, str]:
        cmd_name = 'share_lane'

        if not self.is_executing:
            self.phase = ActionPhase.FAILED
            return True, f"Failed to activate {cmd_name}"

        if self.lat_plan_status == LateralPlanStatus.INVALID_FATAL:
            self.phase = ActionPhase.FAILED
            return True, f"Failed to generate plan for {cmd_name}"

        if self.lat_plan_status == LateralPlanStatus.NONE:
            return False, f"Waiting for feasible {cmd_name} plan"

        # Clear when past the maneuver end and no more intruders
        has_intruders = bool(scene_data.vehicle_data.get(traffic_type="oncoming", get_intruders=True))
        if self.maneuver_end_idx is not None and route_index >= self.maneuver_end_idx and not has_intruders:
            self.phase = ActionPhase.CLEARED
            return True, f"Completed {cmd_name} — shared segment cleared"

        return False, f"Running {cmd_name}"

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update_state(
        self,
        *,
        scene_data         : SceneData,
        planner_state      : PlannerState,
        lat_planner_result : Optional[LatPlannerResult] = None,
    ) -> CommandStatus:
        share_lane_cmd = Action.SHARE_LANE

        # ── 1. Step condition state machines ─────────────────────────
        condition_statuses, command_reasons, replan_reasons, has_blocking_conditions = \
            self._step_conditions(scene_data, planner_state)

        # ── 2. Early-exit: condition-driven replan ────────────────────
        if replan_reasons:
            self.phase = ActionPhase.FAILED
            return CommandStatus(
                cur_cmd=share_lane_cmd,
                phase=self.phase,
                is_blocked=False,
                conditions_status=condition_statuses,
                reasons=replan_reasons,
            )

        # ── 3. Update lateral plan status ────────────────────────────
        if lat_planner_result is not None:
            if lat_planner_result.is_empty_plan:
                self.plan_fail_count += 1
                self.lat_plan_status = (
                    LateralPlanStatus.INVALID_FATAL
                    if self.plan_fail_count >= self.max_plan_failures
                    else LateralPlanStatus.INVALID_TRANSIENT
                )
            else:
                self.lat_plan_status = LateralPlanStatus.VALID
                self.plan_fail_count = 0
                if lat_planner_result.is_new_plan:
                    self.maneuver_end_idx = lat_planner_result.goal_idx

        # ── 4. Advance the state machine ─────────────────────────────
        completed, base_reason = self.step(planner_state.route_index, scene_data)
        command_reasons.append(base_reason)

        has_blocking_conditions = False if completed else has_blocking_conditions

        return CommandStatus(
            cur_cmd=share_lane_cmd,
            phase=self.phase,
            is_blocked=has_blocking_conditions,
            conditions_status=condition_statuses,
            reasons=command_reasons,
        )
