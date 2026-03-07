"""State machine helpers for high-level trajectory planner commands."""

import carla
import numpy as np

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Set, Tuple, Dict, Union
from collections import deque

from config import GlobalConfig
from agents.navigation.local_planner import RoadOption
from privileged_route_planner import PlannerState
from team_code.scene_descriptor.scene_descriptor import SceneData
from team_code.actor_prediction.motion_prediction import PredictionData
from team_code.scene_analyzer.parsers.ego_plan_pydantic_models import ConditionAction

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
        condition_status = None
        if self.condition.obj_type == "stop_sign":
            is_blocking, reason = self._step_stop_sign(
                config=config,
                scene_data=scene_data,
                planner_state=planner_state,
            )
            needs_replan = self.phase == ActionPhase.CLEARED

            condition_status = ConditionStatus(
                condition=self.condition,
                phase=self.phase,
                is_blocking=is_blocking,
                reason=reason,
                needs_replan=needs_replan,
            )
        elif self.condition.obj_type == "traffic_light":
            is_blocking, reason = self._step_traffic_light(
                config=config,
                scene_data=scene_data,
                planner_state=planner_state,
            )
            needs_replan = self.phase == ActionPhase.CLEARED

            condition_status = ConditionStatus(
                condition=self.condition,
                phase=self.phase,
                is_blocking=is_blocking,
                reason=reason,
                needs_replan=needs_replan,
            )
        elif self.condition.obj_type == "obstacle":
            is_blocking, reason = self._step_obstacle(
                config=config,
                scene_data=scene_data
            )
            needs_replan = self.phase == ActionPhase.CLEARED

            condition_status = ConditionStatus(
                condition=self.condition,
                phase=self.phase,
                is_blocking=is_blocking,
                reason=reason,
                needs_replan=needs_replan,
            )
        elif self.condition.obj_type == "pedestrian":
            is_blocking, reason = self._step_pedestrian(
                config=config,
                scene_data=scene_data
            )
            needs_replan = self.phase == ActionPhase.CLEARED

            condition_status = ConditionStatus(
                condition=self.condition,
                phase=self.phase,
                is_blocking=is_blocking,
                reason=reason,
                needs_replan=needs_replan,
            )

        return condition_status

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
            self.phase = ActionPhase.CLEARED
            return False, "No stop sign detected"

        if self.condition.id != next_ss.id:
            self.phase = ActionPhase.CLEARED
            return False, f"Stop sign {self.condition.id} does not exist"

        if next_ss.id in planner_state.cleared_stop_sign_ids:
            self.phase = ActionPhase.CLEARED
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
            self.phase = ActionPhase.CLEARED
            return False, "No traffic light detected"

        if self.condition.id != next_tl.id:
            self.phase = ActionPhase.CLEARED
            return False, f"Traffic light {self.condition.id} does not exist"

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

        # Get the nearest obstacle
        if not ego_obstacles:
            self.phase = ActionPhase.CLEARED
            return False, f"No obstacle detected"

        next_obstacle = ego_obstacles[0]

        if self.condition.id != next_obstacle.id:
            self.phase = ActionPhase.CLEARED
            return False, f"Obstacle {self.condition.id} does not exist"


        # NOTE: Not tracking cleared obstacles, may be important

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
            self.phase = ActionPhase.CLEARED
            return False, f"No pedestrian detected"

        next_ped = scene_data.ped_data[0]

        if self.condition.id != next_ped.id:
            self.phase = ActionPhase.CLEARED
            return False, f"Pedestrian {self.condition.id} does not exist"

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

class YieldForSM(BaseSM):
    # TODO: IMPLEMENT YIELD_FOR SM
    def __init__(self, condition : ConditionCommand):
        super().__init__()
        self.condition = condition
        self.wait_ticks = 0

    def update_condition(
        self,
        new_condition : ConditionCommand
    ):
        self.reset()
        self.wait_ticks = 0
        self.condition = new_condition

    def step(
        self,
        *,
        config,
        scene_data : SceneData,
        planner_state : PlannerState,
    ) -> ConditionStatus:
        condition_status = None
        if self.condition.obj_type == "stop_sign":
            is_blocking, reason = self._step_stop_sign(
                config=config,
                scene_data=scene_data,
                planner_state=planner_state,
            )
            condition_status = ConditionStatus(
                condition=self.condition,
                phase=self.phase,
                is_blocking=is_blocking,
                reason=reason,
            )
        return condition_status

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
            self.phase = ActionPhase.CLEARED
            return False, "No stop sign detected"

        if self.condition.id != next_ss.id:
            self.phase = ActionPhase.CLEARED
            return False, f"Stop sign {self.condition.id} does not exist"

        if next_ss.id in planner_state.cleared_stop_sign_ids:
            self.phase = ActionPhase.CLEARED
            return False, "Stop sign already cleared"

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
            return False, "Finished waiting at stop sign"

        return True, reason

class BaseActionSM(BaseSM):
    """Base class for simple action state machines."""
    def __init__(self, config : GlobalConfig):
        super().__init__()
        self.config = config

        self.conditions_registry : Dict[int, Union[StopForSM, YieldForSM]] = {} # Key = actor_id, value = Condition SM
        self.cleared_actor_ids : Set[int] = set()
        self.collision_events : deque = deque(maxlen=self.config.collision_data_max_entries)

    def reset(self) -> None:
        """Return to the IDLE state and clear any derived-machine bookkeeping."""
        super().reset()

        self.conditions_registry.clear()
        # TODO/NOTE: KEEPING CLEARED ACTOR REGISTRY TO AVOID ENDLESS LOOPS
        self.cleared_actor_ids.clear()

        self.collision_events.clear()

    def update_conditions(self, conditions : List[ConditionCommand]):
        def is_same_cond(cond_a : ConditionCommand, cond_b : ConditionCommand) -> bool:
            return (
                cond_a.condition_action == cond_b.condition_action and
                cond_a.obj_type == cond_b.obj_type and
                cond_a.traffic_type == cond_b.traffic_type and
                cond_a.importance == cond_b.importance
            )

        for cond in conditions:
            if cond.id not in self.conditions_registry:
                if cond.id not in self.cleared_actor_ids:
                    condition_sm = StopForSM(condition=cond) if cond.condition_action == ConditionAction.STOP_FOR else YieldForSM(condition=cond)
                    # TODO: VALIDATE CONDITION BEFORE ADDING TO REGISTRY
                    self.conditions_registry[cond.id] = condition_sm
            else:
                condition_sm = self.conditions_registry[cond.id]
                if not is_same_cond(condition_sm.condition, cond):
                    # TODO: MIGHT WANT TO CONSIDER REMOVING STALE CONDITIONS, CURRENTLY CONDITIONS REMOVED ONLY WHEN CONDITION CLEARS
                    condition_sm.update_condition(cond)

                    self.conditions_registry[cond.id] = condition_sm

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
        config,
        scene_data : SceneData,
        planner_state : PlannerState,
    ) -> CommandStatus:
        condition_statuses : List[ConditionStatus] = []
        command_reasons : List[str] = []
        replan_reasons : List[str] = []

        has_blocking_conditions : bool = False

        # Update condition state machines first
        all_conditions = list(self.conditions_registry.values())
        for cond_sm in all_conditions:
            cond_status : ConditionStatus = cond_sm.step(
                config=config,
                scene_data=scene_data,
                planner_state=planner_state
            )
            # TODO: FIX ONCE ALL THE OTHER CONDITIONS HAVE BEEN ADDED
            # TODO: CONDITIONS REGISTRY DOES NOT EMPTY FULLY IF INCORRECT ID IS ADDED, LIKELY ARTIFACT OF CURRENT INCOMPLETE IMPLEMENTATION
            if cond_status is None:
                continue

            has_blocking_conditions = has_blocking_conditions or cond_status.is_blocking
            condition_statuses.append(cond_status)

            # Condition has finished
            if cond_status.phase == ActionPhase.CLEARED:
                # Remove from conditions registry
                # TODO: DECIDE WHETHER IT'S WORTH TO TRIM CONDITIONS LIST
                # self.conditions_registry.pop(cond_status.condition.id)

                # Update cleared actor id registry
                self.cleared_actor_ids.add(cond_status.condition.id)

                # Check if any conditions have set the replan flag
                if cond_status.needs_replan:
                    replan_reasons.append(cond_status.reason)

            # Update command reasons
            command_reasons.append(cond_status.reason)

        # Update main command state if conditions blocking completion
        # TODO: CURRENTLY BLOCKS BASED ON EXISTENCE OF CONDITIONS, MIGHT WANNA REVISIT

        if scene_data.collision_data:
            print(f'\n\nFOLLOW ROUTE SM ADDING COLLISION EVENT\n\n')
            self.collision_events.append(scene_data.collision_data[0])

        # Directly set main state to CLEARED if any earlier conditions have set the replan flag
        if len(replan_reasons) > 0:
            self.phase = ActionPhase.CLEARED
            return CommandStatus(
                cur_cmd=Action.FOLLOW_ROUTE,
                phase=self.phase,
                is_blocked=False,
                conditions_status=condition_statuses,
                reasons=replan_reasons
            )

        # Otherwise, step main command and advance state machine as normal
        completed, base_reason = self.step(has_blocking_conditions)
        command_reasons.append(base_reason)

        return CommandStatus(
            cur_cmd=Action.FOLLOW_ROUTE,
            phase=self.phase,
            is_blocked=has_blocking_conditions,
            conditions_status=condition_statuses,
            reasons=command_reasons
        )

# ---------------------------------------------------------------------------
# Intersection / Turn Handling
# ---------------------------------------------------------------------------

class TurnSM(BaseActionSM):
    """Tracks intersection traversal for turn actions."""
    def __init__(self, config : GlobalConfig):
        super().__init__(config)
        self.prev_end_idx = -1

    def reset(self) -> None:
        super().reset()
        self.prev_end_idx = -1

    def step(self, turn_dir: RoadOption, route_index: int, scene_data: SceneData) -> Tuple[bool, str]:
        """
        :param route_index: Current index along ego's route.
        :returns: (completed, reason)
        """
        if turn_dir == RoadOption.LEFT:
            cmd_name = 'turn_left'
        elif turn_dir == RoadOption.RIGHT:
            cmd_name = 'turn_right'
        else:
            cmd_name = 'turn_straight'

        # Check if any intersection/turn data exists
        intersection_data = scene_data.route_data.intersection_data if scene_data.route_data else None
        if intersection_data is None and not self.is_executing:
            # No intersection to track and command not currently executing, reset and replan
            self.phase = ActionPhase.CLEARED
            return True, "No turns in upcoming route, replan"

        # Set state to executing if idle
        if intersection_data is not None and self.is_idle:
            route_cmd, start_idx, end_idx, _ = intersection_data.entry

            # Reset and replan due to mismatched commands
            if route_cmd != turn_dir:
                if route_cmd == RoadOption.LEFT:
                    route_turn_dir = 'Turn LEFT'
                elif route_cmd == RoadOption.RIGHT:
                    route_turn_dir = 'Turn RIGHT'
                else:
                    route_turn_dir = 'Turn STRAIGHT'

                self.phase = ActionPhase.CLEARED
                return True, f"Mismatched turn commands, called {cmd_name} when route requires {route_turn_dir} maneuver"

            self.prev_end_idx = end_idx
            self.phase = ActionPhase.EXECUTING

        if self.phase is ActionPhase.EXECUTING:
            # Index tracking error, reset and replan
            if self.prev_end_idx == -1:
                self.phase = ActionPhase.CLEARED
                return True, "Not tracking turn maneuver, replan"

            # Check if current position has passed end point of turn maneuver
            if route_index >= self.prev_end_idx:
                self.prev_end_idx = -1
                self.phase = ActionPhase.CLEARED
                return True, f"Completed {cmd_name} maneuver and cleared intersection segment"

        return False, f"Running {cmd_name}"

    def update_state(
        self,
        *,
        config,
        scene_data : SceneData,
        planner_state : PlannerState,
        turn_dir : RoadOption,
    ) -> CommandStatus:
        condition_statuses : List[ConditionStatus] = []
        command_reasons : List[str] = []
        replan_reasons : List[str] = []

        has_blocking_conditions : bool = False
        if turn_dir == RoadOption.LEFT:
            turn_cmd = Action.TURN_LEFT
        elif turn_dir == RoadOption.RIGHT:
            turn_cmd = Action.TURN_RIGHT
        else:
            turn_cmd = Action.TURN_STRAIGHT

        # Update condition state machines first
        all_conditions = list(self.conditions_registry.values())
        for cond_sm in all_conditions:
            cond_status : ConditionStatus = cond_sm.step(
                config=config,
                scene_data=scene_data,
                planner_state=planner_state
            )
            # TODO: FIX ONCE ALL THE OTHER CONDITIONS HAVE BEEN ADDED
            # TODO: CONDITIONS REGISTRY DOES NOT EMPTY FULLY IF INCORRECT ID IS ADDED, LIKELY ARTIFACT OF CURRENT INCOMPLETE IMPLEMENTATION
            if cond_status is None:
                continue

            has_blocking_conditions = has_blocking_conditions or cond_status.is_blocking
            condition_statuses.append(cond_status)

            # Condition has finished
            if cond_status.phase == ActionPhase.CLEARED:
                # Remove from conditions registry
                # TODO: DECIDE WHETHER IT'S WORTH TO TRIM CONDITIONS LIST
                # self.conditions_registry.pop(cond_status.condition.id)

                # Update cleared actor id registry
                self.cleared_actor_ids.add(cond_status.condition.id)

                # Check if any conditions have set the replan flag
                if cond_status.needs_replan:
                    replan_reasons.append(cond_status.reason)

            # Update command reasons
            command_reasons.append(cond_status.reason)

        # Update main command state if conditions blocking completion
        # TODO: CURRENTLY BLOCKS BASED ON EXISTENCE OF CONDITIONS, MIGHT WANNA REVISIT
        if scene_data.collision_data:
            self.collision_events.append(scene_data.collision_data[0])

        # Directly set main state to CLEARED if any earlier conditions have set the replan flag
        if len(replan_reasons) > 0:
            self.phase = ActionPhase.CLEARED
            return CommandStatus(
                cur_cmd=turn_cmd,
                phase=self.phase,
                is_blocked=False,
                conditions_status=condition_statuses,
                reasons=replan_reasons
            )

        # Otherwise, step main command and advance state machine as normal
        completed, base_reason = self.step(
            turn_dir, planner_state.route_index, scene_data
        )
        command_reasons.append(base_reason)

        # Reset blocking conditions, if high-level command is completed
        has_blocking_conditions = False if completed else has_blocking_conditions

        return CommandStatus(
            cur_cmd=turn_cmd,
            phase=self.phase,
            is_blocked=has_blocking_conditions,
            conditions_status=condition_statuses,
            reasons=command_reasons
        )
# ---------------------------------------------------------------------------
# Lane Change Handling
# ---------------------------------------------------------------------------

class LaneChangeSM(BaseActionSM):
    """Tracks lane change progression and planning horizons."""
    def __init__(self, config : GlobalConfig):
        super().__init__(config)
        self.prev_end_idx = -1

    def reset(self) -> None:
        super().reset()
        self.prev_end_idx = -1

    def step(self, lc_dir: RoadOption, route_index: int, scene_data: SceneData) -> Tuple[bool, str]:
        """
        :returns: (plan_s_goal_m, completed, reason)
        """
        cmd_name = 'change_lane_left' if lc_dir == RoadOption.CHANGELANELEFT else 'change_lane_right'

        # Check if any lane change data exists
        lane_change_data = scene_data.route_data.lane_change_data if scene_data.route_data else None

        # TODO: REVISIT THIS WHEN HANDLING MANEUVERING AROUND CYCLISTS, CURRENTLY SAYING IF NO LANE CHANGES IN ROUTE WE SHOULD REPLAN, DOESNT ACCOUNT FOR ROUTE CHANGES
        if lane_change_data is None:
            # No lane change segment currently active, reset and replan
            self.phase = ActionPhase.CLEARED
            return True, "No lane changes in upcoming route, reevaluate"

        # Set state to executing if idle
        if self.is_idle:
            route_cmd, start_idx, end_idx = lane_change_data.entry

            # Reset and replan due to mismatched commands
            if route_cmd != lc_dir:
                if route_cmd == RoadOption.CHANGELANELEFT:
                    route_lc_dir = 'Change lane LEFT'
                elif route_cmd == RoadOption.CHANGELANERIGHT:
                    route_lc_dir = 'Change lane RIGHT'
                else:
                    route_lc_dir = 'Go STRAIGHT'

                self.phase = ActionPhase.CLEARED
                return True, f"Mismatched lane change commands, called {cmd_name} when route requires {route_lc_dir} maneuver"

            self.prev_end_idx = end_idx
            self.phase = ActionPhase.EXECUTING

        if self.phase is ActionPhase.EXECUTING:
            # Index tracking error, reset and replan
            if self.prev_end_idx == -1:
                self.phase = ActionPhase.CLEARED
                return True, "Not tracking lane change maneuver, replan"

            # Check if current position has passed end point of lane change maneuver
            if route_index >= self.prev_end_idx:
                self.prev_end_idx = -1
                self.phase = ActionPhase.CLEARED
                return True, f"Completed {cmd_name} maneuver and cleared lane change segment"

        return False, f"Running {cmd_name}"

    def update_state(
        self,
        *,
        config,
        scene_data : SceneData,
        planner_state : PlannerState,
        lc_dir : RoadOption,
    ) -> CommandStatus:
        condition_statuses : List[ConditionStatus] = []
        command_reasons : List[str] = []
        replan_reasons : List[str] = []

        has_blocking_conditions : bool = False
        lc_cmd = Action.CHANGE_LANE_LEFT if lc_dir == RoadOption.CHANGELANELEFT else Action.CHANGE_LANE_RIGHT

        # Update condition state machines first
        all_conditions = list(self.conditions_registry.values())
        for cond_sm in all_conditions:
            cond_status : ConditionStatus = cond_sm.step(
                config=config,
                scene_data=scene_data,
                planner_state=planner_state
            )
            # TODO: FIX ONCE ALL THE OTHER CONDITIONS HAVE BEEN ADDED
            # TODO: CONDITIONS REGISTRY DOES NOT EMPTY FULLY IF INCORRECT ID IS ADDED, LIKELY ARTIFACT OF CURRENT INCOMPLETE IMPLEMENTATION
            if cond_status is None:
                continue

            has_blocking_conditions = has_blocking_conditions or cond_status.is_blocking
            condition_statuses.append(cond_status)

            # Condition has finished
            if cond_status.phase == ActionPhase.CLEARED:
                # Remove from conditions registry
                # TODO: DECIDE WHETHER IT'S WORTH TO TRIM CONDITIONS LIST
                # self.conditions_registry.pop(cond_status.condition.id)

                # Update cleared actor id registry
                self.cleared_actor_ids.add(cond_status.condition.id)

                # Check if any conditions have set the replan flag
                if cond_status.needs_replan:
                    replan_reasons.append(cond_status.reason)

            # Update command reasons
            command_reasons.append(cond_status.reason)

        # Update main command state if conditions blocking completion
        # TODO: CURRENTLY BLOCKS BASED ON EXISTENCE OF CONDITIONS, MIGHT WANNA REVISIT

        if scene_data.collision_data:
            self.collision_events.append(scene_data.collision_data[0])

        # Directly set main state to CLEARED if any earlier conditions have set the replan flag
        if len(replan_reasons) > 0:
            self.phase = ActionPhase.CLEARED
            return CommandStatus(
                cur_cmd=lc_cmd,
                phase=self.phase,
                is_blocked=False,
                conditions_status=condition_statuses,
                reasons=replan_reasons
            )

        # Otherwise, step main command and advance state machine as normal
        completed, base_reason = self.step(
            lc_dir, planner_state.route_index, scene_data
        )
        command_reasons.append(base_reason)

        # Reset blocking conditions, if high-level command is completed
        has_blocking_conditions = False if completed else has_blocking_conditions

        return CommandStatus(
            cur_cmd=lc_cmd,
            phase=self.phase,
            is_blocked=has_blocking_conditions,
            conditions_status=condition_statuses,
            reasons=command_reasons
        )
# ---------------------------------------------------------------------------
# Overtake Handling
# ---------------------------------------------------------------------------

class OvertakeSM(BaseActionSM):
    """Tracks temporary route changes during overtakes."""
    def __init__(self, config : GlobalConfig):
        super().__init__(config)
        self.cur_route_changes: Optional[Tuple[int, int]] = None # (start_idx, end_idx)
        self.target_end_idx: Optional[int] = None
        self.last_intrusion_idx: Optional[int] = None
        self.end_locked: bool = False

    def reset(self) -> None:
        super().reset()
        self.cur_route_changes = None
        self.target_end_idx: Optional[int] = None
        self.last_intrusion_idx: Optional[int] = None
        self.end_locked: bool = False

    @property
    def needs_route_adjustment(self) -> bool:
        """
        :returns: True if we should plan a new overtake path segment.
        """
        if self.cur_route_changes is None:
            return True

        if self.target_end_idx is None:
            return True

        return self.target_end_idx > self.cur_route_changes[-1]

    def set_route_changes(self, route_change_idxs: Optional[Tuple[int, int]]):
        if route_change_idxs is not None:
            self.cur_route_changes = route_change_idxs
            self.phase = ActionPhase.EXECUTING

    def get_target_end_idx(
        self,
        *,
        config : GlobalConfig,
        planner_state : PlannerState,
        scene_data : SceneData,
        prediction_data : PredictionData,
        actor_registry : Dict[str, Set[int]],
        target_distance_initial : float,
        buffer_distance : float,
    ) -> int:
        if self.end_locked and self.target_end_idx is not None:
            return self.target_end_idx

        route_index = planner_state.route_index
        route_len = planner_state.route_len

        initial_end_idx = min(
            route_len - 1,
            route_index + int(target_distance_initial * config.points_per_meter)
        )
        buffer_pts = int(buffer_distance * config.points_per_meter)

        furthest_idx = None
        obs_ids = actor_registry["obstacles"]
        cyclist_ids = actor_registry["cyclist"]

        ego_obstacles = scene_data.obstacle_data.ego_obstacles
        for obs_data in ego_obstacles:
            print(f'\t(overtake) FOUND OBSTACLE ID, id: {obs_data.id}, distance: {obs_data.relative_distance}, intrusion_idx: {obs_data.intrusion_idx}')
            if furthest_idx is None:
                furthest_idx = obs_data.intrusion_idx
            else:
                furthest_idx = max(furthest_idx, obs_data.intrusion_idx)


        # if obs_ids:
        #     print(f'\n(overtake) FOUND OBSTACLE IDS FROM LLM, len: {len(obs_ids)}')
        #     for obs_data in scene_data.obstacle_data:
        #         if obs_data.id in obs_ids and obs_data.intrusion_idx is not None:
        #             print(f'\t(overtake) FOUND OBSTACLE ID, id: {obs_data.id}, distance: {obs_data.relative_distance}, intrusion_idx: {obs_data.intrusion_idx}')
        #             if furthest_idx is None:
        #                 furthest_idx = obs_data.intrusion_idx
        #             else:
        #                 furthest_idx = max(furthest_idx, obs_data.intrusion_idx)

        # if cyclist_ids:
        #     print(f'\n(cyclist) FOUND CYCLIST IDS FROM LLM, len: {len(cyclist_ids)}')
        #     all_cyclist_data = scene_data.vehicle_data.get(traffic_type="leading", vehicle_types={"cyclist"}, vehicle_ids=cyclist_ids)
        #     for cyc_data in all_cyclist_data:
        #         print(f'\t(overtake) FOUND CYCLIST ID, id: {cyc_data.id}, distance: {cyc_data.relative_distance}, intrusion_idx: {cyc_data.intrusion_idx}')
        #         if furthest_idx is None:
        #             furthest_idx = cyc_data.intrusion_idx
        #         else:
        #             furthest_idx = max(furthest_idx, cyc_data.intrusion_idx)
        all_cyclist_data = scene_data.vehicle_data.get(traffic_type="leading", vehicle_types={"cyclist"})
        for cyc_data in all_cyclist_data:
            print(f'\n\n(overtake_sm) FOUND CYCLIST ID, id: {cyc_data.id}, speed: {cyc_data.speed}, distance: {cyc_data.relative_distance}, intrusion_idx: {cyc_data.intrusion_idx}')

            # Check if cyclist overlaps with ego route
            if cyc_data.id in prediction_data.all_actor_overlaps:
                cyclist_overlap = prediction_data.all_actor_overlaps.get(cyc_data.id)[0]

                # Get predicted bounding box at approximately 2s
                num_frames = 40

                if num_frames in cyclist_overlap.frame_occupancies:
                    bb_end_idx = cyclist_overlap.frame_occupancies.get(num_frames)[1]

                    print(f'\n\nDISTANCE TO TIMED OVERLAP END INDEX: {(bb_end_idx * 20) / self.config.points_per_meter}')
                    route_end_idx = bb_end_idx * 20 + route_index
                    if furthest_idx is None:
                        furthest_idx = route_end_idx
                    else:
                        furthest_idx = max(furthest_idx, route_end_idx)

            # if cyc_data.intrusion_idx is not None:
            #     if furthest_idx is None:
            #         furthest_idx = cyc_data.intrusion_idx
            #     else:
            #         furthest_idx = max(furthest_idx, cyc_data.intrusion_idx)

        print(f'\n\nINITIAL IDX')
        print(f'\tINITIAL DIST: {(initial_end_idx - route_index) / self.config.points_per_meter}, INITIAL END IDX: {initial_end_idx}')
        if furthest_idx is not None:
            self.last_intrusion_idx = furthest_idx

            print(f'\n\nFURTHEST_IDX')
            print(f'\tBUFFER DIST: {((furthest_idx + buffer_pts) - route_index) / self.config.points_per_meter} BUFER IDX: {furthest_idx + buffer_pts}')
            target_end_idx = min(initial_end_idx, furthest_idx + buffer_pts)
            if self.target_end_idx is None:
                self.target_end_idx = target_end_idx
            else:
                self.target_end_idx = max(self.target_end_idx, target_end_idx)
        else:
            if self.target_end_idx is None:
                self.target_end_idx = initial_end_idx

        if self.last_intrusion_idx is not None:
            if route_index >= self.last_intrusion_idx:
                self.end_locked = True

        print(f'\n\nFINAL OVERTAKE INDEX: {self.target_end_idx}')
        return self.target_end_idx

    def step(self, overtake_dir: RoadOption, route_index: int, scene_data: SceneData) -> Tuple[bool, str]:
        cmd_name = 'overtake_left' if overtake_dir == RoadOption.CHANGELANELEFT else 'overtake_right'

        # Check if any overtake adjustments have been made
        if self.cur_route_changes is None:
            # Unable to change route, reset and replan
            self.phase = ActionPhase.CLEARED
            return True, "Unable to modify route, replan"

        if self.phase is ActionPhase.EXECUTING:
            # Planning error, reset and replan
            if self.cur_route_changes is None:
                self.phase = ActionPhase.CLEARED
                return True, "Not tracking overtake maneuver, replan"

            # Check if current position has passed end point of overtake maneuver
            _, end_idx = self.cur_route_changes
            if route_index > end_idx:
                self.cur_route_changes = None
                self.phase = ActionPhase.CLEARED
                return True, f"Completed {cmd_name} maneuver and cleared overtake segment"

        return False, f"Running {cmd_name}"

    def update_state(
        self,
        *,
        config,
        scene_data : SceneData,
        planner_state : PlannerState,
        overtake_dir : RoadOption,
    ) -> CommandStatus:
        condition_statuses : List[ConditionStatus] = []
        command_reasons : List[str] = []
        replan_reasons : List[str] = []

        has_blocking_conditions : bool = False
        overtake_cmd = Action.OVERTAKE_LEFT if overtake_dir == RoadOption.CHANGELANELEFT else Action.OVERTAKE_RIGHT

        # Update condition state machines first
        all_conditions = list(self.conditions_registry.values())
        for cond_sm in all_conditions:
            cond_status : ConditionStatus = cond_sm.step(
                config=config,
                scene_data=scene_data,
                planner_state=planner_state
            )
            # TODO: FIX ONCE ALL THE OTHER CONDITIONS HAVE BEEN ADDED
            # TODO: CONDITIONS REGISTRY DOES NOT EMPTY FULLY IF INCORRECT ID IS ADDED, LIKELY ARTIFACT OF CURRENT INCOMPLETE IMPLEMENTATION
            if cond_status is None:
                continue

            has_blocking_conditions = has_blocking_conditions or cond_status.is_blocking
            condition_statuses.append(cond_status)

            # Condition has finished
            if cond_status.phase == ActionPhase.CLEARED:
                # Remove from conditions registry
                # TODO: DECIDE WHETHER IT'S WORTH TO TRIM CONDITIONS LIST
                # self.conditions_registry.pop(cond_status.condition.id)

                # Update cleared actor id registry
                self.cleared_actor_ids.add(cond_status.condition.id)

                # Check if any conditions have set the replan flag
                if cond_status.needs_replan:
                    replan_reasons.append(cond_status.reason)

            # Update command reasons
            command_reasons.append(cond_status.reason)

        # Update main command state if conditions blocking completion
        # TODO: CURRENTLY BLOCKS BASED ON EXISTENCE OF CONDITIONS, MIGHT WANNA REVISIT

        if scene_data.collision_data:
            self.collision_events.append(scene_data.collision_data[0])

        # Directly set main state to CLEARED if any earlier conditions have set the replan flag
        if len(replan_reasons) > 0:
            self.phase = ActionPhase.CLEARED
            return CommandStatus(
                cur_cmd=overtake_cmd,
                phase=self.phase,
                is_blocked=False,
                conditions_status=condition_statuses,
                reasons=replan_reasons
            )

        # Otherwise, step main command and advance state machine as normal
        completed, base_reason = self.step(
            overtake_dir, planner_state.route_index, scene_data
        )
        command_reasons.append(base_reason)

        # Reset blocking conditions, if high-level command is completed
        has_blocking_conditions = False if completed else has_blocking_conditions

        return CommandStatus(
            cur_cmd=overtake_cmd,
            phase=self.phase,
            is_blocked=has_blocking_conditions,
            conditions_status=condition_statuses,
            reasons=command_reasons
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
    """Tracks pull-over maneuvers and route restoration bookkeeping."""
    def __init__(self, config : GlobalConfig):
        super().__init__(config)
        self.cur_route_changes: Optional[Tuple[int, int, int]] = None # (start_idx, wait_idx, end_idx)
        self.sub_phase : PullOverPhase = PullOverPhase.IDLE

    def reset(self) -> None:
        super().reset()
        self.cur_route_changes = None
        self.sub_phase = PullOverPhase.IDLE

    @property
    def needs_route_adjustment(self) -> bool:
        return self.cur_route_changes is None

    def set_route_changes(self, route_change_idxs: Optional[Tuple[int, int, int]]):
        if route_change_idxs is not None:
            self.cur_route_changes = route_change_idxs
            self.phase = ActionPhase.EXECUTING
            self.sub_phase = PullOverPhase.YIELDING

    def step(
        self,
        pull_over_dir : RoadOption,
        route_index : int,
        scene_data: SceneData,
        vehicle_id : int,
        vehicle_passed : bool,
        has_vehicle : bool,
    ) -> Tuple[bool, str]:
        cmd_name = 'pull_over_left' if pull_over_dir == RoadOption.CHANGELANELEFT else 'pull_over_right'

        # Check if any pull over adjustments have been made
        if self.cur_route_changes is None:
            # No changes made to route, reset and replan
            self.phase = ActionPhase.CLEARED
            return True, "Unable to modify route, replan"

        if self.phase is ActionPhase.EXECUTING:
            # Planning error, reset and replan
            if self.cur_route_changes is None:
                self.phase = ActionPhase.CLEARED
                return True, "Not tracking pull over maneuver, replan"

            start_idx, wait_idx, end_idx = self.cur_route_changes
            if self.sub_phase is PullOverPhase.YIELDING:
                # Currently yielding to vehicle
                if route_index >= wait_idx:
                    self.sub_phase = PullOverPhase.WAITING
                    return False, f"Waiting for vehicle {vehicle_id} to pass"

                return False, f"Pulling over for vehicle {vehicle_id}"

            elif self.sub_phase is PullOverPhase.WAITING:
                if vehicle_passed or not has_vehicle:
                    self.sub_phase = PullOverPhase.RETURNING
                    return False, f"Vehicle {vehicle_id} has passed, resuming route"

                return False, f"Waiting for vehicle {vehicle_id} to pass"

            elif self.sub_phase is PullOverPhase.RETURNING:
                if route_index >= end_idx:
                    self.phase = ActionPhase.CLEARED
                    self.sub_phase = PullOverPhase.IDLE
                    self.cur_route_changes = None
                    return True, f"Completed {cmd_name} maneuver and cleared pull over segment"

                return False, f"Vehicle {vehicle_id} has passed, resuming route"

        return False, f"Running {cmd_name}"

    def update_state(
        self,
        *,
        config,
        scene_data : SceneData,
        planner_state : PlannerState,
        pull_over_dir : RoadOption,
        vehicle_id : int,
        vehicle_passed : bool,
        has_vehicle : bool
    ) -> CommandStatus:
        condition_statuses : List[ConditionStatus] = []
        command_reasons : List[str] = []
        replan_reasons : List[str] = []

        has_blocking_conditions : bool = False
        pull_over_cmd = Action.PULL_OVER_LEFT if pull_over_dir == RoadOption.CHANGELANELEFT else Action.PULL_OVER_RIGHT

        # Update condition state machines first
        all_conditions = list(self.conditions_registry.values())
        for cond_sm in all_conditions:
            cond_status : ConditionStatus = cond_sm.step(
                config=config,
                scene_data=scene_data,
                planner_state=planner_state
            )
            # TODO: FIX ONCE ALL THE OTHER CONDITIONS HAVE BEEN ADDED
            # TODO: CONDITIONS REGISTRY DOES NOT EMPTY FULLY IF INCORRECT ID IS ADDED, LIKELY ARTIFACT OF CURRENT INCOMPLETE IMPLEMENTATION
            if cond_status is None:
                continue

            has_blocking_conditions = has_blocking_conditions or cond_status.is_blocking
            condition_statuses.append(cond_status)

            # Condition has finished
            if cond_status.phase == ActionPhase.CLEARED:
                # Remove from conditions registry
                # TODO: DECIDE WHETHER IT'S WORTH TO TRIM CONDITIONS LIST
                # self.conditions_registry.pop(cond_status.condition.id)

                # Update cleared actor id registry
                self.cleared_actor_ids.add(cond_status.condition.id)

                # Check if any conditions have set the replan flag
                if cond_status.needs_replan:
                    replan_reasons.append(cond_status.reason)

            # Update command reasons
            command_reasons.append(cond_status.reason)

        # Update main command state if conditions blocking completion
        # TODO: CURRENTLY BLOCKS BASED ON EXISTENCE OF CONDITIONS, MIGHT WANNA REVISIT
        if scene_data.collision_data:
            self.collision_events.append(scene_data.collision_data[0])

        # Directly set main state to CLEARED if any earlier conditions have set the replan flag
        if len(replan_reasons) > 0:
            self.phase = ActionPhase.CLEARED
            return CommandStatus(
                cur_cmd=pull_over_cmd,
                phase=self.phase,
                is_blocked=False,
                conditions_status=condition_statuses,
                reasons=replan_reasons
            )

        # Otherwise, step main command and advance state machine as normal
        completed, base_reason = self.step(
            pull_over_dir, planner_state.route_index, scene_data, vehicle_id, vehicle_passed, has_vehicle
        )
        command_reasons.append(base_reason)

        # Reset blocking conditions, if high-level command is completed
        has_blocking_conditions = False if completed else has_blocking_conditions

        return CommandStatus(
            cur_cmd=pull_over_cmd,
            phase=self.phase,
            is_blocked=has_blocking_conditions,
            conditions_status=condition_statuses,
            reasons=command_reasons
        )

class ShareLaneSM(BaseActionSM):
    """Tracks temporary route changes during lane shares."""

    def __init__(self, config : GlobalConfig):
        super().__init__(config)
        self.cur_route_changes: Optional[Tuple[int, int]] = None # (start_idx, end_idx)
        self.target_end_idx: Optional[int] = None
        self.last_intrusion_idx: Optional[int] = None
        self.end_locked: bool = False

    def reset(self) -> None:
        super().reset()
        self.cur_route_changes = None
        self.target_end_idx: Optional[int] = None
        self.last_intrusion_idx: Optional[int] = None
        self.end_locked: bool = False

    @property
    def needs_route_adjustment(self) -> bool:
        """
        :returns: True if we should plan a new lane share path segment.
        """
        if self.cur_route_changes is None:
            return True

        if self.target_end_idx is None:
            return True

        # return self.target_end_idx > self.cur_route_changes[-1]
        return True

    def set_route_changes(self, route_change_idxs: Optional[Tuple[int, int]]):
        if route_change_idxs is not None:
            self.phase = ActionPhase.EXECUTING
            self.cur_route_changes = route_change_idxs

    def get_target_end_idx(
        self,
        *,
        config : GlobalConfig,
        planner_state : PlannerState,
        scene_data : SceneData,
        prediction_data : PredictionData,
        actor_registry : Dict[str, Set[int]],
        target_distance_initial : float,
        buffer_distance : float,
    ) -> int:
        # if self.end_locked and self.target_end_idx is not None:
        #     return self.target_end_idx

        route_index = planner_state.route_index
        route_len = planner_state.route_len

        initial_end_idx = min(
            route_len - 1,
            route_index + int(target_distance_initial * config.points_per_meter)
        )
        buffer_pts = int(buffer_distance * config.points_per_meter)

        furthest_idx = None

        # Get intruders
        all_intruder_data = scene_data.vehicle_data.get(traffic_type="oncoming", get_intruders=True)
        if self.target_end_idx is None:
            self.target_end_idx = initial_end_idx

        if all_intruder_data:
            self.target_end_idx = initial_end_idx

        print(f'\n\nFINAL SHARE LANE INDEX: {self.target_end_idx}')
        return self.target_end_idx

    def step(self, route_index: int, scene_data: SceneData) -> Tuple[bool, str]:
        # Check if any lane share adjustments have been made
        if self.cur_route_changes is None:
            # Unable to change route, reset and replan
            self.phase = ActionPhase.CLEARED
            return True, "Unable to modify route, replan"

        if self.phase is ActionPhase.EXECUTING:
            # Planning error, reset and replan
            if self.cur_route_changes is None:
                self.phase = ActionPhase.CLEARED
                return True, "Not tracking lane share maneuver, replan"

            # Check if current position has passed end point of lane share maneuver
            _, end_idx = self.cur_route_changes
            if route_index > end_idx and not scene_data.vehicle_data.get(traffic_type="oncoming", get_intruders=True):
                self.cur_route_changes = None
                self.phase = ActionPhase.CLEARED
                return True, f"Completed share_lane maneuver and cleared shared segment with no more intruding vehicles"

        return False, f"Running share_lane"

    def update_state(
        self,
        *,
        config,
        scene_data : SceneData,
        planner_state : PlannerState,
    ) -> CommandStatus:
        condition_statuses : List[ConditionStatus] = []
        command_reasons : List[str] = []
        replan_reasons : List[str] = []

        has_blocking_conditions : bool = False
        share_lane_cmd = Action.SHARE_LANE

        # Update condition state machines first
        all_conditions = list(self.conditions_registry.values())
        for cond_sm in all_conditions:
            cond_status : ConditionStatus = cond_sm.step(
                config=config,
                scene_data=scene_data,
                planner_state=planner_state
            )
            # TODO: FIX ONCE ALL THE OTHER CONDITIONS HAVE BEEN ADDED
            # TODO: CONDITIONS REGISTRY DOES NOT EMPTY FULLY IF INCORRECT ID IS ADDED, LIKELY ARTIFACT OF CURRENT INCOMPLETE IMPLEMENTATION
            if cond_status is None:
                continue

            has_blocking_conditions = has_blocking_conditions or cond_status.is_blocking
            condition_statuses.append(cond_status)

            # Condition has finished
            if cond_status.phase == ActionPhase.CLEARED:
                # Remove from conditions registry
                # TODO: DECIDE WHETHER IT'S WORTH TO TRIM CONDITIONS LIST
                # self.conditions_registry.pop(cond_status.condition.id)

                # Update cleared actor id registry
                self.cleared_actor_ids.add(cond_status.condition.id)

                # Check if any conditions have set the replan flag
                if cond_status.needs_replan:
                    replan_reasons.append(cond_status.reason)

            # Update command reasons
            command_reasons.append(cond_status.reason)

        # Update main command state if conditions blocking completion
        # TODO: CURRENTLY BLOCKS BASED ON EXISTENCE OF CONDITIONS, MIGHT WANNA REVISIT
        if scene_data.collision_data:
            self.collision_events.append(scene_data.collision_data[0])

        # Directly set main state to CLEARED if any earlier conditions have set the replan flag
        if len(replan_reasons) > 0:
            self.phase = ActionPhase.CLEARED
            return CommandStatus(
                cur_cmd=share_lane_cmd,
                phase=self.phase,
                is_blocked=False,
                conditions_status=condition_statuses,
                reasons=replan_reasons
            )

        # Otherwise, step main command and advance state machine as normal
        completed, base_reason = self.step(
            planner_state.route_index, scene_data
        )
        command_reasons.append(base_reason)

        # Reset blocking conditions, if high-level command is completed
        has_blocking_conditions = False if completed else has_blocking_conditions

        return CommandStatus(
            cur_cmd=share_lane_cmd,
            phase=self.phase,
            is_blocked=has_blocking_conditions,
            conditions_status=condition_statuses,
            reasons=command_reasons
        )
