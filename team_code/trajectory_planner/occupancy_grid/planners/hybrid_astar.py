import heapq
import numpy as np

from math import sqrt, cos, sin, atan2, pi
from dataclasses import dataclass
from typing import Tuple, List, Dict, Set, Optional

from config import GlobalConfig
from trajectory_planner.occupancy_grid.sensor_data_processor import BEVGrid

def _normalize_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + pi) % (2 * pi) - pi

@dataclass(frozen=True, slots=True)
class HybridState:
    """Continuous vehicle state used during Hybrid A* search."""
    x: float
    y: float
    theta: float

class HybridAStar:
    def __init__(self, grid : BEVGrid):
        self.config = GlobalConfig()
        self.grid = grid

        # KBM Parameters
        self._rear_wheel_base = float(self.config.rear_wheel_base)
        self._front_wheel_base = float(self.config.front_wheel_base)
        self._wheel_base_sum = self._rear_wheel_base + self._front_wheel_base
        if abs(self._wheel_base_sum) < 1e-6:
            self._wheel_base_sum = 1e-6
        self._steering_gain = float(self.config.steering_gain)

        # Hybrid A* Parameters
        self._hybrid_step = 1.0  # meters travelled per expansion
        self._hybrid_theta_bins = 72
        self._hybrid_goal_xy_tol = 1.5  # meters
        self._hybrid_goal_theta_tol = np.deg2rad(30.0)
        self._collision_check_step = self.grid.resolution * 0.5
        self._steering_options = np.array([-0.6, -0.3, 0.0, 0.3, 0.6], dtype=np.float32)
        self._turn_penalty = 0.2

    def plan_path(
        self,
        static_occupancy_map : np.ndarray,
        static_cost_map : np.ndarray,
        start_point_ego : np.ndarray,
        goal_point_ego : np.ndarray,
    ) -> Tuple[List[HybridState], float]:
        start_state = HybridState(start_point_ego[0], start_point_ego[1], 0.0)
        goal_heading = atan2(goal_point_ego[1] - start_point_ego[1], goal_point_ego[0] - start_point_ego[0])
        goal_state = HybridState(goal_point_ego[0], goal_point_ego[1], goal_heading)

        hybrid_states, path_cost = self._run_hybrid_astar(
            static_occupancy_map,
            static_cost_map,
            start_state,
            goal_state
        )

        if not hybrid_states:
            return [], np.empty((0, 3), dtype=np.float32)

        path_grid: List[Tuple[int, int]] = []
        grid_coords = []
        for state in hybrid_states:
            indices = self._state_to_grid_indices(state.x, state.y, static_occupancy_map.shape)
            if indices is None:
                continue
            path_grid.append(indices)
            grid_coords.append((state.x, state.y))

        return path_grid

    def _state_to_grid_indices(
        self,
        x_val: float,
        y_val: float,
        map_shape: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        H, W = map_shape
        grid_x, grid_y = self.grid.world_to_grid(
            np.array([x_val], dtype=np.float32),
            np.array([y_val], dtype=np.float32)
        )
        i = int(grid_x[0])
        j = int(grid_y[0])
        if i < 0 or i >= H or j < 0 or j >= W:
            return None
        return (i, j)

    def _collision_free(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        occupancy_map: np.ndarray
    ) -> bool:
        dist = sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        steps = max(int(dist / self._collision_check_step), 1)
        for step in range(steps + 1):
            t = step / steps
            xi = x0 + (x1 - x0) * t
            yi = y0 + (y1 - y0) * t
            indices = self._state_to_grid_indices(xi, yi, occupancy_map.shape)
            if indices is None:
                return False
            if occupancy_map[indices[0], indices[1]] <= 0:
                return False
        return True

    def _propagate_state(
        self,
        state: HybridState,
        steer: float
    ) -> HybridState:
        wheel_angle = self._steering_gain * float(steer)
        slip = np.arctan(self._rear_wheel_base / self._wheel_base_sum * np.tan(wheel_angle))
        dx = self._hybrid_step * cos(state.theta + slip)
        dy = self._hybrid_step * sin(state.theta + slip)
        dtheta = self._hybrid_step / self._rear_wheel_base * sin(slip)
        return HybridState(state.x + dx, state.y + dy, _normalize_angle(state.theta + dtheta))

    def _run_hybrid_astar(
        self,
        static_occupancy_map : np.ndarray,
        static_cost_map : np.ndarray,
        start_state : HybridState,
        goal_state : HybridState,
        w_cost : float = 1.0
    ) -> Tuple[List[HybridState], float]:
        """
        Hybrid A* search using the vehicle kinematics.

        Returns:
            path_states: list[(x, y, theta)] from start to goal. [] if no path.
            total_cost: g-score of the goal (np.inf if not found)
        """

        if static_occupancy_map is None:
            raise ValueError("static_occupancy_map must be provided")

        if static_cost_map is None:
            static_cost_map = np.zeros_like(static_occupancy_map, dtype=np.float32)

        theta_bin_size = 2 * pi / float(self._hybrid_theta_bins)

        def state_key(state: HybridState) -> Optional[Tuple[int, int, int]]:
            indices = self._state_to_grid_indices(state.x, state.y, static_occupancy_map.shape)
            if indices is None:
                return None
            theta_norm = _normalize_angle(state.theta)
            theta_idx = int(np.floor((theta_norm + pi) / theta_bin_size)) % self._hybrid_theta_bins
            return (indices[0], indices[1], theta_idx)

        goal_x, goal_y, goal_theta = goal_state.x, goal_state.y, goal_state.theta

        def heuristic(state: HybridState) -> float:
            pos_dist = sqrt((goal_x - state.x) ** 2 + (goal_y - state.y) ** 2)
            heading_diff = abs(_normalize_angle(goal_theta - state.theta))
            return pos_dist + 0.5 * heading_diff

        start_indices = self._state_to_grid_indices(start_state.x, start_state.y, static_occupancy_map.shape)
        goal_indices = self._state_to_grid_indices(goal_x, goal_y, static_occupancy_map.shape)
        if (
            start_indices is None or goal_indices is None or
            static_occupancy_map[start_indices[0], start_indices[1]] <= 0 or
            static_occupancy_map[goal_indices[0], goal_indices[1]] <= 0
        ):
            return [], np.inf

        start_key = state_key(start_state)
        if start_key is None:
            return [], np.inf

        cmax = float(static_cost_map.max()) if static_cost_map.size else 1.0
        if cmax <= 0:
            cmax = 1.0
        cost_norm = static_cost_map.astype(np.float32) / cmax

        open_heap: List[Tuple[float, Tuple[int, int, int]]] = []
        g_cost: Dict[Tuple[int, int, int], float] = {start_key: 0.0}
        parent: Dict[Tuple[int, int, int], Optional[Tuple[int, int, int]]] = {start_key: None}
        states: Dict[Tuple[int, int, int], HybridState] = {start_key: start_state}

        heapq.heappush(open_heap, (heuristic(start_state), start_key))
        closed: Set[Tuple[int, int, int]] = set()

        while open_heap:
            _, current_key = heapq.heappop(open_heap)
            if current_key in closed:
                continue

            closed.add(current_key)

            current_state = states[current_key]

            pos_err = sqrt((goal_x - current_state.x) ** 2 + (goal_y - current_state.y) ** 2)
            heading_err = abs(_normalize_angle(goal_theta - current_state.theta))
            if pos_err <= self._hybrid_goal_xy_tol and heading_err <= self._hybrid_goal_theta_tol:
                # reconstruct path
                path_states: List[HybridState] = []
                key_cursor: Optional[Tuple[int, int, int]] = current_key
                while key_cursor is not None:
                    path_states.append(states[key_cursor])
                    key_cursor = parent[key_cursor]
                path_states.reverse()
                return path_states, g_cost[current_key]

            current_cost = g_cost[current_key]

            for steer in self._steering_options:
                next_state = self._propagate_state(current_state, float(steer))
                next_key = state_key(next_state)
                if next_key is None or next_key in closed:
                    continue
                if not self._collision_free(current_state.x, current_state.y, next_state.x, next_state.y, static_occupancy_map):
                    continue

                cell_idx = self._state_to_grid_indices(next_state.x, next_state.y, static_occupancy_map.shape)
                if cell_idx is None:
                    continue

                cell_cost = cost_norm[cell_idx[0], cell_idx[1]]
                step_cost = self._hybrid_step * (1.0 + w_cost * cell_cost) + self._turn_penalty * abs(steer)
                tentative = current_cost + step_cost

                if tentative < g_cost.get(next_key, np.inf):
                    g_cost[next_key] = tentative
                    parent[next_key] = current_key
                    states[next_key] = next_state
                    f_score = tentative + heuristic(next_state)
                    heapq.heappush(open_heap, (f_score, next_key))

        return [], np.inf


