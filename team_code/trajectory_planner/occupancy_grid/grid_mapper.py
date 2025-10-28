import carla
import cv2
import heapq
import numpy as np
import open3d as o3d

from math import sqrt
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Set, Dict, List, Any

from team_code.trajectory_planner.occupancy_grid.sensor_data_processor import WorldMappingProcessor, BEVGrid
from team_code.scene_descriptor.camera_interface import CameraInterface

@dataclass(frozen=True, slots=True)
class Maps:
    occupancy_map       : np.ndarray
    instance_map        : np.ndarray
    semantic_map        : np.ndarray
    base_cost_map       : np.ndarray
    dynamic_cost_map    : np.ndarray
    total_cost_map      : np.ndarray

class GridMapper:
    def __init__(
        self,
        ego_vehicle : carla.Vehicle,
        x_min : float = -30.0,
        x_max : float = 70.0,
        y_min : float = -30.0,
        y_max : float = 30.0,
        grid_resolution: float = 0.5,
    ):
        self.ego_vehicle = ego_vehicle
        self.grid = BEVGrid(
            x_min = x_min,
            x_max = x_max,
            y_min = y_min,
            y_max = y_max,
            resolution = grid_resolution
        )

        # Initialize processor with world coordinates
        vehicle_transform = self.ego_vehicle.get_transform()
        world_origin = (vehicle_transform.location.x, vehicle_transform.location.y)

        self.processor = WorldMappingProcessor(
            self.ego_vehicle,
            grid_size=(self.grid.H, self.grid.W),
            grid_resolution=grid_resolution,
            world_origin=world_origin
        )

        # Sensors
        self.cameras : Dict[str, CameraInterface] = {}
        self.camera_sensor : Optional[CameraInterface] = None
        self.lidar_sensor : Optional[carla.Sensor] = None

        self.point_list = o3d.geometry.PointCloud()

        # TESTING PAYLOAD
        self.payload :Dict[str, Any] = {
            'static_occupancy_map' : None,
            'static_cost_map' : None,
            'start_point_ego' : None,
            'goal_point_ego' : None
        }

    def setup_sensors(
        self,
        cameras : List[Tuple[str, carla.Sensor]],
        lidar : carla.Sensor
    ) -> None:
        for tag, camera_obj in cameras:
            self.cameras[tag] = CameraInterface(camera_obj)

        self.lidar_sensor = lidar

    def set_camera_observations(
        self,
        images: List[Tuple[str, np.ndarray]]
    ) -> None:
        """
        Update camera observations with new image data.

        Args:
            images: List of tuples containing (camera_tag, image_data)
        """
        for tag, image in images:
            if tag in self.cameras:
                self.cameras[tag].set_image(image)
            else:
                print(f"Warning: Camera tag '{tag}' not found in registered cameras.")

    def update_maps(
        self,
        lidar_data: Dict,
        route_points_world: np.ndarray
    ) -> Maps:
        ego_tf = self.ego_vehicle.get_transform()

        # Get base cost maps
        road_cost_map, static_cost_map, dynamic_cost_map = self.processor.get_base_cost_maps(
            lidar_data,
            ego_tf,
            self.grid
        )
        occupancy_map = np.ones_like(road_cost_map, dtype=np.uint8)
        occupancy_map[road_cost_map > 0] = 0
        occupancy_map[static_cost_map > 0] = 0
        # occupancy_map[dynamic_cost_map > 0] = 0

        # Augment static cost map with route information
        static_cost_map_route = self.processor.augment_static_cost_map(
            road_cost_map,
            static_cost_map,
            self.ego_vehicle,
            route_points_world,
            self.grid
        )

        # total_cost_map = np.maximum(static_cost_map_route, dynamic_cost_map)

        # cost_map = self.processor.get_cost_map(
        #     occupancy_map
        # )

        # occupancy_map = self.processor.get_occupancy_map_multiview_camera(
        #     self.cameras,
        #     self.lidar_sensor,
        #     lidar_data,
        #     ego_tf,
        #     self.grid,
        # )

        # TESTING PAYLOAD
        self.payload['static_occupancy_map'] = occupancy_map
        self.payload['static_cost_map'] = static_cost_map_route

        return Maps(
            occupancy_map=occupancy_map,
            instance_map=None,
            semantic_map=None,
            base_cost_map=None,
            dynamic_cost_map=None,
            total_cost_map=static_cost_map_route
        )

    def generate_astar_path(
        self,
        occupancy_map: np.ndarray,           # 0=occupied, 1=free
        cost_map: np.ndarray,             # higher = more costly (e.g., 0..255)
        start_world : np.ndarray,
        goal_world : np.ndarray,
    ):
        # Get ego transformation matrices
        ego_tf = self.ego_vehicle.get_transform()
        T_world_wrt_ego = np.array(ego_tf.get_inverse_matrix(), dtype=np.float32) # [4x4]
        T_ego_wrt_world = np.array(ego_tf.get_matrix(), dtype=np.float32) # [4x4]

        # Convert world points to ego frame
        world_pts_3d = np.vstack([start_world, goal_world])

        # print(f'world_pts_3d: {world_pts_3d}')
        # print(f'world_pts_3d shape: {world_pts_3d.shape}')

        ones = np.ones((world_pts_3d.shape[0], 1), dtype=np.float32)
        world_pts_4d = np.hstack([world_pts_3d, ones])

        # print(f'world_pts_4d: {world_pts_4d}')
        # print(f'world_pts_4d shape: {world_pts_4d.shape}')

        # Convert ego points to grid frame
        ego_pts_2d = (world_pts_4d @ T_world_wrt_ego.T)[:, :2]

        # TESTING PAYLOAD
        self.payload['start_point_ego'] = ego_pts_2d[0, :]
        self.payload['goal_point_ego'] = ego_pts_2d[1, :]

        # print(f'ego points: {ego_pts_2d}')
        # print(f'ego points shape: {ego_pts_2d.shape}')

        i, j = self.grid.world_to_grid(ego_pts_2d[:, 0], ego_pts_2d[:, 1])

        start_point_grid = (i[0], j[0])
        goal_point_grid = (i[1], j[1])

        # print(f'start_point_grid: {start_point_grid}')
        # print(f'goal_point_grid: {goal_point_grid}')

        astar_path_grid_list, path_cost = self.astar_with_cost(
            occupancy_map,
            cost_map,
            start_point_grid,
            goal_point_grid
        )

        # Convert grid path points to ego
        astar_path_grid = np.array(astar_path_grid_list, dtype=np.float32)
        # print(f'astar_path_grid shape: {astar_path_grid.shape}')

        x, y = self.grid.grid_to_world(astar_path_grid[:, 0], astar_path_grid[:, 1])
        ones = np.ones(x.shape[0], dtype=np.float32)
        path_ego_4d = np.stack([x, y, ones, ones], axis=-1)
        # print(f'path_ego_4d shape: {path_ego_4d.shape}')

        # Convert ego points to world frame
        path_world_3d = (path_ego_4d @ T_ego_wrt_world.T)[:, :3]

        # print(f'path_world_3d shape: {path_world_3d.shape}')

        return astar_path_grid_list, path_world_3d

    def astar_with_cost(
        self,
        occupancy_map: np.ndarray,           # 0=occupied, 1=free
        cost_map: np.ndarray,             # higher = more costly (e.g., 0..255)
        start_rc: tuple,                 # (r, c)
        goal_rc: tuple,                  # (r, c)
        w_cost: float = 1.0,             # weight for costmap influence
        allow_diag: bool = True,         # 8-connected if True, 4-connected if False
    ):
        """
        Returns:
            path_rc: list[(r,c)] from start to goal (inclusive). [] if no path.
            total_cost: g-score of the goal (np.inf if not found)
        """
        H, W = occupancy_map.shape

        def inb(r, c): return (0 <= r < H) and (0 <= c < W)
        def free(r, c): return occupancy_map[r, c] > 0

        sr, sc = start_rc
        gr, gc = goal_rc
        # if not (inb(sr, sc) and inb(gr, gc) and free(sr, sc) and free(gr, gc)):
        #     print(f'NOT FREE OR IN BOUNDS, EXITING')
        #     return [], np.inf

        if not (inb(sr, sc) and inb(gr, gc)):
            print(f'NOT IN BOUNDS, EXITING')
            return [], np.inf

        # Movement model
        if allow_diag:
            moves = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
                    (-1,-1,sqrt(2)),(-1,1,sqrt(2)),(1,-1,sqrt(2)),(1,1,sqrt(2))]
        else:
            moves = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0)]

        # Heuristic: octile distance (admissible with min step cost = 1)
        def h_octile(r, c, gr, gc):
            dr, dc = abs(gr - r), abs(gc - c)
            D, D2 = 1.0, sqrt(2)
            return D * (dr + dc) + (D2 - 2 * D) * min(dr, dc)

        # Normalize costmap to [0,1] once
        cmax = float(cost_map.max()) if cost_map.size else 1.0
        if cmax <= 0: cmax = 1.0
        cost_norm = cost_map.astype(np.float32) / cmax

        # Arrays for speed
        N = H * W
        to_idx = lambda r, c: r * W + c
        to_rc  = lambda i: (i // W, i % W)

        g = np.full(N, np.inf, dtype=np.float32)
        parent = np.full(N, -1, dtype=np.int32)

        s_idx = to_idx(sr, sc)
        g[s_idx] = 0.0

        open_heap = []
        heapq.heappush(open_heap, (h_octile(sr, sc, gr, gc), s_idx))

        closed = np.zeros(N, dtype=bool)

        while open_heap:
            f_curr, i = heapq.heappop(open_heap)
            if closed[i]:
                continue
            closed[i] = True

            r, c = to_rc(i)
            if (r, c) == (gr, gc):
                # reconstruct
                path = []
                cur = i
                while cur != -1:
                    pr, pc = to_rc(cur)
                    path.append((pr, pc))
                    cur = parent[cur]
                path.reverse()
                return path, float(g[i])

            g_i = g[i]

            for dr, dc, dist in moves:
                nr, nc = r + dr, c + dc
                # if not inb(nr, nc) or not free(nr, nc):
                #     continue
                if not inb(nr, nc):
                    continue

                # No diagonal corner-cutting
                if allow_diag and dr != 0 and dc != 0:
                    if not (free(r, nc) and free(nr, c)):
                        continue

                j = to_idx(nr, nc)

                # Step cost = geometric distance * (1 + w_cost * normalized cost at neighbor)
                step_cost = dist * (1.0 + w_cost * cost_norm[nr, nc])
                tentative_g = g_i + step_cost

                if tentative_g < g[j]:
                    g[j] = tentative_g
                    parent[j] = i
                    f = tentative_g + h_octile(nr, nc, gr, gc)
                    heapq.heappush(open_heap, (f, j))

        # No path
        print(f'NO PATH')
        return [], np.inf
