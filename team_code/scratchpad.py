import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

from trajectory_planner.occupancy_grid.planners.hybrid_astar import HybridAStar
from local_planner.lateral.algos import AStar
from local_planner.lateral.config_specs import LatGridSpec, LatAlgoSpec

def smooth_path(
    path,
    weight_data: float = 0.4,
    weight_smooth: float = 0.4,
    tolerance: float = 1e-4,
    max_iterations: int = 1000,
) -> np.ndarray:
    """
    Smooth a 2D path using gradient descent.

    Args:
        path: Iterable of (row, col) or (y, x) points. Shape (N, 2).
        weight_data: Keeps the smoothed path close to the original points.
        weight_smooth: Controls how strongly we smooth (coupling between neighbours).
        tolerance: Convergence threshold (sum of absolute changes per iteration).
        max_iterations: Safety cap on iterations.

    Returns:
        new_path: Smoothed path, same shape as input, dtype float64.
    """
    path = np.asarray(path, dtype=np.float64)
    new_path = np.copy(path)

    change = tolerance
    it = 0
    while change >= tolerance and it < max_iterations:
        change = 0.0
        # Skip endpoints so start/goal remain fixed
        for i in range(1, len(path) - 1):
            for j in range(2):  # y, x (or row, col)
                aux = new_path[i, j]
                new_path[i, j] += (
                    weight_data * (path[i, j] - new_path[i, j])
                    + weight_smooth * (
                        new_path[i - 1, j]
                        + new_path[i + 1, j]
                        - 2.0 * new_path[i, j]
                    )
                )
                change += abs(aux - new_path[i, j])
        it += 1

    return new_path

file_name = '/home/carla/carla_garage/path_planning_data/AccidentTwoWays_26_0/payload_0300.pkl'
with open(file_name, 'rb') as f:
    payload = pickle.load(f)

static_occupancy_map = payload['static_occupancy_map']
static_cost_map = payload['static_cost_map']
dynamic_cost_map = payload['dynamic_cost_map']
start_point_ego = payload['start_point_ego']
goal_point_ego = payload['goal_point_ego']
forecasted_bbs = payload['forecasted_bbs']
ego_forecasted_bbs = payload['ego_forecasted_bbs']

route_subset = payload['route_subset']
path_world_3d = payload['path_world_3d']
route_pts = payload['route_points']
start_idx = payload['start_idx']
goal_idx = payload['goal_idx']

# goal_point_ego[0] -= 24.0

# grid = LatGridSpec()

# start_time = time.time()

# planner = AStar(algo_spec=LatAlgoSpec())

# start_node = grid.world_to_grid(route_subset[0][0], route_subset[0][1])
# goal_node = grid.world_to_grid(route_subset[-1][0], route_subset[-1][1])
# print(f'goal_point_ego: {(route_subset[-1][0], route_subset[-1][1])}')

# path_grid, _ = planner.run(
#     occupancy_map=static_occupancy_map,
#     cost_map=static_cost_map,
#     start_node=start_node,
#     goal_node=goal_node
# )

# # planner = HybridAStar(grid)
# # path_grid = planner.plan_path(
# #     static_occupancy_map,
# #     static_cost_map,
# #     start_point_ego,
# #     goal_point_ego
# # )

# print("--- Execution Time: %s seconds ---" % (time.time() - start_time))

# # Convert path from (row, col) to (x=col, y=row) for plotting
# pts = np.asarray([(c_, r_) for (r_, c_) in path_grid], dtype=np.int32)

# print(f'pts_shape: {pts.shape}')

# # Thickness (similar meaning to your OpenCV code)
# thickness = max(1, int(round((0.4 / 0.5) * 1.0)))
# line_width = thickness
# marker_size = thickness * 3

# # Create Matplotlib figure with three subplots
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ax_occ, ax_static, ax_dynamic = axes

# # 1. Occupancy map (grayscale)
# im_occ = ax_occ.imshow(static_occupancy_map, cmap='gray', origin='upper')
# if pts.size > 0:
#     ax_occ.plot(pts[:, 0], pts[:, 1], 'g-', linewidth=line_width)
#     ax_occ.plot(pts[0, 0], pts[0, 1], 'ro', markersize=marker_size, label='Start')
#     ax_occ.plot(pts[-1, 0], pts[-1, 1], 'bo', markersize=marker_size, label='Goal')
# ax_occ.set_title("Occupancy Map")
# ax_occ.set_axis_off()
# ax_occ.set_aspect('equal')

# # 2. Static cost map (turbo colormap)
# im_static = ax_static.imshow(static_cost_map, cmap='turbo', origin='upper')
# if pts.size > 0:
#     ax_static.plot(pts[:, 0], pts[:, 1], 'k-', linewidth=line_width)
#     ax_static.plot(pts[0, 0], pts[0, 1], 'wo', markersize=marker_size)
#     ax_static.plot(pts[-1, 0], pts[-1, 1], 'wo', markersize=marker_size)

# ax_static.set_title("Static Cost Map")
# ax_static.set_axis_off()
# ax_static.set_aspect('equal')

# # # 3. Dynamic cost map (turbo colormap)
# im_dynamic = ax_dynamic.imshow(dynamic_cost_map, cmap='turbo', origin='upper')
# if pts.size > 0:
#     ax_dynamic.plot(pts[:, 0], pts[:, 1], 'k-', linewidth=line_width)
#     ax_dynamic.plot(pts[0, 0], pts[0, 1], 'wo', markersize=marker_size)
#     ax_dynamic.plot(pts[-1, 0], pts[-1, 1], 'wo', markersize=marker_size)
# ax_dynamic.set_title("Dynamic Cost Map")
# ax_dynamic.set_axis_off()
# ax_dynamic.set_aspect('equal')

# # Add a shared colorbar for the cost maps (static + dynamic)
# # You can change the mappable here if you want the scale from dynamic instead.
# cbar = fig.colorbar(im_static)
# cbar.set_label("Cost")

# # Optional: legend for the first subplot (start/goal)
# ax_occ.legend(loc='lower right')

# # Smooth in grid coordinates (row, col)
# smoothed_path = smooth_path(path_grid)

# # For plotting, convert (row, col) -> (x, y) = (col, row)
# pts_raw = np.asarray([(c_, r_) for (r_, c_) in path_grid], dtype=np.float64)
# pts_smooth = np.asarray([(c_, r_) for (r_, c_) in smoothed_path], dtype=np.float64)

# # Example: overlay both on your Matplotlib plots
# thickness = max(1, int(round((0.4 / 0.5) * 1.0)))
# line_width = thickness
# marker_size = (thickness * 4) ** 2  # scatter uses points^2

# # Suppose `axes[0]`, `axes[1]`, `axes[2]` are as in your previous Matplotlib code
# # Raw path (e.g., dotted)
# if pts_raw.size > 0:
#     axes[0].plot(
#         pts_raw[:, 0], pts_raw[:, 1],
#         linewidth=line_width, color='black', linestyle='--', label='Raw Path'
#     )

#     # Smoothed path (solid)
#     axes[0].plot(
#         pts_smooth[:, 0], pts_smooth[:, 1],
#         linewidth=line_width + 1, color='yellow', label='Smoothed Path'
#     )

#     # Start / goal markers from smoothed path
#     axes[0].scatter(pts_smooth[0, 0], pts_smooth[0, 1], s=marker_size, c='red', label='Start')
#     axes[0].scatter(pts_smooth[-1, 0], pts_smooth[-1, 1], s=marker_size, c='blue', label='Goal')

# axes[1].legend(loc='lower right')
# plt.tight_layout()
# plt.show()

print(f'path_world_3d shape: {path_world_3d.shape}')
print(f'route_subset shape: {route_subset.shape}')

print(f'start_idx: {start_idx}')
print(f'route_pts[start]: {route_pts[start_idx]}')
print(f'route_subset start: {route_subset[0]}')
print(f'path_world_3d start: {path_world_3d[0]}')

print(f'goal idx: {goal_idx}')
print(f'route_pts[goal]: {route_pts[goal_idx]}')
print(f'route_subset[goal]: {route_subset[-1]}')
print(f'path_world[goal]: {path_world_3d[-1]}')

route_len = goal_idx - start_idx + 1

desired = np.arange(route_len)
orig = np.arange(path_world_3d.shape[0])

x_interp = np.interp(desired, orig, path_world_3d[:, 0])
y_interp = np.interp(desired, orig, path_world_3d[:, 1])
z_interp = np.interp(desired, orig, path_world_3d[:, 2])

route_interp = np.column_stack([x_interp, y_interp, z_interp])

print(f'route_interp shape: {route_interp.shape}')
print(f'global route subset shape: {route_pts[start_idx:goal_idx + 1].shape}')

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection in some setups

# def subsample_points(points: np.ndarray, max_points: int = 200):
#     """
#     Subsample a (N, 3) array to at most max_points, preserving endpoints.
#     Returns (subsampled_points, indices_used).
#     """
#     n = points.shape[0]
#     if n <= max_points:
#         idx = np.arange(n)
#     else:
#         idx = np.linspace(0, n - 1, max_points, dtype=int)
#     return points[idx], idx

# # Assuming you already have:
# # path_world_3d, route_subset, route_interp, route_pts, start_idx, goal_idx

# # Global route segment for comparison
# global_route_segment = route_pts[start_idx:goal_idx + 1]

# # Subsample all four
# max_points = 50
# path_sub, path_idx = subsample_points(path_world_3d, max_points=25)
# subset_sub, subset_idx = subsample_points(route_subset, max_points=50)
# interp_sub, interp_idx = subsample_points(route_interp, max_points=50)
# global_sub, global_idx = subsample_points(global_route_segment, max_points=50)

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Original path from your planner
# ax.plot(
#     path_sub[:, 0], path_sub[:, 1], path_sub[:, 2],
#     marker='o', linestyle='-', label='path_world_3d (local path)'
# )

# # Route subset you started from
# ax.plot(
#     subset_sub[:, 0], subset_sub[:, 1], subset_sub[:, 2],
#     marker='^', linestyle='--', label='route_subset (local route segment)'
# )

# # Interpolated path aligned to route indices
# ax.plot(
#     interp_sub[:, 0], interp_sub[:, 1], interp_sub[:, 2],
#     marker='x', linestyle='-', label='route_interp (interpolated path)'
# )

# # Global route (for context)
# ax.plot(
#     global_sub[:, 0], global_sub[:, 1], global_sub[:, 2],
#     marker='.', linestyle=':', label='global route segment'
# )

# # Mark start and goal on the interpolated path
# ax.scatter(
#     route_interp[0, 0], route_interp[0, 1], route_interp[0, 2],
#     s=80, marker='s', label='interp start'
# )
# ax.scatter(
#     route_interp[-1, 0], route_interp[-1, 1], route_interp[-1, 2],
#     s=80, marker='D', label='interp goal'
# )

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Subsampled Paths: world path vs route subset vs interpolated path')
# ax.legend()
# ax.view_init(elev=30, azim=120)  # adjust view angle if needed

# plt.figure(figsize=(8, 8))
# plt.plot(path_sub[:, 0], path_sub[:, 1], marker='o', linestyle='-', label='path_world_3d')
# plt.plot(subset_sub[:, 0], subset_sub[:, 1], marker='^', linestyle='--', label='route_subset')
# plt.plot(interp_sub[:, 0], interp_sub[:, 1], marker='x', linestyle='-', label='route_interp')
# plt.plot(global_sub[:, 0], global_sub[:, 1], marker='.', linestyle=':', label='global route segment')
# plt.scatter(route_interp[0, 0], route_interp[0, 1], s=80, marker='s', label='interp start')
# plt.scatter(route_interp[-1, 0], route_interp[-1, 1], s=80, marker='D', label='interp goal')
# plt.axis('equal')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Top-down view (X-Y)')
# plt.legend()

# plt.tight_layout()
# plt.show()

import numpy as np

def cumulative_arclength(points: np.ndarray) -> np.ndarray:
    """
    points: (N, 3)
    returns: (N,) cumulative distance along the polyline starting at 0
    """
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    return s

def resample_path_to_match_route(path_world_3d: np.ndarray,
                                 route_subset: np.ndarray) -> np.ndarray:
    """
    Resample path_world_3d so that:
      - It has the same number of points as route_subset
      - Points are spaced according to the arc-length profile of route_subset
    """
    # Arc-length coordinates for both
    s_path  = cumulative_arclength(path_world_3d)      # shape (P,)
    s_route = cumulative_arclength(route_subset)       # shape (R,)

    # Normalized param along route_subset: 0..1
    t_route = s_route / s_route[-1] if s_route[-1] > 0 else s_route

    # Target arc-lengths along the path_world_3d
    s_target = t_route * s_path[-1] if s_path[-1] > 0 else t_route

    # Interpolate x, y, z by arc length
    x_interp = np.interp(s_target, s_path, path_world_3d[:, 0])
    y_interp = np.interp(s_target, s_path, path_world_3d[:, 1])
    z_interp = np.interp(s_target, s_path, path_world_3d[:, 2])

    route_interp = np.column_stack([x_interp, y_interp, z_interp])
    return route_interp

import matplotlib.pyplot as plt

route_subset = route_pts[start_idx:goal_idx + 1]
route_interp = resample_path_to_match_route(path_world_3d, route_subset)

plt.figure(figsize=(8, 8))
plt.plot(route_subset[:, 0], route_subset[:, 1],
         marker='.', linestyle=':', label='original route_subset (dense)')
plt.plot(path_world_3d[:, 0], path_world_3d[:, 1],
         marker='o', linestyle='-', label='path_world_3d (coarse planner)')
plt.plot(route_interp[:, 0], route_interp[:, 1],
         marker='x', linestyle='-', label='route_interp (resampled path)')

plt.axis('equal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Local route vs planner path vs resampled path')
plt.legend()
plt.tight_layout()
plt.show()

