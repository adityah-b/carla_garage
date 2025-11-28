import cv2
import time
import carla
import pickle
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from matplotlib.patches import Polygon

from actor_prediction.geometric_utils import GeometricUtils
from trajectory_planner.occupancy_grid.planners.hybrid_astar import HybridAStar

file_name = '/home/carla/carla_garage/path_planning_data/AccidentTwoWays_26_0/payload_0030.pkl'
with open(file_name,'rb') as f:
    payload = pickle.load(f)

static_occupancy_map = payload['static_occupancy_map']
static_cost_map = payload['static_cost_map']
dynamic_cost_map = payload['dynamic_cost_map']
forecasted_bbs = payload['forecasted_bbs']
ego_forecasted_bbs = payload['ego_forecasted_bbs']
route_subset = payload['route_subset']

occ_img = (static_occupancy_map * 255).astype(np.uint8)
occ_bgr = cv2.cvtColor(occ_img, cv2.COLOR_GRAY2BGR)
heat_img = cv2.applyColorMap(static_cost_map, cv2.COLORMAP_TURBO)
dynamic_heat_img = cv2.applyColorMap(dynamic_cost_map, cv2.COLORMAP_TURBO)


ego_key = list(ego_forecasted_bbs.keys())[-1]
ego_forecasted_bbs_arr = ego_forecasted_bbs[ego_key]

ego_forecasted_bbs_carla : List[carla.BoundingBox] = []
for bb_arr in ego_forecasted_bbs_arr:
    loc = carla.Location(
        x=float(bb_arr[0]),
        y=float(bb_arr[1]),
        z=float(bb_arr[2])
    )
    ext = carla.Vector3D(
        x=float(bb_arr[3]),
        y=float(bb_arr[4]),
        z=float(bb_arr[5])
    )
    bounding_box = carla.BoundingBox(loc, ext)
    bounding_box.rotation.yaw = float(bb_arr[6])

    ego_forecasted_bbs_carla.append(bounding_box)

DELTA_TIMESTEP = 1.0 / 20.0
TIMES = np.arange(len(ego_forecasted_bbs_carla), dtype=np.float32) * DELTA_TIMESTEP

@dataclass
class CollisionInterval:
    start_idx : int
    end_idx : int

    start_bbox_a : carla.BoundingBox
    start_bbox_b : carla.BoundingBox

    end_bbox_a : carla.BoundingBox
    end_bbox_b : carla.BoundingBox

def _close_collision_gaps(
    collision_mask : List[bool],
    max_collision_gap : int
) -> List[bool]:
    N = len(collision_mask)
    out = collision_mask[:]
    if max_collision_gap <= 0:
        return out

    i = 0
    while i < N:
        # Skip true regions
        if out[i]:
            i += 1
            continue

        # Start of false region
        j = i
        while j < N and not out[j]:
            j += 1

        # [i, j) is a false run. Fill it iff:
        #   - length <= max_collision_gap
        #   - True before i and True at j
        in_bounds = i > 0 and j < N

        # Fill gaps
        if (j - i) <= max_collision_gap and in_bounds and out[i - 1] and out[j]:
            for k in range(i, j):
                out[k] = True

        # Iterate from next interval
        i = j

    return out

def find_collision_intervals(
    bounding_boxes_a : List[carla.BoundingBox],
    bounding_boxes_b : List[carla.BoundingBox],
    *,
    max_delta_time_offset : int = 1,
    max_delta_time_collision_gap : int = 2,
) -> List[CollisionInterval]:
    N = len(bounding_boxes_a)
    if N == 0:
        return []

    # Get per-frame collisions
    collided = [False] * N
    for i in range(N):
        # Get candidate indices in B list around i
        j0 = max(0, i - max_delta_time_offset)
        j1 = min(N - 1, i + max_delta_time_offset)
        hit = False
        bb_a = bounding_boxes_a[i]

        # Test small time window region
        for j in range(j0, j1):
            if GeometricUtils.check_obb_intersection(bb_a, bounding_boxes_b[j]):
                hit = True
                break
        collided[i] = hit

    # Close collision gaps to avoid oscillating collision booleans
    closed = _close_collision_gaps(collided, max_collision_gap=max_delta_time_collision_gap)

    # Extract all collision intervals
    collision_intervals : List[CollisionInterval] = []
    i = 0
    while i < N:
        # Skip non-collision regions
        if not closed[i]:
            i += 1
            continue

        # Start of collision run (i.e. true region)
        start_idx = i

        # Find end of collision interval
        while i < N and closed[i]:
            i += 1

        end_idx = i - 1
        interval = CollisionInterval(
            start_idx,
            end_idx,
            bounding_boxes_a[start_idx],
            bounding_boxes_b[start_idx],
            bounding_boxes_a[end_idx],
            bounding_boxes_b[end_idx]
        )
        collision_intervals.append(interval)

    return collision_intervals

# -----------------------------------------------------------------------------
# Route preprocessing
# -----------------------------------------------------------------------------
def precompute_route_geometry(
    route_points: np.ndarray
) -> Dict[str, np.ndarray]:
    assert route_points.ndim == 2 and route_points.shape[1] == 3, f"route_points must be (N, 3), got {route_points.shape}"
    route_xy = np.asarray(route_points[:, :2], dtype=np.float32)

    seg_starts = route_xy[:-1]                       # A_i
    seg_vecs   = route_xy[1:] - route_xy[:-1]        # v_i = B_i - A_i
    seg_len    = np.linalg.norm(seg_vecs, axis=1)    # |v_i|
    seg_len2   = seg_len**2                          # |v_i|^2
    s_at_vtx   = np.concatenate(([0.0], np.cumsum(seg_len))).astype(np.float32)

    return {
        "route_xy": route_xy,
        "A": seg_starts,
        "v": seg_vecs,
        "L": seg_len,
        "L2": seg_len2,
        "s_at_vtx": s_at_vtx
    }

# -----------------------------------------------------------------------------
# Vectorized projection of many points using a small segment window
# -----------------------------------------------------------------------------
def project_points_to_route_s(
    points_xy: np.ndarray,
    geom: dict,
    window: int = 2
) -> np.ndarray:
    """
    For each point:
      1) find nearest route vertex (argmin ||P - R_i||),
      2) test the segments in [i-window, i+window],
      3) pick the segment whose projection is closest to P,
      4) convert that projection to arclength s.
    """
    P = np.asarray(points_xy, dtype=np.float32)          # (M,2)
    route = geom["route_xy"]                             # (N,2)
    A, v, L, L2, s_at = geom["A"], geom["v"], geom["L"], geom["L2"], geom["s_at_vtx"]
    nseg = L.shape[0]
    assert nseg > 0, "Route must have ≥1 segment."

    # (1) nearest vertex per point
    d2 = np.sum((P[:, None, :] - route[None, :, :])**2, axis=2)  # (M,N)
    i_v = np.argmin(d2, axis=1)                                  # (M,)

    # (2) candidate segments around that vertex
    K = 2*int(window) + 1
    start = np.clip(i_v - window, 0, nseg - 1)[:, None]          # (M,1)
    cand  = np.clip(start + np.arange(K)[None, :], 0, nseg - 1)  # (M,K)

    A_c = A[cand]                  # (M,K,2)
    v_c = v[cand]                  # (M,K,2)
    L_c = L[cand]                  # (M,K)
    L2_c = L2[cand]                # (M,K)

    # (3) orthogonal projection onto each candidate segment
    #     u* = ((P-A)·v) / (v·v), u ∈ [0,1]
    w    = P[:, None, :] - A_c                                     # (M,K,2)
    num  = np.sum(w * v_c, axis=2)                                 # (M,K)  (w·v)
    denom = np.where(L2_c > 1e-12, L2_c, 1.0)                      # avoid /0
    u    = np.clip(num / denom, 0.0, 1.0)                          # (M,K)

    proj = A_c + u[..., None] * v_c                                # (M,K,2)
    dist2 = np.sum((P[:, None, :] - proj)**2, axis=2)              # (M,K)
    dist2 = np.where(L2_c > 1e-12, dist2, dist2 + 1e12)            # reject degenerate segs

    kbest = np.argmin(dist2, axis=1)                               # (M,)
    row   = np.arange(P.shape[0])
    u_b   = u[row, kbest]                                          # (M,)
    i_s   = cand[row, kbest]                                       # (M,)

    # (4) projection → arclength: s = s(i) + u*|v_i|
    s = s_at[i_s] + u_b * L[i_s]
    return s.astype(np.float32)

# -----------------------------------------------------------------------------
# OBB corners (vectorized) → world XY
# -----------------------------------------------------------------------------
def obb_corners_world_xy(rows7: np.ndarray, use_ue4_yaw: bool = True) -> np.ndarray:
    """
    rows7: (T,7) with [x,y,z, ex,ey,ez, yaw_deg] (CARLA extents: ex,ey are half-dims).
    UE/CARLA yaw is clockwise-positive; negate to convert to standard CCW math.
    Returns: (T,4,2) world corners ordered [+ex,+ey], [+ex,-ey], [-ex,-ey], [-ex,+ey].
    """
    rows7 = np.asarray(rows7, dtype=np.float32)
    assert rows7.ndim == 2 and rows7.shape[1] == 7

    cx, cy = rows7[:, 0], rows7[:, 1]
    ex, ey = rows7[:, 3], rows7[:, 4]
    yaw = rows7[:, 6]
    if use_ue4_yaw:
        yaw = -yaw

    th = np.deg2rad(yaw).astype(np.float32)
    c, s = np.cos(th), np.sin(th)
    R = np.stack([np.stack([c, -s], axis=1), np.stack([s, c], axis=1)], axis=1)  # (T,2,2)

    local = np.stack([
        np.stack([+ex, +ey], axis=1),
        np.stack([+ex, -ey], axis=1),
        np.stack([-ex, -ey], axis=1),
        np.stack([-ex, +ey], axis=1),
    ], axis=1).astype(np.float32)

    world = local @ np.transpose(R, (0, 2, 1))                        # rotate
    world += np.stack([cx, cy], axis=1)[:, None, :]                   # translate
    return world  # (T,4,2)

# -----------------------------------------------------------------------------
# Convert vehicles → s-interval bands
# -----------------------------------------------------------------------------


def compute_vehicle_s_bands(
    predictions: Dict[int, np.ndarray],
    route_points: np.ndarray,
    *,
    window: int = 2,
    use_ue4_yaw: bool = True,
    all_conditions: Optional[Dict[int, Tuple[str, str, float]]] = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    For each vehicle (T,7), project its four OBB corners per time to s, then
    take min/max across corners → s_min(t), s_max(t).

    `all_conditions` lets callers describe how to extend and weight the costs
    for specific actors using (action_type, actor_type, importance):
        - action_type ∈ {"yield_for", "watch_out_for"}
        - actor_type  ∈ {"vehicle", "cyclist", "ped"}
        - importance  ∈ [0.0, 1.0]

    "yield_for" emphasises waiting until the actor passes the conflict region
    by padding the time dimension primarily after the collision interval.
    "watch_out_for" builds symmetric time padding and a larger spatial buffer
    around the collision band. Actor type and importance scale the collision
    cost and dilation to reflect perceived risk.
    """
    geom = precompute_route_geometry(route_points)
    out: Dict[int, Dict[str, np.ndarray]] = {}

    condition_lookup: Dict[int, Tuple[str, str, float]] = {}
    if all_conditions:
        for actor_id, condition in all_conditions.items():
            if not isinstance(condition, tuple) or len(condition) != 3:
                raise ValueError(
                    "Conditions must be tuples of (action_type, actor_type, importance)."
                )
            action_type, actor_type, importance = condition
            if action_type not in {"yield_for", "watch_out_for"}:
                raise ValueError(
                    f"Unsupported action_type '{action_type}' for actor {actor_id}."
                )
            if actor_type not in {"vehicle", "cyclist", "ped"}:
                raise ValueError(
                    f"Unsupported actor_type '{actor_type}' for actor {actor_id}."
                )
            if not (0.0 <= float(importance) <= 1.0):
                raise ValueError(
                    f"Importance must be in [0.0, 1.0], got {importance} for actor {actor_id}."
                )
            condition_lookup[int(actor_id)] = (
                action_type,
                actor_type,
                float(importance),
            )

    # Number of discrete time indices to extend before/after the true
    # collision interval for each action type.
    time_extension_steps = {
        # "yield_for": (2, 10),      # wait longer after the collision
        "yield_for": (20, 0),      # wait longer after the collision
        "watch_out_for": (5, 5),   # symmetric caution window
    }

    # Spatial dilation (in metres) based on actor type and action flavour.
    base_space_dilation = 0.5
    space_dilation_by_actor_type = {"vehicle": 0.6, "cyclist": 0.9, "ped": 1.1}
    space_dilation_by_action = {"yield_for": 0.3, "watch_out_for": 0.8}

    # Collision cost scaling based on actor type + importance.
    collision_cost_scale = {"vehicle": 1.0, "cyclist": 1.2, "ped": 1.4}

    all_collision_intervals : Dict[int, List[CollisionInterval]] = {}

    # Find collision intervals
    for vid, arr in predictions.items():
        bb_list_carla : List[carla.BoundingBox] = [
            carla.BoundingBox(
                carla.Location(
                    x=float(bb_arr[0]),
                    y=float(bb_arr[1]),
                    z=float(bb_arr[2])
                ),
                carla.Vector3D(
                    x=float(bb_arr[3]),
                    y=float(bb_arr[4]),
                    z=float(bb_arr[5])
                )
            ) for bb_arr in arr
        ]

        collision_intervals = find_collision_intervals(
            ego_forecasted_bbs_carla,
            bb_list_carla
        )
        if len(collision_intervals) > 0:
            all_collision_intervals[vid] = collision_intervals

    # Project collision intervals
    for vid, arr in predictions.items():
        if vid not in all_collision_intervals:
            continue

        collision_interval = all_collision_intervals[vid][0]
        start_idx = collision_interval.start_idx
        end_idx = collision_interval.end_idx

        action_type: Optional[str] = None
        actor_type: Optional[str] = None
        importance: float = 0.0
        if vid in condition_lookup:
            action_type, actor_type, importance = condition_lookup[vid]

        # Determine time padding driven by the action type (if any)
        pad_before = pad_after = 0
        if action_type is not None:
            pad_before, pad_after = time_extension_steps[action_type]

        extended_start = max(0, start_idx - pad_before)
        extended_end = min(arr.shape[0] - 1, end_idx + pad_after)

        extended_indices = np.arange(extended_start, extended_end + 1, dtype=np.int32)
        arr_extended = arr[extended_indices]
        T_ext = arr_extended.shape[0]

        corners_ext = obb_corners_world_xy(arr_extended, use_ue4_yaw=use_ue4_yaw)
        s_all_ext = project_points_to_route_s(
            corners_ext.reshape(-1, 2), geom, window=window
        ).reshape(T_ext, 4)

        s_min_ext = np.min(s_all_ext, axis=1)
        s_max_ext = np.max(s_all_ext, axis=1)

        collision_mask = (extended_indices >= start_idx) & (extended_indices <= end_idx)
        time_distance_steps = np.zeros_like(extended_indices, dtype=np.float32)
        if collision_mask.any():
            # Distance in discrete steps to the nearest point inside the collision interval
            before_dist = np.maximum(0, start_idx - extended_indices)
            after_dist = np.maximum(0, extended_indices - end_idx)
            time_distance_steps = np.maximum(before_dist, after_dist).astype(np.float32)

        action_dilation = space_dilation_by_action.get(action_type, 0.0)
        actor_dilation = space_dilation_by_actor_type.get(actor_type, 0.0)
        space_dilation = base_space_dilation + action_dilation + actor_dilation
        cost_scale = collision_cost_scale.get(actor_type, 1.0) * (1.0 + 0.75 * importance)

        out[vid] = {
            "s_min": s_min_ext.astype(np.float32),
            "s_max": s_max_ext.astype(np.float32),
            "start_idx": int(extended_start),
            "end_idx": int(extended_end),
            "collision_start_idx": int(start_idx),
            "collision_end_idx": int(end_idx),
            "collision_mask": collision_mask.astype(np.bool_),
            "time_indices": extended_indices.astype(np.int32),
            "time_distance_steps": time_distance_steps.astype(np.float32),
            "time_extension_steps_before": int(pad_before),
            "time_extension_steps_after": int(pad_after),
            "space_dilation": float(space_dilation),
            "action_type": action_type,
            "actor_type": actor_type,
            "importance": float(importance),
            "cost_scale": float(cost_scale),
        }

    return out

# -----------------------------------------------------------------------------
# Build a single convex polygon on s–T from (t, s_min/s_max)
# -----------------------------------------------------------------------------
def st_polygons_from_bands(
    t: np.ndarray,
    s_min: np.ndarray,
    s_max: np.ndarray
) -> List[Polygon]:
    """
    Construct a convex s–T occupancy polygon from a contiguous collision window.

    Inputs
    ------
    t      : (K,) strictly (or non-decreasing) time samples for the collision span
    s_min  : (K,) lower longitudinal bound per time (meters)
    s_max  : (K,) upper longitudinal bound per time (meters)

    Returns
    -------
    [Polygon] containing one matplotlib.patches.Polygon that traces:
        (t, s_max) forward in time, then (t, s_min) backward, closed.
    If K == 0, returns [].
    If K == 1, returns a very thin time "sliver" polygon centered at t[0].
    """
    # --- shape & dtype hygiene ---
    t = np.asarray(t, dtype=np.float32).reshape(-1)
    smin = np.asarray(s_min, dtype=np.float32).reshape(-1)
    smax = np.asarray(s_max, dtype=np.float32).reshape(-1)

    print(f't: {t}, s_min: {s_min}, s_max: {s_max}')

    assert t.size == smin.size == smax.size, "t, s_min, s_max must have same length"
    K = t.size
    if K == 0:
        return []

    # Ensure smin <= smax pointwise (defensive against upstream ordering errors)
    lower = np.minimum(smin, smax)
    upper = np.maximum(smin, smax)

    # Ensure time is non-decreasing; sort if necessary (should already be true)
    if K > 1 and not np.all(np.diff(t) >= 0):
        order = np.argsort(t)
        t = t[order]
        lower = lower[order]
        upper = upper[order]

    # Handle degenerate 1-sample case by creating a tiny rectangle in time
    if K == 1:
        # Heuristic sliver: 0.001 s wide (adjust if your time units differ)
        dt = np.float32(1e-3)
        tt = np.array([t[0] - dt, t[0] + dt], dtype=np.float32)
        low = np.array([lower[0], lower[0]], dtype=np.float32)
        up  = np.array([upper[0], upper[0]], dtype=np.float32)
    else:
        tt = t
        low = lower
        up  = upper

    # Build convex polygon: top edge goes forward in time along s_max,
    # bottom edge returns backward in time along s_min.
    top = np.column_stack([tt, up])                # (K or 2, 2)
    bot = np.column_stack([tt[::-1], low[::-1]])   # (K or 2, 2)
    coords = np.vstack([top, bot])                 # (2K or 4, 2)

    poly = Polygon(coords, closed=True)
    return [poly]

# -----------------------------------------------------------------------------
# Plotting: obstacle polygons only (no ego line)
# -----------------------------------------------------------------------------
def plot_st_obstacles_only(t: np.ndarray,
                           bands_by_vid: Dict[int, Dict[str, np.ndarray]],
                           *,
                           face_alpha: float = 0.10,
                           edge_width: float = 1.2
                          ) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (vid, bands) in enumerate(bands_by_vid.items()):
        smin, smax = bands["s_min"], bands["s_max"]
        start_idx, end_idx = bands['start_idx'], bands['end_idx']
        polys = st_polygons_from_bands(t[start_idx:end_idx+1], smin, smax)
        if not polys:
            continue

        color = 'red'
        hatch = "////"
        label = f"veh {vid}"

        for j, poly in enumerate(polys):
            # use a light face + hatch + colored edge (matches your example vibe)
            poly.set_facecolor(color if color else "none")
            poly.set_alpha(face_alpha if color else 0.0)
            poly.set_edgecolor(color if color else "black")
            poly.set_linewidth(edge_width)
            poly.set_hatch(hatch)
            ax.add_patch(poly)

            # Place a label near the first polygon of this vehicle
            if j == 0 and label:
                # choose midpoint of first run for label position
                xy = poly.get_path().vertices.mean(axis=0)
                ax.text(xy[0], xy[1], label, ha="left", va="center")

    # Full time span we want visible regardless of polygon extents
    t = np.asarray(t, dtype=np.float32).ravel()
    t_min = float(np.min(t))
    t_max = float(np.max(t))
    if not np.isfinite(t_min) or not np.isfinite(t_max):
        t_min, t_max = 0.0, 1.0
    if t_max <= t_min:
        # Degenerate case (e.g., single time sample) – give a tiny sliver
        eps = 1e-3
        t_min, t_max = t_min - eps, t_max + eps

    ax.set_xlabel("time (s)")
    ax.set_ylabel("s (m)")
    ax.set_title("s–T obstacle occupancy (convex polygons)")
    ax.grid(True, linewidth=0.6, alpha=0.5)

    # ax.autoscale(enable=True, axis='y', tight=True)
    ax.set_ylim(0.0, 40.0)
    ax.set_xlim(t_min, t_max)
    ax.margins(x=0)  # optional: remove extra padding on x

    # ax.autoscale(enable=True)
    return fig, ax

def _rotation_matrix_from_yaw(yaw_deg: float, use_ue4_yaw: bool = True) -> np.ndarray:
    """
    Returns a 2x2 rotation matrix for yaw (degrees).

    In Unreal/UE4 (used by CARLA), yaw increases clockwise when viewed from above.
    To plot in standard math coordinates (CCW +x to the right, +y up), we flip the sign.
    Set use_ue4_yaw=False if your yaw is already CCW.
    """
    theta = np.deg2rad(-yaw_deg if use_ue4_yaw else yaw_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]], dtype=np.float32)

def _box_corners_xy(x: float, y: float, ex: float, ey: float, yaw_deg: float, use_ue4_yaw: bool = True) -> np.ndarray:
    """
    Compute the 4 XY corners of an oriented box (top-down view), starting at front-left and going clockwise.
    Returns array of shape (4, 2).
    """
    # Local (vehicle) frame corners with extents (half-length ex, half-width ey)
    # Order: FL, FR, RR, RL (front-left first), good for drawing a nice arrow later.
    local = np.array([[ ex,  ey],
                      [ ex, -ey],
                      [-ex, -ey],
                      [-ex,  ey]], dtype=np.float32)

    R = _rotation_matrix_from_yaw(yaw_deg, use_ue4_yaw=use_ue4_yaw)
    world = (local @ R.T) + np.array([x, y], dtype=np.float32)
    return world

def _subsample_forecast(arr: np.ndarray,
                        *,
                        pred_hz: int = 20,
                        plot_hz: int = 4,
                        stride: int | None = None,
                        max_horizon_sec: float | None = None,
                        include_last: bool = True) -> np.ndarray:
    """
    Returns a subsampled (T',7) version of arr.
      - By default: from 20Hz -> 4Hz (every 5th frame).
      - Optionally cap the horizon in seconds before subsampling.
      - Ensures the final row is kept (for labeling) if include_last is True.
    """
    assert arr.ndim == 2 and arr.shape[1] == 7, f"Expected (T,7), got {arr.shape}"

    # Optionally trim horizon first
    if max_horizon_sec is not None:
        max_T = int(np.ceil(max_horizon_sec * pred_hz))
        arr = arr[:max_T]

    # Decide stride
    s = int(stride) if stride is not None else max(1, int(np.ceil(pred_hz / plot_hz)))
    arr_sub = arr[::s]

    # Make sure the last prediction is present (useful for ID label and final arrow)
    if include_last and arr_sub.shape[0] > 0 and not np.array_equal(arr_sub[-1], arr[-1]):
        arr_sub = np.vstack([arr_sub, arr[-1][None, :]])

    return arr_sub

def _subsample_route(route_xy: np.ndarray,
                     *,
                     pred_hz: int = 20,
                     plot_hz: int = 4,
                     stride: Optional[int] = None,
                     max_horizon_sec: Optional[float] = None,
                     include_last: bool = True) -> np.ndarray:
    """Subsample (T,2) route points to match plotting cadence."""
    route_xy = np.asarray(route_xy[:, :2])
    # assert route_xy.ndim == 2 and route_xy.shape[1] == 2, f"Expected (T,2), got {route_xy.shape}"

    T = route_xy.shape[0]
    if max_horizon_sec is not None:
        max_T = int(np.ceil(max_horizon_sec * pred_hz))
        route_xy = route_xy[:max_T]

    s = int(stride) if stride is not None else max(1, int(np.ceil(pred_hz / plot_hz)))
    sub = route_xy[::s]

    if include_last and sub.shape[0] > 0 and not np.array_equal(sub[-1], route_xy[-1]):
        sub = np.vstack([sub, route_xy[-1][None, :]])

    # Drop any non-finite rows just in case
    finite_mask = np.isfinite(sub).all(axis=1)
    return sub[finite_mask]

# ---------- main plot ----------
def plot_forecasted_bounding_boxes(
    predictions: Dict[int, np.ndarray],
    ax: Optional[plt.Axes] = None,
    *,
    use_ue4_yaw: bool = True,
    draw_heading: bool = True,
    draw_center_traj: bool = True,
    alpha_start: float = 0.85,
    alpha_end: float = 0.20,
    # subsampling for forecasts
    pred_hz: int = 20,
    plot_hz: int = 4,
    stride: Optional[int] = None,
    max_horizon_sec: Optional[float] = None,
    # NEW: route points
    route_xy: Optional[np.ndarray] = None,
    route_match_subsampling: bool = True,
    route_marker_size: float = 18.0,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot forecasted BEV bounding boxes and (optionally) a route as green points.

    Args:
        predictions: {vehicle_id: (T,7)} where row = [x,y,z, ex,ey,ez, yaw_deg]
                     ex/ey are half-length/half-width.
        route_xy: (N,2) array of XY route points to draw in green (points).
        route_match_subsampling: If True, subsample route with the same cadence as forecasts.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Forecasted vehicle bounding boxes (top-down)")

    all_xy_list = []

    # --- plot route (green points) first so vehicles draw on top ---
    if route_xy is not None:
        if route_match_subsampling:
            route_plot = _subsample_route(
                route_xy, pred_hz=pred_hz, plot_hz=plot_hz,
                stride=stride, max_horizon_sec=max_horizon_sec, include_last=True
            )
        else:
            route_plot = np.asarray(route_xy)
            assert route_plot.ndim == 2 and route_plot.shape[1] == 2, f"route_xy must be (N,2), got {route_plot.shape}"
            finite_mask = np.isfinite(route_plot).all(axis=1)
            route_plot = route_plot[finite_mask]

        if route_plot.size:
            ax.scatter(route_plot[:, 0], route_plot[:, 1],
                       s=route_marker_size, c="green", marker="o",
                       linewidths=0.0, alpha=0.95, label="route")
            all_xy_list.append(route_plot)

    # --- plot vehicles ---
    for vid, arr in predictions.items():
        arr = np.asarray(arr)
        if arr.ndim != 2 or arr.shape[1] != 7:
            raise ValueError(f"Vehicle {vid}: expected array of shape (T, 7), got {arr.shape}.")

        arr = _subsample_forecast(arr, pred_hz=pred_hz, plot_hz=plot_hz, stride=stride,
                                  max_horizon_sec=max_horizon_sec, include_last=True)

        all_xy_list.append(arr[:, :2])

        # draw center trajectory first to lock color
        line_color = None
        if draw_center_traj:
            line_handle, = ax.plot(arr[:, 0], arr[:, 1], linestyle='--', linewidth=1.25)
            line_color = line_handle.get_color()
        else:
            line_handle, = ax.plot([], [])
            line_color = line_handle.get_color()
            line_handle.remove()

        T = arr.shape[0]
        alphas = np.linspace(alpha_start, alpha_end, T) if T > 1 else np.array([alpha_start], dtype=float)

        for t, (x, y, z, ex, ey, ez, yaw_deg) in enumerate(arr):
            corners = _box_corners_xy(x, y, ex, ey, yaw_deg, use_ue4_yaw=use_ue4_yaw)
            poly = Polygon(corners, closed=True, facecolor=line_color, edgecolor=line_color,
                           alpha=float(alphas[t]), linewidth=1.25)
            ax.add_patch(poly)

            if draw_heading:
                R = _rotation_matrix_from_yaw(yaw_deg, use_ue4_yaw=use_ue4_yaw)
                front_world = (np.array([ex, 0.0], dtype=np.float32) @ R.T) + np.array([x, y], dtype=np.float32)
                ax.plot([x, front_world[0]], [y, front_world[1]], linewidth=1.4, color=line_color)

        ax.text(arr[-1, 0], arr[-1, 1], str(vid), fontsize=9, ha='center', va='center')

    # --- autoscale ---
    if all_xy_list:
        all_xy = np.concatenate(all_xy_list, axis=0)
        pad = 2.0
        xmin, ymin = all_xy.min(axis=0) - pad
        xmax, ymax = all_xy.max(axis=0) + pad
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    # optional legend if route is present
    if route_xy is not None:
        ax.legend(loc="best", frameon=True)

    # ax.invert_xaxis()
    return fig, ax

def build_st_from_route(
    route_xy: np.ndarray,
    *,
    pred_hz: int = 20,
    timestamps: Optional[np.ndarray] = None,
    eps_stationary: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert (N,2) XY route samples into s–t (and finite-diff v,a).

    Args:
        route_xy: (N,2) array of x,y points sampled in time order.
        pred_hz:  sampling rate (ignored if timestamps provided).
        timestamps: optional (N,) array of seconds for each point.
        eps_stationary: distances below this are treated as zero (de-jitter).

    Returns:
        t: (N,) time vector in seconds.
        s: (N,) cumulative arclength in meters (s[0] = 0).
        v: (N,) speed estimate in m/s.
        a: (N,) acceleration estimate in m/s^2.
    """
    pts = np.asarray(route_xy, dtype=np.float32)
    assert pts.ndim == 2 and pts.shape[1] == 2, f"route_xy must be (N,2), got {pts.shape}"

    # Build time vector
    if timestamps is None:
        dt = 1.0 / float(pred_hz)
        t = np.arange(len(pts), dtype=np.float32) * dt
    else:
        t = np.asarray(timestamps, dtype=np.float32)
        assert t.shape == (len(pts),)

    # Cumulative arclength along the polyline
    dxy = np.diff(pts, axis=0)
    step = np.linalg.norm(dxy, axis=1)
    step[step < eps_stationary] = 0.0   # kill jitter
    s = np.concatenate(([0.0], np.cumsum(step))).astype(np.float32)

    # Derivatives (speeds/accels); gradient handles non-uniform t if provided
    v = np.gradient(s, t, edge_order=2).astype(np.float32)
    a = np.gradient(v, t, edge_order=2).astype(np.float32)
    return t, s, v, a

def plot_st(t: np.ndarray, s: np.ndarray, *, title: str = "s–t (route)") -> None:
    """Simple s vs t plot."""
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(t, s, linewidth=1.8)
    ax.set_xlabel("time t (s)")
    ax.set_ylabel("arc length s (m)")
    ax.set_title(title)
    ax.grid(True, linewidth=0.6, alpha=0.5)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Grids
# ──────────────────────────────────────────────────────────────────────────────
def make_grids(
    s_max: float,
    ds: float,
    T_horizon: float,
    dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform s–T grids."""
    t = np.arange(0.0, T_horizon + 1e-9, dt, dtype=np.float32)     # (K,)
    s = np.arange(0.0, s_max + 1e-9, ds, dtype=np.float32)       # (S,)
    return t, s

# ──────────────────────────────────────────────────────────────────────────────
# Costmap: dynamic obstacles (bands) → floating grid
# bands_by_vid: {vid: {"s_min": (K,), "s_max": (K,), ...}}, aligned to t
# ──────────────────────────────────────────────────────────────────────────────
def build_costmap_from_bands(
    t: np.ndarray,
    s: np.ndarray,
    bands_by_vid: Dict[int, Dict[str, np.ndarray]],
    *,
    collision_cost: float = 1.0,
    yield_decay_exponent: float = 0.9,
    watch_out_decay_exponent: float = 1.2,
    default_decay_exponent: float = 1.1,
) -> np.ndarray:
    """
    Rasterize collision bands (per-vehicle) into a floating cost map cost[k,i].

    Inputs
    ------
    t : (K,) float32
        Global time grid for the planner. Indices 0..K-1.
    s : (S,) float32
        Global arclength grid for the planner. Indices 0..S-1.
    bands_by_vid : dict
        For each vehicle id:
          {
            "s_min": (L,) float32,
            "s_max": (L,) float32,
            "start_idx": int,
            "end_idx": int,
            "collision_start_idx": int,
            "collision_end_idx": int,
            "collision_mask": (L,) bool,
            "time_indices": (L,) int,
            "time_distance_steps": (L,) float32,
            "time_extension_steps_before": int,
            "time_extension_steps_after": int,
            "space_dilation": float,
            "action_type": Optional[str],
            "actor_type": Optional[str],
            "importance": float,
            "cost_scale": float,
          }
        where L = end_idx - start_idx + 1 and arrays are aligned to that subrange.

    Output
    ------
    cost : (K, S) float32
        Non-negative costs for each (t_k, s_i) cell. Occupied cells receive the
        highest cost, while extension regions receive a decaying penalty.

    Notes
    -----
    - This function assumes the collision arrays are already trimmed to the interval
      [start_idx, end_idx] (as `compute_vehicle_s_bands` does).
    - If a band’s interval partially falls outside the planner’s [0, K-1] time window,
      it is clipped gracefully.
    - The main collision interval is treated as a hard cost while the
      action-driven time extension and spatial dilation smoothly decay towards
      zero cost at their limits.
    - Decay exponents are derived from the action type to shape how quickly the
      time-dependent decay falls off on either side of the collision window.
    """
    t = t.reshape(-1)
    s = s.reshape(-1)
    K, S = t.size, s.size

    ds = float(s[1] - s[0]) if S > 1 else 1.0

    cost = np.zeros((K, S), dtype=np.float32)
    if not bands_by_vid:
        return cost

    decay_exponent_lookup = {
        "yield_for": float(yield_decay_exponent),
        "watch_out_for": float(watch_out_decay_exponent),
    }

    for vid, band in bands_by_vid.items():
        smin_local = band["s_min"]
        smax_local = band["s_max"]

        k_start = int(band["start_idx"])  # inclusive
        k_end = int(band["end_idx"])    # inclusive

        # Length checks and early outs
        L = k_end - k_start + 1
        if L <= 0 or smin_local.size == 0 or smax_local.size == 0:
            continue
        if smin_local.size != L or smax_local.size != L:
            raise ValueError(
                f"band {vid}: expected len(s_min)==len(s_max)==end_idx-start_idx+1 "
                f"({L}), got {smin_local.size} and {smax_local.size}"
            )

        # Clip the interval to the planner horizon [0, K-1]
        k0 = max(0, k_start)
        k1 = min(K - 1, k_end)
        if k1 < k0:
            continue  # completely outside

        span = (k1 - k0 + 1)             # how many rows to paint
        offset = k0 - k_start
        idx_hi = offset + span

        smin_slice = smin_local[offset:idx_hi]
        smax_slice = smax_local[offset:idx_hi]

        # Convert s-intervals to column index ranges on the global S grid
        j_lo = np.floor(smin_slice / ds)
        j_hi = np.floor(smax_slice / ds)

        j_lo = np.clip(j_lo, 0, S - 1).astype(np.int32)
        j_hi = np.clip(j_hi, 0, S - 1).astype(np.int32)

        action_type = band.get("action_type")
        collision_mask = band.get("collision_mask")
        time_indices = band.get("time_indices")
        time_distances = band.get("time_distance_steps")
        space_dilation = float(band.get("space_dilation", 0.0))
        collision_start_idx = int(band.get("collision_start_idx", k_start))
        collision_end_idx = int(band.get("collision_end_idx", k_end))
        ext_before = int(band.get("time_extension_steps_before", 0))
        ext_after = int(band.get("time_extension_steps_after", 0))
        cost_scale = float(band.get("cost_scale", 1.0))
        importance = float(band.get("importance", 0.0))
        actor_type = band.get("actor_type")
        decay_exponent = decay_exponent_lookup.get(action_type, float(default_decay_exponent))
        decay_exponent = float(max(decay_exponent - 0.2 * importance, 1e-6))
        actual_before = max(0, min(ext_before, collision_start_idx - k0))
        actual_after = max(0, min(ext_after, k1 - collision_end_idx))
        if actual_before > 0:
            before_norm_denom = float(max(actual_before - 1, 1))
        else:
            before_norm_denom = 1.0
        if actual_after > 0:
            after_norm_denom = float(max(actual_after - 1, 1))
        else:
            after_norm_denom = 1.0

        for r in range(span):
            lo = int(j_lo[r])
            hi = int(j_hi[r])
            if hi < lo:
                continue

            s_values = s[lo:hi + 1]
            s_min_val = float(smin_slice[r])
            s_max_val = float(smax_slice[r])

            s_min_dil = s_min_val - space_dilation
            s_max_dil = s_max_val + space_dilation

            j_lo_dil = int(np.clip(np.floor(s_min_dil / ds), 0, S - 1))
            j_hi_dil = int(np.clip(np.floor(s_max_dil / ds), 0, S - 1))
            if j_hi_dil < j_lo_dil:
                continue

            # Update with dilated indices if they extend beyond the initial
            # discretisation.
            if j_lo_dil < lo or j_hi_dil > hi:
                lo = min(lo, j_lo_dil)
                hi = max(hi, j_hi_dil)
                s_values = s[lo:hi + 1]

            # Collision vs extension handling
            is_collision = True
            if collision_mask is not None and time_indices is not None:
                idx = offset + r
                if 0 <= idx < collision_mask.size:
                    is_collision = bool(collision_mask[idx])

            if is_collision:
                base_cost = collision_cost * cost_scale
            else:
                if action_type is None or time_distances is None or time_indices is None:
                    continue

                idx = offset + r
                if not (0 <= idx < time_distances.size and 0 <= idx < time_indices.size):
                    continue

                time_distance = float(time_distances[idx])
                if time_distance <= 0.0:
                    base_cost = collision_cost * cost_scale
                else:
                    time_idx_val = int(time_indices[idx])
                    if time_idx_val < collision_start_idx:
                        denom = actual_before
                        norm_denom = before_norm_denom
                    else:
                        denom = actual_after
                        norm_denom = after_norm_denom

                    if denom <= 0:
                        continue

                    if denom == 1:
                        # Only a single extension step → treat as immediate drop to zero.
                        norm = float(time_distance >= 1.0)
                    else:
                        norm = (time_distance - 1.0) / norm_denom

                    norm = np.clip(norm, 0.0, 1.0)
                    time_decay = (1.0 - norm) ** decay_exponent
                    if time_decay <= 0.0:
                        continue
                    base_cost = collision_cost * cost_scale * time_decay

            # Apply spatial decay outside the true collision band.
            s_min_collision = s_min_val
            s_max_collision = s_max_val

            s_values = s[lo:hi + 1]
            inside_collision = (s_values >= s_min_collision) & (s_values <= s_max_collision)

            if space_dilation <= 1e-6:
                # No dilation → all cells inside collision band take the base cost.
                row_costs = np.where(inside_collision, base_cost, 0.0)
            else:
                lower_dist = np.clip(s_min_collision - s_values, a_min=0.0, a_max=None)
                upper_dist = np.clip(s_values - s_max_collision, a_min=0.0, a_max=None)
                dist = lower_dist + upper_dist
                space_decay = 1.0 - np.clip(dist / space_dilation, 0.0, 1.0)
                row_costs = base_cost * space_decay
                row_costs = np.where(inside_collision, base_cost, row_costs)

            if np.all(row_costs <= 0.0):
                continue

            row_slice = slice(lo, hi + 1)
            cost[k0 + r, row_slice] = np.maximum(cost[k0 + r, row_slice], row_costs.astype(np.float32))

    return cost

# ──────────────────────────────────────────────────────────────────────────────
# Envelope (v_max_diag): prune vertices above s = (s_max/T) * t
# ──────────────────────────────────────────────────────────────────────────────
def envelope_mask_under_diagonal(
    t: np.ndarray,
    s: np.ndarray,
    slope : float,
) -> np.ndarray:
    """valid[k,i] is True iff (k,i) lies under/on the diagonal from (0,0)→(T,s_max)."""
    K, S = t.size, s.size
    s_env = slope * t.astype(np.float64)                           # (K,)
    i_max = np.floor((s_env) / float(s[1]-s[0])).astype(int)
    i_max = np.clip(i_max, -1, S - 1)
    i_grid = np.arange(S)[None, :]
    valid = (i_grid <= i_max[:, None]) & (i_max[:, None] >= 0)

    return valid

# ──────────────────────────────────────────────────────────────────────────────
# Inflation → costmap
#   Inflate occupancy by (r_s meters, r_t seconds) and assign a cost.
#   Result cost[k,i] ∈ [0, cost_obstacle], 0 outside inflated region.
# ──────────────────────────────────────────────────────────────────────────────
def inflate_to_costmap(
    occ: np.ndarray,
    ds: float, dt: float,
    r_s: float, r_t: float,
    cost_obstacle: float = 1.0
) -> np.ndarray:
    """
    Simple rectangular (Chebyshev) inflation without scipy:
    mark any cell within r_s in s and r_t in t of an occupied cell.
    """
    K, S = occ.shape
    rs = int(np.ceil(r_s / ds))
    rt = int(np.ceil(r_t / dt))
    if rs <= 0 and rt <= 0:
        return occ.astype(np.float32) * cost_obstacle

    infl = occ.copy()
    # Separate 2D dilation via sliding window (bounded slicing; no wrap)
    for dk in range(-rt, rt + 1):
        k_src_lo = max(0, -dk)
        k_src_hi = min(K, K - dk)
        k_dst_lo = max(0, dk)
        k_dst_hi = min(K, K + dk)
        # For each time-shift, do s-interval expansions
        slice_src = infl[k_src_lo:k_src_hi]
        tgt = infl[k_dst_lo:k_dst_hi]
        # Expand in s with a rolling OR over width 2*rs+1
        if rs > 0:
            # Left-to-right and right-to-left cumulative OR trick
            left = slice_src.copy()
            for _ in range(rs):
                left[:, 1:] |= left[:, :-1]
            right = slice_src.copy()
            for _ in range(rs):
                right[:, :-1] |= right[:, 1:]
            expanded = left | right
        else:
            expanded = slice_src
        # Combine into destination time slice
        infl[k_dst_lo:k_dst_hi] |= expanded

    return infl.astype(np.float32) * cost_obstacle

# ──────────────────────────────────────────────────────────────────────────────
# Neighbour generator with v/accel/jerk constraints
#   State carries previous Δi to enforce accel/jerk during expansion.
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
from typing import List, Tuple

import numpy as np
from typing import List, Tuple

def findNeighbours(
    occupancy_grid: np.ndarray,
    s_idx: int,
    t_idx: int,
    v_cur: float,
    a_cur: float,
    ds_res: float,
    dt_res: float,
    v_max: float,
    a_max: float,
    j_max: float
) -> List[Tuple[int, int, float, float]]:
    """
    Vectorized successor generator for a layered s–T grid.
    Monotonicity (no backwards s) is enforced; waiting (Δs=0) is allowed if feasible.

    Returns a list of (next_s_idx, next_t_idx, next_v, next_a).

    Math (discrete to continuous mapping)
    -------------------------------------
      di = next_s_idx - s_idx                         (integer step in s-index, di >= 0)
      ds = di * ds_res
      dt = dt_res
      next_v = ds / dt
      next_a = (next_v - v_cur) / dt
      next_j = (next_a - a_cur) / dt

    Constraints → bounds on di
    --------------------------
      1) speed:     next_v ≤ v_max
            ⇒ di ≤ floor((v_max * dt)/ds_res)
      2) accel:     |next_a| ≤ a_max
            ⇒ di ∈ [ ceil((v_cur*dt - a_max*dt^2)/ds_res),
                     floor((v_cur*dt + a_max*dt^2)/ds_res) ]
      3) jerk:      |next_j| ≤ j_max  ⇔  |next_a - a_cur| ≤ j_max * dt
            ⇒ next_a ∈ [a_cur - j_max*dt, a_cur + j_max*dt]
            ⇒ di ∈ [ ceil((v_cur*dt + (a_cur - j_max*dt)*dt^2)/ds_res),
                     floor((v_cur*dt + (a_cur + j_max*dt)*dt^2)/ds_res) ]

    We intersect all di-ranges, clamp to grid, then mask by occupancy.

    Notes
    -----
    - If you disallow waiting, just set di_min = max(di_min, 1).
    - The occupancy grid can already include the velocity-envelope mask.
    """
    neighbours: List[Tuple[int, int, float, float]] = []

    # grid extents
    T, S = occupancy_grid.shape

    # next_t_idx = t_idx + 1
    next_t_idx = t_idx + int(dt_res / (t[1] - t[0]))
    if next_t_idx >= T:
        return neighbours

    ds = float(ds_res)
    dt = float(dt_res)

    # --- convert constraints to integer bounds on di ---
    # Speed bound
    di_max_speed = int(np.floor((v_max * dt) / ds + 1e-12))

    # Acceleration bound → di range
    #   di = (v_cur*dt + next_a*dt^2) / ds
    di_min_acc = int(np.ceil(((v_cur * dt) - (a_max * dt * dt)) / ds - 1e-12))
    di_max_acc = int(np.floor(((v_cur * dt) + (a_max * dt * dt)) / ds + 1e-12))

    # Jerk bound → next_a ∈ [a_cur - j_max*dt, a_cur + j_max*dt] → di range
    a_lo = a_cur - j_max * dt
    a_hi = a_cur + j_max * dt
    di_min_jerk = int(np.ceil(((v_cur * dt) + (a_lo * dt * dt)) / ds - 1e-12))
    di_max_jerk = int(np.floor(((v_cur * dt) + (a_hi * dt * dt)) / ds + 1e-12))

    # Intersect all ranges, also enforce monotonicity (di >= 0) and grid bound
    di_min = max(0, di_min_acc, di_min_jerk)
    di_max = min(di_max_speed, di_max_acc, di_max_jerk, S - 1 - s_idx)

    if di_min > di_max:
        return neighbours

    # ---- candidate destinations ----
    di_vec  = np.arange(di_min, di_max + 1, dtype=np.int32)       # (N,)
    next_s  = s_idx + di_vec                                      # (N,)
    next_v  = (di_vec.astype(np.float32) * ds) / dt               # slope in m/s  (N,)

    # ---- diagonal sampling inside the slab (one per grid row) ----
    # absolute s at samples: s0 + v_edge * Δt_k, where Δt_k = k*grid_dt
    stride = int(dt_res / (t[1] - t[0]))
    dt_steps = (np.arange(stride + 1, dtype=np.float32) * float(t[1] - t[0]))[:, None]  # (R,1)
    s0 = (s_idx * ds)
    s_diag = s0 + dt_steps * next_v[None, :]                      # (R,N) meters
    # print(f'\nMeasurements\n')
    # print(f'di_vec: {di_vec}')
    # print(f'next_s: {next_s}')
    # print(f'stride: {stride}')
    # print(f'v_edge: {v_edge}')
    # print(f'dt_steps: {dt_steps}')
    # print(f's0: {s0}')
    # print(f's_diag: {s_diag}')

    # station indices via floor, clamped
    i_diag = np.floor(s_diag / ds).astype(np.int32)               # (R,N)
    i_diag = np.clip(i_diag, 0, S - 1)

    i_diag = np.vstack([i_diag, next_s])

    # Row indices: t_idx + k for k in [0, 1, ..., stride + 1]
    rows = t_idx + np.arange(stride + 2, dtype=np.int32)          # (R,)
    rows = np.clip(rows, 0, T - 1)  # Clamp to valid time indices

    # print(f'rows: {rows}')
    # print(f'i_diag shape: {i_diag.shape}')
    # print(f'i_diag:\n{i_diag}')

    # Occupancy along the diagonal samples
    hit = occupancy_grid[rows[:, None], i_diag]                   # (R,N)
    keep = ~hit.any(axis=0)                                       # (N,)
    # print(f'hit: {hit}')
    # print(f'keep: {keep}')

    if not np.any(keep):
        return neighbours

    di_vec = di_vec[keep]
    next_s = next_s[keep]

    # next_s = next_s[keep]
    # v_edge = v_edge[keep]

    # # --- candidate next s-indices ---
    # di_vec = np.arange(di_min, di_max + 1, dtype=np.int32)
    # next_s = s_idx + di_vec

    # # --- occupancy pruning (False = free) ---
    # free_mask = ~occupancy_grid[next_t_idx, next_s]
    # if not np.any(free_mask):
    #     return neighbours

    # di_vec = di_vec[free_mask]
    # next_s = next_s[free_mask]

    # --- compute kinematics for the survivors (redundant with bounds, but safe) ---
    # next_v = (di_vec.astype(np.float32) * ds) / dt
    # next_a = (next_v - v_cur) / dt
    next_v = next_v[keep]
    next_a = (next_v - v_cur) / dt

    # assemble results
    neighbours = [(int(sj), next_t_idx, float(vj), float(aj))
                  for sj, vj, aj in zip(next_s, next_v, next_a)]
    return neighbours

import heapq
from typing import List, Tuple, Dict, Optional
import numpy as np

def dijkstra_st(
    occupancy_grid: np.ndarray,   # (K,S) bool; True = occupied (hard)
    s_start_idx: int,
    s_goal_idx: int,
    ds_res: float,
    dt_res: float,
    v0: float,                    # initial velocity at (t=0, s=s_start_idx)
    a0: float = 0.0,              # initial acceleration at (t=0, s=s_start_idx)
    v_max: float = 15.0,
    a_max: float = 4.0,
    j_max: float = 6.0,
    w_vel: float = 1.0,
    w_acc: float = 1.0,
    w_jerk: float = 1.0,
    allow_wait: bool = True,
) -> List[Tuple[int, int, float, float]]:
    """
    Dijkstra on a time-layered s–T grid with dynamics enforced by `findNeighbours`.

    State we store and key by (all integers):
        (t_idx, s_idx, di_prev, ai_prev)
      where di_prev = (s_idx - s_{idx-1}) in indices (0..),
            ai_prev = (di_prev - di_prevprev) in indices (can be negative).
      These let us derive:
            v_cur = di_prev * ds / dt,
            a_cur = ai_prev * ds / dt^2,
            j_next = ((di_next - di_prev - ai_prev) * ds) / dt^3.

    Returns:
        List[(t_idx, s_idx, v, a)] along the optimal path in chronological order,
        or [] if no solution was found.

    Notes:
    - The step cost is evaluated at the *landing* node (next state).
    - If you want soft obstacles, pass a *costmap* instead of a boolean grid and
      replace the last term accordingly (e.g., cost += costmap[next_t, next_s]).
    """
    occ = occupancy_grid.astype(bool)
    K, S = occ.shape
    dt = float(dt_res)
    ds = float(ds_res)

    # --- helpers to go between discrete (di/ai) and continuous (v/a) ---
    def v_from_di(di: int) -> float:
        return (di * ds) / dt

    def a_from_ai(ai: int) -> float:
        return (ai * ds) / (dt * dt)

    # --- initial discrete state (quantize v0, a0 onto the grid) ---
    # di_prev0 ≈ round(v0*dt/ds), ai_prev0 ≈ round(a0*dt^2/ds)
    di_prev0 = int(round((v0 * dt) / ds))
    ai_prev0 = int(round((a0 * dt * dt) / ds))

    # Guard: start cell must be free
    if not (0 <= s_start_idx < S) or occ[0, s_start_idx]:
        return []

    # Priority queue entries: (cost_so_far, tie_breaker, state_key)
    # state_key = (t_idx, s_idx, di_prev, ai_prev)
    pq: List[Tuple[float, int, Tuple[int, int, int, int]]] = []
    tie = 0

    start_key = (0, s_start_idx, di_prev0, ai_prev0)
    heapq.heappush(pq, (0.0, tie, start_key))
    tie += 1

    dist: Dict[Tuple[int, int, int, int], float] = {start_key: 0.0}
    parent: Dict[Tuple[int, int, int, int], Optional[Tuple[int, int, int, int]]] = {start_key: None}

    # For reconstruction we also store continuous (v,a) at each node (computed once)
    v_cache: Dict[Tuple[int, int, int, int], float] = {start_key: v_from_di(di_prev0)}
    a_cache: Dict[Tuple[int, int, int, int], float] = {start_key: a_from_ai(ai_prev0)}

    # Main loop
    while pq:
        cost_u, _, key_u = heapq.heappop(pq)
        if cost_u != dist.get(key_u, np.inf):
            continue  # stale

        t_idx, s_idx, di_prev, ai_prev = key_u
        # Goal: any arrival with s >= s_goal_idx
        if s_idx >= s_goal_idx:
            # reconstruct path
            path: List[Tuple[int, int, float, float]] = []
            cur = key_u
            while cur is not None:
                t_c, s_c, di_c, ai_c = cur
                path.append((t_c, s_c, v_from_di(di_c), a_from_ai(ai_c)))
                cur = parent[cur]
            path.reverse()
            return path

        # No more time layers
        if t_idx + 1 >= K:
            continue

        # Build neighbours using your dynamics checker
        v_cur = v_from_di(di_prev)
        a_cur = a_from_ai(ai_prev)

        nbrs = findNeighbours(
            occupancy_grid=occ,
            s_idx=s_idx,
            t_idx=t_idx,
            v_cur=v_cur,
            a_cur=a_cur,
            ds_res=ds,
            dt_res=dt,
            v_max=v_max,
            a_max=a_max,
            j_max=j_max,
        )
        # print(f'nbrs: {nbrs}')

        # Expand
        for next_s_idx, next_t_idx, v_next, a_next in nbrs:
            # derive discrete jumps
            di_next = next_s_idx - s_idx
            ai_next = di_next - di_prev
            # jerk at this landing state (continuous)
            j_next = (a_next - a_cur) / dt

            # step cost evaluated at the landing node (next)
            step_cost = (
                w_vel * (v_next - v_max) ** 2 +
                w_acc * (a_next ** 2) +
                w_jerk * (j_next ** 2) +
                255.0 * float(occ[next_t_idx, next_s_idx])  # zero for hard-occupied grids (already filtered)
            )

            key_v = (next_t_idx, next_s_idx, di_next, ai_next)
            new_cost = cost_u + step_cost

            if new_cost < dist.get(key_v, np.inf):
                dist[key_v] = new_cost
                parent[key_v] = key_u
                v_cache[key_v] = v_next
                a_cache[key_v] = a_next
                heapq.heappush(pq, (new_cost, tie, key_v))
                tie += 1

    # No path
    return []

# ──────────────────────────────────────────────────────────────────────────────
# Quick visualizer: costmap/occupancy + envelope
# ──────────────────────────────────────────────────────────────────────────────
def plot_st_map(t: np.ndarray, s: np.ndarray,
                costmap: np.ndarray, envelope_slope: float,
                title: str = "s–T occupancy/costmap",
                path: Optional[List[Tuple[int, int, float, float]]] = None,
                start_idx: Optional[int] = None,
                goal_idx: Optional[int] = None):
    Tgrid, Sgrid = np.meshgrid(t, s, indexing="ij")  # (K,S)
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.pcolormesh(Tgrid, Sgrid, costmap.astype(float),
                       shading="nearest", cmap="inferno", vmin=0.0, vmax=max(1.0, float(costmap.max())))
    fig.colorbar(im, ax=ax, label="cost")

    # Envelope
    Tfin = float(t[-1])
    ax.plot([0.0, Tfin], [0.0, envelope_slope * Tfin],
            "--", lw=1.5, c="tab:red", label="envelope")

    # Optional: overlay start/goal guides on s-axis (t=0 / any t)
    if start_idx is not None:
        ax.scatter([t[0]], [s[start_idx]], s=60, c="tab:blue",
                   edgecolors="white", zorder=4, label="start")
    if goal_idx is not None:
        # show horizontal goal line for reference
        ax.axhline(s[goal_idx], color="tab:green", lw=1.2, alpha=0.6, label="goal s")

    # Optional: overlay the Dijkstra path
    if path and len(path) > 0:
        t_idx_path = np.array([p[0] for p in path], dtype=int)
        s_idx_path = np.array([p[1] for p in path], dtype=int)
        ax.plot(t[t_idx_path], s[s_idx_path],
                lw=2.5, color="tab:blue", zorder=5, label="planned path")
        ax.scatter(t[t_idx_path], s[s_idx_path],
                   s=18, color="tab:blue", zorder=6)

    ax.set_xlabel("time t (s)")
    ax.set_ylabel("arc length s (m)")
    ax.set_ylim(0.0, 40.0 + 1e-3)
    ax.set_title(title)
    ax.legend(loc="best", frameon=True)
    ax.grid(True, lw=0.6, alpha=0.5)
    return fig, ax

# ------------------------------
# NEW: extract s,v,a,j from path
# ------------------------------
def extract_profiles_from_path(
    path: List[Tuple[int, int, float, float]],
    t: np.ndarray,
    s: np.ndarray,
    dt_res: float
) -> Dict[str, np.ndarray]:
    """
    Convert a Dijkstra path [(t_idx, s_idx, v, a), ...] into time-aligned profiles.

    Returns dict with:
      t: (N,) time
      s: (N,) arclength (m)
      v: (N,) velocity (m/s)
      a: (N,) acceleration (m/s^2)
      j: (N,) jerk (m/s^3), j[0]=0 by convention
    """
    if not path:
        return {"t": np.array([]), "s": np.array([]), "v": np.array([]),
                "a": np.array([]), "j": np.array([])}

    t_idx = np.array([p[0] for p in path], dtype=int)
    s_idx = np.array([p[1] for p in path], dtype=int)
    v     = np.array([p[2] for p in path], dtype=float)
    a     = np.array([p[3] for p in path], dtype=float)

    tt = t[t_idx]
    ss = s[s_idx]

    j = np.zeros_like(a)
    if a.size > 1:
        j[1:] = (a[1:] - a[:-1]) / float(dt_res)

    return {"t": tt, "s": ss, "v": v, "a": a, "j": j}

# ---------------------------------
# NEW: plot s, v, a, j over time
# ---------------------------------
def plot_st_profiles(
    t: np.ndarray,
    s: np.ndarray,
    path: List[Tuple[int, int, float, float]],
    dt_res: float,
    title: str = "Speed profile (s, v, a, j)"
):
    """
    Plot s(t), v(t), a(t), j(t) from a Dijkstra path and RETURN the profiles.

    Returns:
      fig, axs, profiles_dict (with keys 't','s','v','a','j')
    """
    profiles = extract_profiles_from_path(path, t, s, dt_res)
    tt, ss, vv, aa, jj = profiles["t"], profiles["s"], profiles["v"], profiles["a"], profiles["j"]

    fig, axs = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    fig.suptitle(title)

    axs[0].plot(tt, ss, marker="o")
    axs[0].set_ylabel("s (m)")
    axs[0].grid(True, lw=0.6, alpha=0.5)

    axs[1].plot(tt, vv, marker="o")
    axs[1].set_ylabel("v (m/s)")
    axs[1].grid(True, lw=0.6, alpha=0.5)

    axs[2].plot(tt, aa, marker="o")
    axs[2].set_ylabel("a (m/s²)")
    axs[2].grid(True, lw=0.6, alpha=0.5)

    axs[3].plot(tt, jj, marker="o")
    axs[3].set_ylabel("j (m/s³)")
    axs[3].set_xlabel("time t (s)")
    axs[3].grid(True, lw=0.6, alpha=0.5)

    return fig, axs, profiles

# if __name__ == "__main__":
#     slope_env = 20.0  # m/s (your example)

#     # Dynamics limits & weights
#     v0   = 0.0
#     a0   = 0.0
#     vmax = slope_env           # safe choice; can be tighter
#     amax = 25.0                 # m/s^2 (example)
#     jmax = 50.0                # m/s^3 (example)
#     w_vel, w_acc, w_jerk = 1.0, 0.2, 0.05

#     # ---------------- 1) grids ----------------
#     # ds = 0.1
#     # dt = 0.05
#     ds = 1.0
#     dt = 0.25
#     t, s = make_grids(s_max=40.0, ds=ds, T_horizon=5.0, dt=dt)
#     K, S = t.size, s.size
#     ds_res, dt_res = ds, 2 * np.sqrt((2 * ds) / amax).round(2)
#     print(f'dt_res: {dt_res}')

#     # ---------------- 2) dynamic obstacles → cost map ----------------
#     demo_conditions: Dict[int, Tuple[str, str, float]] = {}
#     if forecasted_bbs:
#         demo_ids = list(forecasted_bbs.keys())
#         for idx, actor_id in enumerate(demo_ids):
#             # action_type = "yield_for" if idx == 0 else "watch_out_for"
#             action_type = "yield_for"
#             # actor_type = "vehicle" if idx % 2 == 0 else "ped"
#             actor_type = "vehicle"
#             importance = 0.8 if idx == 0 else 0.4
#             demo_conditions[int(actor_id)] = (action_type, actor_type, importance)

#     bands = compute_vehicle_s_bands(
#         forecasted_bbs,      # {vid: (T,7)}
#         route_subset,        # (N,2)
#         window=2,
#         use_ue4_yaw=False,
#         all_conditions=demo_conditions
#     )
#     # If your bands dict has start/end indices as shown earlier, use the interval-aware
#     # painter; otherwise keep your existing builder.
#     cost_dyn = build_costmap_from_bands(t, s, bands_by_vid=bands)
#     occ_dyn = cost_dyn > 1e-3

#     # ---------------- 3) envelope → valid mask ----------------
#     valid = envelope_mask_under_diagonal(t, s, slope=slope_env)  # your signature uses 'slope='
#     occ_env = ~valid  # treat outside envelope as blocked

#     # ---------------- 4) combine & (optionally) inflate ----------------
#     occ_total = occ_dyn | occ_env  # boolean occupancy used by neighbours/Dijkstra
#     cost_total = np.maximum(cost_dyn, occ_env.astype(np.float32))
#     _fig_dyn, _ax_dyn = plot_st_map(
#         t,
#         s,
#         cost_dyn,
#         envelope_slope=slope_env,
#         title="Dynamic actor costmap (demo)",
#     )
#     # occ_total = occ_env  # boolean occupancy used by neighbours/Dijkstra

#     # ---------------- 5) Dijkstra setup ----------------
#     # Start at smallest free s at t=0 (or just s_idx=0 if you prefer)
#     s_start_idx = 0

#     # Goal: reach highest s (right edge of grid)
#     s_goal_idx = S - 1


#     # ---------------- 6) run Dijkstra ----------------
#     start_time = time.time()
#     path = dijkstra_st(
#         occupancy_grid=cost_total,
#         s_start_idx=s_start_idx,
#         s_goal_idx=s_goal_idx,
#         ds_res=ds_res,
#         dt_res=dt_res,
#         v0=v0,
#         a0=a0,
#         v_max=vmax,
#         a_max=amax,
#         j_max=jmax,
#         w_vel=w_vel,
#         w_acc=w_acc,
#         w_jerk=w_jerk,
#         allow_wait=True
#     )
#     print("--- Execution Time: %s seconds ---" % (time.time() - start_time))

#     if len(path) == 0:
#         print("Dijkstra: no feasible path found.")

#     # ---------------- 7) visualize ----------------
#     fig, ax = plot_st_map(
#         t, s, cost_total, envelope_slope=slope_env,
#         title="s–T costmap with diagonal envelope + Dijkstra path",
#         path=path,
#         start_idx=(path[0][1] if path else None),
#         goal_idx=S - 1
#     )
#     # plt.show()

#     if path:
#         fig2, axs2, prof = plot_st_profiles(
#             t=t, s=s, path=path, dt_res=dt_res,
#             title="Planned longitudinal profiles"
#         )
#         # Access arrays if you need them programmatically:
#         # prof["t"], prof["s"], prof["v"], prof["a"], prof["j"]
#     else:
#         print("No path -> no profiles to plot.")

#     plt.show()

from local_planner.longitudinal.config_specs import *
from local_planner.longitudinal.long_planner import LongPlanner
from actor_prediction.collision_checker import CollisionChecker, CollisionInterval

def get_collision_intervals(
    predictions: Dict[int, np.ndarray],
    route_points: np.ndarray,
    *,
    window: int = 2,
    use_ue4_yaw: bool = True,
    all_conditions: Optional[Dict[int, Tuple[str, str, float]]] = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    For each vehicle (T,7), project its four OBB corners per time to s, then
    take min/max across corners → s_min(t), s_max(t).

    `all_conditions` lets callers describe how to extend and weight the costs
    for specific actors using (action_type, actor_type, importance):
        - action_type ∈ {"yield_for", "watch_out_for"}
        - actor_type  ∈ {"vehicle", "cyclist", "ped"}
        - importance  ∈ [0.0, 1.0]

    "yield_for" emphasises waiting until the actor passes the conflict region
    by padding the time dimension primarily after the collision interval.
    "watch_out_for" builds symmetric time padding and a larger spatial buffer
    around the collision band. Actor type and importance scale the collision
    cost and dilation to reflect perceived risk.
    """
    geom = precompute_route_geometry(route_points)
    out: Dict[int, Dict[str, np.ndarray]] = {}

    condition_lookup: Dict[int, Tuple[str, str, float]] = {}
    if all_conditions:
        for actor_id, condition in all_conditions.items():
            if not isinstance(condition, tuple) or len(condition) != 3:
                raise ValueError(
                    "Conditions must be tuples of (action_type, actor_type, importance)."
                )
            action_type, actor_type, importance = condition
            if action_type not in {"yield_for", "watch_out_for"}:
                raise ValueError(
                    f"Unsupported action_type '{action_type}' for actor {actor_id}."
                )
            if actor_type not in {"vehicle", "cyclist", "ped"}:
                raise ValueError(
                    f"Unsupported actor_type '{actor_type}' for actor {actor_id}."
                )
            if not (0.0 <= float(importance) <= 1.0):
                raise ValueError(
                    f"Importance must be in [0.0, 1.0], got {importance} for actor {actor_id}."
                )
            condition_lookup[int(actor_id)] = (
                action_type,
                actor_type,
                float(importance),
            )

    # Number of discrete time indices to extend before/after the true
    # collision interval for each action type.
    time_extension_steps = {
        # "yield_for": (2, 10),      # wait longer after the collision
        "yield_for": (20, 0),      # wait longer after the collision
        "watch_out_for": (5, 5),   # symmetric caution window
    }

    # Spatial dilation (in metres) based on actor type and action flavour.
    base_space_dilation = 0.5
    space_dilation_by_actor_type = {"vehicle": 0.6, "cyclist": 0.9, "ped": 1.1}
    space_dilation_by_action = {"yield_for": 0.3, "watch_out_for": 0.8}

    # Collision cost scaling based on actor type + importance.
    collision_cost_scale = {"vehicle": 1.0, "cyclist": 1.2, "ped": 1.4}

    all_collision_intervals : Dict[int, List[CollisionInterval]] = {}

    # Find collision intervals
    for vid, arr in predictions.items():
        bb_list_carla : List[carla.BoundingBox] = [
            carla.BoundingBox(
                carla.Location(
                    x=float(bb_arr[0]),
                    y=float(bb_arr[1]),
                    z=float(bb_arr[2])
                ),
                carla.Vector3D(
                    x=float(bb_arr[3]),
                    y=float(bb_arr[4]),
                    z=float(bb_arr[5])
                )
            ) for bb_arr in arr
        ]

        # collision_intervals = find_collision_intervals(
        #     ego_forecasted_bbs_carla,
        #     bb_list_carla
        # )
        collision_intervals = CollisionChecker._find_collision_intervals(
            ego_forecasted_bbs_carla,
            bb_list_carla
        )
        if len(collision_intervals) > 0:
            all_collision_intervals[vid] = collision_intervals

    return all_collision_intervals

if __name__ == "__main__":
    st_algo_spec = STAlgoSpec(
        A_max=24.0,
        J_max=50.0,
        W_vel=1.0,
        W_acc=1.0,
        W_jerk=1.0,
        ds_grid=1.0,
        dt_grid=0.25
    )
    st_grid_spec = STGridSpec(
        S_max=40.0,
        ds=1.0,
        T_max=6.0,
        dt=0.25
    )
    long_planner = LongPlanner(st_grid_spec=st_grid_spec, st_algo_spec=st_algo_spec, algo_name='dijkstra', sim_freq=20)

    demo_conditions: Dict[int, Tuple[str, str, float]] = {}
    if forecasted_bbs:
        demo_ids = list(forecasted_bbs.keys())
        for idx, actor_id in enumerate(demo_ids):
            # action_type = "yield_for" if idx == 0 else "watch_out_for"
            action_type = "yield_for"
            # actor_type = "vehicle" if idx % 2 == 0 else "ped"
            actor_type = "vehicle"
            importance = 0.8 if idx == 0 else 0.4
            demo_conditions[int(actor_id)] = (action_type, actor_type, importance)

    actor_collisions = get_collision_intervals(
        forecasted_bbs,      # {vid: (T,7)}
        route_subset,        # (N,2)
        window=2,
        use_ue4_yaw=False,
        all_conditions=demo_conditions
    )

    long_planner.run_step(
        route_subset,
        0.0,
        13.89,
        actor_collisions=actor_collisions,
        plan_tick_counter=20
    )

# if __name__ == "__main__":
#     N = 40
#     t_demo = np.arange(N) / 20.0
#     # toy route: gentle curve
#     x = 0.6 * t_demo * 20.0
#     y = 0.1 * (t_demo * 20.0) ** 1.1
#     route_xy = np.column_stack([x, y]).astype(np.float32)

#     t, s, v, a = build_st_from_route(route_xy, pred_hz=20)
#     plot_st(t, s)

# if __name__ == "__main__":

#     final_img = np.hstack([occ_bgr, heat_img, dynamic_heat_img])
#     cv2.namedWindow("BirdView Map", cv2.WINDOW_NORMAL)
#     cv2.imshow("BirdView Map", final_img)
#     # cv2.waitKey(10000)
#     cv2.waitKey(0)

#     all_forecasted_bbs = forecasted_bbs | ego_forecasted_bbs
#     fig, ax = plot_forecasted_bounding_boxes(all_forecasted_bbs, use_ue4_yaw=False, route_xy=route_subset)
#     plt.show()

# # -----------------------------------------------------------------------------
# # Demo (creates two polygons: a cut-in ahead, and an obstacle behind)
# # -----------------------------------------------------------------------------
# if __name__ == "__main__":

#     # Compute s-bands (min/max over OBB corners); optional pad_s adds safety margin in s
#     bands = compute_vehicle_s_bands(
#         forecasted_bbs,
#         route_subset,
#         window=2,
#         use_ue4_yaw=False
#     )

#     # Plot polygons only (no ego line)
#     fig, ax = plot_st_obstacles_only(TIMES, bands)
#     plt.show()

# def create_polyline(
#     route_points : np.ndarray
# ) -> np.ndarray:
#     route_xy = np.asarray(route_points[:, :2])
#     segments = np.diff(route_xy, axis=0)
#     seg_lens = np.linalg.norm(segments, axis=1)
#     s = np.concatenate(([0.0], np.cumsum(seg_lens)), dtype=np.float32)
#     return s
