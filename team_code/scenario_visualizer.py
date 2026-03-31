"""
Offline visualizer for ScenarioLogger v2 records.

Usage:
    python scenario_visualizer.py <log_dir> [--fps 10] [--out video.mp4] [--no-filter]

Interactive mode opens three synchronised windows:

    Scene    — Perception BEV (left) + Prediction BEV (right)
               Fixed 40 m half-span centred on ego — scale never changes.
               Slider + play/pause button live here.

    Planning — Top row : ST space-time plan (left)  · v(t) and a(t) profiles (right)
               Bottom row: SL costmap + paths (left) · L, L', L'' profiles (right)

    Control  — Speed history: actual (solid) overlaid with ST planned (dashed)
               + Throttle · Brake · Steer histories with step cursor

Controls (any window):
    ← / →   – step one frame at a time
    Space   – play / pause
    q       – close all windows
"""

import argparse
import gzip
import json
import sys
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon, FancyArrowPatch, Circle
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ─────────────────────────────────────────────────────────────────────────────
# Color palette
# ─────────────────────────────────────────────────────────────────────────────
BG_COLOR        = "#1a1a2e"
ROAD_COLOR      = "#2c2c54"
ROUTE_COLOR     = "#e0e0e0"
EGO_COLOR       = "#00b4d8"
LEADING_COLOR   = "#f4a261"
TRAILING_COLOR  = "#ffd166"
ONCOMING_COLOR  = "#ef233c"
CROSSING_COLOR  = "#c77dff"
PED_COLOR       = "#06d6a0"
OBS_COLOR       = "#adb5bd"
TL_RED          = "#ef233c"
TL_YELLOW       = "#ffd166"
COLLISION_COLOR = "#ef233c"
OVERLAP_COLOR   = "#f4a261"
SAFE_PRED_COLOR = "#90e0ef"
EGO_PRED_COLOR  = "#48cae4"
PLACEHOLDER_CLR = "#4a4e69"
TEXT_COLOR      = "#e0e0e0"
GRID_COLOR      = "#2d2d44"

TRAFFIC_TYPE_COLORS = {
    "leading":  LEADING_COLOR,
    "trailing": TRAILING_COLOR,
    "oncoming": ONCOMING_COLOR,
    "crossing": CROSSING_COLOR,
}


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def rotate_corners(corners: np.ndarray, pos: np.ndarray, yaw: float) -> np.ndarray:
    """
    Transform bounding-box corners from actor-local frame to world frame.

    Args:
        corners : (4, 2) array  – each row is [y_local, x_local] (CARLA convention:
                                  x=forward, y=right, stored as [y, x])
        pos     : (2,)  array  – world [x, y] of actor centre
        yaw     : float        – heading in radians (CARLA: clockwise from +X axis)

    Returns:
        (4, 2) world-frame corners as [[wx, wy], ...]
    """
    x_local = corners[:, 1]   # forward component
    y_local = corners[:, 0]   # lateral component

    cos_y, sin_y = np.cos(yaw), np.sin(yaw)

    # CARLA left-handed → negate sin terms for standard plot axes
    wx = pos[0] + x_local * cos_y - y_local * sin_y
    wy = pos[1] + x_local * sin_y + y_local * cos_y

    return np.column_stack([wx, wy])


def corners_from_serialized_bbox(box: list) -> np.ndarray:
    """
    Reconstruct world-frame corners from a serialized bbox.

    box format: [x, y, z, extent_x, extent_y, yaw_deg]
      extent_x – forward half-length
      extent_y – lateral half-width
      yaw_deg  – heading in degrees (CARLA convention)

    Local corners follow the [lateral, forward] convention used throughout:
      [[ ey,  ex], [ ey, -ex], [-ey, -ex], [-ey,  ex]]
    """
    x, y, _z, ex, ey, yaw_deg = box
    local_corners = np.array([
        [ ey,  ex],
        [ ey, -ex],
        [-ey, -ex],
        [-ey,  ex],
    ])
    return rotate_corners(local_corners, np.array([x, y]), np.radians(yaw_deg))


def bbox_patch(corners_world: np.ndarray, color: str, alpha: float = 0.85,
               lw: float = 1.2, fill: bool = True, hatch: str = "") -> Polygon:
    """Return a matplotlib Polygon for one rotated bounding box."""
    return Polygon(
        corners_world,
        closed=True,
        facecolor=color if fill else "none",
        edgecolor=color,
        alpha=alpha,
        linewidth=lw,
        hatch=hatch,
    )


def extrapolate_trajectory(x: float, y: float, heading: float, speed: float,
                           n_steps: int = 15, dt: float = 0.1) -> np.ndarray:
    """
    Constant-velocity linear extrapolation of an actor's future positions.

    Returns (n_steps, 2) array of [x, y].
    """
    steps = np.arange(1, n_steps + 1)
    xs = x + speed * np.cos(heading) * dt * steps
    ys = y + speed * np.sin(heading) * dt * steps
    return np.column_stack([xs, ys])


# ─────────────────────────────────────────────────────────────────────────────
# Main visualizer
# ─────────────────────────────────────────────────────────────────────────────

class ScenarioVisualizer:
    """
    Offline visualizer for ScenarioLogger v2 per-frame log directories.

    Four panels:
      ┌──────────────────────┬──────────────────┐
      │  Main BEV            │  Predicted        │
      │  (scene overview)    │  Motions BEV      │
      ├──────────────────────┼──────────────────┤
      │  ST Plan             │  SL Plan          │
      │  (longitudinal)      │  (lateral)        │
      └──────────────────────┴──────────────────┘
    """

    # Stored prediction frame rate (matches bicycle_frame_rate in config.py)
    PRED_HZ       = 20
    # How many seconds of stored predictions to draw (or up to collision start)
    DRAW_HORIZON_S      = 2.0
    DRAW_HORIZON_FRAMES = int(DRAW_HORIZON_S * PRED_HZ)  # 40 frames

    # Legacy constants kept for any remaining fallback path
    PRED_SECONDS  = 2.0
    PRED_DT       = 0.1
    PRED_N_STEPS  = int(PRED_SECONDS / PRED_DT)

    def __init__(self, records_path: str, filter_active: bool = True) -> None:
        self.records_path = records_path
        self._load(records_path, filter_active=filter_active)
        self._precompute_history()

        self._playing  = False
        self._step_idx = 0

        # Set by run_interactive()
        self._figs: list            = []
        self._slider                = None
        self._btn_play              = None
        self._ax_main               = None
        self._ax_pred               = None
        self._ax_plans              = None
        self._ax_st                 = None
        self._ax_sl                 = None
        self._ax_st_profiles: tuple = ()  # (ax_s, ax_v, ax_a)
        self._ax_sl_profiles: tuple = ()  # (ax_L, ax_Lp, ax_Lpp)
        self._ax_ctrl: tuple        = ()  # (ax_spd, ax_thr, ax_brk, ax_str)

    # ── data loading ─────────────────────────────────────────────────────────

    def _load(self, dir_path: str, filter_active: bool = True) -> None:
        """Load per-frame JSON.gz files from a lon_log_frames/ directory."""
        frames_dir = os.path.join(dir_path, "lon_log_frames")
        meta_path  = os.path.join(dir_path, "lon_log_meta.json")

        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self.meta = {}

        if not os.path.isdir(frames_dir):
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

        frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".json.gz"))

        def _is_active(fd: dict) -> bool:
            if fd.get("ego_plan", {}).get("action"):
                return True
            if fd.get("scene_context", {}).get("collisions"):
                return True
            return False

        self.states          = []
        self.lights          = []
        self.route_boxes     = []
        self.ego_actions     = []
        self.adv_actions     = []
        self.planner_states  = []
        self.scene_contexts  = []
        self.prediction_data = []
        self.long_results    = []
        self.lat_results     = []
        self.ego_plans       = []
        self.high_level_behs = []

        skipped = 0
        for fname in frame_files:
            fpath = os.path.join(frames_dir, fname)
            with gzip.open(fpath, "rt", encoding="utf-8") as f:
                fd = json.load(f)

            if filter_active and not _is_active(fd):
                skipped += 1
                continue

            self.states.append(fd.get("state", {}))
            self.lights.append(fd.get("lights", {}))
            self.route_boxes.append(fd.get("route_boxes", {}))
            self.ego_actions.append(fd.get("ego_actions", {}))
            self.adv_actions.append(fd.get("adv_actions", {}))
            self.planner_states.append(fd.get("planner_state", {}))
            self.scene_contexts.append(fd.get("scene_context", {}))
            self.prediction_data.append(fd.get("prediction_data", {}))
            self.long_results.append(fd.get("long_planner_result", {}))
            self.lat_results.append(fd.get("lat_planner_result", {}))
            self.ego_plans.append(fd.get("ego_plan", {}))
            self.high_level_behs.append(fd.get("high_level_beh", {}))

        self.n_steps = len(self.states)
        filter_str = f", filtered {skipped} inactive" if filter_active else ""
        print(f"Loaded {self.n_steps} frames from '{frames_dir}'{filter_str}  "
              f"(town: {self.meta.get('town', '?')}, route: {self.meta.get('index', '?')})")

    def _precompute_history(self) -> None:
        """Cache per-step scalars used by the Control window."""
        n = self.n_steps
        self._hist_speed    = np.array([self._sctx(k).get("ego", {}).get("speed", 0.0) for k in range(n)])
        self._hist_throttle = np.array([self._ego_act(k).get("throttle", 0.0) for k in range(n)])
        self._hist_brake    = np.array([self._ego_act(k).get("brake",    0.0) for k in range(n)])
        self._hist_steer    = np.array([self._ego_act(k).get("steer",    0.0) for k in range(n)])

        # dt per logged step: profile time (sec) → logged step offset
        # logging_freq logged steps = logging_freq sim ticks; sim runs at fps Hz
        # dt_sim = 1/fps, so dt_logged_step = logging_freq * dt_sim = logging_freq / fps
        # Default: logging_freq=10, fps=20 → 0.5 s/step
        logging_freq = int(self.meta.get("logging_freq", 10))
        fps          = float(self.meta.get("fps", 20.0))
        self._dt_logged_step = logging_freq / fps   # seconds per logged step

    # ── accessor helpers ──────────────────────────────────────────────────────

    def _get(self, lst: list, idx: int, default=None):
        return lst[idx] if idx < len(lst) else default

    def _state(self, i):    return self._get(self.states, i, {})
    def _lights(self, i):   return self._get(self.lights, i, {})
    def _route(self, i):    return self._get(self.route_boxes, i, {})
    def _pstate(self, i):   return self._get(self.planner_states, i, {})
    def _sctx(self, i):     return self._get(self.scene_contexts, i, {})
    def _pred(self, i):     return self._get(self.prediction_data, i, {})
    def _long(self, i):     return self._get(self.long_results, i, {})
    def _lat(self, i):      return self._get(self.lat_results, i, {})
    def _ego_act(self, i):  return self._get(self.ego_actions, i, {})
    def _ego_plan(self, i): return self._get(self.ego_plans, i, {})
    def _hlbeh(self, i):    return self._get(self.high_level_behs, i, {})

    # ── coordinate helpers ────────────────────────────────────────────────────

    def _ego_center(self, i: int) -> Optional[np.ndarray]:
        state = self._state(i)
        pos = state.get("pos")
        if not pos:
            return None
        return np.array(pos[0][0])  # shape [1, N, 2] → ego is index 0

    # Fixed half-span for BEV panels (meters).  Both Perception and Prediction
    # always show this same square around the ego so the scene scale is stable.
    SCENE_HALF_SPAN: float = 40.0

    def _fixed_roi(self, i: int) -> Tuple[float, float, float, float]:
        """Fixed-size ROI centred on ego — scene scale never changes."""
        ego = self._ego_center(i)
        if ego is None:
            cx, cy = 0.0, 0.0
        else:
            cx, cy = float(ego[0]), float(ego[1])
        h = self.SCENE_HALF_SPAN
        return cx - h, cx + h, cy - h, cy + h

    # ── drawing primitives ────────────────────────────────────────────────────

    def _draw_route(self, ax, i: int) -> None:
        """Draw route centerline and bounding boxes."""
        # Centerline from planner state
        route_pts = self._pstate(i).get("route_points_ahead", [])
        if len(route_pts) >= 2:
            rp = np.array(route_pts)
            ax.plot(rp[:, 0], rp[:, 1], color=ROUTE_COLOR, lw=1.2,
                    alpha=0.4, zorder=1, linestyle="--")

        # Route boxes (wider road approximation)
        rb = self._route(i)
        poses = rb.get("pos", [])
        yaws  = rb.get("yaw", [])
        exts  = rb.get("extent", [])
        if poses and exts:
            poses = np.array(poses)[0]  # (N, 2)
            yaws  = np.array(yaws)[0]   # (N, 1)
            exts  = np.array(exts)[0]   # (N, 4, 2)
            patches = []
            for k in range(len(poses)):
                corners = rotate_corners(exts[k], poses[k], float(yaws[k, 0]))
                patches.append(Polygon(corners, closed=True))
            pc = PatchCollection(patches, facecolor=ROAD_COLOR, alpha=0.18,
                                 edgecolor="none", zorder=0)
            ax.add_collection(pc)

    def _draw_traffic_lights(self, ax, i: int) -> None:
        """Draw traffic light trigger zones as coloured circles."""
        lt = self._lights(i)
        poses  = lt.get("pos", [])
        states = lt.get("state", [])
        if not poses:
            return
        poses  = np.array(poses)[0]   # (N, 2)
        states = np.array(states)[0]  # (N, 1)
        for k in range(len(poses)):
            state = int(states[k, 0])
            color = TL_RED if state == 0 else TL_YELLOW
            circle = Circle(poses[k], radius=1.5, color=color, alpha=0.9, zorder=4)
            ax.add_patch(circle)
            ax.annotate("🔴" if state == 0 else "🟡",
                        poses[k], fontsize=7, ha="center", va="center",
                        color=TEXT_COLOR, zorder=5)

    def _draw_actors_from_state(self, ax, i: int,
                                highlight_ids: Optional[Dict[str, str]] = None) -> None:
        """
        Draw all actors (ego + background) from the raw simulation state.

        highlight_ids: {str(actor_id): color} – override color for flagged actors
        """
        state = self._state(i)
        poses  = state.get("pos")
        yaws   = state.get("yaw")
        exts   = state.get("extent")
        ids    = state.get("id")
        types  = state.get("type")
        vels   = state.get("vel")

        if not poses:
            return

        poses  = np.array(poses)[0]   # (N, 2)
        yaws   = np.array(yaws)[0]    # (N, 1)
        exts   = np.array(exts)[0]    # (N, 4, 2)
        ids    = np.array(ids)[0]     # (N, 1)
        vels   = np.array(vels)[0]    # (N, 2)

        # Traffic type colors from scene context (richer data)
        sctx = self._sctx(i)
        veh_color_map = {}  # id → color
        for veh in sctx.get("vehicles", []):
            tt = veh.get("traffic_type", "leading")
            veh_color_map[veh["id"]] = TRAFFIC_TYPE_COLORS.get(tt, LEADING_COLOR)

        for k in range(len(poses)):
            actor_id = int(ids[k, 0])
            corners  = rotate_corners(exts[k], poses[k], float(yaws[k, 0]))

            if k == 0:  # ego
                color = highlight_ids.get(str(actor_id), EGO_COLOR) \
                        if highlight_ids else EGO_COLOR
                lw    = 2.0
                alpha = 1.0
            else:
                color = highlight_ids.get(str(actor_id),
                                          veh_color_map.get(actor_id, LEADING_COLOR)) \
                        if highlight_ids else veh_color_map.get(actor_id, LEADING_COLOR)
                lw    = 1.2
                alpha = 0.85

            patch = bbox_patch(corners, color, alpha=alpha, lw=lw)
            ax.add_patch(patch)

            # Velocity arrow
            speed = float(np.linalg.norm(vels[k]))
            if speed > 0.3:
                yaw = float(yaws[k, 0])
                dx  = np.cos(yaw) * min(speed * 0.4, 4.0)
                dy  = np.sin(yaw) * min(speed * 0.4, 4.0)
                ax.annotate("", xy=(poses[k, 0] + dx, poses[k, 1] + dy),
                            xytext=(poses[k, 0], poses[k, 1]),
                            arrowprops=dict(arrowstyle="->", color=color,
                                            lw=1.0, mutation_scale=8),
                            zorder=6)

    def _draw_pedestrians(self, ax, i: int) -> None:
        sctx = self._sctx(i)
        ego  = self._ego_center(i)
        if ego is None:
            return
        for ped in sctx.get("pedestrians", []):
            rel = ped.get("relative_position", [0, 0])
            px  = ego[0] + rel[0]
            py  = ego[1] + rel[1]
            circle = Circle([px, py], radius=0.6, color=PED_COLOR, alpha=0.9, zorder=5)
            ax.add_patch(circle)

    def _draw_obstacles(self, ax, i: int) -> None:
        sctx = self._sctx(i)
        for obs in sctx.get("obstacles", []):
            corners_raw = obs.get("bbox_corners_xy")
            if not corners_raw:
                continue
            color = ONCOMING_COLOR if obs.get("obstructs_ego") else OBS_COLOR
            # bbox_corners_xy is serialized as (1, 4, 2); take first entry
            corners = np.array(corners_raw[0])  # (4, 2) world frame
            patch = bbox_patch(corners, color, alpha=0.7, lw=1.5, fill=False)
            ax.add_patch(patch)

    # ── panel renderers ───────────────────────────────────────────────────────

    def _render_main_bev(self, ax, i: int) -> None:
        """Panel 1: full scene overview."""
        ax.set_facecolor(BG_COLOR)
        ax.set_title("Scene Overview", color=TEXT_COLOR, fontsize=9, pad=3)

        self._draw_route(ax, i)
        self._draw_traffic_lights(ax, i)
        self._draw_actors_from_state(ax, i)
        self._draw_pedestrians(ax, i)
        self._draw_obstacles(ax, i)

        # Ego label
        ego = self._ego_center(i)
        if ego is not None:
            ax.text(ego[0], ego[1] + 2.5, "EGO", color=EGO_COLOR,
                    fontsize=6, ha="center", zorder=10)

        # Axes  — CARLA Y is south, so invert so north is up (swap ymin/ymax)
        xmin, xmax, ymin, ymax = self._fixed_roi(i)
        ax.set_xlim(xmax, xmin)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.tick_params(colors=TEXT_COLOR, labelsize=6)
        ax.set_xlabel("X (m)", color=TEXT_COLOR, fontsize=7)
        ax.set_ylabel("Y (m, south↓)", color=TEXT_COLOR, fontsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.4, alpha=0.5)

        # Legend
        legend_elems = [
            mpatches.Patch(color=EGO_COLOR,      label="Ego"),
            mpatches.Patch(color=LEADING_COLOR,  label="Leading"),
            mpatches.Patch(color=TRAILING_COLOR, label="Trailing"),
            mpatches.Patch(color=ONCOMING_COLOR, label="Oncoming"),
            mpatches.Patch(color=CROSSING_COLOR, label="Crossing"),
            mpatches.Patch(color=PED_COLOR,      label="Pedestrian"),
            mpatches.Patch(color=TL_RED,         label="Red TL"),
        ]
        ax.legend(handles=legend_elems, loc="upper right", fontsize=5,
                  facecolor="#22223b", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
                  framealpha=0.8)

    # ── predicted-bbox helpers ────────────────────────────────────────────────

    @staticmethod
    def _frames_from_intervals(intervals: list) -> set:
        """Return all frame indices covered by a list of {start_idx, end_idx} dicts."""
        frames = set()
        for ci in intervals:
            frames.update(range(ci["start_idx"], ci["end_idx"] + 1))
        return frames

    @staticmethod
    def _boundary_frames(intervals: list) -> set:
        """Return only the start and end frame of each interval."""
        frames = set()
        for ci in intervals:
            frames.add(ci["start_idx"])
            frames.add(ci["end_idx"])
        return frames

    def _draw_predicted_actor(
        self,
        ax,
        boxes: list,
        collision_intervals: list,
        overlap_interval: dict,
        base_color: str,
    ) -> None:
        """
        Draw one actor's predicted trajectory from stored bounding boxes.

        Trajectory path
        ───────────────
        Drawn segment-by-segment; each segment is coloured by whether either
        of its endpoint frames falls inside a collision or overlap interval:
          • collision  → COLLISION_COLOR
          • overlap    → OVERLAP_COLOR
          • safe       → base_color (dimmed)

        Bounding boxes
        ──────────────
        To avoid stacking many filled polygons into an opaque blob, bounding
        box outlines are drawn only at the *boundary* frames of each interval
        (the start_idx and end_idx of every collision/overlap interval).
        This shows the actor's shape at the point the event begins and ends
        without cluttering the intermediate frames.
        """
        if not boxes:
            return

        n = len(boxes)

        # Determine how many frames to draw:
        #   up to DRAW_HORIZON_FRAMES OR up to (but not including) the first
        #   collision start, whichever is shorter.
        first_col_start = min(
            (iv["start_idx"] for iv in collision_intervals),
            default=self.DRAW_HORIZON_FRAMES,
        )
        draw_n = min(self.DRAW_HORIZON_FRAMES, first_col_start, n)

        if draw_n == 0:
            return

        centers = np.array([[boxes[k][0], boxes[k][1]] for k in range(draw_n)])

        # Frame-index sets for path / box coloring within the draw window
        col_frames = self._frames_from_intervals(collision_intervals)
        ov_frames  = (
            set(range(overlap_interval["time_start_idx"],
                      overlap_interval["time_end_idx"] + 1))
            if overlap_interval.get("is_valid") else set()
        )

        # ── bounding box outlines at every predicted frame ──────────────────
        for k in range(draw_n):
            t = k / max(draw_n - 1, 1)          # 0 → 1 (near → far)
            alpha = max(0.12, 0.55 * (1.0 - 0.7 * t))  # fades from ~0.55 to ~0.17

            if k in col_frames:
                color, lw = COLLISION_COLOR, 1.6
            elif k in ov_frames:
                color, lw = OVERLAP_COLOR, 1.2
            else:
                color, lw = base_color, 0.7

            corners = corners_from_serialized_bbox(boxes[k])
            ax.add_patch(bbox_patch(corners, color, alpha=alpha, lw=lw, fill=False))

        # ── trajectory path (centre dots connecting box centres) ─────────────
        if draw_n > 1:
            ax.plot(centers[:, 0], centers[:, 1],
                    color=base_color, lw=0.6, alpha=0.35, zorder=6)

        # ── endpoint marker at the last drawn frame ──────────────────────────
        end_k = draw_n - 1
        end_color = (COLLISION_COLOR if end_k in col_frames
                     else OVERLAP_COLOR if end_k in ov_frames
                     else base_color)
        ax.scatter(centers[-1, 0], centers[-1, 1], s=14,
                   color=end_color, alpha=0.8, marker="x", zorder=8)

    def _render_predicted_motions(self, ax, i: int) -> None:
        """
        Panel 2: predicted trajectories drawn from stored bounding boxes.

        Each actor's trajectory is drawn frame-by-frame using the boxes
        serialized by ScenarioLogger.  Collision and overlap intervals are
        used directly (by frame index) to choose per-segment colors and to
        decide which bounding-box shapes to draw.

        Falls back to the scene_context data when no predicted boxes are
        stored (e.g. when motion prediction is disabled).
        """
        ax.set_facecolor(BG_COLOR)
        ax.set_title("Predicted Motions", color=TEXT_COLOR, fontsize=9, pad=3)

        pred            = self._pred(i)
        actor_collisions = pred.get("actor_collisions", {})
        actor_overlaps   = pred.get("actor_overlaps",   {})

        collision_ids = set(actor_collisions.keys())
        overlap_ids   = {k for k, v in actor_overlaps.items() if v.get("is_valid")}

        # Highlight current (non-predicted) actor boxes on the base scene
        highlight: Dict[str, str] = {}
        ids_arr = self._state(i).get("id")
        if ids_arr:
            ids_arr = np.array(ids_arr)[0]
            for k in range(len(ids_arr)):
                aid = str(int(ids_arr[k, 0]))
                if aid in collision_ids:
                    highlight[aid] = COLLISION_COLOR
                elif aid in overlap_ids:
                    highlight[aid] = OVERLAP_COLOR

        self._draw_route(ax, i)
        self._draw_actors_from_state(ax, i, highlight_ids=highlight)

        ego_bbs  = pred.get("ego_forecasted_bbs",  [])
        veh_bbs  = pred.get("veh_forecasted_bbs",  {})
        ped_bbs  = pred.get("ped_forecasted_bbs",  {})
        has_pred = bool(ego_bbs or veh_bbs or ped_bbs)

        if has_pred:
            # ── ego ─────────────────────────────────────────────────────────
            if ego_bbs:
                self._draw_predicted_actor(
                    ax, ego_bbs,
                    collision_intervals=[],
                    overlap_interval={},
                    base_color=EGO_PRED_COLOR,
                )

            # ── vehicles ────────────────────────────────────────────────────
            for aid, boxes in veh_bbs.items():
                sctx_veh   = next(
                    (v for v in self._sctx(i).get("vehicles", [])
                     if str(v["id"]) == aid), {}
                )
                tt         = sctx_veh.get("traffic_type", "leading")
                base_color = TRAFFIC_TYPE_COLORS.get(tt, LEADING_COLOR)

                self._draw_predicted_actor(
                    ax, boxes,
                    collision_intervals=actor_collisions.get(aid, []),
                    overlap_interval=actor_overlaps.get(aid, {}),
                    base_color=base_color,
                )

            # ── pedestrians ─────────────────────────────────────────────────
            for aid, boxes in ped_bbs.items():
                self._draw_predicted_actor(
                    ax, boxes,
                    collision_intervals=[],
                    overlap_interval={},
                    base_color=PED_COLOR,
                )

        else:
            ax.text(0.02, 0.98, "⚠ No stored predictions",
                    transform=ax.transAxes, color=PLACEHOLDER_CLR, fontsize=6,
                    va="top", style="italic")

        # ── summary + axes ───────────────────────────────────────────────────
        n_col = len(collision_ids)
        n_ov  = len(overlap_ids)
        ax.text(0.02, 0.02, f"Collisions: {n_col}  |  Overlaps: {n_ov}",
                transform=ax.transAxes, color=TEXT_COLOR, fontsize=6.5, va="bottom",
                bbox=dict(facecolor="#22223b", edgecolor=GRID_COLOR, alpha=0.7))

        legend_elems = [
            mpatches.Patch(color=EGO_PRED_COLOR,   label="Ego predicted"),
            mpatches.Patch(color=SAFE_PRED_COLOR,   label="Vehicle predicted"),
            mpatches.Patch(color=COLLISION_COLOR,   label="Collision"),
            mpatches.Patch(color=OVERLAP_COLOR,     label="Overlap"),
            mpatches.Patch(color=PED_COLOR,         label="Pedestrian predicted"),
        ]
        ax.legend(handles=legend_elems, loc="upper right", fontsize=5,
                  facecolor="#22223b", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
                  framealpha=0.8)

        xmin, xmax, ymin, ymax = self._fixed_roi(i)
        ax.set_xlim(xmax, xmin)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.tick_params(colors=TEXT_COLOR, labelsize=6)
        ax.set_xlabel("X (m)", color=TEXT_COLOR, fontsize=7)
        ax.set_ylabel("Y (m, south↓)", color=TEXT_COLOR, fontsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.4, alpha=0.5)

    def _render_st_plan(self, ax, i: int) -> None:
        """Panel 3: Space-Time (ST) longitudinal plan diagram.

        Background: cost_map heatmap (T×S grid, inferno colormap).
        Overlay: planned s(t) trajectory + collision/overlap bands.
        """
        ax.set_facecolor(BG_COLOR)
        ax.set_title("ST Plan  (Longitudinal)", color=TEXT_COLOR, fontsize=9, pad=3)
        ax.set_xlabel("Time  t  (s)", color=TEXT_COLOR, fontsize=7)
        ax.set_ylabel("Route station  s  (m)", color=TEXT_COLOR, fontsize=7)
        ax.tick_params(colors=TEXT_COLOR, labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)

        long = self._long(i)
        pred = self._pred(i)
        st   = long.get("st_maps")

        # ---- ST costmap heatmap --------------------------------------------
        if st:
            T_arr    = np.array(st["T_arr"])          # (T,)
            S_arr    = np.array(st["S_arr"])          # (S,)
            cost_map = np.array(st["cost_map"])       # (T, S)
            occ_map  = np.array(st["occupancy_map"])  # (T, S)

            # cost_map is (T, S): T on X axis, S on Y axis
            Tgrid, Sgrid = np.meshgrid(T_arr, S_arr, indexing="ij")
            ax.pcolormesh(Tgrid, Sgrid, cost_map,
                          shading="nearest", cmap="inferno",
                          vmin=0, vmax=255, alpha=0.8, zorder=1)

            # Hard-blocked cells as brighter overlay
            occ_mask = occ_map > 0
            if occ_mask.any():
                ax.pcolormesh(Tgrid, Sgrid,
                              np.where(occ_mask, 255.0, np.nan),
                              shading="nearest", cmap="Wistia",
                              vmin=0, vmax=255, alpha=0.50, zorder=2)

            t_horizon = float(T_arr[-1])
            s_horizon = float(S_arr[-1])
        else:
            t_horizon = self.PRED_SECONDS * 2
            s_horizon = 60.0
            ax.text(0.5, 0.55, "No ST map logged",
                    transform=ax.transAxes, ha="center", va="center",
                    color=PLACEHOLDER_CLR, fontsize=8, style="italic",
                    bbox=dict(facecolor="#22223b", edgecolor=PLACEHOLDER_CLR,
                              alpha=0.5, boxstyle="round"))

        ax.set_xlim(0, t_horizon)
        ax.set_ylim(0, s_horizon)

        # ---- Planned s(t) trajectory (remaining portion only) ---------------
        time_profile = long.get("time_profile", [])
        s_profile    = long.get("s_profile",    [])
        step_idx     = int(long.get("profile_step_idx", 0))
        if time_profile and s_profile:
            t_arr = np.array(time_profile)[step_idx:]
            s_arr = np.array(s_profile)[step_idx:]
            if len(t_arr) >= 2:
                # Offset so remaining plan starts at (t=0, s=0) — matching costmap origin
                t_arr = t_arr - t_arr[0]
                s_arr = s_arr - s_arr[0]
                ax.plot(t_arr, s_arr, color=EGO_COLOR, lw=2.0,
                        zorder=10, label="Ego s(t)")

        # ---- Collision interval bands (time axis) ---------------------------
        actor_collisions = pred.get("actor_collisions", {})
        # for aid, intervals in actor_collisions.items():
        #     for ci in intervals:
        #         t_start = ci["start_idx"] * self.PRED_DT
        #         t_end   = ci["end_idx"]   * self.PRED_DT
        #         ax.axvspan(t_start, min(t_end, t_horizon),
        #                    alpha=0.30, color=COLLISION_COLOR, zorder=3)

        # ---- Lane overlap bands (space axis) --------------------------------
        actor_overlaps = pred.get("actor_overlaps", {})
        # s_route = self._pstate(i).get("s_route_ahead", [])
        # for aid, ov in actor_overlaps.items():
        #     if not ov.get("is_valid"):
        #         continue
        #     si = ov.get("space_start_idx", -1)
        #     ei = ov.get("space_end_idx",   -1)
        #     if si >= 0 and ei >= si and s_route:
        #         sr = np.array(s_route)
        #         s0 = float(sr[min(si, len(sr) - 1)])
        #         s1 = float(sr[min(ei, len(sr) - 1)])
        #         ax.axhspan(s0, min(s1, s_horizon),
        #                    alpha=0.25, color=OVERLAP_COLOR, zorder=3)

        # ---- Current state info overlay ------------------------------------
        cost      = long.get("plan_cost")
        tgt_speed = long.get("target_speed")
        info_lines = []
        if tgt_speed is not None:
            info_lines.append(f"Target: {tgt_speed:.1f} m/s")
        if cost is not None and cost < 1e9:
            info_lines.append(f"Cost: {cost:.2f}")
        n_col = len(actor_collisions)
        n_ov  = sum(1 for v in actor_overlaps.values() if v.get("is_valid"))
        info_lines.append(f"Col: {n_col}  Ov: {n_ov}")

        ax.text(0.02, 0.97, "\n".join(info_lines), transform=ax.transAxes,
                color=TEXT_COLOR, fontsize=5.5, va="top",
                bbox=dict(facecolor="#22223b", edgecolor=GRID_COLOR, alpha=0.7))

        ax.grid(True, color=GRID_COLOR, linewidth=0.4, alpha=0.4, zorder=4)

        legend_handles = [mpatches.Patch(color=EGO_COLOR, label="Planned s(t)")]
        # if actor_collisions:
        #     legend_handles.append(
        #         mpatches.Patch(color=COLLISION_COLOR, alpha=0.4, label="Collision window"))
        # if actor_overlaps:
        #     legend_handles.append(
        #         mpatches.Patch(color=OVERLAP_COLOR, alpha=0.4, label="Overlap band"))
        ax.legend(handles=legend_handles, fontsize=5, facecolor="#22223b",
                  edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, loc="lower right")

    def _render_sl_plan(self, ax, i: int) -> None:
        """
        Panel 4: SL costmap with Dijkstra seed, convex corridor, and QP path.

        When sl_maps data is stored (from LatPlannerResult), renders:
          • cost_map heatmap (inferno, dark background)
          • convex corridor shaded band + dashed bounds
          • Dijkstra seed path (cyan dashed)
          • QP spline (green solid)
          • reference centreline at l=0

        Falls back to a placeholder text when no plan data is logged.
        """
        ax.set_facecolor(BG_COLOR)
        ax.set_title("SL Plan  (Lateral)", color=TEXT_COLOR, fontsize=9, pad=3)
        ax.set_xlabel("Route station  s  (m)", color=TEXT_COLOR, fontsize=7)
        ax.set_ylabel("Lateral offset  l  (m)", color=TEXT_COLOR, fontsize=7)
        ax.tick_params(colors=TEXT_COLOR, labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)

        lat = self._lat(i)
        sl  = lat.get("sl_maps")

        if not sl:
            ax.text(0.5, 0.5, "No SL plan data logged",
                    transform=ax.transAxes, ha="center", va="center",
                    color=PLACEHOLDER_CLR, fontsize=8, style="italic",
                    bbox=dict(facecolor="#22223b", edgecolor=PLACEHOLDER_CLR,
                              alpha=0.5, boxstyle="round"))
            return

        S_arr    = np.array(sl["S_arr"])
        L_arr    = np.array(sl["L_arr"])
        cost_map = np.array(sl["cost_map"])   # (S, L)

        # ── costmap heatmap ──────────────────────────────────────────────────
        Sgrid, Lgrid = np.meshgrid(S_arr, L_arr, indexing="ij")
        ax.pcolormesh(Sgrid, Lgrid, cost_map,
                      shading="nearest", cmap="inferno", vmin=0, vmax=255)

        # ── reference centreline ─────────────────────────────────────────────
        ax.axhline(y=float(sl.get("L_ref", 0.0)), color="white",
                   lw=0.6, alpha=0.3, linestyle=":")

        dp_raw  = lat.get("dp_path")
        cor_raw = lat.get("corridor")
        qp_raw  = lat.get("qp_path")

        # ── convex corridor ──────────────────────────────────────────────────
        if dp_raw and cor_raw:
            dp  = np.array(dp_raw)
            cor = np.array(cor_raw)
            s_dp = dp[:, 0]
            lb   = cor[:, 0]
            ub   = cor[:, 1]
            ax.fill_between(s_dp, lb, ub, color="white", alpha=0.12,
                            label="Corridor")
            ax.plot(s_dp, lb, color="white", lw=0.8, alpha=0.5, linestyle="--")
            ax.plot(s_dp, ub, color="white", lw=0.8, alpha=0.5, linestyle="--")

        # ── Dijkstra seed ────────────────────────────────────────────────────
        if dp_raw:
            dp = np.array(dp_raw)
            ax.plot(dp[:, 0], dp[:, 1],
                    color="cyan", lw=1.2, linestyle="--",
                    marker=".", markersize=3, label="Dijkstra seed")

        # ── QP spline ────────────────────────────────────────────────────────
        if qp_raw:
            qp = np.array(qp_raw)
            ax.plot(qp[:, 0], qp[:, 1],
                    color="#00FF00", lw=2.0, zorder=10, label="QP spline")

        # ── axes bounds from data ─────────────────────────────────────────────
        ax.set_xlim(S_arr[0], S_arr[-1])
        ax.set_ylim(L_arr[0], L_arr[-1])

        # ── info + legend ─────────────────────────────────────────────────────
        info = (f"s_dist: {lat.get('s_distance', 0.0):.1f} m  "
                f"new: {lat.get('is_new_plan', False)}")
        ax.text(0.02, 0.97, info, transform=ax.transAxes,
                color=TEXT_COLOR, fontsize=5.5, va="top",
                bbox=dict(facecolor="#22223b", edgecolor=GRID_COLOR, alpha=0.7))

        ax.legend(fontsize=5, facecolor="#22223b",
                  edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, loc="upper right")

    def _render_plans_panel(self, ax, i: int) -> None:
        """
        Text panel shown alongside the Scene BEV panels.

        Displays EgoPlan (top half) and HighLevelBehaviour (bottom half) as
        formatted monospace text.  Each block is clipped to its half so the
        panel never overflows the window regardless of content length.
        """
        ax.set_facecolor("#16213e")
        ax.set_title("AI Plans", color=TEXT_COLOR, fontsize=9, pad=3)
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ego_plan = self._ego_plan(i)
        hl_beh   = self._hlbeh(i)

        _FONT  = 7.5
        _MONO  = "monospace"
        _PAD   = dict(boxstyle="round,pad=0.35", facecolor="#1a1a2e",
                      edgecolor=GRID_COLOR, alpha=0.9)

        # ── divider line ─────────────────────────────────────────────────────
        ax.axhline(0.5, color=GRID_COLOR, lw=0.8, alpha=0.6)

        # ── EgoPlan (top half, y in [0.5, 1.0]) ──────────────────────────────
        ep_lines = ["EGO PLAN"]
        ep_text  = ego_plan.get("_text")
        if ep_text:
            ep_lines.extend(ep_text.split("\n"))
        else:
            ep_lines.append("(none logged)")

        ax.text(
            0.03, 0.98, "\n".join(ep_lines),
            transform=ax.transAxes,
            color=TEXT_COLOR, fontsize=_FONT,
            va="top", ha="left", family=_MONO,
            clip_on=True,
            bbox=_PAD,
        )

        # ── HighLevelBehaviour (bottom half, y in [0.0, 0.5]) ────────────────
        hl_lines = ["HIGH-LEVEL BEHAVIOUR"]
        hl_text  = hl_beh.get("_text")
        if hl_text:
            hl_lines.extend(hl_text.split("\n"))
        else:
            hl_lines.append("(none logged)")

        ax.text(
            0.03, 0.48, "\n".join(hl_lines),
            transform=ax.transAxes,
            color=TEXT_COLOR, fontsize=_FONT,
            va="top", ha="left", family=_MONO,
            clip_on=True,
            bbox=_PAD,
        )

    def _render_info_overlay(self, fig, i: int) -> None:
        """Render a text info strip above the figure."""
        sctx     = self._sctx(i)
        pstate   = self._pstate(i)
        ego_act  = self._ego_act(i)
        long     = self._long(i)

        ego_spd    = sctx.get("ego", {}).get("speed", 0.0)
        tgt_spd    = long.get("target_speed", 0.0)
        steer      = ego_act.get("steer",    0.0)
        throttle   = ego_act.get("throttle", 0.0)
        brake      = ego_act.get("brake",    0.0)
        ri         = pstate.get("route_index", 0)
        rl         = pstate.get("route_len",   0)
        sl         = sctx.get("traffic", {}).get("speed_limit", 0.0)

        tl_info = sctx.get("traffic", {}).get("next_traffic_light")
        tl_str  = (f"TL {tl_info['state']} @ {tl_info['distance']:.0f} m"
                   if tl_info else "no TL")

        title = (
            f"Step {i}/{self.n_steps - 1}"
            f"  |  Town: {self.meta.get('town', '?')}"
            f"  |  Route {ri}/{rl}"
            f"  |  Speed: {ego_spd:.1f} m/s  (target {tgt_spd:.1f})"
            f"  |  SpeedLimit: {sl:.0f}"
            f"  |  {tl_str}"
            f"  |  Steer: {steer:+.2f}  Thr: {throttle:.2f}  Brk: {brake:.2f}"
        )
        fig.suptitle(title, color=TEXT_COLOR, fontsize=7.5, y=0.995)

    # ── ST plan profiles (v(t), a(t)) ─────────────────────────────────────────

    def _render_st_profiles(self, axes, i: int) -> None:
        """
        Right column of the Planning window ST section:
          ax_s  — planned station s(t)
          ax_v  — planned velocity v(t)
          ax_a  — planned acceleration a(t) derived as dv/dt
        """
        ax_s, ax_v, ax_a = axes
        labels = ["s (m)\nStation", "v (m/s)\nVelocity", "a (m/s²)\nAccel"]
        colors = [ROUTE_COLOR, EGO_COLOR, LEADING_COLOR]

        for ax, label, color in zip(axes, labels, colors):
            ax.cla()
            ax.set_facecolor(BG_COLOR)
            ax.set_ylabel(label, color=TEXT_COLOR, fontsize=6)
            ax.tick_params(colors=TEXT_COLOR, labelsize=5)
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID_COLOR)
            ax.grid(True, color=GRID_COLOR, lw=0.3, alpha=0.5)

        ax_s.set_title("ST Profiles", color=TEXT_COLOR, fontsize=8, pad=3)
        ax_a.set_xlabel("Time t (s)", color=TEXT_COLOR, fontsize=6)
        ax_s.set_xticklabels([])
        ax_v.set_xticklabels([])

        long     = self._long(i)
        t        = long.get("time_profile", [])
        vel      = long.get("vel_profile",  [])
        s        = long.get("s_profile",    [])
        step_idx = int(long.get("profile_step_idx", 0))

        if len(t) < 2 or len(vel) < 2:
            ax_s.text(0.5, 0.5, "No ST plan", transform=ax_s.transAxes,
                      ha="center", va="center",
                      color=PLACEHOLDER_CLR, fontsize=7, style="italic")
            return

        # Slice to remaining portion and offset time to start at 0
        t_arr = np.array(t)[step_idx:]
        v_arr = np.array(vel)[step_idx:]
        if len(t_arr) < 2:
            return
        t_arr = t_arr - t_arr[0]

        dt    = np.diff(t_arr)
        dt    = np.where(dt < 1e-6, 1e-6, dt)
        a_arr = np.concatenate([np.diff(v_arr) / dt, [0.0]])

        # s(t)
        if s and len(s) >= step_idx + 2:
            s_arr = np.array(s)[step_idx:]
            s_arr = s_arr - s_arr[0]
            ax_s.plot(t_arr, s_arr, color=ROUTE_COLOR, lw=1.4)
            ax_s.set_xlim(t_arr[0], t_arr[-1])

        # v(t)
        tgt = long.get("target_speed")
        ax_v.plot(t_arr, v_arr, color=EGO_COLOR, lw=1.5, label="Planned v")
        if tgt is not None:
            ax_v.axhline(tgt, color="white", lw=0.8, ls="--", alpha=0.7,
                         label=f"Target {tgt:.1f} m/s")
        ax_v.legend(fontsize=5, facecolor="#22223b", edgecolor=GRID_COLOR,
                    labelcolor=TEXT_COLOR)
        ax_v.set_xlim(t_arr[0], t_arr[-1])

        # a(t)
        ax_a.plot(t_arr, a_arr, color=LEADING_COLOR, lw=1.4)
        ax_a.axhline(0, color=GRID_COLOR, lw=0.6)
        ax_a.set_xlim(t_arr[0], t_arr[-1])

    # ── SL plan profiles (L, L', L'') ────────────────────────────────────────

    def _render_sl_profiles(self, axes, i: int) -> None:
        """
        Right column of the Planning window SL section:
          axes = (ax_L, ax_Lp, ax_Lpp) — lateral offset, heading, curvature vs S.
        """
        ax_L, ax_Lp, ax_Lpp = axes
        labels = ["L (m)\nOffset", "L'\nHeading", "L''\nCurvature"]
        colors = ["#00FF00", "dodgerblue", "orange"]

        for ax, label, color in zip(axes, labels, colors):
            ax.cla()
            ax.set_facecolor(BG_COLOR)
            ax.set_ylabel(label, color=TEXT_COLOR, fontsize=6)
            ax.tick_params(colors=TEXT_COLOR, labelsize=5)
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID_COLOR)
            ax.grid(True, color=GRID_COLOR, lw=0.3, alpha=0.5)

        ax_L.set_title("SL QP Profiles", color=TEXT_COLOR, fontsize=8, pad=3)
        ax_Lpp.set_xlabel("Arc length S (m)", color=TEXT_COLOR, fontsize=6)
        ax_L.set_xticklabels([])
        ax_Lp.set_xticklabels([])

        qp_raw = self._lat(i).get("qp_path")
        if not qp_raw or len(qp_raw) < 2:
            ax_L.text(0.5, 0.5, "No QP path", transform=ax_L.transAxes,
                      ha="center", va="center",
                      color=PLACEHOLDER_CLR, fontsize=7, style="italic")
            return

        qp  = np.array(qp_raw)
        s   = qp[:, 0]
        l   = qp[:, 1]
        ds  = max(s[1] - s[0], 1e-6)
        dl  = np.gradient(l,  ds)
        ddl = np.gradient(dl, ds)

        for ax, dat, color in zip(axes, [l, dl, ddl], colors):
            ax.plot(s, dat, color=color, lw=1.4)
            ax.set_xlim(s[0], s[-1])

    # ── control history window ────────────────────────────────────────────────

    def _render_control_window(self, axes, i: int) -> None:
        """
        4-panel history plot.  Speed panel overlays actual (solid) vs planned
        (dashed) km/h so planner tracking is immediately visible.
        """
        ax_spd, ax_thr, ax_brk, ax_str = axes
        x = np.arange(self.n_steps)

        # ── Speed ─────────────────────────────────────────────────────────────
        ax_spd.cla()
        ax_spd.set_facecolor(BG_COLOR)
        ax_spd.plot(x, self._hist_speed, color=EGO_COLOR, lw=1.0,
                    alpha=0.9, label="Actual (km/h)")

        # Overlay remaining velocity profile as a forward-looking curve
        long_plan = self._long(i)
        t_prof    = long_plan.get("time_profile", [])
        v_prof    = long_plan.get("vel_profile",  [])
        step_idx  = int(long_plan.get("profile_step_idx", 0))
        if t_prof and v_prof and step_idx < len(v_prof):
            t_rem = np.array(t_prof)[step_idx:]
            v_rem = np.array(v_prof)[step_idx:]   # m/s → km/h
            t_rem = t_rem - t_rem[0]                     # offset to start at 0
            x_prof = i + t_rem / self._dt_logged_step    # convert to logged-step axis
            ax_spd.plot(x_prof, v_rem, color=ROUTE_COLOR,
                        lw=1.2, alpha=0.85, ls="--", label="Planned v(t)")

        ax_spd.axvline(i, color="white", lw=1.2, alpha=0.9, zorder=5)
        ax_spd.scatter([i], [self._hist_speed[i]], color="white", s=22, zorder=6)
        ax_spd.set_ylabel("Speed\n(km/h)", color=TEXT_COLOR, fontsize=6)
        ax_spd.legend(fontsize=5, facecolor="#22223b", edgecolor=GRID_COLOR,
                      labelcolor=TEXT_COLOR)
        ax_spd.set_title("Ego Control & Speed History", color=TEXT_COLOR,
                         fontsize=9, pad=3)

        # ── Throttle / Brake / Steer ───────────────────────────────────────────
        other_panels = [
            (ax_thr, self._hist_throttle, "#00FF00",       "Throttle", 0.0, 1.0),
            (ax_brk, self._hist_brake,    COLLISION_COLOR, "Brake",    0.0, 1.0),
            (ax_str, self._hist_steer,    LEADING_COLOR,   "Steer",   -1.0, 1.0),
        ]
        for ax, data, color, label, ymin, ymax in other_panels:
            ax.cla()
            ax.set_facecolor(BG_COLOR)
            ax.plot(x, data, color=color, lw=0.9, alpha=0.8)
            ax.axvline(i, color="white", lw=1.2, alpha=0.9, zorder=5)
            ax.scatter([i], [data[i]], color="white", s=22, zorder=6)
            ax.set_ylabel(label, color=TEXT_COLOR, fontsize=6)
            ax.set_ylim(ymin - 0.05, ymax + 0.05)

        for ax in axes:
            ax.set_xlim(0, max(self.n_steps - 1, 1))
            ax.tick_params(colors=TEXT_COLOR, labelsize=5)
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID_COLOR)
            ax.grid(True, color=GRID_COLOR, lw=0.3, alpha=0.5)

        ax_str.set_xlabel("Step", color=TEXT_COLOR, fontsize=7)
        for ax in (ax_spd, ax_thr, ax_brk):
            ax.set_xticklabels([])

    # ── multi-window setup ────────────────────────────────────────────────────

    def _setup_rcparams(self) -> None:
        matplotlib.rcParams.update({
            "figure.facecolor": BG_COLOR,
            "axes.facecolor":   BG_COLOR,
            "text.color":       TEXT_COLOR,
        })

    def _build_all_figures(self) -> dict:
        """
        Create the three figure windows:

          Scene    — Perception BEV (left) + Prediction BEV (centre) + Plans text (right)
                     Slider + play button live here.

          Planning — Row 0-2: ST plan (left) + ST s/v/a profiles (right)
                     Row 3-5: SL costmap (left) + SL L/L'/L'' profiles (right)

          Control  — Speed (actual ⊕ planned) · Throttle · Brake · Steer
        """
        self._setup_rcparams()

        # ── Scene window ──────────────────────────────────────────────────────
        fig_scene = plt.figure("Scene", figsize=(22, 8), facecolor=BG_COLOR)
        fig_scene.subplots_adjust(left=0.03, right=0.98, top=0.95,
                                  bottom=0.10, wspace=0.12)
        gs_sc    = gridspec.GridSpec(1, 3, figure=fig_scene,
                                     width_ratios=[1.0, 1.0, 0.65])
        ax_main  = fig_scene.add_subplot(gs_sc[0, 0])
        ax_pred  = fig_scene.add_subplot(gs_sc[0, 1])
        ax_plans = fig_scene.add_subplot(gs_sc[0, 2])

        ax_sld  = fig_scene.add_axes([0.05, 0.03, 0.78, 0.025],
                                     facecolor="#22223b")
        slider  = Slider(ax_sld, "Step", 0, max(self.n_steps - 1, 1),
                         valinit=0, valstep=1, color=EGO_COLOR)
        slider.label.set_color(TEXT_COLOR)
        slider.valtext.set_color(TEXT_COLOR)

        ax_btn   = fig_scene.add_axes([0.85, 0.025, 0.10, 0.035],
                                      facecolor="#22223b")
        btn_play = Button(ax_btn, "▶ Play", color="#22223b", hovercolor="#44445e")
        btn_play.label.set_color(TEXT_COLOR)

        # ── Planning window ───────────────────────────────────────────────────
        fig_plan = plt.figure("Planning", figsize=(14, 10), facecolor=BG_COLOR)
        fig_plan.subplots_adjust(left=0.08, right=0.97, top=0.95,
                                 bottom=0.07, hspace=0.55, wspace=0.32)
        gs_pl = gridspec.GridSpec(6, 2, figure=fig_plan,
                                  width_ratios=[1.5, 1.0])
        ax_st      = fig_plan.add_subplot(gs_pl[0:3, 0])    # ST plan — left, top 3 rows
        ax_st_s    = fig_plan.add_subplot(gs_pl[0,   1])    # s(t) profile
        ax_st_v    = fig_plan.add_subplot(gs_pl[1,   1], sharex=ax_st_s)  # v(t)
        ax_st_a    = fig_plan.add_subplot(gs_pl[2,   1], sharex=ax_st_s)  # a(t)
        ax_sl      = fig_plan.add_subplot(gs_pl[3:6, 0])    # SL costmap — left, bottom 3 rows
        ax_sl_L    = fig_plan.add_subplot(gs_pl[3,   1])    # L(s)
        ax_sl_Lp   = fig_plan.add_subplot(gs_pl[4,   1], sharex=ax_sl_L)  # L'(s)
        ax_sl_Lpp  = fig_plan.add_subplot(gs_pl[5,   1], sharex=ax_sl_L)  # L''(s)

        # ── Control window ────────────────────────────────────────────────────
        fig_ctrl = plt.figure("Control", figsize=(12, 7), facecolor=BG_COLOR)
        fig_ctrl.subplots_adjust(left=0.09, right=0.97, top=0.95,
                                 bottom=0.08, hspace=0.10)
        gs_ct  = gridspec.GridSpec(4, 1, figure=fig_ctrl)
        ax_spd = fig_ctrl.add_subplot(gs_ct[0])
        ax_thr = fig_ctrl.add_subplot(gs_ct[1], sharex=ax_spd)
        ax_brk = fig_ctrl.add_subplot(gs_ct[2], sharex=ax_spd)
        ax_str = fig_ctrl.add_subplot(gs_ct[3], sharex=ax_spd)

        return dict(
            figs     = [fig_scene, fig_plan, fig_ctrl],
            scene    = (fig_scene, ax_main, ax_pred, ax_plans, slider, btn_play),
            planning = (fig_plan,
                        ax_st, ax_st_s, ax_st_v, ax_st_a,
                        ax_sl, ax_sl_L, ax_sl_Lp, ax_sl_Lpp),
            control  = (fig_ctrl, ax_spd, ax_thr, ax_brk, ax_str),
        )

    # ── per-frame render ──────────────────────────────────────────────────────

    def _goto_frame(self, i: int) -> None:
        """Clear and redraw all windows at step i."""
        self._step_idx = i

        # Scene (Perception + Prediction — fixed ROI + AI Plans text)
        self._ax_main.cla()
        self._ax_pred.cla()
        self._ax_plans.cla()
        self._render_main_bev(self._ax_main, i)
        self._render_predicted_motions(self._ax_pred, i)
        self._render_plans_panel(self._ax_plans, i)
        self._render_info_overlay(self._ax_main.figure, i)

        # Planning
        for ax in (self._ax_st, self._ax_sl,
                   *self._ax_st_profiles, *self._ax_sl_profiles):
            ax.cla()
        self._render_st_plan(self._ax_st, i)
        self._render_st_profiles(self._ax_st_profiles, i)
        self._render_sl_plan(self._ax_sl, i)
        self._render_sl_profiles(self._ax_sl_profiles, i)

        # Control
        self._render_control_window(self._ax_ctrl, i)

        for fig in self._figs:
            fig.canvas.draw_idle()

    # ── public API ────────────────────────────────────────────────────────────

    def run_interactive(self) -> None:
        """
        Open three synchronised windows: Scene, Planning, and Control.

        Controls (any window):
            ← / →   step one frame
            Space   play / pause
            q       close all windows
        Slider + play button live in the Scene window.
        """
        layout   = self._build_all_figures()
        fig_scene, ax_main, ax_pred, ax_plans, slider, btn_play = layout["scene"]
        (fig_plan, ax_st, ax_st_s, ax_st_v, ax_st_a,
                   ax_sl, ax_sl_L, ax_sl_Lp, ax_sl_Lpp) = layout["planning"]
        fig_ctrl, ax_spd, ax_thr, ax_brk, ax_str       = layout["control"]

        self._figs           = layout["figs"]
        self._ax_main        = ax_main
        self._ax_pred        = ax_pred
        self._ax_plans       = ax_plans
        self._ax_st          = ax_st
        self._ax_sl          = ax_sl
        self._ax_st_profiles = (ax_st_s, ax_st_v, ax_st_a)
        self._ax_sl_profiles = (ax_sl_L, ax_sl_Lp, ax_sl_Lpp)
        self._ax_ctrl        = (ax_spd, ax_thr, ax_brk, ax_str)
        self._slider         = slider
        self._btn_play       = btn_play

        self._goto_frame(0)

        # ── slider ────────────────────────────────────────────────────────────
        def on_slider(val):
            idx = int(slider.val)
            if idx != self._step_idx:
                self._goto_frame(idx)

        slider.on_changed(on_slider)

        # ── play / pause ──────────────────────────────────────────────────────
        _timer = [None]

        def advance(_event=None):
            if self._step_idx < self.n_steps - 1:
                slider.set_val(self._step_idx + 1)   # triggers on_slider
            else:
                self._playing = False
                btn_play.label.set_text("▶ Play")
                if _timer[0]:
                    _timer[0].stop()

        def toggle_play(event):
            self._playing = not self._playing
            if self._playing:
                btn_play.label.set_text("⏸ Pause")
                _timer[0] = fig_scene.canvas.new_timer(interval=150)
                _timer[0].add_callback(advance)
                _timer[0].start()
            else:
                btn_play.label.set_text("▶ Play")
                if _timer[0]:
                    _timer[0].stop()

        btn_play.on_clicked(toggle_play)

        # ── keyboard — all windows ────────────────────────────────────────────
        def on_key(event):
            if event.key == "right":
                slider.set_val(min(self._step_idx + 1, self.n_steps - 1))
            elif event.key == "left":
                slider.set_val(max(self._step_idx - 1, 0))
            elif event.key == " ":
                toggle_play(None)
            elif event.key in ("q", "escape"):
                for fig in self._figs:
                    plt.close(fig)

        for fig in self._figs:
            fig.canvas.mpl_connect("key_press_event", on_key)

        plt.show()

    def render_video(self, output_path: str, fps: int = 10) -> None:
        """
        Render all steps to an mp4.  Combined layout mirrors the three windows:
          Row 0: Perception BEV  | Prediction BEV
          Row 1: ST plan         | ST v(t)  | SL costmap  | L(s)
          Row 2: (speed history) | ST a(t)  | (empty)     | L'(s)
        """
        self._setup_rcparams()

        fig = plt.figure(figsize=(22, 14), facecolor=BG_COLOR)
        fig.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.06,
                            hspace=0.50, wspace=0.30)
        gs = gridspec.GridSpec(3, 4, figure=fig,
                               height_ratios=[2.2, 1.5, 1.0],
                               width_ratios=[1.2, 1.0, 1.2, 1.0])

        ax_main   = fig.add_subplot(gs[0, 0])
        ax_pred   = fig.add_subplot(gs[0, 1])
        ax_st     = fig.add_subplot(gs[0, 2])
        ax_sl     = fig.add_subplot(gs[0, 3])
        ax_st_v   = fig.add_subplot(gs[1, 0])
        ax_st_a   = fig.add_subplot(gs[2, 0])
        ax_sl_L   = fig.add_subplot(gs[1, 1])
        ax_sl_Lp  = fig.add_subplot(gs[2, 1])
        ax_spd    = fig.add_subplot(gs[1:3, 2])   # speed history spans 2 rows
        ax_ctrl   = fig.add_subplot(gs[1, 3])     # throttle/brake/steer combined

        def animate(k):
            for ax in (ax_main, ax_pred, ax_st, ax_sl,
                       ax_st_v, ax_st_a, ax_sl_L, ax_sl_Lp,
                       ax_spd, ax_ctrl):
                ax.cla()
            self._render_main_bev(ax_main, k)
            self._render_info_overlay(fig, k)
            self._render_predicted_motions(ax_pred, k)
            self._render_st_plan(ax_st, k)
            self._render_st_profiles((ax_st_v, ax_st_a), k)
            self._render_sl_plan(ax_sl, k)
            self._render_sl_profiles((ax_sl_L, ax_sl_Lp,
                                      ax_sl_Lp), k)   # reuse Lp for Lpp slot
            # Speed history
            x = np.arange(self.n_steps)
            ax_spd.set_facecolor(BG_COLOR)
            ax_spd.plot(x, self._hist_speed, color=EGO_COLOR, lw=0.9,
                        label="Actual (km/h)")
            if self._hist_planned_speed_kmh.any():
                ax_spd.plot(x, self._hist_planned_speed_kmh,
                            color=ROUTE_COLOR, lw=0.8, ls="--",
                            label="Planned (km/h)")
            ax_spd.axvline(k, color="white", lw=1.1, alpha=0.9)
            ax_spd.set_xlim(0, max(self.n_steps - 1, 1))
            ax_spd.legend(fontsize=5, facecolor="#22223b",
                          edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
            ax_spd.tick_params(colors=TEXT_COLOR, labelsize=5)
            ax_spd.set_ylabel("Speed (km/h)", color=TEXT_COLOR, fontsize=6)
            ax_spd.set_title("Speed", color=TEXT_COLOR, fontsize=7)
            for spine in ax_spd.spines.values():
                spine.set_edgecolor(GRID_COLOR)
            # Input history (throttle green, brake red, steer orange — overlaid)
            ax_ctrl.set_facecolor(BG_COLOR)
            ax_ctrl.plot(x, self._hist_throttle, color="#00FF00", lw=0.7,
                         alpha=0.8, label="Thr")
            ax_ctrl.plot(x, self._hist_brake,    color=COLLISION_COLOR,
                         lw=0.7, alpha=0.8, label="Brk")
            ax_ctrl.plot(x, self._hist_steer * 0.5 + 0.5,  # remap –1…1→0…1
                         color=LEADING_COLOR, lw=0.7, alpha=0.8, label="Str")
            ax_ctrl.axvline(k, color="white", lw=1.1, alpha=0.9)
            ax_ctrl.set_xlim(0, max(self.n_steps - 1, 1))
            ax_ctrl.set_ylim(-0.05, 1.05)
            ax_ctrl.legend(fontsize=5, facecolor="#22223b",
                           edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
            ax_ctrl.tick_params(colors=TEXT_COLOR, labelsize=5)
            ax_ctrl.set_title("Inputs", color=TEXT_COLOR, fontsize=7)
            for spine in ax_ctrl.spines.values():
                spine.set_edgecolor(GRID_COLOR)

        writer = FFMpegWriter(fps=fps, metadata={"title": "Scenario"})
        anim   = FuncAnimation(fig, animate, frames=self.n_steps, blit=False)
        print(f"Rendering {self.n_steps} frames to '{output_path}' @ {fps} fps …")
        anim.save(output_path, writer=writer, dpi=100,
                  savefig_kwargs={"facecolor": BG_COLOR})
        plt.close(fig)
        print("Done.")


# ─────────────────────────────────────────────────────────────────────────────
# SL projection helper
# ─────────────────────────────────────────────────────────────────────────────

def _project_to_sl(
    path_xy: np.ndarray,
    ref_xy:  np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Project a 2-D world path onto the Frenet (s, l) frame defined by ref_xy.

    Returns (s_values, l_values) or (None, None) on failure.
    """
    if len(ref_xy) < 2 or len(path_xy) < 1:
        return None, None

    # Arc-length parameterisation of the reference
    deltas = np.diff(ref_xy, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    s_ref = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    s_vals, l_vals = [], []
    for pt in path_xy:
        # Find closest reference point
        dists = np.linalg.norm(ref_xy - pt, axis=1)
        k     = int(np.argmin(dists))

        # Arc-length station
        s = float(s_ref[k])

        # Signed lateral offset (cross-product sign)
        if k < len(ref_xy) - 1:
            tang = ref_xy[k + 1] - ref_xy[k]
        else:
            tang = ref_xy[k] - ref_xy[k - 1]
        perp = pt - ref_xy[k]
        cross = tang[0] * perp[1] - tang[1] * perp[0]
        l = float(np.sign(cross) * np.linalg.norm(perp))

        s_vals.append(s)
        l_vals.append(l)

    return np.array(s_vals), np.array(l_vals)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Offline visualizer for ScenarioLogger v2 records."
    )
    parser.add_argument("records", help="Path to log directory containing lon_log_frames/")
    parser.add_argument("--out",   help="Output video path (mp4). "
                                        "If omitted, opens interactive window.")
    parser.add_argument("--fps",   type=int, default=10,
                        help="Frames per second for video output (default: 10)")
    parser.add_argument("--filter",    dest="filter_active", action="store_true",  default=True,
                        help="Only load frames with an active Action or collision (default).")
    parser.add_argument("--no-filter", dest="filter_active", action="store_false",
                        help="Load all logged frames without filtering.")
    args = parser.parse_args()

    if not os.path.exists(args.records):
        print(f"Error: path not found: {args.records}", file=sys.stderr)
        sys.exit(1)

    vis = ScenarioVisualizer(args.records, filter_active=args.filter_active)

    if args.out:
        vis.render_video(args.out, fps=args.fps)
    else:
        vis.run_interactive()


if __name__ == "__main__":
    main()
