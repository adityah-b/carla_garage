import carla
import numpy as np

from scipy import ndimage
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass, field

from .geometric_utils import GeometricUtils

@dataclass
class CollisionInterval:
    start_idx : int
    end_idx : int

    collision_bboxes_a : List[carla.BoundingBox]
    collision_bboxes_b : List[carla.BoundingBox]

@dataclass
class LaneOverlapInterval:
    is_valid : bool

    time_start_idx : int = -1
    time_end_idx : int = -1

    space_start_idx : int = -1
    space_end_idx : int = -1

    actor_tbb_bboxes : List[carla.BoundingBox] = field(default_factory=list)
    actor_ebb_bboxes : List[carla.BoundingBox] = field(default_factory=list)
    route_subset_bboxes : List[carla.BoundingBox] = field(default_factory=list)

    tbb_frame_occupancies : Dict[int, Tuple[int, int]] = field(default_factory=dict)
    ebb_frame_occupancies : Dict[int, Tuple[int, int]] = field(default_factory=dict)

class CollisionChecker:
    _structure_cache : Dict[int, np.ndarray] = {}

    @staticmethod
    def _close_collision_gaps(
        collision_mask : np.ndarray,
        max_collision_gap : int
    ) -> np.ndarray:
        """
        Fills gaps between True regions using morphological closing.
        Optimized for many repeated calls on small arrays.

        Args:
            collision_mask: Boolean numpy array
            max_collision_gap: Maximum gap size to fill

        Returns:
            Boolean numpy array with gaps filled
        """
        if max_collision_gap <= 0:
            return collision_mask

        # Get or create cached structure
        if max_collision_gap not in CollisionChecker._structure_cache:
            CollisionChecker._structure_cache[max_collision_gap] = np.ones(max_collision_gap + 1, dtype=bool)

        structure = CollisionChecker._structure_cache[max_collision_gap]
        return ndimage.binary_closing(collision_mask, structure=structure)

    @staticmethod
    def _find_collision_intervals(
        bounding_boxes_a : List[carla.BoundingBox],
        bounding_boxes_b : List[carla.BoundingBox],
        *,
        max_frame_offset : int = 1,
        max_collision_gap : int = 1,
        get_longest : bool = True
    ) -> List[CollisionInterval]:
        N = len(bounding_boxes_a)
        if N == 0:
            return []

        # Get per-frame collisions
        collided = np.zeros(N, dtype=bool)
        for i in range(N):
            j0 = max(0, i - max_frame_offset)
            j1 = min(N - 1, i + max_frame_offset)
            bb_a = bounding_boxes_a[i]

            for j in range(j0, j1 + 1):
                if GeometricUtils.check_obb_intersection_2d(bb_a, bounding_boxes_b[j]):
                    collided[i] = True
                    break

        # Close collision gaps to avoid oscillating collision booleans
        closed = CollisionChecker._close_collision_gaps(collided, max_collision_gap=max_collision_gap)

        # No collisions
        if not np.any(closed):
            return []

        # Extract all collision intervals
        padded = np.concatenate([[False], closed, [False]])
        diff = np.diff(padded.astype(int))

        # Start indices: where False→True (diff == 1)
        starts = np.where(diff == 1)[0]

        # End indices: where True→False (diff == -1)
        ends = np.where(diff == -1)[0]

        if get_longest:
            # Find longest interval
            lengths = ends - starts
            longest_idx = np.argmax(lengths)

            start_idx = starts[longest_idx]
            end_idx = ends[longest_idx]

            return [CollisionInterval(
                start_idx,
                end_idx,
                bounding_boxes_a,
                bounding_boxes_b
            )]

        # Find all intervals
        collision_intervals : List[CollisionInterval] = []
        for start_idx, end_idx in zip(starts, ends):
            interval = CollisionInterval(
                int(start_idx),
                int(end_idx),
                bounding_boxes_a,
                bounding_boxes_b
            )
            collision_intervals.append(interval)

        return collision_intervals

    # TODO: CURRENTLY DOING LONGEST (CONSERVATIVE) INTERVAL, MAY BE WORTH LOOKING INTO LIST OF ALL OVERLAP INTERVALS LATER
    @staticmethod
    def _find_overlap_interval(
        actor_tbb_bboxes : List[carla.BoundingBox],
        actor_ebb_bboxes : List[carla.BoundingBox],
        route_subset_bboxes : List[carla.BoundingBox],
        ignore_tbb : bool = False,
    ) -> LaneOverlapInterval:
        num_actor_boxes = len(actor_tbb_bboxes)
        num_route_boxes = len(route_subset_bboxes)
        if num_actor_boxes == 0 or num_route_boxes == 0:
            return LaneOverlapInterval(is_valid=False)

        has_overlap_intervals = False

        # Global overlap bounds
        global_ebb_time_min = float('inf')
        global_ebb_time_max = float('-inf')

        global_tbb_time_min = float('inf')
        global_tbb_time_max = float('-inf')

        global_route_min = float('inf')
        global_route_max = float('-inf')

        # Per-frame occupancies
        ebb_frame_occupancies = {}
        tbb_frame_occupancies = {}

        # Get per-frame collisions
        for i in range(num_actor_boxes):
            ebb_bb = actor_ebb_bboxes[i]
            tbb_bb = actor_tbb_bboxes[i]

            # Track current frame overlap bounds
            current_frame_ebb_route_min = float('inf')
            current_frame_ebb_route_max = float('-inf')

            current_frame_tbb_route_min = float('inf')
            current_frame_tbb_route_max = float('-inf')

            has_frame_overlap = False

            for j in range(num_route_boxes):
                route_bb = route_subset_bboxes[j]

                # Check if EBB overlaps
                if GeometricUtils.check_obb_intersection_2d(ebb_bb, route_bb):
                    has_overlap_intervals = True
                    has_frame_overlap = True

                    # Update EBB per-frame bounds
                    current_frame_ebb_route_min = min(current_frame_ebb_route_min, j)
                    current_frame_ebb_route_max = max(current_frame_ebb_route_max, j)

                    # Update Global Bounds (actor indices are temporal, route indices are spatial)
                    global_ebb_time_min = min(global_ebb_time_min, i)
                    global_ebb_time_max = max(global_ebb_time_max, i)

                    global_route_min = min(global_route_min, j)
                    global_route_max = max(global_route_max, j)

                    # Since EBB overlapped, check if TBB also overlaps
                    if not ignore_tbb and GeometricUtils.check_obb_intersection_2d(tbb_bb, route_bb):
                        global_tbb_time_min = min(global_tbb_time_min, i)
                        global_tbb_time_max = max(global_tbb_time_max, i)

                        current_frame_tbb_route_min = min(current_frame_tbb_route_min, j)
                        current_frame_tbb_route_max = max(current_frame_tbb_route_max, j)

            # Save bounds if there was an EBB overlap this frame
            if has_frame_overlap:
                ebb_frame_occupancies[i] = (int(current_frame_ebb_route_min), int(current_frame_ebb_route_max))

                # Only save TBB bounds if overlap exists
                if current_frame_tbb_route_min != float('inf'):
                    tbb_frame_occupancies[i] = (int(current_frame_tbb_route_min), int(current_frame_tbb_route_max))

        # No overlaps in any frame
        if not has_overlap_intervals:
            return LaneOverlapInterval(is_valid=False)

        return LaneOverlapInterval(
            is_valid=True,
            time_start_idx=global_ebb_time_min,
            time_end_idx=global_ebb_time_max,
            space_start_idx=global_route_min,
            space_end_idx=global_route_max,
            actor_tbb_bboxes=actor_tbb_bboxes,
            actor_ebb_bboxes=actor_ebb_bboxes,
            route_subset_bboxes=route_subset_bboxes,
            tbb_frame_occupancies=tbb_frame_occupancies,
            ebb_frame_occupancies=ebb_frame_occupancies,
        )

    @staticmethod
    def predict_actor_collisions(
        ego_bb_preds : List[carla.BoundingBox],
        actor_predictions : Dict[int, List[carla.BoundingBox]], # K=actor id, V=predicted BBs
        *,
        max_frame_offset : int = 1,
        max_collision_gap : int = 1,
        get_longest : bool = True
    ) -> Dict[int, List[CollisionInterval]]: # K=actor id, V=collision intervals
        all_collision_intervals : Dict[int, List[CollisionInterval]] = {}
        for actor_id, actor_bb_preds in actor_predictions.items():
            collision_intervals = CollisionChecker._find_collision_intervals(
                bounding_boxes_a=ego_bb_preds,
                bounding_boxes_b=actor_bb_preds,
                max_frame_offset=max_frame_offset,
                max_collision_gap=max_collision_gap,
                get_longest=get_longest
            )
            if len(collision_intervals) > 0:
                all_collision_intervals[actor_id] = collision_intervals

        return all_collision_intervals

    @staticmethod
    def get_lane_overlaps(
        veh_tbb_predictions : Dict[int, List[carla.BoundingBox]], # K=actor id, V=predicted BBs,
        veh_ebb_predictions : Dict[int, List[carla.BoundingBox]],
        ped_predictions : Dict[int, List[carla.BoundingBox]],
        route_subset_bbs : List[carla.BoundingBox],
    ) -> Dict[int, LaneOverlapInterval]: # K=actor id, V=collision interval
        all_overlap_intervals : Dict[int, LaneOverlapInterval] = {}

        # Vehicles
        for veh_id in veh_tbb_predictions.keys():
            veh_tbbs = veh_tbb_predictions[veh_id]
            veh_ebbs = veh_ebb_predictions[veh_id]
            overlap_interval = CollisionChecker._find_overlap_interval(
                actor_tbb_bboxes=veh_tbbs,
                actor_ebb_bboxes=veh_ebbs,
                route_subset_bboxes=route_subset_bbs,
            )
            if overlap_interval.is_valid:
                all_overlap_intervals[veh_id] = overlap_interval

        # Pedestrians
        for ped_id, ped_bbs in ped_predictions.items():
            overlap_interval = CollisionChecker._find_overlap_interval(
                actor_tbb_bboxes=ped_bbs,
                actor_ebb_bboxes=ped_bbs,
                route_subset_bboxes=route_subset_bbs,
                ignore_tbb=True
            )
            if overlap_interval.is_valid:
                all_overlap_intervals[ped_id] = overlap_interval

        return all_overlap_intervals

    @staticmethod
    def get_lane_intrusions(
        actors : List[carla.Actor],
        route_bboxes : List[carla.BoundingBox],
        *,
        bbox_inflation : float = 1.0,
    ) -> Dict[int, int]: # K=actor_id, V=intrusion_idx
        intruding_actors = {}

        for i, route_bb in enumerate(route_bboxes):
            for actor in actors:
                actor_id = actor.id

                # Construct actor bounding box in world frame
                actor_tf = actor.get_transform()

                loc = actor_tf.location
                rot = actor_tf.rotation
                extent = actor.bounding_box.extent
                extent.x *= bbox_inflation
                extent.y *= bbox_inflation

                actor_bb = carla.BoundingBox(loc, extent)
                actor_bb.rotation = carla.Rotation(
                    pitch=0,
                    roll=0,
                    yaw=rot.yaw
                )

                # Check if actor overlaps bounding box
                if GeometricUtils.check_obb_intersection_2d(actor_bb, route_bb):
                    if actor_id not in intruding_actors:
                        intruding_actors[actor_id] = i

        return intruding_actors

# TODO: Add collision visualization/plotting for debugging