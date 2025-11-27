import carla
import numpy as np

from scipy import ndimage
from typing import List, Dict
from dataclasses import dataclass

from .geometric_utils import GeometricUtils

@dataclass
class CollisionInterval:
    start_idx : int
    end_idx : int

    collision_bboxes_a : List[carla.BoundingBox]
    collision_bboxes_b : List[carla.BoundingBox]

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
                if GeometricUtils.check_obb_intersection(bb_a, bounding_boxes_b[j]):
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

# TODO: Add collision visualization/plotting for debugging