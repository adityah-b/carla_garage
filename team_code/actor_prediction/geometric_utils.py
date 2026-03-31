import carla
import numpy as np

from typing import Tuple

class GeometricUtils:
    @staticmethod
    def _get_separating_plane(
        relative_position : carla.Vector3D,
        plane_normal : carla.Vector3D,
        obb1 : carla.BoundingBox,
        obb2 : carla.BoundingBox
    ) -> bool:
        """
        Check if there is a separating plane between two oriented bounding boxes (OBBs).

        Args:
            relative_position (carla.Vector3D): The relative position between the two OBBs.
            plane_normal (carla.Vector3D): The normal vector of the plane.
            obb1 (carla.BoundingBox): The first oriented bounding box.
            obb2 (carla.BoundingBox): The second oriented bounding box.

        Returns:
            bool: True if there is a separating plane, False otherwise.
        """
        dot_func = (lambda vec1, vec2 : vec1.dot(vec2))

        # Calculate the projection of the relative position onto the plane normal
        projection_distance = abs(dot_func(relative_position, plane_normal))

        # Calculate the sum of the projections of the OBB extents onto the plane normal
        obb1_projection = (
            abs(dot_func(obb1.rotation.get_forward_vector() * obb1.extent.x, plane_normal)) +
            abs(dot_func(obb1.rotation.get_right_vector() * obb1.extent.y, plane_normal)) +
            abs(dot_func(obb1.rotation.get_up_vector() * obb1.extent.z, plane_normal))
        )

        obb2_projection = (
            abs(dot_func(obb2.rotation.get_forward_vector() * obb2.extent.x, plane_normal)) +
            abs(dot_func(obb2.rotation.get_right_vector() * obb2.extent.y, plane_normal)) +
            abs(dot_func(obb2.rotation.get_up_vector() * obb2.extent.z, plane_normal))
        )

        # Check if the projection distance is greater than the sum of the OBB projections
        return projection_distance > obb1_projection + obb2_projection

    @staticmethod
    def check_obb_intersection(
        obb1 : carla.BoundingBox,
        obb2 : carla.BoundingBox
    ) -> bool:
        """
        Check if two 3D oriented bounding boxes (OBBs) intersect.

        Args:
            obb1 (carla.BoundingBox): The first oriented bounding box.
            obb2 (carla.BoundingBox): The second oriented bounding box.

        Returns:
            bool: True if the two OBBs intersect, False otherwise.
        """
        cross_func = (lambda vec1, vec2 : vec1.cross(vec2))
        relative_position = obb2.location - obb1.location

        # Check for separating planes along the axes of both OBBs
        if (GeometricUtils._get_separating_plane(relative_position, obb1.rotation.get_forward_vector(), obb1, obb2) or
            GeometricUtils._get_separating_plane(relative_position, obb1.rotation.get_right_vector(), obb1, obb2) or
            GeometricUtils._get_separating_plane(relative_position, obb1.rotation.get_up_vector(), obb1, obb2) or
            GeometricUtils._get_separating_plane(relative_position, obb2.rotation.get_forward_vector(), obb1, obb2) or
            GeometricUtils._get_separating_plane(relative_position, obb2.rotation.get_right_vector(), obb1, obb2) or
            GeometricUtils._get_separating_plane(relative_position, obb2.rotation.get_up_vector(), obb1, obb2)):

            return False

        # Check for separating planes along the cross products of the axes of both OBBs
        if (GeometricUtils._get_separating_plane(relative_position, cross_func(obb1.rotation.get_forward_vector(), \
                                                            obb2.rotation.get_forward_vector()), obb1,obb2) or
            GeometricUtils._get_separating_plane(relative_position, cross_func(obb1.rotation.get_forward_vector(), \
                                                            obb2.rotation.get_right_vector()), obb1,obb2) or
            GeometricUtils._get_separating_plane(relative_position, cross_func(obb1.rotation.get_forward_vector(), \
                                                            obb2.rotation.get_up_vector()), obb1,obb2) or
            GeometricUtils._get_separating_plane(relative_position, cross_func(obb1.rotation.get_right_vector(), \
                                                            obb2.rotation.get_forward_vector()), obb1,obb2) or
            GeometricUtils._get_separating_plane(relative_position, cross_func(obb1.rotation.get_right_vector(), \
                                                            obb2.rotation.get_right_vector()), obb1, obb2) or
            GeometricUtils._get_separating_plane(relative_position, cross_func(obb1.rotation.get_right_vector(), \
                                                            obb2.rotation.get_up_vector()), obb1, obb2) or
            GeometricUtils._get_separating_plane(relative_position, cross_func(obb1.rotation.get_up_vector(), \
                                                            obb2.rotation.get_forward_vector()), obb1,obb2) or
            GeometricUtils._get_separating_plane(relative_position, cross_func(obb1.rotation.get_up_vector(), \
                                                            obb2.rotation.get_right_vector()), obb1,obb2) or
            GeometricUtils._get_separating_plane(relative_position, cross_func(obb1.rotation.get_up_vector(), \
                                                            obb2.rotation.get_up_vector()), obb1, obb2)):

            return False

        # If no separating plane is found, the OBBs intersect
        return True

    @staticmethod
    def get_actor_bbox(actor : carla.Actor) -> carla.BoundingBox:
        actor_tf = actor.get_transform()

        loc = actor_tf.location
        rot = actor_tf.rotation
        extent = actor.bounding_box.extent

        actor_bb = carla.BoundingBox(loc, extent)
        actor_bb.rotation = carla.Rotation(
            pitch=0,
            roll=0,
            yaw=rot.yaw
        )

        return actor_bb

    @staticmethod
    def _get_separating_plane_2d(
        relative_position: carla.Vector3D,
        plane_normal: carla.Vector3D,
        obb1: carla.BoundingBox,
        obb2: carla.BoundingBox,
    ) -> bool:
        """
        2-D separating-plane test along plane_normal.

        Identical in structure to _get_separating_plane but uses only the XY
        components of all vectors, so elevation differences between boxes do not
        contribute to the projection and cannot produce false negatives.
        """
        dot2 = lambda v1, v2: v1.x * v2.x + v1.y * v2.y

        projection_distance = abs(dot2(relative_position, plane_normal))

        obb1_projection = (
            abs(dot2(obb1.rotation.get_forward_vector() * obb1.extent.x, plane_normal)) +
            abs(dot2(obb1.rotation.get_right_vector()   * obb1.extent.y, plane_normal))
        )

        obb2_projection = (
            abs(dot2(obb2.rotation.get_forward_vector() * obb2.extent.x, plane_normal)) +
            abs(dot2(obb2.rotation.get_right_vector()   * obb2.extent.y, plane_normal))
        )

        return projection_distance > obb1_projection + obb2_projection

    @staticmethod
    def check_obb_intersection_2d(
        obb1: carla.BoundingBox,
        obb2: carla.BoundingBox,
    ) -> bool:
        """
        2-D OBB overlap test using the Separating Axis Theorem.

        Only the 4 face-normal axes are tested (forward and right of each box).
        The 9 edge-cross-product axes from the 3-D version are unnecessary in 2-D
        and are omitted.  Elevation is ignored entirely, preventing false negatives
        caused by z-drift in the KBM forecast.
        """
        relative_position = obb2.location - obb1.location

        return not (
            GeometricUtils._get_separating_plane_2d(relative_position, obb1.rotation.get_forward_vector(), obb1, obb2) or
            GeometricUtils._get_separating_plane_2d(relative_position, obb1.rotation.get_right_vector(),   obb1, obb2) or
            GeometricUtils._get_separating_plane_2d(relative_position, obb2.rotation.get_forward_vector(), obb1, obb2) or
            GeometricUtils._get_separating_plane_2d(relative_position, obb2.rotation.get_right_vector(),   obb1, obb2)
        )

    @staticmethod
    def get_bbox_corners_xy(bbox: carla.BoundingBox) -> np.ndarray:
        """Returns the 4 corners of a carla.BoundingBox as a (4, 2) XY array.

        Uses CARLA's get_world_vertices with an identity transform.  The 8
        returned vertices are z-paired — adjacent indices share the same XY but
        differ only in z — so stride-2 indexing [0::2] yields the 4 unique XY
        corners without any manual rotation arithmetic.
        """
        verts = bbox.get_world_vertices(carla.Transform())
        return np.array([[v.x, v.y] for v in verts[0::2]])

    @staticmethod
    def frenet_to_cartesian(
        frenet_path: np.ndarray,
        route_xy: np.ndarray,
        route_yaws: np.ndarray,
        s_route: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a path in Frenet coordinates back to Cartesian.

        Args:
            frenet_path:  (N, 2) array of [s, d] values.
            route_xy:     (M, 2) reference path coordinates.
            route_yaws:   (M,)   heading angles (degrees) along the reference path.
            s_route:      (M,)   accumulated arc-length along the reference path.

        Returns:
            xy:   (N, 2) Cartesian positions.
            yaws: (N,)   headings (degrees) recomputed from the Cartesian trajectory.
        """
        s_vals = frenet_path[:, 0]   # (N,)
        d_vals = frenet_path[:, 1]   # (N,)

        # ── 1. Match each s value to its closest route point ─────────────────────
        nearest_idx = np.argmin(
            np.abs(s_vals[:, np.newaxis] - s_route[np.newaxis, :]), axis=1
        )  # (N,)

        # ── 2. Extract reference state at matched points ──────────────────────────
        ref_xy    = route_xy[nearest_idx]                  # (N, 2)
        ref_theta = np.deg2rad(route_yaws[nearest_idx])    # (N,)
        ref_s     = s_route[nearest_idx]                   # (N,)

        # ── 3. Refine along the tangent to sub-sample accuracy ────────────────────
        delta_s = s_vals - ref_s               # (N,)
        t_x = np.cos(ref_theta)                # (N,)
        t_y = np.sin(ref_theta)                # (N,)

        # ── 4. Offset along the normal ────────────────────────────────────────────
        # Flipped signs to fix the lateral coordinate inversion.
        # This calculates the normal 90° CW from the tangent: n = (sin θ, -cos θ)
        n_x =  np.sin(ref_theta)               # (N,)
        n_y = -np.cos(ref_theta)               # (N,)

        # ── 5. Reconstruct Cartesian position ────────────────────────────────────
        x = ref_xy[:, 0] + delta_s * t_x + d_vals * n_x   # (N,)
        y = ref_xy[:, 1] + delta_s * t_y + d_vals * n_y   # (N,)

        xy = np.stack([x, y], axis=1)          # (N, 2)

        # ── 6. Recompute Yaws ────────────────────────────────────────────────────
        dx = np.gradient(x)
        dy = np.gradient(y)
        recomputed_yaws = np.rad2deg(np.arctan2(dy, dx))   # (N,)

        return xy, recomputed_yaws