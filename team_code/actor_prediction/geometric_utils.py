import carla

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