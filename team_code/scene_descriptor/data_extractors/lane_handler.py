import carla
import hashlib
import struct
import numpy as np

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from collections import OrderedDict

from .junction_handler import JunctionHandler
from agents.navigation.global_route_planner import GlobalRoutePlanner

SparseKey = Tuple[int, int, int]

def wp_key(wp : carla.Waypoint) -> SparseKey:
    return (int(wp.road_id), int(wp.section_id), int(wp.lane_id))

class Lanelet:
    def __init__(
        self,
        sparse_points : OrderedDict[SparseKey, carla.Waypoint],
        dense_points : np.ndarray,
        lanelet_id : int,
        dense_waypoints : List[carla.Waypoint] = [],
    ):
        self.sparse_points = sparse_points
        self.dense_points = dense_points
        self.lanelet_id = lanelet_id
        # NOTE, TODO: TEMPORARY FOR DEBUGGING
        self.dense_waypoints = dense_waypoints

    def waypoints_list(self):
        # return list(self.points.values())
        return self.dense_waypoints

    def lanelet_sections(self):
        return set(self.sparse_points.keys())

    # def find(self, wp: carla.Waypoint) -> int:
    #     if not self.dense_points:
    #         raise ValueError("Lanelet has no dense points")

    #     loc = wp.transform.location
    #     best_i = 0
    #     best_d2 = float("inf")

    #     # Use squared distance (no sqrt) for speed
    #     for i, w in enumerate(self.dense_points):
    #         l = w.transform.location
    #         dx, dy, dz = l.x - loc.x, l.y - loc.y, l.z - loc.z
    #         d2 = dx*dx + dy*dy + dz*dz
    #         if d2 < best_d2:
    #             best_d2 = d2
    #             best_i = i

    #     return best_i

    def find(self, wp: carla.Waypoint) -> int:
        if self.dense_points.size == 0:
            raise ValueError("Lanelet has no dense points")

        loc = wp.transform.location
        p = np.array([loc.x, loc.y, loc.z], dtype=self.dense_points.dtype)

        diff = self.dense_points - p  # (N, 3)
        d2 = (diff * diff).sum(axis=1)  # (N,)
        return int(d2.argmin())

    def __len__(self):
        return len(self.sparse_points)

    def __contains__(self, wp: carla.Waypoint) -> bool:
        key = wp_key(wp)
        # TODO: ADD A WAYPOINT.S VALUE COMPARISON WITH FIRST AND LAST LANELET WPS TO AVOID SCENARIOS WHERE VEHICLES BEYOND THE LANELET EDGES ARE ASSIGNED DUE TO SAME KEY
        return key in self.sparse_points

    def contains_wp(
        self,
        wp : carla.Waypoint,
        max_distance : float = 2.0,
    ) -> bool:
        if wp is None:
            return False

        if self.dense_points.size == 0:
            return False

        loc = wp.transform.location
        p = np.array([loc.x, loc.y, loc.z], dtype=self.dense_points.dtype)

        diff = self.dense_points - p
        d2 = np.square(diff).sum(axis=1)
        min_d2 = d2.min()

        # print(f'MIN DIST: {np.sqrt(min_d2)}, MAX_DIST: {max_distance}')

        if min_d2 > max_distance * max_distance:
            return False

        return True

class WaypointUtils:
    WAYPOINT_SAMPLING_RESOLUTION : float = 2.0

    @staticmethod
    def _find_waypoint_entry_exit(
        grp : GlobalRoutePlanner,
        wp : carla.Waypoint
    ) -> Tuple[Optional[carla.Waypoint], Optional[carla.Waypoint]]:
        """
        Find the entry and exit points of the waypoint's corresponding road segment.
        Args:
            waypoint (carla.Waypoint): The waypoint.

        Returns:
            tuple: A tuple containing the entry and exit points of the road segment.
        """
        entry_wp, exit_wp = None, None

        try:
            node_pair = grp._road_id_to_edge[wp.road_id][wp.section_id][wp.lane_id]
        except KeyError:
            print(f'Could not find edge for waypoint {wp.transform.location}')

        if node_pair:
            entry_xyz, exit_xyz = node_pair
            edge = grp._graph.edges[entry_xyz, exit_xyz]
            entry_wp = edge['entry_waypoint']
            exit_wp = edge['exit_waypoint']

        return (entry_wp, exit_wp)

    @staticmethod
    def get_distance(
        wp1 : carla.Waypoint,
        wp2 : carla.Waypoint
    ) -> float:
        """
        Get the distance between two waypoints
        """
        return wp1.transform.location.distance(wp2.transform.location)

    @staticmethod
    def interpolate_between_waypoints(
        source_wp : carla.Waypoint,
        target_wp : carla.Waypoint,
        backward : bool = False
    ) -> List[carla.Waypoint]:
        """
        Interpolate waypoints between source and target waypoints

        Args:
            source_wp (carla.Waypoint): Source waypoint.
            target_wp (carla.Waypoint): Target waypoint.

        Returns:
            list: List of interpolated waypoints between source and target waypoints up to desired distance.
        """
        # print(f'Interpolating between waypoints {source_wp.transform.location} and {target_wp.transform.location}')
        interp_wps = [source_wp]
        cur_wp = source_wp

        if backward:
            step_func = (lambda wp : wp.previous(WaypointUtils.WAYPOINT_SAMPLING_RESOLUTION))
        else:
            step_func = (lambda wp : wp.next(WaypointUtils.WAYPOINT_SAMPLING_RESOLUTION))

        while WaypointUtils.get_distance(cur_wp, target_wp) > WaypointUtils.WAYPOINT_SAMPLING_RESOLUTION and \
                WaypointUtils.get_distance(cur_wp, source_wp) < WaypointUtils.get_distance(target_wp, source_wp):
            # wp_choice = cur_wp.next(WaypointUtils.WAYPOINT_SAMPLING_RESOLUTION)
            wp_choice = step_func(cur_wp)
            if len(wp_choice) > 1:
                max_dot = -1 * np.inf
                target_wp_vec = target_wp.transform.get_forward_vector()
                for wp in wp_choice:
                    wp_vec = wp.transform.get_forward_vector()
                    target_select_wp_dot = target_wp_vec.dot(wp_vec)
                    # Select waypoint with straightest path to target waypoint
                    if target_select_wp_dot > max_dot:
                        max_dot = target_select_wp_dot
                        cur_wp = wp

            elif not wp_choice:
                break

            else:
                cur_wp = wp_choice[0]

            interp_wps.append(cur_wp)

        interp_wps.append(target_wp)

        return interp_wps

    @staticmethod
    def get_waypoint_at_distance(
        source_wp : carla.Waypoint,
        distance : float,
        backward : bool = False
    ) -> carla.Waypoint:
        """
        Get a waypoint at a certain distance from the input waypoint.

        Args:
            waypoint (carla.Waypoint): The input waypoint.
            distance (float): The distance to travel.
            ego_entry_exit_pairs (list): List of entry and exit points of the ego vehicle's route.

        Returns:
            list: List of waypoints at the specified distance.
        """
        # print(f'Getting waypoints at distance {distance} from {waypoint.transform.location}')
        traveled_distance = 0.0
        cur_wp = source_wp
        if backward:
            step_func = (lambda wp : wp.previous(WaypointUtils.WAYPOINT_SAMPLING_RESOLUTION))
        else:
            step_func = (lambda wp : wp.next(WaypointUtils.WAYPOINT_SAMPLING_RESOLUTION))

        while traveled_distance < distance:
            # wp_choice = cur_wp.next(WaypointUtils.WAYPOINT_SAMPLING_RESOLUTION)
            wp_choice = step_func(cur_wp)
            if not wp_choice:
                break
            cur_wp = wp_choice[0]
            traveled_distance = WaypointUtils.get_distance(cur_wp, source_wp)

        return cur_wp

class LaneHandler:
    """
    Identifies all ongoing and oncoming lanes
    """

    # @staticmethod
    # def _to_lanelet(
    #     dense_waypoints : List[carla.Waypoint]
    # ) -> Lanelet:
    #     """
    #     Generate lanelet given waypoint list
    #     """
    #     sparse_lanelet_points = OrderedDict()
    #     for wp in dense_waypoints:
    #         key = (wp.road_id, wp.section_id, wp.lane_id)
    #         sparse_lanelet_points.setdefault(key, wp)

    #     dense_points = np.array(
    #         [
    #             [wp.transform.location.x, wp.transform.location.y, wp.transform.location.z]
    #             for wp in dense_waypoints
    #         ],
    #         dtype=np.float32
    #     )
    #     return Lanelet(sparse_points=sparse_lanelet_points, dense_points=dense_points, dense_waypoints=dense_waypoints)

    @staticmethod
    def _to_lanelet(
        dense_waypoints : List[carla.Waypoint]
    ) -> Lanelet:
        """
        Generate lanelet given waypoint list
        """
        sparse_lanelet_points = OrderedDict()
        dense_xyz = []

        # Lanelet hash id
        h = hashlib.blake2b(digest_size=8)
        h.update(b"LL64v1")

        first_wp = None
        last_wp = None

        for wp in dense_waypoints:
            if first_wp is None:
                first_wp = wp
            last_wp = wp

            loc = wp.transform.location
            dense_xyz.append((loc.x, loc.y, loc.z))

            key = wp_key(wp)
            if key not in sparse_lanelet_points:
                sparse_lanelet_points[key] = wp
                h.update(struct.pack("<iii", *key))


        # Update hash id with lane endpoints
        s0 = first_wp.s
        s1 = last_wp.s

        h.update(struct.pack("<ii", int(round(s0)), int(round(s1))))

        # Generate integer lane id from hash
        lanelet_id = int.from_bytes(h.digest(), byteorder='little', signed=False)

        dense_points = np.asarray(dense_xyz, dtype=np.float32)

        return Lanelet(
            sparse_points=sparse_lanelet_points,
            dense_points=dense_points,
            lanelet_id=lanelet_id,
            dense_waypoints=dense_waypoints
        )

    @staticmethod
    def _gen_wps(
        start_wp : carla.Waypoint,
        wp_step_func,
        junc_step_func,
        max_length : float = 80.0,
        backward : bool = False,
    ) -> List[List[carla.Waypoint]]:
        """
        Generate dense waypoints that extend up to max_length metres from start_wp,
        incorporating full junction crossings when present.
        """
        all_dense_wps : List[List[carla.Waypoint]] = []

        # Look nearby for any junctions
        # NOTE: This also accounts for the case where start_wp is a junction waypoint
        junction_pair = junc_step_func(start_wp, max_length)
        if junction_pair:
            junction_entry_wp, junction_wp = junction_pair
            junction_map = JunctionHandler.create_junction_map(junction_wp)
            junction_connections = JunctionHandler.get_junction_connections(junction_map, junction_entry_wp)

            # If no junction connections, return single dense waypoint list
            if not junction_connections:
                return [[start_wp]]

            # Generate waypoints for each possible path through the junction
            for j_conn in junction_connections:
                anchors = [
                    j_conn.entry_connection,
                    j_conn.entry_junction,
                    j_conn.exit_junction,
                    j_conn.exit_connection
                ]
                if backward:
                    anchors.reverse()

                # if not start_wp.is_junction:
                #     anchors.insert(0, start_wp)

                # NOTE and TODO: Adding start_wp to start of anchors list for bidirectional generation
                anchors.insert(0, start_wp)

                # extend beyond junction up to max_length
                cur_dist = WaypointUtils.get_distance(anchors[0], anchors[-1])
                dist_left = max_length - cur_dist
                if dist_left > 0:
                    last_wp = anchors[-1]
                    choices = wp_step_func(last_wp, dist_left)
                    end_wp = choices[0] if choices else WaypointUtils.get_waypoint_at_distance(
                        last_wp, dist_left, backward
                    )
                    anchors.append(end_wp)

                    # end_wp = WaypointUtils.get_waypoint_at_distance(
                    #     last_wp, grp, dist_left, backward
                    # )
                    # anchors.append(end_wp)

                # one dense interpolation pass
                dense_wps = []
                for a, b in zip(anchors[:-1], anchors[1:]):
                    dense_wps.extend(WaypointUtils.interpolate_between_waypoints(a, b, backward))

                if backward:
                    dense_wps.reverse()

                all_dense_wps.append(dense_wps)
        else:
            end_wp_choices = wp_step_func(start_wp, max_length)
            end_wp = end_wp_choices[0] if end_wp_choices else WaypointUtils.get_waypoint_at_distance(start_wp, max_length, backward)
            dense_wps = WaypointUtils.interpolate_between_waypoints(start_wp, end_wp, backward)
            if backward:
                dense_wps.reverse()
            all_dense_wps.append(dense_wps)

        return all_dense_wps

    @staticmethod
    def generate_lanelet(
        start_wp : carla.Waypoint,
        max_length : float = 80.0,
        bidirectional : bool = False,
        backward : bool = False,
    ) -> List[Lanelet]:
        """
        Generate lanelets that extend up to max_length metres from start_wp,
        incorporating full junction crossings when present.
        """
        lanelets : List[Lanelet] = []

        if bidirectional:
            fwd_dense_lanelets = LaneHandler._gen_wps(
                start_wp=start_wp,
                wp_step_func=(lambda wp, dist : wp.next(dist)),
                junc_step_func=(JunctionHandler.get_next_junction),
                max_length=(max_length / 2),
                backward=False
            )

            bwd_dense_lanelets = LaneHandler._gen_wps(
                start_wp=start_wp,
                wp_step_func=(lambda wp, dist : wp.previous(dist)),
                junc_step_func=(JunctionHandler.get_previous_junction),
                max_length=(max_length / 2),
                backward=True
            )

            for fwd_lanelet in fwd_dense_lanelets:
                for bwd_lanelet in bwd_dense_lanelets:
                    dense_lanelet = bwd_lanelet[:-1] + fwd_lanelet
                    lanelets.append(LaneHandler._to_lanelet(dense_lanelet))

            return lanelets

        wp_step_func = (lambda wp, dist : wp.previous(dist)) if backward else (lambda wp, dist : wp.next(dist))
        junc_step_func = (JunctionHandler.get_previous_junction) if backward else (JunctionHandler.get_next_junction)

        dense_lanelets = LaneHandler._gen_wps(start_wp=start_wp, wp_step_func=wp_step_func, junc_step_func=junc_step_func, max_length=max_length, backward=backward)
        for dense_lanelet in dense_lanelets:
            lanelets.append(LaneHandler._to_lanelet(dense_lanelet))

        return lanelets

    # @staticmethod
    # def generate_lanelet(
    #     start_wp : carla.Waypoint,
    #     grp : GlobalRoutePlanner,
    #     max_length : float = 80.0,
    #     backward : bool = False
    # ) -> List[Lanelet]:
    #     """
    #     Generate lanelets that extend up to max_length metres from start_wp,
    #     incorporating full junction crossings when present.
    #     """
    #     wp_step = (lambda wp, dist : wp.previous(dist)) if backward else (lambda wp, dist : wp.next(dist))
    #     find_junc = (JunctionHandler.get_previous_junction) if backward else (JunctionHandler.get_next_junction)

    #     lanelets = []
    #     # Look nearby for any junctions
    #     junction_pair = find_junc(start_wp, max_length)
    #     if junction_pair:
    #         # print(f'{LaneHandler.generate_lanelet.__name__}: Junction')
    #         junction_entry_wp, junction_wp = junction_pair
    #         junction_map = JunctionHandler.create_junction_map(junction_wp)
    #         junction_connections = JunctionHandler.get_junction_connections(junction_map, junction_entry_wp)

    #         # If no junction connections, return single waypoint lanelet
    #         if not junction_connections:
    #             lanelets.append(LaneHandler._to_lanelet([start_wp]))
    #             return lanelets

    #         # for each possible path through the junction...
    #         for j_conn in junction_connections:
    #             anchors = [
    #                 j_conn.entry_connection,
    #                 j_conn.entry_junction,
    #                 j_conn.exit_junction,
    #                 j_conn.exit_connection
    #             ]
    #             if backward:
    #                 anchors.reverse()

    #             if not start_wp.is_junction:
    #                 anchors.insert(0, start_wp)

    #             # extend beyond junction up to max_length
    #             cur_dist = WaypointUtils.get_distance(start_wp, anchors[-1])
    #             dist_left = max_length - cur_dist
    #             if dist_left > 0:
    #                 last_wp = anchors[-1]
    #                 # choices = wp_step(last_wp, dist_left)
    #                 # end_wp = choices[0] if choices else WaypointUtils.get_waypoint_at_distance(
    #                 #     last_wp, grp, dist_left, backward
    #                 # )
    #                 # anchors.append(end_wp)
    #                 end_wp = WaypointUtils.get_waypoint_at_distance(
    #                     last_wp, dist_left, backward
    #                 )
    #                 anchors.append(end_wp)

    #             # one dense interpolation pass
    #             dense_lanelet = []
    #             for a, b in zip(anchors[:-1], anchors[1:]):
    #                 dense_lanelet.extend(WaypointUtils.interpolate_between_waypoints(a, b, grp, backward))

    #             if backward:
    #                 dense_lanelet.reverse()

    #             lanelets.append(LaneHandler._to_lanelet(dense_lanelet))
    #     else:
    #         # no junction: anchor = [start → end]
    #         # print(f'{LaneHandler.generate_lanelet.__name__}: No junction')
    #         end_wp_choices = wp_step(start_wp, max_length)
    #         end_wp = end_wp_choices[0] if end_wp_choices else WaypointUtils.get_waypoint_at_distance(start_wp, max_length, backward)
    #         dense_lanelet = WaypointUtils.interpolate_between_waypoints(start_wp, end_wp, grp, backward)
    #         if backward:
    #             dense_lanelet.reverse()
    #         lanelets.append(LaneHandler._to_lanelet(dense_lanelet))

    #     return lanelets

    # @staticmethod
    # def generate_lanelet(
    #     start_wp : carla.Waypoint,
    #     grp : GlobalRoutePlanner,
    #     max_length : float = 50.0
    # ) -> List[Lanelet]:
    #     """
    #     Generate lanelets that extend up to max_length metres from start_wp,
    #     incorporating full junction crossings when present.
    #     """
    #     lanelets = []
    #     # Look ahead for any junctions
    #     junction_pair = JunctionHandler.get_next_junction(start_wp, max_length)
    #     if junction_pair:
    #         junction_entry_wp, junction_wp = junction_pair
    #         junction_map = JunctionHandler.create_junction_map(junction_wp)
    #         junction_connections = JunctionHandler.get_junction_connections(junction_map, junction_entry_wp)

    #         # for each possible path through the junction...
    #         for j_conn in junction_connections:
    #             anchors = [
    #                 j_conn.entry_connection,
    #                 j_conn.entry_junction,
    #                 j_conn.exit_junction,
    #                 j_conn.exit_connection
    #             ]
    #             if not start_wp.is_junction:
    #                 anchors.insert(0, start_wp)

    #             # extend beyond junction up to max_length
    #             last_loc = anchors[-1].transform.location
    #             dist_left = max_length - start_wp.transform.location.distance(last_loc)
    #             if dist_left > 0:
    #                 # pick endpoint either via .next() or planner lookup
    #                 choices = anchors[-1].next(dist_left)
    #                 end_wp = choices[0] if choices else WaypointUtils.get_waypoint_at_distance(
    #                     anchors[-1], grp, dist_left
    #                 )
    #                 anchors.append(end_wp)

    #             # one dense interpolation pass
    #             dense_lanelet = []
    #             for a, b in zip(anchors[:-1], anchors[1:]):
    #                 dense_lanelet.extend(WaypointUtils.interpolate_between_waypoints(a, b, grp))

    #             lanelets.append(LaneHandler._to_lanelet(dense_lanelet))
    #     else:
    #         # no junction: anchor = [start → end]
    #         end_wp_choices = start_wp.next(max_length)
    #         end_wp = end_wp_choices[0] if end_wp_choices else WaypointUtils.get_waypoint_at_distance(start_wp, grp, max_length)
    #         dense_lanelet = WaypointUtils.interpolate_between_waypoints(start_wp, end_wp, grp)
    #         lanelets.append(LaneHandler._to_lanelet(dense_lanelet))

    #     return lanelets

    # @staticmethod
    # def get_same_dir_lanes(
    #     waypoint : carla.Waypoint
    # ) -> List[carla.Waypoint]:
    #     """
    #     Gets immediate left and right lanes with the same direction of the road of a wp.

    #     Args:
    #         waypoint (carla.Waypoint): Waypoint to start the search from.

    #     Returns:
    #         list: List of waypoints with the same direction of the road.
    #     """
    #     same_dir_wps = [waypoint]

    #     # Check roads on the right
    #     possible_right_wp = waypoint.get_right_lane()
    #     if possible_right_wp and possible_right_wp.lane_type == carla.LaneType.Driving:
    #         same_dir_wps.append(possible_right_wp)

    #     # Check roads on the left
    #     possible_left_wp = waypoint.get_left_lane()
    #     if possible_left_wp and possible_left_wp.lane_type == carla.LaneType.Driving and possible_left_wp.lane_id * waypoint.lane_id >= 0:
    #         same_dir_wps.insert(0, possible_left_wp)

    #     return same_dir_wps

    @staticmethod
    def _lr_group(wp: carla.Waypoint) -> List[carla.Waypoint]:
        """
        Build [left?, center, right?] around wp
        """
        group = [wp]

        l = wp.get_left_lane()
        if l and l.lane_type == carla.LaneType.Driving and (l.lane_id * wp.lane_id >= 0):
            group.insert(0, l)

        r = wp.get_right_lane()
        if r and r.lane_type == carla.LaneType.Driving:
            group.append(r)

        return group

    @staticmethod
    def _oncoming_group(wp: carla.Waypoint, max_hops : int = 1) -> List[carla.Waypoint]:
        """
        Build oncoming lanes around wp
        """
        other_dir_wp = None

        # Get the first lane of the opposite direction
        left_wp = wp
        lane_hops = 0
        while True:
            possible_left_wp = left_wp.get_left_lane()
            lane_hops += 1
            if possible_left_wp is None:
                break
            if possible_left_wp.lane_id * left_wp.lane_id < 0:
                other_dir_wp = possible_left_wp
                break
            left_wp = possible_left_wp

        if not other_dir_wp:
            return []

        # If first oncoming lane more than max_hops lane away, ignore
        if lane_hops > max_hops:
            return []

        group = []
        # Check roads on the right
        right_wp = other_dir_wp
        while True:
            if right_wp.lane_type == carla.LaneType.Driving:
                group.append(right_wp)
            possible_right_wp = right_wp.get_right_lane()
            if possible_right_wp is None:
                break
            right_wp = possible_right_wp

        return group

    @staticmethod
    def get_same_dir_lanes(
        waypoint: carla.Waypoint,
        route_waypoints: List[carla.Waypoint],
    ) -> Dict[str, object]:
        """
        Route-conditioned same-direction lanes with proper grouping.

        Returns:
          {
            "ref_wp":   carla.Waypoint,              # use for defining left/right meaning (entry/approach)
            "approach": List[carla.Waypoint],        # [left?, ref_wp, right?] at junction entry (or current if not junction)
            "exit":     Optional[List[carla.Waypoint]],  # [left?, exit_wp, right?] at route-selected junction exit
            "seeds":    List[carla.Waypoint],        # dedup union (good for lanelet generation)
          }
        """
        # Non-junction: only one group
        if not waypoint.is_junction:
            entry_corridor = LaneHandler._lr_group(waypoint)
            return {"entry_wp": waypoint, "entry_corridor": entry_corridor, "exit_wp": None, "exit_corridor": None, "seeds": entry_corridor}

        # Junction entry (approach anchor)
        junction_entry_wp = JunctionHandler.get_junction_entry_connection(waypoint)

        # Find target junction exit from global route
        junction_exit_wp = next((wp for wp in route_waypoints if not wp.is_junction), None)

        # print(f'\n\nWaypoints')
        # print(f'\tJUNCTION PTS')
        # print(f'\t\tentry --> lane_id: {junction_entry_wp.lane_id}, road_id: {junction_entry_wp.road_id}')
        # print(f'\t\texit --> lane_id: {junction_exit_wp.lane_id}, road_id: {junction_exit_wp.road_id}')

        entry_corridor = LaneHandler._lr_group(junction_entry_wp)
        exit_corridor = LaneHandler._lr_group(junction_exit_wp) if junction_exit_wp else None

        # print(f'\n\nCORRIDOR POINTS')
        # print(f'\tENTRY')
        # for i, entry_wp in enumerate(entry_corridor):
        #     print(f'\t\tcorridor {i} --> lane_id: {entry_wp.lane_id}, road_id: {entry_wp.road_id}')

        # print(f'\tEXIT')
        # for i, exit_wp in enumerate(exit_corridor):
        #     print(f'\t\tcorridor {i} --> lane_id: {exit_wp.lane_id}, road_id: {exit_wp.road_id}')

        # Unique lanelet generation points
        seeds: List[carla.Waypoint] = []
        seen = set()
        for wp in entry_corridor:
            k = wp_key(wp)
            if k not in seen:
                seen.add(k)
                seeds.append(wp)
        if exit_corridor:
            for wp in exit_corridor:
                k = wp_key(wp)
                if k not in seen:
                    seen.add(k)
                    seeds.append(wp)

        return {"entry_wp": junction_entry_wp, "entry_corridor": entry_corridor, "exit_wp": junction_exit_wp, "exit_corridor": exit_corridor, "seeds": seeds}

    @staticmethod
    def get_opposite_dir_lanes(
        waypoint : carla.Waypoint,
        route_waypoints: List[carla.Waypoint],
        oncoming_angle : float = 30.0,
        max_hops : int = 1,
    # ) -> Dict[str, object]:
    ) -> List[carla.Waypoint]:
        """
        Gets all the lanes with opposite direction of the road of a wp
        Ordered from the center lane to the edge one (from inwards to outwards)

        Args:
            waypoint (carla.Waypoint): Waypoint to start the search from.

        Returns:
            list: List of waypoints with opposite direction of the road.
        """
        # Non-junction: only one group
        if not waypoint.is_junction:
            entry_corridor = LaneHandler._oncoming_group(waypoint, max_hops)
            return entry_corridor
            # return {"entry_wp": waypoint, "entry_corridor": entry_corridor, "exit_wp": None, "exit_corridor": None, "seeds": entry_corridor}

        # Junction entry
        junction_entry_wp = JunctionHandler.get_junction_entry_connection(waypoint)
        target_junction_entry_vec = junction_entry_wp.transform.get_forward_vector()

        # Find target junction exit from global route
        junction_exit_wp = next((wp for wp in route_waypoints if not wp.is_junction), None)
        target_junction_exit_vec = junction_exit_wp.transform.get_forward_vector()

        oncoming_wps = []

        # oncoming_wps.extend(LaneHandler._oncoming_group(junction_entry_wp, max_hops))
        # if junction_exit_wp:
        #     oncoming_wps.extend(LaneHandler._oncoming_group(junction_exit_wp, max_hops))

        # Create junction map
        junction_map = JunctionHandler.create_junction_map(waypoint)

        # Get all junction connections
        seen = set()
        dot_threshold = np.cos(np.deg2rad(oncoming_angle))
        for junction_connections in junction_map.values():
            for j_conn in junction_connections:
                junc_entry_vec = j_conn.entry_junction.transform.get_forward_vector()

                entry_oncoming_wrt_target_entry = (junc_entry_vec.dot(target_junction_entry_vec) <= -dot_threshold)
                entry_oncoming_wrt_target_exit  = (junc_entry_vec.dot(target_junction_exit_vec)  <= -dot_threshold)

                if not (entry_oncoming_wrt_target_entry or entry_oncoming_wrt_target_exit):
                    continue

                # Unique lanelet generation points
                k = wp_key(j_conn.exit_junction)
                if k not in seen:
                    seen.add(k)
                    oncoming_wps.append(j_conn.exit_junction)

        return oncoming_wps

    @staticmethod
    def get_cross_dir_lanes(
        waypoint : carla.Waypoint,
        crossing_angle : float = 30.0,
    ) -> List[carla.Waypoint]:
        """
        Gets all the lanes that are perpendicular to the direction of the road of a wp
        """
        start_wp = waypoint
        if waypoint.is_junction:
            start_wp = JunctionHandler.get_junction_entry_connection(waypoint)

        if start_wp is None:
            return []

        # Cross directional lanes are only present at junctions
        junction_pair = JunctionHandler.get_next_junction(start_wp)
        if not junction_pair:
            return []

        ego_entry_wp, junction_wp = junction_pair
        ego_wp_vec = ego_entry_wp.transform.get_forward_vector()

        junction_map = JunctionHandler.create_junction_map(junction_wp)

        cross_wps = []

        # Get all junction connections
        seen = set()
        dot_threshold = np.cos(np.deg2rad(90 - crossing_angle))
        for junction_connections in junction_map.values():
            for j_conn in junction_connections:
                junc_entry_vec = j_conn.entry_junction.transform.get_forward_vector()
                junc_exit_vec = j_conn.exit_junction.transform.get_forward_vector()

                # Crossing lanelets are perpendicular to ego's lanelet
                if np.abs(junc_entry_vec.dot(ego_wp_vec)) >= dot_threshold:
                    continue
                # if np.abs(junc_exit_vec.dot(ego_wp_vec)) >= dot_threshold:
                #     continue

                # Unique lanelet generation points
                k = wp_key(j_conn.exit_junction)
                if k not in seen:
                    seen.add(k)
                    cross_wps.append(j_conn.exit_junction)

        return cross_wps
