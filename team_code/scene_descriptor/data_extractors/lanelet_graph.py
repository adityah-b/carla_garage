import carla
import numpy as np
import h5py
import networkx as nx

from typing import Dict, List, Tuple, OrderedDict, Set, Optional
from collections import OrderedDict as ODict

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption
from .lane_handler import Lanelet

# Structured dtype for HDF5 datasets
WP_DTYPE = np.dtype([
    ("road_id",     np.int32),
    ("section_id",  np.int32),
    ("lane_id",     np.int32),
    ("s",           np.float32),
])

def pos_to_wp(world_map : carla.Map, pos):
    # Reconstruct Waypoint from XODR. Section comes from map.
    carla_loc = carla.Location(
        x = pos[0], y=pos[1], z=pos[2]
    )
    wp = world_map.get_waypoint(carla_loc)
    # (Optional) sanity check
    # assert wp.section_id == int(rec["section_id"])
    return wp

class LaneletGraph:
    """
    A graph structure parallel to GlobalRoutePlanner's graph but with Lanelet objects.
    Maintains the same node IDs and connectivity for compatibility.
    """

    def __init__(self, world_map: carla.Map, sampling_resolution: float = 2.0):
        """
        Initialize the lanelet graph with a CARLA map.

        Args:
            world_map: CARLA map object
            sampling_resolution: Distance between waypoints in meters
        """
        self.world_map = world_map
        self.sampling_resolution = sampling_resolution
        self.route_planner = GlobalRoutePlanner(world_map, sampling_resolution)

        # Create a new graph with the same structure but Lanelet objects
        self.graph = nx.DiGraph()

        # Mapping from waypoint to lanelet for quick actor categorization
        self.waypoint_to_lanelet = {}  # Maps (road_id, section_id, lane_id) to (node1, node2)

        # Build the lanelet graph
        self._build_lanelet_graph()

    def _build_lanelet_graph(self):
        """
        Build a graph with the same structure as GlobalRoutePlanner but with Lanelet objects.
        """
        original_graph = self.route_planner._graph

        # Copy nodes from original graph
        for node_id, node_data in original_graph.nodes(data=True):
            self.graph.add_node(node_id, **node_data)

        # Process each edge and create Lanelet objects
        for n1, n2, edge_data in original_graph.edges(data=True):
            lanelet = self._create_lanelet_from_edge(edge_data)

            # Create new edge with Lanelet object
            new_edge_data = edge_data.copy()
            new_edge_data['lanelet'] = lanelet

            self.graph.add_edge(n1, n2, **new_edge_data)

            # Update waypoint mapping for quick lookups
            if lanelet and edge_data['type'] == RoadOption.LANEFOLLOW:
                for wp in lanelet.dense_points:
                    key = (wp.road_id, wp.section_id, wp.lane_id)
                    if key not in self.waypoint_to_lanelet:
                        self.waypoint_to_lanelet[key] = []
                    self.waypoint_to_lanelet[key].append((n1, n2))

    def _create_lanelet_from_edge(self, edge_data: Dict) -> Optional[Lanelet]:
        """
        Create a Lanelet object from edge data.

        Args:
            edge_data: Dictionary containing edge information from GlobalRoutePlanner

        Returns:
            Lanelet object or None if edge doesn't represent a lane
        """
        # Only create lanelets for actual lane segments
        if edge_data['type'] not in [RoadOption.LANEFOLLOW, RoadOption.CHANGELANELEFT,
                                      RoadOption.CHANGELANERIGHT]:
            return None

        entry_wp = edge_data.get('entry_waypoint')
        exit_wp = edge_data.get('exit_waypoint')
        path = edge_data.get('path', [])

        if not entry_wp:
            return None

        # Build sparse points (key waypoints)
        sparse_points = ODict()

        # Add entry waypoint
        entry_key = (entry_wp.road_id, entry_wp.section_id, entry_wp.lane_id)
        sparse_points[entry_key] = entry_wp

        # Add exit waypoint if it exists and is different
        if exit_wp:
            exit_key = (exit_wp.road_id, exit_wp.section_id, exit_wp.lane_id)
            if exit_key != entry_key:
                sparse_points[exit_key] = exit_wp

        # Build dense points (all waypoints)
        dense_points = []

        # Add entry
        dense_points.append(entry_wp)

        # Add path waypoints
        for wp in path:
            if wp not in dense_points:  # Avoid duplicates
                dense_points.append(wp)

        # Add exit if it exists and isn't already added
        if exit_wp and exit_wp not in dense_points:
            dense_points.append(exit_wp)

        return Lanelet(sparse_points, dense_points)

    def find_lanelet(
        self,
        target_wp: carla.Waypoint
    ) -> Optional[Tuple[int, int, Lanelet]]:
        """
        Find the lanelet that contains the given waypoint.

        Args:
            actor_waypoint: Waypoint of an actor

        Returns:
            Tuple of (node1, node2, lanelet) or None if not found
        """
        key = (target_wp.road_id, target_wp.section_id, target_wp.lane_id)

        # Check if we have any edges for this road/section/lane
        if key not in self.waypoint_to_lanelet:
            return None

        # Find the best matching lanelet based on distance
        best_match = None
        best_distance = float('inf')

        for n1, n2 in self.waypoint_to_lanelet[key]:
            edge_data = self.graph.edges[n1, n2]
            lanelet = edge_data.get('lanelet')

            if lanelet and target_wp in lanelet:
                # Find distance to closest point in lanelet
                idx = lanelet.find(target_wp)
                wp = lanelet.dense_points[idx]
                distance = target_wp.transform.location.distance(wp.transform.location)

                if distance < best_distance:
                    best_distance = distance
                    best_match = (n1, n2, lanelet)

        return best_match

    def get_connected_lanelets(self, node1: int, node2: int, distance: float,
                              direction: str = 'forward') -> List[Lanelet]:
        """
        Get a chain of connected lanelets starting from a given edge.

        Args:
            node1: Starting node
            node2: End node of starting edge
            distance: Desired length of lanelet chain in meters
            direction: 'forward', 'backward', or 'both'

        Returns:
            List of connected Lanelet objects
        """
        lanelets : List[Lanelet] = []
        accumulated_distance = 0.0

        if direction in ['forward', 'both']:
            # Traverse forward
            current_node = node2
            visited = {(node1, node2)}

            while accumulated_distance < distance:
                # Get successors
                successors = list(self.graph.successors(current_node))
                if not successors:
                    break

                # Find the best successor (preferring LANEFOLLOW)
                best_successor = None
                for succ in successors:
                    edge_data = self.graph.edges[current_node, succ]
                    if edge_data['type'] == RoadOption.LANEFOLLOW:
                        if (current_node, succ) not in visited:
                            best_successor = succ
                            break

                if not best_successor:
                    # Try any unvisited successor
                    for succ in successors:
                        if (current_node, succ) not in visited:
                            best_successor = succ
                            break

                if not best_successor:
                    break

                # Add the lanelet
                edge_data = self.graph.edges[current_node, best_successor]
                lanelet = edge_data.get('lanelet')

                if lanelet:
                    lanelets.append(lanelet)
                    # Calculate distance
                    for i in range(len(lanelet.dense_points) - 1):
                        wp1 = lanelet.dense_points[i]
                        wp2 = lanelet.dense_points[i + 1]
                        accumulated_distance += wp1.transform.location.distance(
                            wp2.transform.location
                        )

                visited.add((current_node, best_successor))
                current_node = best_successor

        if direction in ['backward', 'both']:
            # Traverse backward
            accumulated_distance = 0.0
            current_node = node1
            visited = {(node1, node2)}
            backward_lanelets = []

            while accumulated_distance < distance:
                # Get predecessors
                predecessors = list(self.graph.predecessors(current_node))
                if not predecessors:
                    break

                # Find the best predecessor (preferring LANEFOLLOW)
                best_predecessor = None
                for pred in predecessors:
                    edge_data = self.graph.edges[pred, current_node]
                    if edge_data['type'] == RoadOption.LANEFOLLOW:
                        if (pred, current_node) not in visited:
                            best_predecessor = pred
                            break

                if not best_predecessor:
                    # Try any unvisited predecessor
                    for pred in predecessors:
                        if (pred, current_node) not in visited:
                            best_predecessor = pred
                            break

                if not best_predecessor:
                    break

                # Add the lanelet
                edge_data = self.graph.edges[best_predecessor, current_node]
                lanelet = edge_data.get('lanelet')

                if lanelet:
                    backward_lanelets.insert(0, lanelet)
                    # Calculate distance
                    for i in range(len(lanelet.dense_points) - 1):
                        wp1 = lanelet.dense_points[i]
                        wp2 = lanelet.dense_points[i + 1]
                        accumulated_distance += wp1.transform.location.distance(
                            wp2.transform.location
                        )

                visited.add((best_predecessor, current_node))
                current_node = best_predecessor

            # Prepend backward lanelets
            lanelets = backward_lanelets + lanelets

        # Add the initial lanelet
        initial_edge = self.graph.edges[node1, node2]
        initial_lanelet = initial_edge.get('lanelet')
        if initial_lanelet:
            if direction == 'backward':
                lanelets.append(initial_lanelet)
            else:
                lanelets.insert(len(backward_lanelets) if direction == 'both' else 0,
                              initial_lanelet)

        sparse_wps = ODict()
        dense_wps = []
        for lanelet in lanelets:
            sparse_wps.update(lanelet.sparse_points)
            dense_wps.extend(lanelet.dense_points)

        return Lanelet(sparse_points=sparse_wps, dense_points=dense_wps)

    def categorize_actors(self, actors: List[carla.Actor]) -> Dict[carla.Actor, Tuple[int, int, Lanelet]]:
        """
        Categorize multiple actors based on their lanelet membership.

        Args:
            actors: List of CARLA actors

        Returns:
            Dictionary mapping actors to their (node1, node2, lanelet) tuples
        """
        categorized = {}

        for actor in actors:
            # Get actor's current waypoint
            actor_location = actor.get_location()
            actor_waypoint = self.world_map.get_waypoint(actor_location)

            # Find corresponding lanelet
            lanelet_info = self.find_lanelet_for_actor(actor_waypoint)

            if lanelet_info:
                categorized[actor] = lanelet_info

        return categorized

    def save_to_h5(self, filename: str):
        with h5py.File(filename, "w") as f:
            graph_group   = f.create_group('graph')
            nodes_group   = graph_group.create_group('nodes')
            edges_group   = graph_group.create_group('edges')
            metadata_group = f.create_group('metadata')

            metadata_group.attrs['sampling_resolution'] = self.sampling_resolution
            metadata_group.attrs['num_nodes'] = self.graph.number_of_nodes()
            metadata_group.attrs['num_edges'] = self.graph.number_of_edges()

            # Nodes
            for node_id, node_data in self.graph.nodes(data=True):
                node_group = nodes_group.create_group(f'node_{node_id}')
                node_group.attrs['node_id'] = node_id
                if 'vertex' in node_data:
                    node_group.create_dataset('vertex', data=node_data['vertex'])

            # Edges
            edge_idx = 0
            for n1, n2, edge_data in self.graph.edges(data=True):
                edge_name  = f'edge_{edge_idx}'; edge_idx += 1
                edge_group = edges_group.create_group(edge_name)

                # Prefer storing RoadOption as its enum value (int), easier to restore
                edge_group.attrs['node1'] = n1
                edge_group.attrs['node2'] = n2
                edge_group.attrs['type']  = int(edge_data.get('type', RoadOption.VOID).value) \
                                            if hasattr(edge_data.get('type', None), "value") \
                                            else int(RoadOption.VOID.value)
                edge_group.attrs['intersection'] = bool(edge_data.get('intersection', False))
                edge_group.attrs['length'] = int(edge_data.get('length', 0))

                lanelet = edge_data.get('lanelet')
                if not lanelet:
                    continue

                g = edge_group.create_group('lanelet')

                # Save sparse points
                sparse_keys = np.array(list(lanelet.sparse_points.keys()), dtype=np.int32)
                sparse_positions = np.array([
                    [wp.transform.location.x, wp.transform.location.y, wp.transform.location.z]
                    for wp in lanelet.sparse_points.values()
                ])

                # Save dense points
                dense_positions = np.array([
                    [
                        wp.transform.location.x,
                        wp.transform.location.y,
                        wp.transform.location.z
                    ]
                    for wp in lanelet.dense_points
                ])

                g.create_dataset('sparse_keys', data=sparse_keys, compression="gzip", chunks=True)
                g.create_dataset('sparse_positions', data=sparse_positions, compression="gzip", chunks=True)

                # DENSE: full list of waypoint records
                g.create_dataset('dense_positions', data=dense_positions, compression="gzip", chunks=True)

                g.attrs['num_dense_points']  = len(lanelet.dense_points)
                g.attrs['num_sparse_points'] = len(lanelet.sparse_points)

        print(f"Saved lanelet graph with {self.graph.number_of_nodes()} nodes "
            f"and {self.graph.number_of_edges()} edges to {filename}")

    @classmethod
    def load_from_h5(cls, world_map: carla.Map, filename: str) -> "LaneletGraph":
        self = object.__new__(cls)
        self.world_map = world_map
        # sampling_resolution is for info; you can recompute if needed
        self.route_planner = None
        self.sampling_resolution = None
        self.graph = nx.DiGraph()
        self.waypoint_to_lanelet = {}

        with h5py.File(filename, "r") as f:
            meta = f["metadata"]
            self.sampling_resolution = float(meta.attrs.get("sampling_resolution", 2.0))

            # Nodes
            nodes_group = f["graph/nodes"]
            for name in nodes_group:
                ng = nodes_group[name]
                node_id = int(ng.attrs["node_id"])
                if "vertex" in ng:
                    self.graph.add_node(node_id, vertex=tuple(ng["vertex"][...]))
                else:
                    self.graph.add_node(node_id)

            # Edges
            edges_group = f["graph/edges"]
            for name in edges_group:
                eg = edges_group[name]
                n1 = int(eg.attrs["node1"])
                n2 = int(eg.attrs["node2"])
                type_val = int(eg.attrs["type"])
                edge_type = RoadOption(type_val)
                intersection = bool(eg.attrs["intersection"])
                length = int(eg.attrs["length"])

                edge_data = dict(type=edge_type, intersection=intersection, length=length)

                # Rebuild lanelet (if present)
                lanelet = None
                if "lanelet" in eg:
                    lg = eg["lanelet"]
                    dense_positions = lg["dense_positions"][...]
                    sparse_positions = lg["sparse_positions"][...]
                    sparse_keys = lg["sparse_keys"][...]

                    # Reconstruct Waypoints
                    dense_points = [pos_to_wp(world_map, pos) for pos in dense_positions]
                    sparse_points = ODict()
                    for i, pos in enumerate(sparse_positions):
                        wp = pos_to_wp(world_map, pos)
                        key = tuple(int(x) for x in sparse_keys[i])
                        sparse_points[key] = wp

                    lanelet = Lanelet(sparse_points, dense_points)
                    edge_data["lanelet"] = lanelet

                    # Update reverse index for quick lookup
                    if edge_type == RoadOption.LANEFOLLOW:
                        for wp in dense_points:
                            key = (wp.road_id, wp.section_id, wp.lane_id)
                            self.waypoint_to_lanelet.setdefault(key, []).append((n1, n2))

                self.graph.add_edge(n1, n2, **edge_data)

        return self


