import carla
import os

from pathlib import Path

from team_code.scene_descriptor.data_extractors.lanelet_graph import LaneletGraph

def main():
    """
    Example usage of the LaneletGraph.
    """

    ws_dir = Path(os.environ['WORK_DIR'])
    lanelet_dir = ws_dir.joinpath('team_code/scene_descriptor/lanelets')
    print(f'Saving lanelets to {lanelet_dir}')

    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(1000)
    settings = carla.WorldSettings(
        synchronous_mode=True,
        fixed_delta_seconds=0.1,
        deterministic_ragdolls=True,
        no_rendering_mode=False,
        spectator_as_ego=False,
    )
    client.get_world().apply_settings(settings)
    map_names = [
        'Town12', 'Town13'
    ]

    for map_name in map_names:
        h5_file_path = lanelet_dir / (map_name + '.h5')

        print(f'Generating {map_name}.h5 lanelets.')

        # Get the world and map
        world = client.load_world(map_name, reset_settings=False)
        world_map = world.get_map()

        # Create lanelet graph
        print("Building lanelet graph...")
        lanelet_graph = LaneletGraph(world_map, sampling_resolution=2.0)

        print(f"Created graph with {lanelet_graph.graph.number_of_nodes()} nodes "
            f"and {lanelet_graph.graph.number_of_edges()} edges")

        # Save to HDF5
        lanelet_graph.save_to_h5(h5_file_path)

        print(f"\nSaved lanelet graph to {h5_file_path}")

if __name__ == "__main__":
    main()