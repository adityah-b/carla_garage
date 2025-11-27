from typing import List

from .base_formatter import BaseFormatter
from scene_descriptor.data_extractors.obstacle_data_extractor import ObstacleData


class ObstacleFormatter(BaseFormatter):
    @classmethod
    def format_obstacles(
        cls,
        obstacles: List[ObstacleData],
        precision: int = 2,
    ) -> str:
        lines: List[str] = []
        if obstacles:
            lines.append("Obstacle Data:")
            for obs in obstacles:
                lines.append(cls._format_obstacle(obs, indent="\t", precision=precision))

        return "\n".join(lines)

    @classmethod
    def _format_obstacle(
        cls,
        obstacle: ObstacleData,
        indent: str = "",
        precision: int = 2,
    ) -> str:
        f = cls.fmt
        return (
            f"{indent}Obstacle ID: {obstacle.id}, "
            f"Relative Position: {f(obstacle.relative_position, precision)}, "
            f"Relative Distance: {f(obstacle.relative_distance, precision)}"
        )