from typing import List

from .base_formatter import BaseFormatter
from scene_descriptor.data_extractors.obstacle_data_extractor import ObstacleData, ObstacleDataEntry


class ObstacleFormatter(BaseFormatter):
    @classmethod
    def format_obstacles(
        cls,
        obstacle_data: ObstacleData,
        precision: int = 2,
    ) -> str:
        lines: List[str] = []
        ego_obstacles = obstacle_data.ego_obstacles
        if ego_obstacles:
            lines.append("Obstacle Data:")
            for obs in ego_obstacles:
                lines.append(cls._format_obstacle(obs, indent="\t", precision=precision))

        return "\n".join(lines)

    @classmethod
    def _format_obstacle(
        cls,
        obstacle: ObstacleDataEntry,
        indent: str = "",
        precision: int = 2,
    ) -> str:
        f = cls.fmt

        parts = [f"{indent}Obstacle ID: {obstacle.id}"]

        # parts.append(f"Type: {obstacle.obstacle.type_id}")
        parts.append(f"Distance: {f(obstacle.relative_distance, precision)}")

        if obstacle.is_near_junction:
            parts.append(f"Blocking Intersection. Wait Until Cleared")

        return ", ".join(parts)

    @classmethod
    def summarize(
        cls,
        obstacle_data: ObstacleData,
    ) -> str:
        if obstacle_data is None or not obstacle_data.ego_obstacles:
            return ""

        bullets = []

        for obs in obstacle_data.ego_obstacles:
            dist = round(obs.relative_distance)
            if obs.is_near_junction:
                bullets.append(
                    f"[CAUTION] A stationary obstacle is {dist} m ahead near an intersection and blocks the route."
                )
            else:
                bullets.append(
                    f"[CAUTION] A stationary obstacle is {dist} m ahead and blocks the route."
                )

        return "Obstacle Context:\n" + "\n".join(f"- {b}" for b in bullets)