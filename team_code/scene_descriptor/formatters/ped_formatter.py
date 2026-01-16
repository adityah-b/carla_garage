from typing import List
from .base_formatter import BaseFormatter
from scene_descriptor.data_extractors.ped_data_extractor import PedestrianData

class PedestrianFormatter(BaseFormatter):
    @classmethod
    def format_pedestrians(
        cls,
        peds : List[PedestrianData],
        precision : int = 2
    ) -> str:
        lines : List[str] = []
        if peds:
            lines.append("Pedestrian Data:")
            for p_data in peds:
                lines.append(cls._format_pedestrian_data(p_data, indent="\t", precision=precision))

        return "\n".join(lines)

    @classmethod
    def _format_pedestrian_data(
        cls,
        ped_data : PedestrianData,
        indent : str = "",
        precision : int = 2
    ) -> str:
        f = cls.fmt
        parts = [
            f"{indent}Pedestrian ID: {ped_data.id}",
            f"Distance: {f(ped_data.relative_distance, precision)}",
        ]

        if ped_data.is_on_road:
            parts.append("Status: Crossing road")

        return ", ".join(parts)

