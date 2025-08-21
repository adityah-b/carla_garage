from .ego_plan_parser import EgoPlan, EgoPlanParser
from .hl_beh_parser import HighLevelBehaviour, HighLevelBehaviourParser

class SceneParser:
    def parse_hl_beh(
        self,
        text : str
    ) -> HighLevelBehaviour:
        return HighLevelBehaviourParser.parse(text)

    def parse_ego_plan(
        self,
        text : str
    ) -> EgoPlan:
        return EgoPlanParser.parse(text)
