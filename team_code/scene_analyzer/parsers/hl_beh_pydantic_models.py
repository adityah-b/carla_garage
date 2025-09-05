from pydantic import BaseModel, Field
from typing import List, Literal

class KeyActor(BaseModel):
    id : int
    obj_type : Literal["vehicle", "ped", "cyclist", "traffic_light", "stop_sign"]
    traffic_type : Literal["leading", "trailing", "oncoming", "cross", "None"]

class HighLevelBehaviour(BaseModel):
    scenario : str = Field(description="one paragraph natural language summary describing the driving scenario from the provided image and text along with the necessary high-level driving behaviours to navigate the scenario")
    key_actors : List[KeyActor]
    reasoning : List[str] = Field(min_length=1, description="Step by step reasoning on why each action and key actor choice is valid")
