import numpy as np

from PIL import Image
from pathlib import Path

from team_code.config import GlobalConfig
from team_code.scene_analyzer.scene_analyzer import SceneAnalyzer

folder = Path('/home/carla/carla_garage/runs/run_20251122_061909')
file_number = '0050'

img_file = folder.joinpath(f'rgb_bounding_boxes_{file_number}.png')
text_file = folder.joinpath(f'scene_context_{file_number}.txt')

scene_img = np.asarray(Image.open(img_file))

with open(text_file, 'r') as f:
    scene_text = f.read()

config = GlobalConfig()

scene_analyzer_config = config.scene_analyzer_config
scene_analyzer = SceneAnalyzer(
    provider=scene_analyzer_config['provider'],
    model_name=scene_analyzer_config['model_name'],
    temperature=scene_analyzer_config['temperature'],
    max_output_tokens=scene_analyzer_config['max_output_tokens'],
)
print(f'scene_text: {scene_text}')

hl_beh = scene_analyzer.get_high_level_behaviour(text=scene_text, image=scene_img)
print(f'\n\nHigh Level Behaviour\n\n')
print(f'\tScenario: {hl_beh.scenario}')
print(f'\tKey Actors: {hl_beh.key_actors}')
print(f'\tReasoning: {hl_beh.reasoning}')

ego_plan = scene_analyzer.get_ego_plan(text=hl_beh.scenario, image=scene_img)
print(f'\n\nEgo Plan\n\n')
print(f"\tPlan: {ego_plan.action.value}")
print(f"\tConditions: {ego_plan.conditions}")
print(f"\tReasoning: {ego_plan.reasoning}")
