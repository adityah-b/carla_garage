# demo_egoplan_parser.py
from pprint import pprint

# If your classes are in another module/file, change this import accordingly:
from ego_plan_parser import (
    EgoPlanParser, Action,
    LongitudinalParams, LateralParams, WaitParams
)

def demo_valid():
    raw = """plan:
accelerate, wait_for, change_lane_left, maintain_speed, decelerate, stop

low_level_actions:
1) accelerate:
   longitudinal_params = {
     'spd': 12.5, 't_head': 1.6, 'f_dist': 3.5
   }
2) wait_for:
   wait_params = {
     'all': [
       { 'gap_ok': { 'lane': 'left', 'gap_time': 1.5, 'gap_dist': 12.0 } },
       { 'spd':  { 'op': '>=', 'value': 12.0 } }
     ],
     'hold_s': 0.4,
     'timeout_s': 2.0,
     'on_timeout': 'continue'
   }
3) change_lane_left:
   lateral_params = {
     'gap_time': 1.5, 'gap_dist': 12.0
   }
4) maintain_speed:
   longitudinal_params = {
     'spd': 13.9, 't_head': 1.5, 'f_dist': 3.5
   }
5) decelerate:
   longitudinal_params = {
     'spd': 0.0, 't_head': 2.0, 'f_dist': 5.0
   }
6) stop:

reasoning:
1. Match target-lane flow before crossing.
2. Keep conservative headway during lane change.
"""
    ego = EgoPlanParser.parse(raw)
    print("=== VALID PLAN ===")
    print("Actions:", [a.value for a in ego.plan.actions])
    print("\nParam blocks (aligned with actions):")
    for i, (a, p) in enumerate(zip(ego.plan.actions, ego.plan.params), 1):
        print(f"{i:>2}) {a.value}: ", end="")
        if isinstance(p, LongitudinalParams):
            print("LongitudinalParams", vars(p))
        elif isinstance(p, LateralParams):
            print("LateralParams", vars(p))
        elif isinstance(p, WaitParams):
            print(f"WaitParams(mode={p.mode}, hold_s={p.hold_s}, timeout_s={p.timeout_s}, on_timeout={p.on_timeout})")
            print("    conds:")
            for c in p.conds:
                pprint(vars(c), indent=8, width=120)
        else:
            print(p)  # None for STOP

    # Lightweight assertions
    assert ego.plan.actions[0] == Action.ACCELERATE
    assert isinstance(ego.plan.params[0], LongitudinalParams)
    assert ego.plan.actions[1] == Action.WAIT_FOR
    assert isinstance(ego.plan.params[1], WaitParams) and ego.plan.params[1].mode == "all"
    assert ego.plan.actions[2] == Action.CHANGE_LANE_LEFT
    assert isinstance(ego.plan.params[2], LateralParams)
    assert ego.plan.actions[-1] == Action.STOP and ego.plan.params[-1] is None
    print("✅ Valid plan parsed and validated.\n")


def demo_invalid_wait_both_all_any():
    raw = """plan:
wait_for

low_level_actions:
1) wait_for:
   wait_params = {
     'all': [ { 'time': { 'seconds': 0.5 } } ],
     'any': [ { 'spd': { 'op': '>=', 'value': 5.0 } } ]
   }

reasoning:
- invalid both all and any
"""
    print("=== INVALID wait_for (both all & any) ===")
    try:
        EgoPlanParser.parse(raw)
    except ValueError as e:
        print("Caught expected error:", e, "\n")


def demo_unknown_action():
    raw = """plan:
fly

low_level_actions:
1) fly:
   longitudinal_params = { 'spd': 5.0, 't_head': 1.5, 'f_dist': 3.0 }

reasoning:
- unknown action name
"""
    print("=== UNKNOWN ACTION ===")
    try:
        EgoPlanParser.parse(raw)
    except ValueError as e:
        print("Caught expected error:", e, "\n")


if __name__ == "__main__":
    demo_valid()
    demo_invalid_wait_both_all_any()
    demo_unknown_action()