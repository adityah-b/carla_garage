 #!/bin/bash

# PDM-Lite agent
# export TEAM_AGENT=$WORK_DIR/team_code/autopilot.py

# PDM-Lite agent with data collection
export TEAM_AGENT=$WORK_DIR/team_code/data_agent.py

# export ROUTES=$LEADERBOARD_ROOT/data/routes_validation.xml
export ROUTES=$WORK_DIR/data/50x36_Town13/SignalizedJunctionLeftTurn/9_0.xml

# export ROUTES_SUBSET=13
export REPETITIONS=1

export DEBUG_CHALLENGE=1
export CHALLENGE_TRACK_CODENAME=MAP
export CHECKPOINT_ENDPOINT="${LEADERBOARD_ROOT}/results.json"
export RECORD_PATH=
export RESUME=1
export SAVE_PATH="$WORK_DIR/debug_visualizations/test_run"

export TOWN=13
export REPETITION=0

export RECORD_EXPERT_AGENT=1

#!/bin/bash

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--debug-checkpoint=${DEBUG_CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
# --routes-subset=${ROUTES_SUBSET} \
