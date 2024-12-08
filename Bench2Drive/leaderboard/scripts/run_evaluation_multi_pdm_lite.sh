export CARLA_ROOT=/mnt/lustre/work/geiger/gwb438/hiwi/eval_pdm_lite_bench2drive/carla
export WORK_DIR=/mnt/lustre/work/geiger/gwb438/hiwi/eval_pdm_lite_bench2drive/Bench2Drive
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

#!/bin/bash
BASE_PORT=30000
BASE_TM_PORT=50000
IS_BENCH2DRIVE=True
BASE_ROUTES=leaderboard/data/bench2drive220
TEAM_AGENT=leaderboard/team_code/autopilot.py
# Must set YOUR_CKPT_PATH
TEAM_CONFIG=your_team_agent_ckpt.pth
# TEAM_CONFIG=Bench2DriveZoo/adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py+YOUR_CKPT_PATH/uniad_base_b2d.pth
BASE_CHECKPOINT_ENDPOINT=eval_bench2drive220
PLANNER_TYPE=traj
ALGO=pdm_lite
SAVE_PATH=./eval_bench2drive220_${ALGO}_${PLANNER_TYPE}

if [ ! -d "${ALGO}_b2d_${PLANNER_TYPE}" ]; then
    mkdir ${ALGO}_b2d_${PLANNER_TYPE}
    echo -e "\033[32m Directory ${ALGO}_b2d_${PLANNER_TYPE} created. \033[0m"
else
    echo -e "\033[32m Directory ${ALGO}_b2d_${PLANNER_TYPE} already exists. \033[0m"
fi

# Check if the split_xml script needs to be executed
if [ ! -f "${BASE_ROUTES}_${ALGO}_${PLANNER_TYPE}_split_done.flag" ]; then
    echo -e "****************************\033[33m Attention \033[0m ****************************"
    echo -e "\033[33m Running split_xml.py \033[0m"
    TASK_NUM=4 # 8*H100, 1 task per gpu
    python tools/split_xml.py $BASE_ROUTES $TASK_NUM $ALGO $PLANNER_TYPE
    touch "${BASE_ROUTES}_${ALGO}_${PLANNER_TYPE}_split_done.flag"
    echo -e "\033[32m Splitting complete. Flag file created. \033[0m"
else
    echo -e "\033[32m Splitting already done. \033[0m"
fi

echo -e "**************\033[36m Please Manually adjust GPU or TASK_ID \033[0m **************"
# Example, 8*H100, 1 task per gpu
# GPU_RANK_LIST=(0 1 2 3 4 5 6 7)
# TASK_LIST=(0 1 2 3 4 5 6 7)
GPU_RANK_LIST=(0 0 0 0)
TASK_LIST=(0 1 2 3)
echo -e "\033[32m GPU_RANK_LIST: $GPU_RANK_LIST \033[0m"
echo -e "\033[32m TASK_LIST: $TASK_LIST \033[0m"
echo -e "***********************************************************************************"

length=${#GPU_RANK_LIST[@]}
for ((i=0; i<$length; i++ )); do
    if [[ $i -eq 1 ]]; then
        PORT=$((BASE_PORT + i * 150))
        TM_PORT=$((BASE_TM_PORT + i * 150))
        ROUTES="${BASE_ROUTES}_${TASK_LIST[$i]}_${ALGO}_${PLANNER_TYPE}.xml"
        CHECKPOINT_ENDPOINT="${ALGO}_b2d_${PLANNER_TYPE}/${BASE_CHECKPOINT_ENDPOINT}_${TASK_LIST[$i]}.json"
        GPU_RANK=${GPU_RANK_LIST[$i]}
        echo -e "\033[32m ALGO: $ALGO \033[0m"
        echo -e "\033[32m PLANNER_TYPE: $PLANNER_TYPE \033[0m"
        echo -e "\033[32m TASK_ID: $i \033[0m"
        echo -e "\033[32m PORT: $PORT \033[0m"
        echo -e "\033[32m TM_PORT: $TM_PORT \033[0m"
        echo -e "\033[32m CHECKPOINT_ENDPOINT: $CHECKPOINT_ENDPOINT \033[0m"
        echo -e "\033[32m GPU_RANK: $GPU_RANK \033[0m"
        echo -e "\033[32m bash leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK \033[0m"
        echo -e "***********************************************************************************"
        bash -e leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK 2>&1 > ${BASE_ROUTES}_${TASK_LIST[$i]}_${ALGO}_${PLANNER_TYPE}.log &
        sleep 5
    fi
done
wait