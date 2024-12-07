# Configuration Notes

A series of personal notes for running various parts of `carla_garage`

## Running the PDM-Lite Expert Agent
### Description
The expert agent relies on `leaderboard_autopilot` and `scenario_runner_autopilot` which are slightly modified versions of the original modules.

So we need to correctly use these when running expert agent code. Most issues will arise from incorrect `PYTHONPATH` configurations. So I created environment configuration files in the `garage_2` conda environment.

**Environment Configuration Files**
```
$CONDA_PREFIX/etc/conda/configurations/base_env.sh <-- Base Environment (using default leaderboard and scenario_runner)

$CONDA_PREFIX/etc/conda/configurations/expert_env.sh <-- Expert Environment (using leaderboard_autopilot and scenario_runner_autopilot)
```

### Usage

Aliases are defined in `$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh` to activate these environments.
- `use_base_env` for base environment
- `use_expert_env` for expert environment

Calling `conda deactivate` automatically unsets the environment variables and removes the aliases. Code is found in `$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh`


### Additional Notes
Setting and unsetting the environments using the method above likely messes up `PYTHONPATH` but not in immediate plans to fix. Just use sparingly when switching between environment configurations.

TODO: This is an easy fix, just need to get on it

## Leaderboard Environment Variables

**ROUTES_SUBSET** -> Integer value to select specific route (based on `route id`) to run given a set of routes.

**CHECKPOINT_ENDPOINT** -> Path to a `.json` file where recorded Leaderboard metrics are stored for a given run
- **IMPORTANT** If a `CHECKPOINT_ENDPOINT` file already **exists**, the evaluation code will simply **exit**

### `carla_garage` Specific Environment Variables

**TOWN** -> Town name in which expert agent is run. This is used for naming purposes when saving the recorded visualizations from a particular run

**REPETITION** -> Repetition number for a given run. This is used for naming purposes when saving the recorded visualizations from a particular run

## Personal Environment Variables

**RECORD_EXPERT_AGENT** -> Boolean to control whether to record .log files for expert agent runs. Generally set to false since these files are massive due to accessing standard sensor data + privileged data