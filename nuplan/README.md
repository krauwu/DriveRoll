# nuPlan Closed-Loop Interface

<p align="center">
  <img src="../assets/sim.gif" width="95%" alt="Closed-Loop Autoregressive Generation">
</p>

`nuplan/` connects a DWM-compatible generator to nuPlan scenarios for closed-loop generative simulation. The interface uses nuPlan DB frames for warm-up, builds simulator-aligned conditions, and then generates camera observations during rollout.

This suite is intended for **closed-loop simulation**, not offline benchmark conversion. For offline evaluation, use `UniADBench/` or `DGGT/`.

## What this suite does

```text
nuPlan scenario state
  -> DB warm-up frames
  -> ego trajectory rollout
  -> condition construction
  -> DWM / PAS generation
  -> generated camera observations
```

The current runner supports predefined local ego trajectories. The trajectory source can be extended to user-provided trajectories or planner outputs.

## Structure

```text
nuplan/
├── launch.py                    # High-level multi-DB launcher
└── sim_tools/
    ├── config_paths.yaml        # Dataset, checkpoint, and runtime paths
    ├── configs/nuplan.json      # DWM generation config used by the simulator
    ├── configs/simulation/      # nuPlan simulation configs
    ├── nuplan_sim_DB.py         # Main simulation runner
    ├── info_tool_DB.py          # nuPlan metadata and rolling-buffer utilities
    ├── prepare_cond_gpu.py      # Condition construction and point projection
    ├── generator.py             # Streaming DWM generator wrapper
    ├── optimized_projection.py  # GPU point-projection utilities
    └── common.py                # Visualization and geometry helpers
```

## Environment

The nuPlan interface requires two parts:

- the DWM-compatible generation environment used by `DWM/`;
- the official [nuPlan devkit](https://github.com/motional/nuplan-devkit) and its dataset dependencies.

The generation environment can follow `requirements_dwm.txt`. The nuPlan devkit should be installed according to the official nuPlan instructions.

When `DWM/` and `nuplan/` are sibling directories, `sim_tools/config_paths.yaml` can use the relative source path:

```yaml
dwm_path: "../DWM/src"
```

## Required resources

| Resource | Where to configure | Notes |
| --- | --- | --- |
| nuPlan DB files | `sim_tools/config_paths.yaml` or `NUPLAN_DB_DIR` | The scenario DBs to simulate. |
| nuPlan maps | `sim_tools/config_paths.yaml` | Required by the nuPlan scenario builder. |
| nuPlan sensor blobs | `sim_tools/config_paths.yaml` | Used for warm-up frames and DB context. |
| DWM source tree | `dwm_path` | Used to import the generator and condition pipeline. |
| DWM / PAS checkpoint | `sim_tools/configs/nuplan.json` | The model used for closed-loop generation. |
| Base diffusion model | `sim_tools/configs/nuplan.json` | Should match the selected generation checkpoint. |
| Text annotations | `sim_tools/nuplan_text.json` or config path | Optional text conditioning used by the runner. |

## Configuration

Edit the path config first:

```text
sim_tools/config_paths.yaml
```

Important fields include:

```text
nuplan_data_root
nuplan_maps_root
nuplan_db_files
blob_path
dwm_path
gen_cfg
gen_img_log
ray_temp_dir
```

Then edit the generation config:

```text
sim_tools/configs/nuplan.json
```

Common fields to update:

```text
pipeline.pretrained_model_name_or_path
pipeline.model_checkpoint_path
pipeline.common_config
pipeline.inference_config
```

## Run

The default launcher reads DB stems from `DB_STEMS` in `launch.py`. Each stem should match a DB file under `NUPLAN_DB_DIR`.

```bash
cd nuplan
python launch.py
```

The launcher also supports runtime overrides through environment variables:

```bash
export NUPLAN_DB_DIR=<directory-containing-nuplan-db-files>
export NUPLAN_GPU_IDS=0
export STAGE3_OUTPUT_ROOT=<output-root>
python launch.py
```

For one-off debugging, the runner can be called directly:

```bash
export NUPLAN_SIM_CONFIG_PATHS=$PWD/sim_tools/config_paths.yaml
export NUPLAN_DB_FILES=<path-to-log.db>
export STAGE3_OUTPUT_DIR=<output-dir>
python sim_tools/nuplan_sim_DB.py
```

## Trajectory behavior

The default simulation flow is:

```text
DB warm-up frames
  -> predefined local ego trajectory
  -> rolling condition update
  -> autoregressive generation
```

The predefined trajectory logic is implemented in `get_traj(...)` in `sim_tools/nuplan_sim_DB.py`. Current presets include:

```text
straight
right_then_back
cosine
```

To support a planner or website-side user trajectory, replace the trajectory source inside the runner while keeping the same condition-building interface.

## Outputs

Outputs are written under `STAGE3_OUTPUT_ROOT` or `STAGE3_OUTPUT_DIR`. A typical run may produce generated images, videos, condition previews, and simulation logs.
