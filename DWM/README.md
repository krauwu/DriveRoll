# DWM Generation Suite

<p align="center">
  <img src="../assets/argen.gif" width="100%" alt="DWM autoregressive generation demo">
</p>

`DWM/` contains the OpenDWM-style generation code used to autoregressively rollout driving videos for this benchmark. It includes several reproduced generator settings, including DreamForge-style generation, LiVE-style generation, DFoT / diffusion-forcing rollout, and rollingforcing inference pipeline.
## Environment

The DWM environment should generally follow the installation workflow of [OpenDWM](https://github.com/SenseTime-FVG/OpenDWM). Please first prepare a CUDA / PyTorch / Diffusers environment compatible with OpenDWM and your checkpoints.

For reproducibility, we provide the package versions used in our experiments:

```text
../requirements_dwm.txt
```

This file can be used as a reference environment specification, but users may adapt package versions according to their CUDA stack and local hardware. The generation-side code does not require custom CUDA operators.

Before running DWM scripts, expose the source tree:

```bash
cd DWM
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

## Data

DWM generation uses the full nuScenes dataset and the 12Hz nuScenes-style annotation files.

| Resource | Link |
| --- | --- |
| nuScenes dataset | [Official nuScenes dataset](https://www.nuscenes.org/download) |
| 12Hz nuScenes-style annotations | [Download](https://pan.baidu.com/s/107dd2wyuG-7tnIFDgh5mCg?pwd=4930) |

Update dataset paths in the corresponding config before running preview or training.

## Checkpoints

Each reproduced method has its own config folder and checkpoint setting. The base model is [SD 3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)

| Method / setting | Config folder | Checkpoint |
| --- | --- | --- |
| DreamForge reproduction | `configs/dreamforge/` | [Download](https://pan.baidu.com/s/1p2iU5Vwc09pkgzoVV1qJUw?pwd=4930) |
| LiVE reproduction | `configs/live/` | [Download](https://pan.baidu.com/s/1eOiYTQN6LjD-uEkP2kH75Q?pwd=4930) |
| DFoT / rolling inference | `configs/dfot/`, `configs/rolling/` | [Download](https://pan.baidu.com/s/15FEkpFSi4BqsqA5hdlysjw?pwd=4930) |
| nuPlan-oriented DWM config | `configs/nuplan-dwm/` | [Download](https://pan.baidu.com/s/1LkWjWA0KfaCfEnk8q_WMBA?pwd=4930) |

Checkpoint paths are configured inside each JSON config. Common fields include:

```text
pipeline.pretrained_model_name_or_path
pipeline.model_checkpoint_path
```

## Config folders

```text
configs/
├── dreamforge/     DreamForge-style reproduced generation configs
├── live/           LiVE-style reproduced generation configs
├── dfot/           DFoT / diffusion-forcing configs
├── rolling/        Rolling autoregressive preview configs
└── nuplan-dwm/     nuPlan-oriented DWM configs
```

## Run preview

Use the provided scripts after editing local paths inside the script:

```bash
bash scripts/rolling_case.sh
```

For direct execution:

```bash
torchrun --nproc_per_node=1 src/dwm/preview.py \
  --config-path <config.json> \
  --output-path <output_dir>
```

## Training

Training uses the OpenDWM-style training entry:

```bash
torchrun --nproc_per_node=<num_gpus> src/dwm/train.py \
  --config-path <config.json> \
  --output-path <output_dir>
```

The trained or reproduced checkpoints can be used by:

- `DWM/` for offline preview and rolling generation;
- `EasyTry/` for interactive visual debugging;
- `UniADBench/` for downstream perception / planning evaluation;
- `DGGT/` for geometry-oriented evaluation;
- `nuPlan/` for closed-loop generative simulation.

## Output convention

Generated outputs should keep the metadata needed by downstream tools:

- camera order;
- frame rate;
- sample token or scene token;
- generated images / videos / arrays;
- ego pose or trajectory metadata when geometry evaluation is required.

If a new generator changes the output format, update the converters in `UniADBench/` and `DGGT/` accordingly.
