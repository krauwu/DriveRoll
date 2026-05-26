# Self-Forcing Suite

`Self_Forcing/` extends the DWM codebase with ODE-pair generation, ODE distillation, and DMD / self-forcing training. It is intended for experiments that reduce autoregressive rollout cost while preserving generation quality.

The suite keeps a DWM-compatible package layout, so most dataset, model, metric, and preview utilities mirror `DWM/src/dwm`.

## Directory structure

```text
Self_Forcing/
├── configs/
│   ├── generate_ode/      # Teacher ODE-pair generation configs
│   ├── train_ode/         # ODE distillation configs
│   └── train_dmd/         # DMD / self-forcing training configs
├── scripts/
│   ├── generate_nuscenes_ode_pairs.py
│   ├── generate_nuscenes_ode_pairs_with_ref.py
│   ├── train_dmd.sh
│   ├── train_debug1card.sh
│   └── visualize_camera_trajectory.py
└── src/dwm/
    ├── datasets/          # DWM datasets plus ODE-specific nuScenes adapters
    ├── models/            # DWM models plus VAE / VQ / point-cloud modules
    ├── pipelines/         # CTSD and rolling-ref pipelines
    ├── train_ode.py       # ODE distillation entry point
    ├── train_dmd.py       # DMD / self-forcing training entry point
    ├── train_dmd_dist.py  # Distributed DMD entry point
    ├── preview.py
    └── preview_dist.py
```

## Environment

Use the same base environment as DWM:

```text
../requirements_dwm.txt
```

Before running the suite:

```bash
cd Self_Forcing
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

If your config requires external Waymo or FVD utilities, add those source trees to `PYTHONPATH`.

## Checkpoints

Self-forcing experiments usually involve multiple checkpoint roles:

| Checkpoint role | Used by | Notes |
| --- | --- | --- |
| DMD / self-forcing model | autoregressive generation | [Download](https://pan.baidu.com/s/1CWN1YOd80ZXsG47RslC-cQ?pwd=4930), Self_Forcing/configs/train_dmd/chunk=12_ref=3_stride=4_iter=3.json |
| Base diffusion model | Model initialization | Should match the selected DWM config. |

Update checkpoint paths inside the selected JSON config before running.

## Workflow

The intended workflow is:

```text
Teacher checkpoint
  -> generate ODE pairs
  -> train ODE-distilled model
  -> train DMD / self-forcing model
  -> preview / downstream evaluation
```

### Generate ODE pairs

Use the reference generator scripts:

```bash
python scripts/generate_nuscenes_ode_pairs_with_ref.py \
  --config_path <generate_ode_config.json> \
  --output_folder <ode_pair_output_dir> \
  --num_samples <num_samples>
```

Common configs are stored in:

```text
configs/generate_ode/
```

### Train with ODE distillation

```bash
torchrun --nproc_per_node=<num_gpus> -m dwm.train_ode \
  --config-path <train_ode_config.json> \
  --output-path <output_dir> \
  --data-folder <ode_pair_output_dir>
```

Common configs are stored in:

```text
configs/train_ode/
```

### Train with DMD / self-forcing

The suite provides a script template:

```bash
bash scripts/train_dmd.sh <num_gpus>
```

The script is a launcher template. Before running, update:

- the config path;
- output directory;
- conda or virtual environment activation;
- GPU list;
- checkpoint paths inside the selected JSON config.

Common configs are stored in:

```text
configs/train_dmd/
```

## Outputs

Self-Forcing outputs can be consumed by downstream evaluation suites:

- `DGGT/` expects preview outputs plus token / pose metadata for geometry evaluation;
- `UniADBench/` expects generated videos or `.npy` preview arrays aligned with nuScenes-style metadata;
- `EasyTry/` can be used for qualitative rolling inspection if the config is adapted to the trained model.
