# DGGT Geometry Benchmark

`DGGT/` evaluates generated driving videos with a DGGT-style geometry model. It is used to inspect view consistency, pose-related quality, and geometry-aware reconstruction behavior of generated results.

This suite is complementary to `UniADBench/`: `UniADBench/` focuses on downstream autonomy metrics, while `DGGT/` focuses on geometry-oriented quality.

## Structure

```text
DGGT/
├── benchmark.py                    # Direct DGGT benchmark entry point
├── scripts/
│   ├── convert_sf_to_benchmark.py  # Prepare, convert, benchmark, and aggregate backend
│   └── run_benchmark_pipeline.sh   # Pipeline wrapper
├── datasets/                       # Dataset source loaders and preprocessing utilities
├── dggt/                           # DGGT model implementation
└── utils/                          # Camera, geometry, logging, video, and visualization helpers
```

The `dggt/` subdirectory contains the model code. The `scripts/` directory contains the benchmark-facing pipeline used by this repository.

## Environment

Use the DGGT reference environment file:

```text
../requirements_dggt.txt
```

DGGT may require geometry-related dependencies. If installation fails, follow the official [DGGT installation guide](https://github.com/xiaomi-research/dggt#installation) and use `requirements_dggt.txt` as the version reference for this benchmark release.

## Checkpoint & Data Setup

The official Waymo checkpoint can be used as the default DGGT model:

> [!NOTE]
> DGGT official Waymo checkpoint: [model_latest_waymo.pt](https://huggingface.co/xiaomi-research/dggt/resolve/main/model_latest_waymo.pt?download=true).

Update the checkpoint path in the command or wrapper script before running.

### Setup

1. Create the `pretrained` directory and place the checkpoint:

```bash
mkdir -p pretrained
# Download model_latest_waymo.pt and place it as:
# ./pretrained/model_latest_waymo.pt
```

2. Create the `benchmark_input` directory and download [metadata](https://pan.baidu.com/s/1Bqz4CoKWth-u3aiDL6B9ag?pwd=4930):


```bash
mkdir -p benchmark_input
# Download metadata from URL and place it as:
# ./benchmark_input/metadata
```

3. Place the folder you want to evaluate under `benchmark_input`:

```bash
# Example: copy your eval data folder into benchmark_input
# ./benchmark_input/<your_eval_folder>
```


## Inputs

DGGT expects generated outputs and pose metadata from a DWM or Self-Forcing rollout.

| Input | Description |
| --- | --- |
| Generated preview output | Generated frames or preview outputs to evaluate. |
| Pose metadata | Scene-level camera / ego pose metadata used for alignment. |
| DGGT checkpoint | The model checkpoint used by `benchmark.py`. |
| Camera list | The camera IDs included in evaluation. |

## Pipeline

The high-level wrapper is:

```bash
cd DGGT
bash scripts/run_benchmark_pipeline.sh --sf_preview_dir <your_eval_folder> --ckpt_path pretrained/model_latest_waymo.pt --num_scenes 150 --cams 1
```

The wrapper is a template. Edit the generated-output path, pose-metadata path, checkpoint path, output directory, scene count, and camera list before running.

Internally, the pipeline follows:

```text
prepare metadata
  -> convert generated outputs
  -> run DGGT benchmark
  -> aggregate metrics
```

For debugging, call `scripts/convert_sf_to_benchmark.py` stage by stage.

## Outputs

The benchmark output directory stores converted inputs, per-scene results, and aggregated metric files. File names depend on the source name, selected cameras, and sampling parameters.
