# UniADBench

`UniADBench/` evaluates generated multi-view driving videos with a UniAD / Bench2Drive-style downstream evaluation stack. It converts generated results into pseudo-nuScenes format, runs UniAD evaluation, and summarizes perception, mapping, occupancy, motion, and planning metrics when supported by the selected UniAD config.

<p align="center">
  <img src="../assets/case1.gif" width="100%" alt="UniAD generation demo">
</p>

<p align="center">
  <img src="../assets/case2.gif" width="100%" alt="UniAD generation demo">
</p>

> [!NOTE]
> Current methods still struggle to preserve consistent vehicle identity during close-range interactions, especially when vehicles pass by each other.

## Structure

```text
UniADBench/
├── cfg/                            # Pipeline and UniAD evaluation configs
├── tasks/                          # Reusable pipeline stages
├── data_tools/                     # nuScenes metadata and scene grouping helpers
├── UniAD/                          # UniAD / Bench2Drive-compatible adapter stack
├── generate_uniad_pkls.py          # Convert generated outputs to UniAD-compatible PKLs
├── run_uniad.py                    # Run UniAD evaluation
├── eval_long_sequence_pipeline.py  # Long-sequence one-click pipeline
├── eval_scene_based_pipeline.py    # Scene-group one-click pipeline
├── plot_metrics.py                 # Plot metric curves
├── plot_compare.py                 # Compare metric summaries
└── view_result.py                  # Inspect evaluation outputs
```

## Environment

Use the UniADBench reference environment file:

```text
../requirements_uniad.txt
```

This suite follows the Bench2Drive / UniAD ecosystem. Install the UniAD extensions according to the setup files inside `UniAD/` and the upstream [Bench2DriveZoo / UniAD guide](https://github.com/Thinklab-SJTU/Bench2DriveZoo/tree/uniad/vad).

## Checkpoint

Use the official UniAD Stage-2 E2E checkpoint unless a different evaluation checkpoint is intended:

> [!NOTE]
> UniAD official checkpoint: [uniad_base_e2e.pth](https://github.com/OpenDriveLab/UniAD/releases/download/v1.0.1/uniad_base_e2e.pth).

Place the checkpoint under the path expected by the selected UniAD config, or pass it through the pipeline configuration.

## Inputs

The pipeline expects generated outputs aligned with the nuScenes / 12Hz annotation format.

| Input | Description |
| --- | --- |
| nuScenes root | Original nuScenes dataset root. |
| 12Hz annotation files | Annotation files used by the benchmark stack. |
| Generated video or preview output | DWM / Self-Forcing generation result. |
| Token metadata | Sample-token or scene-token files used for alignment. |
| UniAD checkpoint and config | The downstream evaluation model and config. |

## Pipelines

Two high-level pipelines are provided.

```text
long-sequence pipeline:
  generated output -> PKL conversion -> UniAD evaluation -> metric plots

scene-based pipeline:
  generated output -> grouped PKL conversion -> UniAD evaluation -> scene-level summary
```

Run the corresponding entry after editing the config block near the top of each script:

```bash
cd UniADBench
python eval_long_sequence_pipeline.py --master-port <port>
python eval_scene_based_pipeline.py --master-port <port>
```

The scripts are templates. Update dataset paths, generated-output paths, UniAD root, config path, checkpoint path, output directory, GPU IDs, and port before running.

## Lower-level commands

Convert generated outputs:

```bash
python generate_uniad_pkls.py -cfg cfg/nuscenes_default.yaml
```

Run UniAD evaluation:

```bash
python run_uniad.py \
  --uniad-root <path-to-UniAD> \
  --config-path <uniad_config.py> \
  --checkpoint <uniad_checkpoint.pth> \
  --pkl-dir <generated_pkl_dir> \
  --output-dir <eval_output_dir> \
  --gpu-ids 0,1,2,3
```

## Outputs

Typical outputs include aggregated JSON metrics, optional plots, scene-level CSV summaries, and pseudo-nuScenes intermediate files.
