# EasyTry Interactive Demo

`EasyTry/` provides a lightweight Gradio demo for local inspection of rolling generation. It is useful for checking whether a DWM config, checkpoint, dataset segment, and action-conditioned rollout behave as expected before running large-scale evaluation.

This suite is for **interactive debugging and visualization**. It is not used for metric reporting.

## What this suite does

```text
load rolling config
  -> select a segment
  -> preview the next condition
  -> confirm an action
  -> generate the next frame
```

The UI uses a two-click confirmation design for directional controls. The first click previews the pending condition, and the second click commits generation with that action.

## Structure

```text
EasyTry/
├── app.py                 # Gradio app entry point
├── rolling_demo.json      # Example rolling-generation config
├── debug_smoke.py         # Optional terminal smoke test
└── tools/
    ├── roll/              # Dataset state and generation agents
    └── vis.py             # Visualization utilities
```

## Environment

Use the same DWM-compatible environment as `DWM/`:

```text
../requirements_dwm.txt
```

Expose both the demo directory and the DWM source tree before running the app:

```bash
cd EasyTry
export PYTHONPATH=$PWD:$PWD/../DWM/src:$PYTHONPATH
```

## Configuration

The default config template is:

```text
rolling_demo.json
```

Before running the app, update the paths in the config, including:

- nuScenes dataset root;
- 12Hz annotation files;
- base diffusion model path;
- DWM checkpoint path;
- FVD / metric checkpoint path, if metrics are enabled;
- output directory.

## Run

```bash
cd EasyTry
python app.py
```

Open the Gradio URL printed in the terminal.

## Notes

- Use this demo for quick qualitative inspection.
- Use `UniADBench/` for downstream perception / planning evaluation.
- Use `DGGT/` for geometry-aware evaluation.
- The default config is a template. Replace all dataset and checkpoint paths before running on a new machine.
