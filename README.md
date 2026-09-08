# Echo-WM T8 nodes for ComfyUI

Echo-WM T8 exposes the real **Echo-WM Flash Preview / 4-step Causal** pipeline
as three ComfyUI nodes. It accepts a first-frame `IMAGE`, prompt, and WASD/IJKL
camera action, and returns ComfyUI's native `VIDEO` object without expanding
the generated movie into an in-memory image batch.

The node package intentionally contains no model dependencies. It launches
`echo_wm/inference_wm_causal.py` with a separate Python interpreter, an argv
list, and `shell=False`. Echo-WM's Torch/CUDA stack therefore cannot overwrite
or import into the ComfyUI process.

## Nodes

- **Isolated Runtime** validates the Echo-WM source tree, dedicated Python,
  Flash checkpoint, Gemma 3 weights, GPU index, and optional FFmpeg path.
- **Action** validates and canonicalizes the public action DSL. The built-in
  one-second and ten-second actions exactly match their generation presets.
- **Causal Flash Generate** runs the 4-step distilled schedule with the bounded
  sink-plus-FIFO cache and returns native `VIDEO` plus JSON metadata.

The default is the smallest smoke test: **256×128, 25 frames, 24 fps**. A
**256×128, 241-frame (~10 second)** low-resolution preset is also included.
Requests above 384×224 or longer than 241 frames require explicit
`allow_high_resource`; invalid shapes, mismatched action durations, low free
system memory, missing weights, and accidental use of ComfyUI's Python are
rejected before loading the model.

## Install

Install this repository in `ComfyUI/custom_nodes` using ComfyUI-Manager or Git:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/Comfyui-Echo-WM-T8.git
```

Prepare Echo-WM in a **different** Python environment. From the upstream
`echo_wm` directory:

```bash
conda create -n echo-wm python=3.11 -y
conda activate echo-wm
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
hf download Echo-Team/Echo-WM --local-dir checkpoints
hf download google/gemma-3-12b-it-qat-q4_0-unquantized \
  --local-dir checkpoints/gemma-3
```

Gemma 3 is gated: accept its license and authenticate with Hugging Face before
downloading. The Runtime node expects this layout unless paths are supplied:

```text
echo_wm/
  inference_wm_causal.py
  checkpoints/
    echo-wm-flash.safetensors
    gemma-3/
      config.json
      *.safetensors
```

Set `echo_wm_root`, the dedicated environment's Python executable, and
`model_root` in **Isolated Runtime**. Blank paths use `ECHO_WM_ROOT`,
`ECHO_WM_PYTHON`, and `ECHO_WM_MODEL_ROOT`, then try common bundled layouts.
`FFMPEG_BIN` is optional and only required for the action-HUD copy.

Open [`workflows/echo_wm_t8_smoke.json`](workflows/echo_wm_t8_smoke.json) for a
minimal graph. Choose matching pairs:

| Generate preset | Action preset | Output |
| --- | --- | --- |
| `smoke_1s_256x128` | `smoke_1s_forward_turn_jump` | 25 frames |
| `low_10s_256x128` | `low_10s_forward_turn_jump` | 241 frames |

Cancelling the ComfyUI job terminates the isolated inference process tree.
Each successful run writes an MP4, a log, and JSON metadata under
`ComfyUI/output/echo_wm_t8`. Partial output is removed on failure or cancel.
The subprocess boundary, path validation, environment handling, and Registry
scanner rationale are documented in [`SECURITY.md`](SECURITY.md).

## Publish to the Comfy Registry

This repository follows the official Registry workflow. Its immutable package
name is `echo-wm-t8`, Publisher ID is `t8star`, and the repository URL is set in
`pyproject.toml`. Create a Registry publishing API key for `t8star`, add it to
the GitHub repository as the Actions secret `REGISTRY_ACCESS_TOKEN`, then run
the **Publish to Comfy registry** workflow. A push to `main` that changes
`pyproject.toml` also triggers it. Bump the semantic version for every release.

Model weights are not distributed by this node package and remain subject to
their respective upstream licenses.
