# Security design

Echo-WM T8 deliberately runs inference in a separate Python process. Echo-WM
requires a Torch/CUDA environment that is incompatible with many ComfyUI
installations, so importing that environment into the ComfyUI process would
make both runtimes less reliable.

## Process boundary

- The configured executable must resolve to an existing file and must not be
  ComfyUI's current Python interpreter.
- The generated argument vector always selects `inference_wm_causal.py` under
  the configured Echo-WM root as its script entry point.
- The command is always passed as an argument list with `shell=False`. Prompt
  text and paths are never interpolated into a shell command.
- The node exposes no free-form command, module, package-install, or
  Python-code input.
- On cancellation or timeout, Windows invokes the literal `taskkill.exe`
  executable with the launched process's numeric PID. POSIX systems signal the
  process group created for that same child. Neither path invokes a shell.

The Registry scanner therefore reports the intentional `subprocess.Popen` and
`subprocess.run` calls even though command injection through node inputs is not
possible in this implementation.

## Trust boundary

The Echo-WM root and dedicated Python path are local operator configuration.
They must point to a trusted Echo-WM installation and environment. A workflow
can carry values for these visible Runtime-node fields, so users must inspect
them before running workflows from untrusted sources. This is the same local
trust decision as selecting a custom-node installation or interpreter; the
node does not attempt to make an untrusted local executable safe.

## Paths and environment

- Echo-WM source, Python, model, and optional FFmpeg paths are resolved and
  checked before inference starts.
- The model root must contain the Flash checkpoint and a Gemma configuration
  plus safetensor shards.
- `ECHO_WM_ROOT`, `ECHO_WM_PYTHON`, `ECHO_WM_MODEL_ROOT`, and `FFMPEG_BIN` are
  optional read-only configuration sources. Explicit node inputs take
  precedence.
- A private copy of the environment is passed to the child with a bounded GPU
  index and the validated runtime paths. The parent ComfyUI environment is not
  mutated.
- Generated input, logs, metadata, and video remain under ComfyUI's configured
  temporary/output directories.

## Resource guards

The default request is 256x128 at 25 frames. Larger spatial requests or more
than 241 frames require explicit high-resource opt-in, and shape, duration,
action schedule, memory, and timeout limits are validated before model load.

## Reporting

Please report a suspected vulnerability through GitHub's private security
advisory feature for this repository rather than publishing exploit details in
a public issue.
