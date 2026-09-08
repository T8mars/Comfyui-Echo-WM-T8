"""ComfyUI nodes for isolated Echo-WM Flash/Causal inference.

The model stack is deliberately never imported into ComfyUI.  A dedicated
Python interpreter launches ``echo_wm/inference_wm_causal.py`` with an argv
list and ``shell=False`` so its Torch/CUDA packages cannot replace ComfyUI's.
"""

from __future__ import annotations

import ctypes
import json
import os
import signal
import subprocess
import sys
import time
import unicodedata
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__:
    from .video import video_from_file
else:  # Supports direct test/validator imports outside ComfyUI's package loader.
    from video import video_from_file


CATEGORY = "Echo-WM T8"
SCHEMA = "echo-wm-t8.comfy.v1"
ALLOWED_ACTION_KEYS = frozenset("wsadikjl")
CAUSAL_TIMESTEPS = (1000, 750, 500, 250)
VIDEO_LOCAL_ATTN_SIZE = 19
VIDEO_SINK_SIZE = 7
VIDEO_CHUNK_SIZE = 3

GENERATION_PRESETS: dict[str, tuple[int, int, int, int]] = {
    "smoke_1s_256x128": (256, 128, 25, 24),
    "low_10s_256x128": (256, 128, 241, 24),
}

ACTION_PRESETS: dict[str, str] = {
    "smoke_1s_forward_turn_jump": "w-6,wl-6,wj-6,w-6",
    "low_10s_forward_turn_jump": "w-60,wl-60,wj-60,w-60",
}


@dataclass(frozen=True)
class EchoWMRuntimeConfig:
    echo_wm_root: str
    python_executable: str
    checkpoint: str
    gemma_root: str
    gpu_index: int
    ffmpeg_bin: str | None

    def public_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(schema=SCHEMA, isolation="external-python", shell=False)
        return result


@dataclass(frozen=True)
class EchoWMActionConfig:
    action_str: str
    total_frames: int
    preset: str

    def public_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, **asdict(self)}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str)


def _normalise_text(value: Any, field: str, *, maximum: int) -> str:
    text = unicodedata.normalize("NFC", str(value)).strip()
    if not text or len(text) > maximum:
        raise ValueError(f"{field} must contain 1-{maximum} characters")
    if "\x00" in text:
        raise ValueError(f"{field} cannot contain NUL characters")
    return text


def parse_action_string(value: str) -> tuple[str, int]:
    """Validate and canonicalize the public WASD/IJKL action DSL."""

    action = "".join(
        unicodedata.normalize("NFC", str(value)).replace("，", ",").split()
    )
    if not action:
        raise ValueError("action string is empty")
    if len(action) > 2048 or "\x00" in action:
        raise ValueError("action string is too long or contains NUL")
    canonical: list[str] = []
    total = 0
    for segment in action.split(","):
        if not segment or "-" not in segment:
            raise ValueError(
                f"invalid action segment {segment!r}; use '<keys>-<frames>'"
            )
        keys_part, duration_part = segment.rsplit("-", 1)
        if not duration_part.isascii() or not duration_part.isdecimal():
            raise ValueError(f"invalid frame count in action segment {segment!r}")
        duration = int(duration_part)
        if not 1 <= duration <= 2400:
            raise ValueError("each action segment must last 1-2400 frames")
        lowered = keys_part.lower()
        if lowered == "none":
            keys = "none"
        else:
            if not lowered or len(lowered) != len(set(lowered)):
                raise ValueError(
                    f"action keys must be non-empty and unique in {segment!r}"
                )
            invalid = sorted(set(lowered) - ALLOWED_ACTION_KEYS)
            if invalid:
                allowed = "".join(sorted(ALLOWED_ACTION_KEYS))
                raise ValueError(
                    f"unknown action keys {invalid}; allowed keys: {allowed}"
                )
            keys = "".join(sorted(lowered))
        canonical.append(f"{keys}-{duration}")
        total += duration
        if total > 2400:
            raise ValueError("action schedule cannot exceed 2400 frames")
    return ",".join(canonical), total


def _is_same_file(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except (FileNotFoundError, OSError):
        return left.resolve() == right.resolve()


def _root_candidates() -> list[Path]:
    candidates: list[Path] = []
    configured = os.environ.get("ECHO_WM_ROOT", "").strip()
    if configured:
        candidates.append(Path(configured))
    node_root = Path(__file__).resolve().parent
    for base in (node_root, *list(node_root.parents)[:6]):
        candidates.extend((base / "echo_wm", base / "runtime" / "echo-wm"))
    return candidates


def _resolve_echo_root(value: str) -> Path:
    raw = str(value).strip()
    candidates = [Path(raw).expanduser()] if raw else _root_candidates()
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "inference_wm_causal.py").is_file():
            return resolved
    supplied = raw or "automatic discovery"
    raise FileNotFoundError(
        f"Echo-WM root is invalid ({supplied}); inference_wm_causal.py was not found"
    )


def _python_candidates(echo_root: Path) -> list[Path]:
    configured = os.environ.get("ECHO_WM_PYTHON", "").strip()
    names = (
        Path("python/python.exe"),
        Path(".venv/Scripts/python.exe"),
        Path(".venv/bin/python"),
        Path("venv/Scripts/python.exe"),
        Path("venv/bin/python"),
    )
    candidates = [Path(configured)] if configured else []
    candidates.extend(echo_root / name for name in names)
    candidates.extend(echo_root.parent / name for name in names)
    return candidates


def _resolve_python(value: str, echo_root: Path) -> Path:
    raw = str(value).strip()
    candidates = [Path(raw).expanduser()] if raw else _python_candidates(echo_root)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            if _is_same_file(resolved, Path(sys.executable)):
                raise ValueError(
                    "python_executable must be a dedicated Echo-WM interpreter, not ComfyUI's Python"
                )
            return resolved
    supplied = raw or "automatic discovery"
    raise FileNotFoundError(f"dedicated Echo-WM Python was not found ({supplied})")


def _model_candidates(echo_root: Path) -> list[Path]:
    configured = os.environ.get("ECHO_WM_MODEL_ROOT", "").strip()
    candidates = [Path(configured)] if configured else []
    candidates.extend(
        (
            echo_root / "checkpoints",
            echo_root.parent / "models" / "echo-wm",
            echo_root.parent.parent / "models" / "echo-wm",
        )
    )
    return candidates


def _model_layout_ready(root: Path) -> bool:
    gemma = root / "gemma-3"
    return (
        (root / "echo-wm-flash.safetensors").is_file()
        and (gemma / "config.json").is_file()
        and any(gemma.glob("*.safetensors"))
    )


def _resolve_model_root(value: str, echo_root: Path) -> Path:
    raw = str(value).strip()
    candidates = [Path(raw).expanduser()] if raw else _model_candidates(echo_root)
    for candidate in candidates:
        resolved = candidate.resolve()
        if _model_layout_ready(resolved):
            return resolved
    supplied = raw or "automatic discovery"
    raise FileNotFoundError(
        "Echo-WM Flash model files are incomplete "
        f"({supplied}); expected echo-wm-flash.safetensors and gemma-3 weights"
    )


def _resolve_optional_executable(value: str, variable: str) -> str | None:
    raw = str(value).strip() or os.environ.get(variable, "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured executable does not exist: {path}")
    return str(path)


def resolve_runtime(
    echo_wm_root: str,
    python_executable: str,
    model_root: str,
    gpu_index: int,
    ffmpeg_bin: str,
) -> EchoWMRuntimeConfig:
    if not 0 <= int(gpu_index) <= 31:
        raise ValueError("gpu_index must be between 0 and 31")
    root = _resolve_echo_root(echo_wm_root)
    python = _resolve_python(python_executable, root)
    models = _resolve_model_root(model_root, root)
    return EchoWMRuntimeConfig(
        echo_wm_root=str(root),
        python_executable=str(python),
        checkpoint=str(models / "echo-wm-flash.safetensors"),
        gemma_root=str(models / "gemma-3"),
        gpu_index=int(gpu_index),
        ffmpeg_bin=_resolve_optional_executable(ffmpeg_bin, "FFMPEG_BIN"),
    )


def _system_memory_gib() -> tuple[float | None, float | None]:
    if os.name == "nt":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            gib = float(1024**3)
            return status.available_physical / gib, status.total_physical / gib
        return None, None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available = os.sysconf("SC_AVPHYS_PAGES") * page_size / 1024**3
        total = os.sysconf("SC_PHYS_PAGES") * page_size / 1024**3
        return float(available), float(total)
    except (AttributeError, OSError, ValueError):
        return None, None


def validate_request(
    *,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    action: EchoWMActionConfig,
    allow_high_resource: bool,
    memory_gib: tuple[float | None, float | None] | None = None,
) -> list[str]:
    if width < 128 or height < 128 or width % 32 or height % 32:
        raise ValueError("width and height must be at least 128 and divisible by 32")
    if width > 1280 or height > 1280:
        raise ValueError("width and height cannot exceed 1280")
    if num_frames < 25 or num_frames > 2401 or (num_frames - 1) % 24:
        raise ValueError("num_frames must be 1 + 24n in the range 25-2401")
    if not 8 <= fps <= 60:
        raise ValueError("fps must be between 8 and 60")
    if not isinstance(action, EchoWMActionConfig):
        raise TypeError("action must come from an Echo-WM Action node")
    if action.total_frames != num_frames - 1:
        raise ValueError(
            f"action schedule has {action.total_frames} frames; this output needs {num_frames - 1}"
        )

    high_spatial = width * height > 384 * 224
    long_rollout = num_frames > 241
    if (high_spatial or long_rollout) and not allow_high_resource:
        raise RuntimeError(
            "request exceeds the guarded preview limit; use 256x128, or explicitly enable high-resource mode"
        )

    available, total = memory_gib if memory_gib is not None else _system_memory_gib()
    if available is not None and available < 36.0:
        raise RuntimeError(
            f"only {available:.1f} GiB system RAM is available; Echo-WM requires at least 36 GiB free"
        )
    if high_spatial and total is not None and total < 90.0 and not allow_high_resource:
        raise RuntimeError(
            "high-resolution Echo-WM inference needs a host with at least 90 GiB RAM"
        )
    warnings: list[str] = []
    if available is not None and available < 44.0:
        warnings.append(
            f"available system RAM is {available:.1f} GiB; paging is possible"
        )
    return warnings


def _output_root() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.get_output_directory()) / "echo_wm_t8"
    except ImportError:
        return Path.cwd() / "output" / "echo_wm_t8"


def _temp_root() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.get_temp_directory()) / "echo_wm_t8"
    except ImportError:
        return Path.cwd() / "temp" / "echo_wm_t8"


def _write_input_image(image: Any, destination: Path) -> None:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("ComfyUI's NumPy/Pillow packages are unavailable") from exc

    ndim = getattr(image, "ndim", None)
    if ndim != 4 or int(image.shape[0]) != 1 or int(image.shape[-1]) not in (3, 4):
        raise ValueError(
            "image must be exactly one ComfyUI IMAGE with RGB or RGBA channels"
        )
    frame = image[0]
    if hasattr(frame, "detach"):
        frame = frame.detach().float().cpu().numpy()
    array = np.asarray(frame)
    if not np.isfinite(array).all():
        raise ValueError("image contains NaN or infinite values")
    rgb = np.rint(np.clip(array[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)
    destination.parent.mkdir(parents=True, exist_ok=False)
    Image.fromarray(rgb).save(destination, format="PNG")


def _check_interrupted() -> None:
    try:
        from comfy.model_management import throw_exception_if_processing_interrupted

        throw_exception_if_processing_interrupted()
    except ImportError:
        return


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    """Terminate inference and its process tree when ComfyUI cancels."""

    if process.poll() is not None:
        return
    if os.name == "nt":
        # taskkill is invoked directly (never through cmd.exe) so descendants
        # such as a late HUD encoder are terminated as well.
        subprocess.run(
            ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            shell=False,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait(timeout=5)


def _tail_log(path: Path, limit: int = 60) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-limit:])


def _run_process(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
    timeout_seconds: float,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    popen_options: dict[str, Any] = {
        "cwd": str(cwd),
        "env": environment,
        "stdin": subprocess.DEVNULL,
        "stdout": None,
        "stderr": subprocess.STDOUT,
        "shell": False,
    }
    if os.name == "nt":
        popen_options.update(
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            startupinfo=_hidden_startup_info(),
        )
    else:
        popen_options["start_new_session"] = True

    started = time.monotonic()
    with log_path.open("wb") as log:
        popen_options["stdout"] = log
        process = subprocess.Popen(command, **popen_options)
        try:
            while True:
                _check_interrupted()
                return_code = process.poll()
                if return_code is not None:
                    break
                if time.monotonic() - started > timeout_seconds:
                    raise TimeoutError("Echo-WM inference exceeded timeout_minutes")
                time.sleep(0.25)
        except BaseException:
            _terminate_process(process)
            raise
    if return_code != 0:
        detail = _tail_log(log_path)
        suffix = f"\n{detail}" if detail else ""
        raise RuntimeError(
            f"Echo-WM Causal process exited with code {return_code}{suffix}"
        )


def _hidden_startup_info():
    if os.name != "nt":
        return None
    info = subprocess.STARTUPINFO()
    info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    info.wShowWindow = subprocess.SW_HIDE
    return info


def _build_command(
    runtime: EchoWMRuntimeConfig,
    action: EchoWMActionConfig,
    *,
    image_path: Path,
    output_path: Path,
    prompt: str,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    seed: int,
    include_audio: bool,
    action_overlay: bool,
) -> list[str]:
    command = [
        runtime.python_executable,
        str(Path(runtime.echo_wm_root) / "inference_wm_causal.py"),
        "--image",
        str(image_path),
        "--prompt",
        prompt,
        "--action-str",
        action.action_str,
        "--checkpoint",
        runtime.checkpoint,
        "--gemma-path",
        runtime.gemma_root,
        "--output",
        str(output_path),
        "--width",
        str(width),
        "--height",
        str(height),
        "--num-frames",
        str(num_frames),
        "--fps",
        str(fps),
        "--timesteps",
        *(str(value) for value in CAUSAL_TIMESTEPS),
        "--video-local-attn-size",
        str(VIDEO_LOCAL_ATTN_SIZE),
        "--video-sink-size",
        str(VIDEO_SINK_SIZE),
        "--video-chunk-size",
        str(VIDEO_CHUNK_SIZE),
        "--seed",
        str(seed),
        "--action-overlay" if action_overlay else "--no-action-overlay",
    ]
    if not include_audio:
        command.append("--no-audio")
    return command


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


class EchoWMRuntime:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "echo_wm_root": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Directory containing inference_wm_causal.py; blank enables discovery.",
                    },
                ),
                "python_executable": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Dedicated Echo-WM Python. ComfyUI's interpreter is rejected.",
                    },
                ),
                "model_root": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Contains echo-wm-flash.safetensors and gemma-3/.",
                    },
                ),
                "gpu_index": ("INT", {"default": 0, "min": 0, "max": 31}),
                "ffmpeg_bin": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional ffmpeg path, needed only for the action HUD overlay.",
                    },
                ),
                "refresh_nonce": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            }
        }

    RETURN_TYPES = ("ECHO_WM_RUNTIME", "STRING")
    RETURN_NAMES = ("runtime", "runtime_metadata")
    FUNCTION = "configure"
    CATEGORY = CATEGORY

    def configure(
        self,
        echo_wm_root: str,
        python_executable: str,
        model_root: str,
        gpu_index: int,
        ffmpeg_bin: str,
        refresh_nonce: int,
    ):
        del refresh_nonce
        runtime = resolve_runtime(
            echo_wm_root, python_executable, model_root, gpu_index, ffmpeg_bin
        )
        return runtime, _json(runtime.public_dict())


class EchoWMAction:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (
                    [*ACTION_PRESETS, "custom"],
                    {"default": "smoke_1s_forward_turn_jump"},
                ),
                "custom_action": (
                    "STRING",
                    {
                        "default": "w-6,wl-6,wj-6,w-6",
                        "multiline": False,
                        "tooltip": "Used only when preset is custom. Syntax: keys-frames,keys-frames.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("ECHO_WM_ACTION", "STRING")
    RETURN_NAMES = ("action", "action_metadata")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, preset: str, custom_action: str):
        if preset not in (*ACTION_PRESETS, "custom"):
            raise ValueError(f"unknown action preset: {preset}")
        source = custom_action if preset == "custom" else ACTION_PRESETS[preset]
        action_str, total = parse_action_string(source)
        action = EchoWMActionConfig(
            action_str=action_str, total_frames=total, preset=preset
        )
        return action, _json(action.public_dict())


class EchoWMCausalGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "runtime": ("ECHO_WM_RUNTIME",),
                "action": ("ECHO_WM_ACTION",),
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {
                        "default": "A cinematic world viewed from a moving camera.",
                        "multiline": True,
                    },
                ),
                "preset": (
                    [*GENERATION_PRESETS, "custom"],
                    {"default": "smoke_1s_256x128"},
                ),
                "width": ("INT", {"default": 256, "min": 128, "max": 1280, "step": 32}),
                "height": (
                    "INT",
                    {"default": 128, "min": 128, "max": 1280, "step": 32},
                ),
                "num_frames": (
                    "INT",
                    {"default": 25, "min": 25, "max": 2401, "step": 24},
                ),
                "fps": ("INT", {"default": 24, "min": 8, "max": 60}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**63 - 1}),
                "include_audio": ("BOOLEAN", {"default": True}),
                "action_overlay": ("BOOLEAN", {"default": False}),
                "allow_high_resource": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Required above 384x224 or 241 frames. Large requests can exhaust system RAM.",
                    },
                ),
                "timeout_minutes": ("INT", {"default": 240, "min": 1, "max": 1440}),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "metadata_json")
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def generate(
        self,
        runtime: EchoWMRuntimeConfig,
        action: EchoWMActionConfig,
        image: Any,
        prompt: str,
        preset: str,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        seed: int,
        include_audio: bool,
        action_overlay: bool,
        allow_high_resource: bool,
        timeout_minutes: int,
    ):
        if not isinstance(runtime, EchoWMRuntimeConfig):
            raise TypeError("runtime must come from an Echo-WM Runtime node")
        prompt = _normalise_text(prompt, "prompt", maximum=5000)
        if preset in GENERATION_PRESETS:
            width, height, num_frames, fps = GENERATION_PRESETS[preset]
        elif preset != "custom":
            raise ValueError(f"unknown generation preset: {preset}")
        warnings = validate_request(
            width=int(width),
            height=int(height),
            num_frames=int(num_frames),
            fps=int(fps),
            action=action,
            allow_high_resource=bool(allow_high_resource),
        )
        _check_interrupted()

        run_id = uuid.uuid4().hex
        output_dir = _output_root()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"echo_wm_t8_{run_id}.mp4"
        log_path = output_dir / f"echo_wm_t8_{run_id}.log"
        temp_dir = _temp_root() / run_id
        input_path = temp_dir / "input.png"
        _write_input_image(image, input_path)

        command = _build_command(
            runtime,
            action,
            image_path=input_path,
            output_path=output_path,
            prompt=prompt,
            width=int(width),
            height=int(height),
            num_frames=int(num_frames),
            fps=int(fps),
            seed=int(seed),
            include_audio=bool(include_audio),
            action_overlay=bool(action_overlay),
        )
        environment = os.environ.copy()
        environment.update(
            PYTHONUTF8="1",
            PYTHONDONTWRITEBYTECODE="1",
            CUDA_VISIBLE_DEVICES=str(runtime.gpu_index),
        )
        if runtime.ffmpeg_bin:
            environment["FFMPEG_BIN"] = runtime.ffmpeg_bin
            ffmpeg_path = Path(runtime.ffmpeg_bin)
            ffprobe = ffmpeg_path.with_name(
                "ffprobe.exe" if ffmpeg_path.suffix.lower() == ".exe" else "ffprobe"
            )
            if ffprobe.is_file():
                environment["FFPROBE_BIN"] = str(ffprobe)

        started_at = datetime.now(timezone.utc)
        started = time.monotonic()
        try:
            _run_process(
                command,
                cwd=Path(runtime.echo_wm_root),
                environment=environment,
                log_path=log_path,
                timeout_seconds=float(timeout_minutes) * 60.0,
            )
            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise RuntimeError("Echo-WM completed without a non-empty MP4 output")
            selected_output = output_path
            overlay_path = output_path.with_name(
                f"{output_path.stem}_action{output_path.suffix}"
            )
            if action_overlay:
                if not overlay_path.is_file() or overlay_path.stat().st_size == 0:
                    raise RuntimeError(
                        "action overlay was requested but its MP4 was not created"
                    )
                selected_output = overlay_path
            cli_metadata_path = output_path.with_suffix(".json")
            cli_metadata: dict[str, Any] = {}
            if cli_metadata_path.is_file():
                try:
                    loaded = json.loads(cli_metadata_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        cli_metadata = loaded
                except (OSError, json.JSONDecodeError):
                    warnings.append("the inference metadata sidecar could not be read")
            metadata = {
                "schema": SCHEMA,
                "mode": "causal_4_step_flash",
                "preset": preset,
                "prompt": prompt,
                "action": action.public_dict(),
                "width": int(width),
                "height": int(height),
                "num_frames": int(num_frames),
                "fps": int(fps),
                "seed": int(seed),
                "include_audio": bool(include_audio),
                "action_overlay": bool(action_overlay),
                "timesteps": list(CAUSAL_TIMESTEPS),
                "cache": {
                    "video_local_attn_size": VIDEO_LOCAL_ATTN_SIZE,
                    "video_sink_size": VIDEO_SINK_SIZE,
                    "video_chunk_size": VIDEO_CHUNK_SIZE,
                },
                "runtime": runtime.public_dict(),
                "output_path": str(selected_output.resolve()),
                "raw_output_path": str(output_path.resolve()),
                "log_path": str(log_path.resolve()),
                "started_at": started_at.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "warnings": warnings,
                "inference": cli_metadata,
            }
            values = (video_from_file(selected_output), _json(metadata))
            return {"ui": {"echo_wm_t8": [metadata]}, "result": values}
        except BaseException:
            _safe_unlink(output_path)
            _safe_unlink(
                output_path.with_name(f"{output_path.stem}_action{output_path.suffix}")
            )
            _safe_unlink(output_path.with_suffix(".json"))
            raise
        finally:
            _safe_unlink(input_path)
            try:
                temp_dir.rmdir()
            except OSError:
                pass


NODE_CLASS_MAPPINGS = {
    "EchoWMRuntimeT8": EchoWMRuntime,
    "EchoWMActionT8": EchoWMAction,
    "EchoWMCausalGenerateT8": EchoWMCausalGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EchoWMRuntimeT8": "Echo-WM T8 · Isolated Runtime",
    "EchoWMActionT8": "Echo-WM T8 · Action",
    "EchoWMCausalGenerateT8": "Echo-WM T8 · Causal Flash Generate",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
