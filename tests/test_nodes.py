from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "echo_wm_t8_test_package"
SPEC = importlib.util.spec_from_file_location(
    PACKAGE_NAME,
    ROOT / "__init__.py",
    submodule_search_locations=[str(ROOT)],
)
assert SPEC is not None and SPEC.loader is not None
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[PACKAGE_NAME] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
NODES = sys.modules[f"{PACKAGE_NAME}.nodes"]


def test_node_registration_and_safe_defaults():
    assert set(PACKAGE.NODE_CLASS_MAPPINGS) == {
        "EchoWMRuntimeT8",
        "EchoWMActionT8",
        "EchoWMCausalGenerateT8",
    }
    generate = NODES.EchoWMCausalGenerate.INPUT_TYPES()["required"]
    assert generate["preset"][1]["default"] == "smoke_1s_256x128"
    assert generate["width"][1]["default"] == 256
    assert generate["height"][1]["default"] == 128
    assert generate["num_frames"][1]["default"] == 25
    assert NODES.EchoWMCausalGenerate.RETURN_TYPES[0] == "VIDEO"
    assert NODES.EchoWMCausalGenerate.OUTPUT_NODE is True


def test_action_presets_and_parser_are_exact():
    smoke, _ = NODES.EchoWMAction().build("smoke_1s_forward_turn_jump", "none-24")
    long, _ = NODES.EchoWMAction().build("low_10s_forward_turn_jump", "none-24")
    custom, _ = NODES.EchoWMAction().build("custom", " WJ-12，none-12 ")
    assert (smoke.action_str, smoke.total_frames) == ("w-6,lw-6,jw-6,w-6", 24)
    assert (long.action_str, long.total_frames) == ("w-60,lw-60,jw-60,w-60", 240)
    assert (custom.action_str, custom.total_frames) == ("jw-12,none-12", 24)


@pytest.mark.parametrize(
    "value",
    ["", "w", "x-24", "w-0", "ww-24", "w-nope", "w-2401"],
)
def test_action_parser_rejects_unsafe_values(value):
    with pytest.raises(ValueError):
        NODES.parse_action_string(value)


def test_runtime_requires_complete_isolated_layout(tmp_path):
    echo_root = tmp_path / "echo_wm"
    echo_root.mkdir()
    (echo_root / "inference_wm_causal.py").write_text("# test", encoding="utf-8")
    python = tmp_path / ("python.exe" if sys.platform == "win32" else "python")
    python.write_bytes(b"test")
    model_root = tmp_path / "models"
    gemma = model_root / "gemma-3"
    gemma.mkdir(parents=True)
    (model_root / "echo-wm-flash.safetensors").write_bytes(b"checkpoint")
    (gemma / "config.json").write_text("{}", encoding="utf-8")
    (gemma / "model.safetensors").write_bytes(b"gemma")

    runtime = NODES.resolve_runtime(str(echo_root), str(python), str(model_root), 1, "")
    assert runtime.gpu_index == 1
    assert runtime.checkpoint.endswith("echo-wm-flash.safetensors")
    assert runtime.public_dict()["shell"] is False
    assert runtime.public_dict()["isolation"] == "external-python"


def test_shape_action_and_resource_guards():
    smoke = NODES.EchoWMActionConfig("w-24", 24, "custom")
    assert (
        NODES.validate_request(
            width=256,
            height=128,
            num_frames=25,
            fps=24,
            action=smoke,
            allow_high_resource=False,
            memory_gib=(64.0, 96.0),
        )
        == []
    )
    with pytest.raises(ValueError, match="schedule"):
        NODES.validate_request(
            width=256,
            height=128,
            num_frames=241,
            fps=24,
            action=smoke,
            allow_high_resource=False,
            memory_gib=(64.0, 96.0),
        )
    high = NODES.EchoWMActionConfig("w-24", 24, "custom")
    with pytest.raises(RuntimeError, match="guarded preview"):
        NODES.validate_request(
            width=1280,
            height=704,
            num_frames=25,
            fps=24,
            action=high,
            allow_high_resource=False,
            memory_gib=(64.0, 96.0),
        )
    with pytest.raises(RuntimeError, match="36 GiB"):
        NODES.validate_request(
            width=256,
            height=128,
            num_frames=25,
            fps=24,
            action=smoke,
            allow_high_resource=False,
            memory_gib=(20.0, 64.0),
        )


def test_command_is_true_causal_flash_and_has_no_shell_tokens(tmp_path):
    runtime = NODES.EchoWMRuntimeConfig(
        echo_wm_root=str(tmp_path),
        python_executable=str(tmp_path / "python"),
        checkpoint=str(tmp_path / "echo-wm-flash.safetensors"),
        gemma_root=str(tmp_path / "gemma-3"),
        gpu_index=0,
        ffmpeg_bin=None,
    )
    action = NODES.EchoWMActionConfig("w-24", 24, "custom")
    command = NODES._build_command(
        runtime,
        action,
        image_path=tmp_path / "input.png",
        output_path=tmp_path / "out.mp4",
        prompt="safe; $(not-a-shell)",
        width=256,
        height=128,
        num_frames=25,
        fps=24,
        seed=42,
        include_audio=True,
        action_overlay=False,
    )
    assert command[1].endswith("inference_wm_causal.py")
    assert command[command.index("--prompt") + 1] == "safe; $(not-a-shell)"
    assert command[
        command.index("--timesteps") + 1 : command.index("--video-local-attn-size")
    ] == [
        "1000",
        "750",
        "500",
        "250",
    ]
    assert "--video-chunk-size" in command and command[-1] == "--no-action-overlay"
    source = (ROOT / "nodes.py").read_text(encoding="utf-8")
    assert "shell=True" not in source
    assert '"shell": False' in source


def test_cancel_interrupt_terminates_launched_process(tmp_path, monkeypatch):
    process = type(
        "FakeProcess",
        (),
        {"pid": 1234, "poll": lambda self: None},
    )()
    popen_options = {}
    terminated = []

    def fake_popen(command, **options):
        popen_options.update(options)
        return process

    class Cancelled(RuntimeError):
        pass

    monkeypatch.setattr(NODES.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        NODES, "_check_interrupted", lambda: (_ for _ in ()).throw(Cancelled())
    )
    monkeypatch.setattr(
        NODES, "_terminate_process", lambda value: terminated.append(value)
    )

    with pytest.raises(Cancelled):
        NODES._run_process(
            ["dedicated-python", "inference_wm_causal.py"],
            cwd=tmp_path,
            environment={},
            log_path=tmp_path / "run.log",
            timeout_seconds=60,
        )
    assert terminated == [process]
    assert popen_options["shell"] is False


def test_windows_termination_uses_fixed_executable_and_numeric_pid(monkeypatch):
    class FakeOS:
        name = "nt"

    class FakeProcess:
        pid = 1234

        def poll(self):
            return None

        def wait(self, timeout):
            assert timeout == 10
            return 0

    calls = []

    def fake_run(command, **options):
        calls.append((command, options))

    monkeypatch.setattr(NODES, "os", FakeOS())
    monkeypatch.setattr(NODES.subprocess, "run", fake_run)
    NODES._terminate_process(FakeProcess())

    assert calls[0][0] == ["taskkill.exe", "/PID", "1234", "/T", "/F"]
    assert calls[0][1]["shell"] is False
    assert calls[0][1]["check"] is False


def test_generate_smoke_returns_native_video_and_metadata(tmp_path, monkeypatch):
    output_root = tmp_path / "output"
    temp_root = tmp_path / "temp"
    monkeypatch.setattr(NODES, "_output_root", lambda: output_root)
    monkeypatch.setattr(NODES, "_temp_root", lambda: temp_root)
    monkeypatch.setattr(NODES, "_system_memory_gib", lambda: (64.0, 96.0))
    monkeypatch.setattr(
        NODES, "video_from_file", lambda path: ("native-video", str(path))
    )
    captured = {}

    def fake_run(command, *, cwd, environment, log_path, timeout_seconds):
        captured.update(
            command=command,
            cwd=cwd,
            environment=environment,
            log_path=log_path,
            timeout_seconds=timeout_seconds,
        )
        output = Path(command[command.index("--output") + 1])
        output.write_bytes(b"mp4")
        output.with_suffix(".json").write_text(
            json.dumps({"mode": "causal_4_step"}), encoding="utf-8"
        )

    monkeypatch.setattr(NODES, "_run_process", fake_run)
    runtime = NODES.EchoWMRuntimeConfig(
        echo_wm_root=str(tmp_path),
        python_executable=str(tmp_path / "python"),
        checkpoint=str(tmp_path / "flash.safetensors"),
        gemma_root=str(tmp_path / "gemma-3"),
        gpu_index=2,
        ffmpeg_bin=None,
    )
    action = NODES.EchoWMActionConfig("w-6,lw-6,jw-6,w-6", 24, "smoke")
    image = np.zeros((1, 32, 32, 3), dtype=np.float32)
    result = NODES.EchoWMCausalGenerate().generate(
        runtime,
        action,
        image,
        "A safe smoke test.",
        "smoke_1s_256x128",
        1280,
        704,
        241,
        30,
        7,
        True,
        False,
        False,
        5,
    )
    video, metadata_json = result["result"]
    metadata = json.loads(metadata_json)
    assert video[0] == "native-video"
    assert (metadata["width"], metadata["height"], metadata["num_frames"]) == (
        256,
        128,
        25,
    )
    assert metadata["mode"] == "causal_4_step_flash"
    assert metadata["inference"]["mode"] == "causal_4_step"
    assert captured["environment"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert not list(temp_root.rglob("input.png"))


def test_registry_metadata_and_example_workflow():
    import tomllib

    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["name"] == "echo-wm-t8"
    assert "comfyui" not in metadata["project"]["name"].lower()
    assert metadata["project"]["urls"]["Repository"] == (
        "https://github.com/T8mars/Comfyui-Echo-WM-T8"
    )
    assert metadata["tool"]["comfy"]["PublisherId"] == "t8star"
    workflow = json.loads(
        (ROOT / "workflows" / "echo_wm_t8_smoke.json").read_text(encoding="utf-8")
    )
    assert {node["type"] for node in workflow["nodes"]} >= {
        "EchoWMRuntimeT8",
        "EchoWMActionT8",
        "EchoWMCausalGenerateT8",
    }
