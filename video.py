"""Small adapter for ComfyUI's native VIDEO value."""

from __future__ import annotations

from pathlib import Path


def video_from_file(path: str | Path):
    """Create a native VIDEO object without decoding frames in this process."""

    resolved = str(Path(path).expanduser().resolve())
    try:
        from comfy_api.input_impl import VideoFromFile

        return VideoFromFile(resolved)
    except ImportError:
        try:
            from comfy_api.latest import InputImpl

            return InputImpl.VideoFromFile(resolved)
        except ImportError as exc:
            raise RuntimeError(
                "Echo-WM T8 requires a current ComfyUI build with native VIDEO support."
            ) from exc


__all__ = ["video_from_file"]
