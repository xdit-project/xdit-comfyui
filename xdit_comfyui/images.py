"""Pixels in and out: Comfy IMAGE/VIDEO tensors, staged input files, preview thumbnails."""

import shutil
import tempfile
from pathlib import Path

try:
    import folder_paths  # type: ignore[reportMissingImports]
except Exception:  # pragma: no cover - only used inside ComfyUI runtime
    folder_paths = None


# Enough that a run in flight, and the one a user may still be comparing against, are
# never swept; small enough that a long session does not fill /tmp.
_SCRATCH_KEEP = 8


def _scratch_mtime(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _prune_scratch_dirs(base: Path, prefix: str, keep: int = _SCRATCH_KEEP):
    """Drop the scratch directories earlier runs left behind.

    Results come back over the worker socket, not off disk, so nothing reads these once
    the run returns. Pruning as the next run starts rather than in a `finally` also
    clears what a crashed or cancelled run left behind.
    """
    try:
        stale = sorted(
            (path for path in base.glob(f"{prefix}*") if path.is_dir()),
            key=_scratch_mtime,
            reverse=True,
        )[keep:]
    except OSError:
        return
    for path in stale:
        shutil.rmtree(path, ignore_errors=True)


def _generation_output_directory():
    base = Path(tempfile.gettempdir()) / "xdit_comfyui"
    base.mkdir(parents=True, exist_ok=True)
    _prune_scratch_dirs(base, "run_")
    return Path(tempfile.mkdtemp(prefix="run_", dir=base))


def _generation_input_staging_directory():
    base = Path(tempfile.gettempdir()) / "xdit_comfyui" / "inputs"
    base.mkdir(parents=True, exist_ok=True)
    _prune_scratch_dirs(base, "in_")
    return Path(tempfile.mkdtemp(prefix="in_", dir=base))


def _placeholder_image_tensor():
    import torch  # type: ignore[reportMissingImports]

    return torch.zeros((1, 64, 64, 3), dtype=torch.float32)


def _is_preset_placeholder_image(image) -> bool:
    import torch  # type: ignore[reportMissingImports]

    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        return False
    if tuple(int(dim) for dim in image.shape[1:3]) != (64, 64):
        return False
    tensor = image.detach().cpu()
    return float(tensor.mean()) == 0.0 and float(tensor.std()) == 0.0


def _load_preset_reference_image(image_input_preset):
    spec = _parse_image_input_preset(image_input_preset)
    if not spec["paths"]:
        return _placeholder_image_tensor()
    return _paths_to_comfy_image(spec["paths"], resize_input_images=spec["resize_input_images"])


def _parse_image_input_preset(image_input_preset):
    if not isinstance(image_input_preset, dict):
        return {"paths": [], "resize_input_images": False, "required": False}
    raw_paths = image_input_preset.get("paths")
    if raw_paths is None:
        paths: list[str] = []
    elif isinstance(raw_paths, str):
        paths = [raw_paths] if raw_paths.strip() else []
    else:
        paths = [str(path) for path in raw_paths if str(path).strip()]
    return {
        "paths": paths,
        "resize_input_images": bool(image_input_preset.get("resize_input_images", False)),
        "required": bool(image_input_preset.get("required", paths)),
    }


def _paths_to_comfy_image(paths: list[str], *, resize_input_images: bool = False):
    import numpy as np
    import torch  # type: ignore[reportMissingImports]
    from PIL import Image

    from .benchmark_data import resolve_benchmark_data_path

    if not paths:
        return _placeholder_image_tensor()

    pil_images = []
    for path in paths:
        resolved = resolve_benchmark_data_path(path)
        src = Path(resolved)
        if not src.is_file():
            raise FileNotFoundError(f"Preset reference image not found: {path}")
        pil_images.append(Image.open(src).convert("RGB"))

    if resize_input_images and pil_images:
        target = pil_images[0].size
        pil_images = [
            img if img.size == target else img.resize(target, Image.Resampling.LANCZOS)
            for img in pil_images
        ]

    if len(pil_images) == 1:
        arr = np.array(pil_images[0], dtype=np.float32) / 255.0
        return torch.from_numpy(arr[np.newaxis, ...])

    max_w = max(img.width for img in pil_images)
    max_h = max(img.height for img in pil_images)
    frames = []
    for img in pil_images:
        if img.width == max_w and img.height == max_h:
            canvas = img
        else:
            canvas = Image.new("RGB", (max_w, max_h))
            canvas.paste(img, (0, 0))
        frames.append(np.array(canvas, dtype=np.float32) / 255.0)
    return torch.from_numpy(np.stack(frames, axis=0))


def _resolve_sample_input_paths(images, preset) -> list[str]:
    if images is not None and not _is_preset_placeholder_image(images):
        return _comfy_image_batch_to_paths(images, _generation_input_staging_directory())
    if isinstance(preset, dict):
        image_input_preset = preset.get("image_input_preset") or {}
        paths = [str(path) for path in (image_input_preset.get("paths") or []) if str(path).strip()]
        if paths:
            return paths
    return []


def _comfy_image_batch_to_paths(image, dest_dir: Path) -> list[str]:
    import numpy as np
    import torch  # type: ignore[reportMissingImports]
    from PIL import Image

    if image is None:
        return []
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a ComfyUI IMAGE tensor")
    if image.ndim != 4:
        raise ValueError(f"expected IMAGE tensor NHWC, got shape {tuple(image.shape)}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    batch = image.detach().cpu().clamp(0, 1)
    for index in range(int(batch.shape[0])):
        arr = (batch[index].numpy() * 255.0).round().astype(np.uint8)
        path = dest_dir / f"input_{index:03d}.png"
        Image.fromarray(arr).save(path)
        paths.append(str(path))
    return paths


def _pil_rgb_array(image):
    """convert() copies even when the mode already matches, which is not free at 1080p."""
    import numpy as np

    return np.asarray(image if getattr(image, "mode", None) == "RGB" else image.convert("RGB"))


def _video_frames_to_numpy_list(video):
    import numpy as np

    if isinstance(video, list):
        if video and isinstance(video[0], list):
            video = video[0]
        if video and hasattr(video[0], "convert"):
            return [_pil_rgb_array(item) for item in video]

    arr = np.asarray(video)
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    if arr.ndim == 5:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[1] in (1, 3):
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.ndim == 4:
        return [arr[index] for index in range(arr.shape[0])]
    if arr.ndim == 3:
        return [arr]
    raise ValueError(
        f"Unsupported xDiT video frame array shape: {getattr(arr, 'shape', type(arr))}"
    )


def _frame_arrays_to_comfy_tensor(frame_arrays):
    """Frames to a Comfy IMAGE batch in one pass over the pixels.

    A video batch is a couple of GiB, so converting and clamping in separate numpy
    steps costs a full copy each time.
    """
    import numpy as np
    import torch  # type: ignore[reportMissingImports]

    batch = torch.from_numpy(np.stack(frame_arrays))
    if batch.dtype == torch.uint8:
        return batch.to(torch.float32).div_(255.0)
    batch = batch.to(torch.float32)
    if float(batch.max()) > 1.0:
        batch.div_(255.0)
    return batch.clamp_(0.0, 1.0)


def _diffusion_output_kind(output):
    if output.images:
        return "image"
    if output.videos:
        return "video"
    return None


def _diffusion_output_to_comfy_image(output):
    frames = []
    if output.images:
        for image, _ in output.get_outputs():
            frames.append(_pil_rgb_array(image))
    elif output.videos:
        videos = list(output.get_outputs())
        if len(videos) > 1:
            raise ValueError("Multiple video outputs are not supported by the Comfy VIDEO output.")
        frames.extend(_video_frames_to_numpy_list(videos[0][0]))
    if not frames:
        raise ValueError("xDiT run produced no image or video frames.")
    return _frame_arrays_to_comfy_tensor(frames)


def _generation_outputs_from_frames(frames, output_kind, fps=24):
    if output_kind == "image":
        if frames.ndim != 4:
            raise ValueError("Expected an image batch with shape [N, H, W, C].")
        return frames, None
    if output_kind != "video":
        raise ValueError(f"Unknown xDiT output kind: {output_kind!r}")
    from fractions import Fraction

    try:
        from comfy_api.latest._input_impl.video_types import VideoFromComponents
        from comfy_api.latest._util.video_types import VideoComponents
    except Exception as exc:
        # Dropping the frames here would throw away a finished run without a word.
        raise RuntimeError(
            f"ComfyUI's video API is unavailable, so the {frames.shape[0]} generated "
            f"frames cannot be returned as a video: {exc}"
        ) from exc

    video = VideoFromComponents(VideoComponents(images=frames, frame_rate=Fraction(int(fps), 1)))
    return None, video
