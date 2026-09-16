import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .runner_contract import (
    SAMPLE_VAE_DESTS,
    preset_args_to_runtime,
    preset_to_generation_widgets,
    preset_to_image_input_preset,
    runner_nproc_from_values,
)
from .runner_contract import (
    runtime_to_loader_widgets as _contract_runtime_to_loader_widgets,
)

_CONFIG_DIR = Path(__file__).resolve().parent / "preset_configs"

_HARDWARE_TAGS = frozenset(
    {
        "gfx942",
        "gfx950",
        "gfx1201",
        "h100",
        "h200",
        "b200",
        "b300",
    }
)

_ARCH_TO_TAG = {
    "gfx942": "gfx942",
    "gfx950": "gfx950",
    "gfx1201": "gfx1201",
}

_DEVICE_NAME_TAGS = [
    (re.compile(r"\bh100\b", re.I), "h100"),
    (re.compile(r"\bh200\b", re.I), "h200"),
    (re.compile(r"\bb200\b", re.I), "b200"),
    (re.compile(r"\bb300\b", re.I), "b300"),
]

_COMFY_IGNORED_BENCHMARK_ARGS = frozenset(
    {
        "warmup_calls",
        "num_iterations",
        "batch_size",
        "dataset_path",
        "output_directory",
        "use_torch_compile",
    }
)


def _canonical_gpu_arch(value: str) -> str:
    """Drop ROCm feature suffixes such as ``:sramecc+:xnack-``."""
    return str(value or "").strip().lower().split(":", 1)[0]


@dataclass(frozen=True)
class BenchmarkPreset:
    name: str
    tags: tuple[str, ...]
    model: str
    gpu_count: int
    args: dict[str, Any]
    source_file: str

    def hardware_tags(self) -> frozenset[str]:
        return frozenset(self.tags)


@dataclass
class ResolvedPreset:
    matched: bool
    preset_name: str | None
    runtime: dict[str, Any]
    gpu_tags: tuple[str, ...] = ()


def _parse_benchmark_model(entry: dict[str, Any]) -> tuple[str, int]:
    preset_name = str(entry.get("name") or "<unknown>")
    model_field = entry.get("model")
    if not isinstance(model_field, str) or not model_field:
        raise ValueError(f"preset {preset_name}: model must be a HuggingFace repo id string")
    gpu_count = entry.get("gpu_count")
    if gpu_count is None:
        gpu_count = runner_nproc_from_values(dict(entry.get("args") or {}))
    return model_field, max(int(gpu_count), 1)


def _args_to_runtime(args: dict[str, Any]) -> dict[str, Any]:
    return preset_args_to_runtime(args)


def _default_runtime(gpu_count: int) -> dict[str, Any]:
    return _args_to_runtime({"ulysses_degree": max(gpu_count, 1)})


def detect_gpu_tags() -> tuple[str, ...]:
    tags: set[str] = set()
    try:
        import torch  # type: ignore[reportMissingImports]

        if not torch.cuda.is_available():
            return tuple()
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            arch = _canonical_gpu_arch(getattr(props, "gcnArchName", "") or "")
            if arch in _ARCH_TO_TAG:
                tags.add(_ARCH_TO_TAG[arch])
            name = getattr(props, "name", "") or ""
            for pattern, tag in _DEVICE_NAME_TAGS:
                if pattern.search(name):
                    tags.add(tag)
    except Exception:
        return tuple()
    return tuple(sorted(tags))


def detect_gpu_devices() -> tuple[dict[str, str | int], ...]:
    devices: list[dict[str, str | int]] = []
    try:
        import torch  # type: ignore[reportMissingImports]

        if not torch.cuda.is_available():
            return tuple()
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            arch = getattr(props, "gcnArchName", "") or ""
            devices.append(
                {
                    "index": index,
                    "name": getattr(props, "name", "") or f"GPU {index}",
                    "arch": arch,
                    "memory_bytes": int(getattr(props, "total_memory", 0) or 0),
                }
            )
    except Exception:
        return tuple()
    return tuple(devices)


def _format_vram_gb(total_bytes: int) -> str:
    if total_bytes <= 0:
        return ""
    gb = total_bytes / (1024**3)
    rounded = round(gb)
    if abs(gb - rounded) < 0.75:
        return f"{int(rounded)}GB"
    return f"{gb:.0f}GB"


def _compact_device_label(device: dict[str, str | int]) -> str:
    arch = str(device.get("arch") or "").strip()
    label = arch or str(device.get("name") or "?")
    vram = _format_vram_gb(int(device.get("memory_bytes") or 0))
    return f"{label} {vram}".strip()


def format_gpu_detection_summary(gpu_tags: tuple[str, ...] | None = None) -> str:
    del gpu_tags
    count = available_gpu_count()
    devices = detect_gpu_devices()

    if count <= 0:
        return "No GPUs detected"

    if len(devices) == 1:
        return f"[0] {_compact_device_label(devices[0])}"

    first = devices[0]
    if all(
        device.get("arch") == first.get("arch")
        and device.get("memory_bytes") == first.get("memory_bytes")
        for device in devices
    ):
        return f"[{count}] {_compact_device_label(first)}"

    return " · ".join(f"[{device['index']}] {_compact_device_label(device)}" for device in devices)


def list_benchmark_hardware_tags() -> tuple[str, ...]:
    tags: set[str] = set()
    for preset in load_benchmark_presets():
        tags.update(preset.tags)
    preferred = ("gfx1201", "gfx942", "gfx950", "h100", "h200", "b200", "b300")
    ordered = [tag for tag in preferred if tag in tags]
    ordered.extend(sorted(tags - set(ordered)))
    return tuple(ordered)


def default_gpu_tag() -> str:
    available = list_benchmark_hardware_tags()
    if not available:
        return "gfx1201"
    detected = set(detect_gpu_tags())
    matching = [tag for tag in available if tag in detected]
    if matching:
        matching.sort(key=lambda tag: (0 if tag.startswith("gfx") else 1, tag))
        return matching[0]
    return available[0]


def normalize_gpu_tag(gpu_tag: str | None) -> str:
    tag = (gpu_tag or "").strip()
    if not tag:
        return default_gpu_tag()
    available = set(list_benchmark_hardware_tags())
    if tag in available:
        return tag
    return default_gpu_tag()


def list_gpu_counts_for_gpu_tag(gpu_tag: str) -> list[int]:
    raw = (gpu_tag or "").strip()
    if not raw:
        return []
    tag = normalize_gpu_tag(raw)
    return sorted({preset.gpu_count for preset in load_benchmark_presets() if tag in preset.tags})


def list_presets_for_gpu_tag(gpu_tag: str, gpu_count: int | None = None) -> list[str]:
    raw = (gpu_tag or "").strip()
    if not raw:
        return []
    tag = normalize_gpu_tag(raw)
    names: list[str] = []
    for preset in load_benchmark_presets():
        if tag not in preset.tags:
            continue
        if gpu_count is not None and preset.gpu_count != max(int(gpu_count), 1):
            continue
        names.append(preset.name)
    return sorted(names)


def list_hardware_matching_preset_names(
    gpu_count: int | None = None,
    gpu_tags: tuple[str, ...] | None = None,
    gpu_tag: str | None = None,
) -> list[str]:
    if gpu_tag is not None:
        return list_presets_for_gpu_tag(gpu_tag, gpu_count)
    tags = set(gpu_tags if gpu_tags is not None else detect_gpu_tags())
    names: list[str] = []
    for preset in load_benchmark_presets():
        hardware = preset.hardware_tags()
        if hardware and not hardware.issubset(tags):
            continue
        if gpu_count is not None and preset.gpu_count != gpu_count:
            continue
        names.append(preset.name)
    return sorted(names)


def available_gpu_count() -> int:
    try:
        import torch  # type: ignore[reportMissingImports]

        return max(int(torch.cuda.device_count()), 0)
    except Exception:
        return 0


def default_gpu_device_ids(gpu_count: int) -> str:
    count = max(int(gpu_count), 1)
    return ",".join(str(device_id) for device_id in range(count))


def normalize_gpu_device_ids(raw: str | None, gpu_count: int) -> str:
    text = (raw or "").strip()
    if not text or text.lower() == "auto":
        return default_gpu_device_ids(gpu_count)
    return text


def parse_gpu_device_ids(raw: str, gpu_count: int) -> list[int]:
    text = normalize_gpu_device_ids(raw, gpu_count)
    try:
        ids = [int(part.strip()) for part in text.replace(" ", "").split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(
            f"gpu_device_ids must be comma-separated GPU indices (e.g. 0,1,2,3); got {raw!r}."
        ) from exc
    if len(ids) != gpu_count:
        raise ValueError(
            f"gpu_device_ids must list exactly {gpu_count} device index(es); got {ids!r}."
        )
    if len(set(ids)) != len(ids):
        raise ValueError(f"gpu_device_ids must not contain duplicates; got {ids!r}.")
    visible = available_gpu_count()
    if visible and any(device_id < 0 or device_id >= visible for device_id in ids):
        raise ValueError(f"gpu_device_ids {ids} out of range for {visible} visible GPU(s).")
    return ids


@lru_cache(maxsize=1)
def load_benchmark_presets() -> tuple[BenchmarkPreset, ...]:
    presets: list[BenchmarkPreset] = []
    if not _CONFIG_DIR.is_dir():
        return tuple()
    for path in sorted(_CONFIG_DIR.glob("*.yaml")):
        with path.open(encoding="utf-8") as handle:
            entries = yaml.safe_load(handle) or []
        for entry in entries:
            model, gpu_count = _parse_benchmark_model(entry)
            tags = tuple(tag for tag in (entry.get("tags") or ()) if tag in _HARDWARE_TAGS)
            preset_name = str(entry["name"])
            if not tags:
                raise ValueError(f"preset {preset_name}: tags contain no recognized hardware")
            args = {
                key: value
                for key, value in dict(entry.get("args") or {}).items()
                if key not in _COMFY_IGNORED_BENCHMARK_ARGS
            }
            presets.append(
                BenchmarkPreset(
                    name=preset_name,
                    tags=tags,
                    model=model,
                    gpu_count=gpu_count,
                    args=args,
                    source_file=path.name,
                )
            )
    return tuple(presets)


def _model_aliases(model: str) -> set[str]:
    aliases = {model}
    if "/" in model:
        aliases.add(model.split("/")[-1])
    return aliases


def _model_version_base(model: str) -> str:
    tail = model.split("/")[-1] if "/" in model else model
    base_tail = re.sub(r"-\d+$", "", tail)
    if "/" in model:
        return f"{model.rsplit('/', 1)[0]}/{base_tail}"
    return base_tail


def _models_match(registry_model: str, benchmark_model: str) -> bool:
    registry_aliases = _model_aliases(registry_model)
    benchmark_aliases = _model_aliases(benchmark_model)
    if registry_aliases & benchmark_aliases or registry_model == benchmark_model:
        return True
    registry_bases = {_model_version_base(alias) for alias in registry_aliases}
    benchmark_bases = {_model_version_base(alias) for alias in benchmark_aliases}
    return bool(registry_bases & benchmark_bases)


def _score_preset(preset: BenchmarkPreset, gpu_count: int, gpu_tags: set[str]) -> int:
    if preset.gpu_count != gpu_count:
        return -1
    hardware = preset.hardware_tags()
    if hardware and not hardware.issubset(gpu_tags):
        return -1
    score = 1 if hardware & gpu_tags else 0
    name = preset.name.lower()
    if gpu_count == 1 and "single_gpu" in name:
        score += 3
    if gpu_count > 1 and f".{gpu_count}gpu." in name:
        score += 5
    if gpu_count > 1 and f"_{gpu_count}gpu" in name:
        score += 5
    if "quantgemm" in name or "sageattn" in name or "spargeattn" in name:
        score -= 2
    return score


def resolve_loader_preset(
    registry_model: str,
    gpu_count: int,
    gpu_tags: tuple[str, ...] | None = None,
) -> ResolvedPreset:
    tags = set(gpu_tags if gpu_tags is not None else detect_gpu_tags())
    candidates = [
        preset for preset in load_benchmark_presets() if _models_match(registry_model, preset.model)
    ]
    best: BenchmarkPreset | None = None
    best_score = -1
    for preset in candidates:
        score = _score_preset(preset, gpu_count, tags)
        if score > best_score:
            best = preset
            best_score = score
    if best is None or best_score < 0:
        return ResolvedPreset(
            matched=False,
            preset_name=None,
            runtime=_default_runtime(gpu_count),
            gpu_tags=tuple(sorted(tags)),
        )
    runtime = _args_to_runtime(best.args)
    return ResolvedPreset(
        matched=True,
        preset_name=best.name,
        runtime=runtime,
        gpu_tags=tuple(sorted(tags)),
    )


PRESET_NONE = "none"
PRESET_CUSTOM = "custom"
PRESET_AUTO = "auto (best for hardware)"
CUSTOM_MODEL_SENTINEL = "Custom (HF repo id)"


def is_manual_preset_choice(preset_choice: str | None) -> bool:
    choice = (preset_choice or PRESET_NONE).strip()
    return choice in (PRESET_NONE, PRESET_CUSTOM, PRESET_AUTO)


def preset_by_name(name: str) -> BenchmarkPreset | None:
    for preset in load_benchmark_presets():
        if preset.name == name:
            return preset
    return None


def runtime_to_loader_widgets(runtime: dict[str, Any]) -> dict[str, Any]:
    return _contract_runtime_to_loader_widgets(runtime)


def preset_to_loader_widgets(preset: BenchmarkPreset) -> dict[str, Any]:
    from .runner_contract import loader_display_widgets

    runtime = _args_to_runtime(preset.args)
    runtime.setdefault("model", preset.model)
    return loader_display_widgets(runtime, preset.model)


def build_preset_spec(
    preset_choice: str,
    gpu_tag: str,
    registry_choices: list[str] | None = None,
) -> dict[str, Any]:
    choice = (preset_choice or PRESET_NONE).strip()
    selected_tag = normalize_gpu_tag(gpu_tag)
    spec: dict[str, Any] = {
        "selected": choice,
        "selected_gpu_tag": selected_tag,
        "preset_name": None,
        "matched": False,
        "gpu_count": None,
        "gpu_tags": list(detect_gpu_tags()),
        "model": None,
        "model_choice": None,
        "runtime_widgets": {},
        "generation_defaults": {},
        "image_input_preset": {"paths": [], "resize_input_images": False, "required": False},
    }
    if is_manual_preset_choice(choice):
        return spec
    preset = preset_by_name(choice)
    if preset is None:
        return spec
    if selected_tag and selected_tag not in preset.tags:
        return spec
    choices = registry_choices or []
    preset_runtime = preset_args_to_runtime(preset.args)
    spec.update(
        {
            "preset_name": preset.name,
            "matched": True,
            "gpu_count": preset_gpu_count(preset),
            "gpu_device_ids": default_gpu_device_ids(preset_gpu_count(preset)),
            "model": preset.model,
            "model_choice": (
                model_choice_for_benchmark_model(preset.model, choices) if choices else preset.model
            ),
            "runtime_widgets": preset_to_loader_widgets(preset),
            "vae_defaults": {
                name: preset_runtime[name]
                for name in SAMPLE_VAE_DESTS
                if preset_runtime.get(name) is not None
            },
            "generation_defaults": preset_to_generation_widgets(preset.args),
            "image_input_preset": preset_to_image_input_preset(preset.args),
        }
    )
    return spec


def list_applicable_presets(
    registry_model: str,
    gpu_count: int,
    gpu_tags: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    tags = set(gpu_tags if gpu_tags is not None else detect_gpu_tags())
    scored: list[tuple[int, BenchmarkPreset]] = []
    for preset in load_benchmark_presets():
        if not _models_match(registry_model, preset.model):
            continue
        score = _score_preset(preset, gpu_count, tags)
        if score >= 0:
            scored.append((score, preset))
    scored.sort(key=lambda item: (-item[0], item[1].name))
    return [
        {
            "name": preset.name,
            "score": score,
            "gpu_count": preset.gpu_count,
            "widgets": preset_to_loader_widgets(preset),
        }
        for score, preset in scored
    ]


def resolve_preset_choice(
    preset_choice: str,
    registry_model: str,
    gpu_count: int,
    gpu_tags: tuple[str, ...] | None = None,
) -> ResolvedPreset | None:
    choice = (preset_choice or PRESET_NONE).strip()
    if is_manual_preset_choice(choice):
        return None
    preset = preset_by_name(choice)
    if preset is None:
        return None
    runtime = _args_to_runtime(preset.args)
    return ResolvedPreset(
        matched=True,
        preset_name=preset.name,
        runtime=runtime,
        gpu_tags=tuple(sorted(set(gpu_tags if gpu_tags is not None else detect_gpu_tags()))),
    )


def preset_gpu_count(preset: BenchmarkPreset) -> int:
    return max(int(preset.gpu_count), 1)


def model_choice_for_benchmark_model(benchmark_model: str, registry_choices: list[str]) -> str:
    if benchmark_model in registry_choices:
        return benchmark_model
    for choice in registry_choices:
        if choice != CUSTOM_MODEL_SENTINEL and _models_match(choice, benchmark_model):
            return choice
    return benchmark_model


def list_preset_names() -> list[str]:
    return sorted({preset.name for preset in load_benchmark_presets()})


def preset_loader_seed(preset: BenchmarkPreset, registry_choices: list[str]) -> dict[str, Any]:
    tag = preset.tags[0] if preset.tags else default_gpu_tag()
    spec = build_preset_spec(preset.name, tag, registry_choices=registry_choices)
    return {
        "model_choice": spec["model_choice"],
        "gpu_count": spec["gpu_count"],
        **spec["runtime_widgets"],
    }


def preset_backed_model_choices(registry_choices: list[str]) -> list[str]:
    registry = [choice for choice in registry_choices if choice != CUSTOM_MODEL_SENTINEL]
    return sorted(registry, key=str.casefold)
