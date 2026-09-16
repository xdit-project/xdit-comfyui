import json
import subprocess
from pathlib import Path

HELPER = Path(__file__).parents[2] / "web" / "xdit_preset_filters.js"
SCHEMA = {
    "gpu_counts_by_tag": {
        "gfx1201": [1, 2, 4],
        "h100": [1, 2, 4, 8],
    },
    "presets_by_tag_and_count": {
        "gfx1201": {
            "1": ["flux.1gpu.rdna4"],
            "2": ["z_image_turbo.2gpu.rdna4"],
            "4": ["flux.usp_1k.4gpu.rdna4"],
        },
        "h100": {
            "1": ["flux.1gpu.hopper"],
            "8": ["flux.usp.hopper"],
        },
    },
}


def _resolve(gpu_tag, gpu_count, preset):
    script = f"""
        import {{ resolvePresetFilters }} from {json.dumps(HELPER.as_uri())};
        console.log(JSON.stringify(resolvePresetFilters(
            {json.dumps(SCHEMA)},
            {json.dumps(gpu_tag)},
            {json.dumps(gpu_count)},
            {json.dumps(preset)},
        )));
    """
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_gpu_tag_change_refreshes_counts_and_presets():
    resolved = _resolve("h100", 2, "z_image_turbo.2gpu.rdna4")
    assert resolved["counts"] == ["1", "2", "4", "8"]
    assert resolved["gpuCount"] == "2"
    assert resolved["presets"] == []
    assert resolved["preset"] == "none"


def test_gpu_count_change_filters_presets():
    resolved = _resolve("gfx1201", 2, "flux.1gpu.rdna4")
    assert resolved["presets"] == ["z_image_turbo.2gpu.rdna4"]
    assert resolved["preset"] == "none"


def test_valid_preset_selection_is_preserved():
    resolved = _resolve("gfx1201", 2, "z_image_turbo.2gpu.rdna4")
    assert resolved["preset"] == "z_image_turbo.2gpu.rdna4"


def test_invalid_count_falls_back_to_first_available_count():
    resolved = _resolve("gfx1201", 8, "flux.1gpu.rdna4")
    assert resolved["gpuCount"] == "1"
    assert resolved["preset"] == "flux.1gpu.rdna4"


def test_preset_trigger_includes_gpu_count():
    script = f"""
        import {{ presetTrigger }} from {json.dumps(HELPER.as_uri())};
        console.log(JSON.stringify(presetTrigger("gfx1201", "2", "preset")));
    """
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == "gfx1201:2:preset"
