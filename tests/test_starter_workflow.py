import json

from xdit_comfyui.api import _EXAMPLE_WORKFLOWS_DIR, _TEMPLATE_MODULE
from xdit_comfyui.starter_workflow import (
    NAMED_WIDGET_VALUES_KEY,
    build_starter_api_prompt,
    build_starter_workflow_dict,
    starter_template_revision,
)

STARTER_NODE_TYPES = frozenset(
    {
        "xDiT.Preset",
        "xDiT.Model",
        "xDiT.Sample",
        "SaveImage",
        "SaveVideo",
    }
)
STARTER_FILENAME = "xDiT-Starter.json"
TEMPLATE_FILENAMES = {"xDiT-Starter.json"}


def test_starter_workflow_uses_comfy_link_arrays():
    workflow = build_starter_workflow_dict()
    assert workflow["revision"] == starter_template_revision(workflow)
    assert isinstance(workflow["revision"], str)
    assert len(workflow["revision"]) == 12
    assert isinstance(workflow.get("id"), str)
    assert workflow["nodes"]
    assert workflow["links"]
    assert isinstance(workflow["links"][0], list)
    assert len(workflow["links"][0]) == 6
    assert len(workflow["links"]) == 6
    assert workflow["last_link_id"] == len(workflow["links"])
    assert "extra" in workflow and "ds" in workflow["extra"]


def test_starter_workflow_revision_stable_for_same_structure():
    first = build_starter_workflow_dict()
    second = build_starter_workflow_dict()
    assert first["revision"] == second["revision"]
    assert first["id"] != second["id"]


def test_starter_workflow_revision_changes_with_preset():
    flux = build_starter_workflow_dict(preset_name="flux.1gpu.rdna4")
    turbo = build_starter_workflow_dict(preset_name="z_image_turbo.1gpu.rdna4")
    assert flux["revision"] != turbo["revision"]


def test_on_disk_starter_revision_matches_content_hash():
    path = _EXAMPLE_WORKFLOWS_DIR / STARTER_FILENAME
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["revision"] == starter_template_revision(on_disk)


def test_starter_workflow_nodes_include_widget_values():
    workflow = build_starter_workflow_dict()
    by_type = {node["type"]: node for node in workflow["nodes"]}
    for class_type in ("xDiT.Model", "xDiT.Sample", "xDiT.Preset"):
        assert by_type[class_type][NAMED_WIDGET_VALUES_KEY]


def test_starter_workflow_carries_no_positional_widget_values():
    """Positions shift as widgets are grouped; only the named map is authoritative."""
    workflow = build_starter_workflow_dict()
    for node in workflow["nodes"]:
        if node["type"].startswith("XDiT"):
            assert node["widgets_values"] == []


def test_starter_workflow_sets_preset_gpu_count_by_name():
    workflow = build_starter_workflow_dict(preset_name="z_image_turbo.2gpu.rdna4")
    preset = next(node for node in workflow["nodes"] if node["type"] == "xDiT.Preset")
    by_name = preset[NAMED_WIDGET_VALUES_KEY]
    assert by_name["gpu_count"] == "2"
    assert by_name["preset"] == "z_image_turbo.2gpu.rdna4"
    assert (
        build_starter_api_prompt(preset_name="z_image_turbo.2gpu.rdna4")["1"]["inputs"]["gpu_count"]
        == "2"
    )


def test_starter_api_prompt_loader_matches_requested_preset():
    """A loader carrying another preset's model makes Sample reject the warmed worker."""
    prompt = build_starter_api_prompt(preset_name="z_image_turbo.2gpu.rdna4")
    loader = prompt["2"]["inputs"]
    assert loader["model"] == "Tongyi-MAI/Z-Image-Turbo"
    assert loader["ulysses_degree"] == 2


def test_starter_api_prompt_samples_a_video_preset_as_video():
    """Node defaults are image-shaped, so a video preset must bring its own frame count."""
    sample = build_starter_api_prompt(preset_name="wan2_2_ti2v_5b.i2v.4gpu.rdna4")["3"]["inputs"]
    assert sample["num_frames"] == 121
    assert sample["height"] == 1280
    assert sample["width"] == 736


def test_starter_workflow_sample_wires_video_to_save_video():
    workflow = build_starter_workflow_dict()
    sample = next(node for node in workflow["nodes"] if node["type"] == "xDiT.Sample")
    assert sample["outputs"][1]["name"] == "video"
    assert sample["outputs"][1]["links"] == [6]
    save_video = next(node for node in workflow["nodes"] if node["type"] == "SaveVideo")
    video_input = next(entry for entry in save_video["inputs"] if entry["name"] == "video")
    assert video_input["link"] == 6


def test_starter_workflow_wires_image_preset_to_sample():
    workflow = build_starter_workflow_dict()
    links = workflow["links"]
    image_link = next(link for link in links if link[0] == 5)
    assert image_link[1] == 1
    assert image_link[2] == 1
    assert image_link[3] == 3
    assert image_link[5] == "IMAGE"
    preset = next(node for node in workflow["nodes"] if node["type"] == "xDiT.Preset")
    assert [output["name"] for output in preset["outputs"]] == [
        "model",
        "images",
        "sample",
    ]
    assert preset["outputs"][1]["links"] == [5]
    sample = next(node for node in workflow["nodes"] if node["type"] == "xDiT.Sample")
    images_input = next(entry for entry in sample["inputs"] if entry["name"] == "images")
    assert images_input["link"] == 5


def test_starter_workflow_sample_images_input_is_socket_only():
    workflow = build_starter_workflow_dict()
    sample = next(node for node in workflow["nodes"] if node["type"] == "xDiT.Sample")
    images_input = next(entry for entry in sample["inputs"] if entry["name"] == "images")
    assert images_input["type"] == "IMAGE"
    assert images_input.get("link") == 5
    assert "widget" not in images_input
    height_widget = next(entry for entry in sample["inputs"] if entry["name"] == "height")
    assert height_widget.get("widget") == {"name": "height"}


def test_starter_workflow_sample_widgets_include_seed_control():
    workflow = build_starter_workflow_dict()
    sample = next(node for node in workflow["nodes"] if node["type"] == "xDiT.Sample")
    values = sample[NAMED_WIDGET_VALUES_KEY]
    assert values["seed"] == 42
    assert values["control_after_generate"] == "randomize"
    assert values["timeout_seconds"] == 900


def test_starter_template_excludes_load_preset_image_node():
    workflow = build_starter_workflow_dict()
    node_types = {node["type"] for node in workflow["nodes"]}
    assert "XDiTLoadPresetImage" not in node_types
    assert node_types == STARTER_NODE_TYPES

    path = _EXAMPLE_WORKFLOWS_DIR / STARTER_FILENAME
    assert path.is_file()
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    disk_types = {node["type"] for node in on_disk["nodes"]}
    assert "XDiTLoadPresetImage" not in disk_types
    assert disk_types == STARTER_NODE_TYPES


def test_example_workflow_dir_has_starter_template():
    path = _EXAMPLE_WORKFLOWS_DIR / STARTER_FILENAME
    assert path.is_file(), f"missing starter template at {path}"


def test_only_the_canonical_starter_template_is_provided():
    assert {path.name for path in _EXAMPLE_WORKFLOWS_DIR.glob("*.json")} == TEMPLATE_FILENAMES


def test_template_module_name_matches_comfy_custom_node():
    assert _TEMPLATE_MODULE == "xdit_comfyui"
