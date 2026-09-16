"""Run the browser extensions against the real server payloads.

The frontend is where most of this pack's behaviour lives, and a renamed widget or a
deleted helper used to surface only as a broken node in the UI. The harness fakes just
enough of litegraph to run the extension lifecycle under node, driving it with the
payloads the server actually returns.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from xdit_comfyui.api import _preset_filter_schema, _preset_preview_payload, _preview_payload
from xdit_comfyui.runner_contract import loader_schema
from xdit_comfyui.runtime_config import (
    _generation_input_types,
    _preset_picker_input_types,
    _runtime_loader_input_types,
)

_REPO = Path(__file__).parents[2]
_HARNESS = Path(__file__).parents[1] / "web_lifecycle_harness.mjs"
_WEB = _REPO / "web"

pytestmark = [
    pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed"),
    pytest.mark.usefixtures("synthetic_preset_catalog"),
]


def test_served_extensions_register_unique_names():
    """ComfyUI imports every top-level JS file; duplicate names abort extension setup."""
    registrations: dict[str, list[str]] = {}
    pattern = re.compile(r"registerExtension\s*\(\s*\{.*?\bname:\s*[\"']([^\"']+)", re.S)
    for path in sorted(_WEB.glob("*.js")):
        for name in pattern.findall(path.read_text()):
            registrations.setdefault(name, []).append(path.name)
    duplicates = {name: files for name, files in registrations.items() if len(files) > 1}
    assert duplicates == {}


def test_sidebar_and_modern_submenu_register(harness):
    report, _ = harness
    assert report["sidebar_tabs"] == [{"id": "xdit", "title": "xDiT", "hasRender": True}]
    assert {"content": "xDiT", "submenuItems": ["Unload all models"]} in report["canvas_menus"]


def test_model_unload_action_is_kept_in_the_sidebar_card_header():
    source = (_WEB / "xdit_residency_sidebar.js").read_text(encoding="utf-8")
    assert 'button.textContent = t("xdit.residency.unload")' in source
    assert "heading.append(button)" in source
    assert "if (loader.warm || loader.parked)" in source


def test_english_localization_uses_comfyui_layout():
    locales = _REPO / "locales" / "en"
    node_defs = json.loads((locales / "nodeDefs.json").read_text(encoding="utf-8"))
    commands = json.loads((locales / "commands.json").read_text(encoding="utf-8"))
    json.loads((locales / "main.json").read_text(encoding="utf-8"))
    assert set(node_defs) == {"xDiT.Preset", "xDiT.Model", "xDiT.Sample"}
    assert node_defs["xDiT.Model"]["inputs"]["scm_policy"]["name"] == "SCM Policy"
    assert "xdit_unloadAllModels" in commands

    schemas = {
        "xDiT.Preset": _preset_picker_input_types(),
        "xDiT.Model": _runtime_loader_input_types(),
        "xDiT.Sample": _generation_input_types(),
    }
    for node_id, schema in schemas.items():
        localized = node_defs[node_id]
        assert localized.get("display_name")
        assert localized.get("description")
        for section in ("required", "optional"):
            for input_name, (_input_type, options) in schema.get(section, {}).items():
                entry = localized["inputs"].get(input_name)
                assert entry and entry.get("name"), f"missing locale name: {node_id}.{input_name}"
                if options.get("tooltip"):
                    assert entry.get("tooltip"), f"missing locale tooltip: {node_id}.{input_name}"


def _object_info():
    specs = {
        "xDiT.Preset": (_preset_picker_input_types(), ["model", "images", "sample"]),
        "xDiT.Model": (_runtime_loader_input_types(), ["model"]),
        "xDiT.Sample": (_generation_input_types(), ["images", "video"]),
    }
    info = {}
    for class_type, (inputs, outputs) in specs.items():
        info[class_type] = {
            "input": inputs,
            "output_name": outputs,
        }
    return info


_MULTI_TRANSFORMER_MODEL = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
# I2V-A14B implies its task; TI2V-5B is the one that asks the user to choose.
_TASK_MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


def _wan_dbcache_request():
    """Wan 2.2 with dbcache: two denoisers whose warmup schedules differ (4 and 2)."""
    return {
        "model": _MULTI_TRANSFORMER_MODEL,
        "cache_method": "dbcache",
        "gpu_count": 1,
        "gpu_device_ids": "0",
    }


def _run_harness(fixtures, node_ui):
    result = subprocess.run(
        ["node", str(_HARNESS), str(fixtures), str(_WEB), node_ui],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(f"harness crashed ({node_ui}):\n{result.stderr}")
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def server_payloads(tmp_path_factory):
    fixtures = tmp_path_factory.mktemp("xdit_web_fixtures")
    payloads = {
        "loader_schema": loader_schema(),
        "preset_schema": _preset_filter_schema(),
        "preset_preview": _preset_preview_payload(
            {"gpu_tag": "gfx1201", "gpu_count": 1, "preset": "flux.1gpu.rdna4"}
        ),
        "preset_preview_multi_gpu": _preset_preview_payload(
            {"gpu_tag": "gfx1201", "gpu_count": 4, "preset": "none"}
        ),
        "loader_preview": _preview_payload(
            {"preset_gpu_tag": "gfx1201", "preset_gpu_count": 1, "preset_choice": "flux.1gpu.rdna4"}
        ),
        # The user switching to a model with a different native resolution and step count.
        "loader_preview_switched": _preview_payload(
            {"model": "Qwen/Qwen-Image", "gpu_count": 1, "gpu_device_ids": "0"}
        ),
        "loader_preview_multi_transformer": _preview_payload(_wan_dbcache_request()),
        # A model that takes a task, which the image models around it do not have.
        "loader_preview_video": _preview_payload(
            {"model": _TASK_MODEL, "gpu_count": 1, "gpu_device_ids": "0"}
        ),
        "object_info": _object_info(),
    }
    for name, payload in payloads.items():
        (fixtures / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    return fixtures, payloads


@pytest.fixture(scope="module")
def harness(server_payloads):
    fixtures, payloads = server_payloads
    return _run_harness(fixtures, "vue"), payloads


@pytest.fixture(scope="module")
def canvas_harness(server_payloads):
    """The same lifecycle on the canvas UI, where the toggle is its own heading."""
    fixtures, _ = server_payloads
    return _run_harness(fixtures, "canvas")


def test_setting_up_the_nodes_logs_nothing(harness):
    report, _ = harness
    assert report["errors"] == []


def test_no_extension_installs_a_polling_interval(harness):
    """Propagation is event-driven; an interval means something reverted to polling."""
    report, _ = harness
    assert report["intervals"] == 0


def test_nodes2_gpu_count_change_refreshes_preset_choices(harness):
    report, payloads = harness
    assert report["after_gpu_count_change"]["gpu_count"] == "4"
    assert (
        report["after_gpu_count_change"]["preset_choices"]
        == payloads["preset_preview_multi_gpu"]["choices"]
    )


def test_the_model_node_leads_with_its_pinned_widgets(harness):
    report, payloads = harness
    pinned = payloads["loader_schema"]["pinned_widgets"]
    shown = report["after_setup"]["model_first_widgets"]
    assert "xdit_info" not in shown
    assert "Unload Model (Free VRAM)" not in shown
    assert pinned[0] in shown


def test_sample_has_no_inline_vram_information(harness):
    report, _ = harness
    setup = report["after_setup"]
    assert "xdit_info" not in setup["sample_first_widgets"]


def test_the_sample_node_leads_with_the_controls_used_every_run(harness):
    """Prompt, size, then sampling -- the rarely-touched knobs live under Advanced."""
    report, _ = harness
    order = report["sample_layout"]["order"]
    primary = [name for name in order if name in {"prompt", "width", "height", "seed"}]
    assert primary == ["prompt", "width", "height", "seed"]
    for buried in ("max_sequence_length", "timeout_seconds"):
        assert order.index(buried) > order.index("guidance_scale")


def test_a_sample_section_heading_sits_above_the_widgets_it_collapses(harness):
    """A heading below its own widgets collapses rows the user sees above the toggle."""
    report, _ = harness
    for group in report["sample_layout"]["groups"]:
        assert group["heading"] >= 0, group
        assert group["heading"] < group["first_widget"], group


def test_the_seed_keeps_its_control_widget_beside_it(canvas_harness):
    """Reordering must not strand control_after_generate at the bottom of the node."""
    order = canvas_harness["sample_layout"]["order"]
    if "control_after_generate" not in order:
        pytest.skip("this ComfyUI stub does not attach a seed control widget")
    assert order.index("control_after_generate") == order.index("seed") + 1


def test_collapsed_groups_hide_their_widgets(harness):
    report, payloads = harness
    cache = next(
        group for group in payloads["loader_schema"]["widget_groups"] if group["id"] == "cache"
    )
    assert cache["collapsed"] is True
    assert set(report["after_setup"]["hidden_cache_widgets"]) == set(cache["widgets"])


def test_picking_a_preset_reaches_the_sample_node(harness):
    report, payloads = harness
    assert report["preset_applied_to_sample"] is True
    expected = payloads["preset_preview"]["preset"]["generation_defaults"]["prompt"]
    assert report["sample_prompt"] == expected


def test_picking_a_preset_records_the_trigger_on_the_model_node(harness):
    """Persisted so a reloaded graph knows its values already came from this preset."""
    report, _ = harness
    expected = f"gfx1201:{report['preset_gpu_count']}:{report['chosen_preset']}"
    assert report["model_preset_trigger"] == expected


def test_one_preset_request_serves_the_preset_and_sample_nodes(harness):
    """Both read generation defaults out of the same response."""
    report, _ = harness
    preview_calls = [
        entry for entry in report["requests"] if entry["url"] == "/xdit/preset/preview"
    ]
    # Two for the connected node (setup + selection), plus setup + count change on
    # the isolated Preset node used by the GPU-count regression check.
    assert len(preview_calls) == 4


def test_graph_loading_does_not_release_inactive_workflow_workers(harness):
    report, _ = harness
    assert not [entry for entry in report["requests"] if entry["url"] == "/xdit/loader/reap"]


def test_the_model_hands_its_input_defaults_to_the_sample_node(harness):
    """The Sample definition carries one model's numbers; the selected model wins."""
    report, payloads = harness
    generation = payloads["loader_preview_switched"]["generation"]
    assert report["sample_model_constraints"]["defaults"] == generation["defaults"]
    assert report["sample_resolution_step"] == generation["resolution_step"]

    values = report["after_model_switch"]["sample_widget_values"]
    for name, expected in generation["defaults"].items():
        if name in values:
            assert values[name] == expected


def test_a_preset_outranks_the_models_own_defaults(harness):
    report, payloads = harness
    preset_defaults = payloads["preset_preview"]["preset"]["generation_defaults"]
    values = report["after_preset"]["sample_widget_values"]
    shared = [
        name
        for name in payloads["loader_preview"]["generation"]["defaults"]
        if name in preset_defaults and name in values
    ]
    assert shared, "the fixture no longer exercises an overlapping default"
    for name in shared:
        assert values[name] == preset_defaults[name]


@pytest.mark.parametrize("stage", ["after_setup", "after_expand"])
def test_each_group_keeps_its_widgets_under_its_own_heading(harness, stage):
    """Headings are created after the widgets they label, so ordering must follow."""
    report, _ = harness
    for group in report[stage]["group_layout"]:
        assert group["heading"] >= 0, f"{group['id']} lost its disclosure heading"
        assert (
            group["heading"] < group["first_widget"]
        ), f"{group['id']} renders its widgets above its heading ({stage})"


def test_expanding_a_group_reveals_its_widgets_below_the_heading(harness):
    report, _ = harness
    revealed = report["after_expand"]["visible_after_heading"]
    assert revealed["first_visible_group_widget"] > revealed["heading"]


def test_the_node_opens_at_its_collapsed_height_and_grows_when_expanded(harness):
    """Nothing recomputes the size on its own, so the node would open fully stretched."""
    report, _ = harness
    collapsed = report["after_setup"]["model_height"]
    expanded = report["after_expand"]["height"]
    assert expanded > collapsed, "expanding a group did not make room for its widgets"
    assert report["height_after_collapse"] == collapsed, "collapsing left the space behind"


def test_sample_advanced_widgets_stay_plainly_visible():

    required = _generation_input_types()["required"]
    assert "Advanced" not in required
    for name in ("resize_input_images", "max_sequence_length", "timeout_seconds"):
        assert required[name][1].get("advanced") is not True


def test_sample_owns_the_vae_decode_widgets():
    from xdit_comfyui.runner_contract import SAMPLE_VAE_DESTS

    required = _generation_input_types()["required"]
    assert "VAE" in required
    for name in SAMPLE_VAE_DESTS:
        assert name in required


def test_dom_widgets_reserve_nonzero_bounded_rows(harness):
    report, _ = harness
    for name, layout in report["dom_layout"].items():
        if name == "preset_previews" and layout["minHeight"] == 0:
            assert layout["maxHeight"] == 0
            continue
        assert layout["minHeight"] >= 32, name
        assert layout["maxHeight"] == layout["minHeight"], name


def test_a_two_transformer_model_can_be_tuned_per_denoiser(harness):
    """Wan 2.2 warms one denoiser for 4 steps and the other for 2; both must be editable."""
    report, payloads = harness
    rows = payloads["loader_preview_multi_transformer"]["cache_transformers"]
    warmups = [row["config"]["max_warmup_steps"] for row in rows]
    assert warmups == [4, 2], "the fixture model no longer has asymmetric cache presets"

    groups = report["after_multi_transformer"]["denoiser_groups"]
    assert len(groups) == len(rows) - 1, "one group per denoiser after the first"
    for group in groups:
        assert not group["heading_hidden"], f"{group['id']} stayed hidden for a model with two"
        assert group["warmup"] == warmups[1]
    assert report["after_multi_transformer"]["max_warmup_steps_widget"] == warmups[0]


def test_leaving_a_video_model_takes_its_task_with_it(harness):
    """`task` only exists for video models, so an image preset must not inherit one."""
    report, _ = harness
    before = report["task_on_video_model"]
    assert before["value"], "the fixture video model no longer offers a task"
    assert not before["hidden"]

    after = report["task_after_leaving_video_model"]
    assert after["value"] == "", "the image preset kept the video model's task"
    assert after["hidden"], "task stayed on a model that has no tasks"


def test_the_editable_cache_widget_holds_the_models_own_value(harness):
    """Left at the global DBCachePreset default it would erase the asymmetry on queue."""
    report, payloads = harness
    baseline = payloads["loader_preview_multi_transformer"]["cache_defaults"]
    assert report["after_multi_transformer"]["max_warmup_steps_widget"] == (
        baseline["max_warmup_steps"]
    )


def test_no_widget_is_left_holding_null(harness):
    """ComfyUI refuses to convert null to the declared type, so the queue rejects the graph."""
    report, _ = harness
    assert report["null_valued_widgets"] == []


def test_the_second_denoiser_group_is_reachable_on_the_canvas_ui_too(canvas_harness):
    """Its heading is the toggle there, which nothing reveals once it has been hidden."""
    groups = canvas_harness["after_multi_transformer"]["denoiser_groups"]
    assert groups
    for group in groups:
        assert not group["heading_hidden"], f"{group['id']} has no heading to click"
    assert canvas_harness["errors"] == []


def test_switching_back_to_one_transformer_hides_the_extra_group(harness):
    report, _ = harness
    groups = report["denoiser_groups_after_switch_back"]
    assert groups, "the schema no longer declares a second denoiser group"
    for group in groups:
        assert group["heading_hidden"], f"{group['id']} is offered for a single-denoiser model"
        assert group["widgets_hidden"]


def test_options_the_model_cannot_use_are_greyed_out(harness):
    """Editing them would only be undone by the queue sanitizer on the way to the worker."""
    report, payloads = harness
    gates = payloads["loader_preview_switched"]["widget_gates"]
    unsupported = {name for name, allowed in gates.items() if not allowed}
    assert "use_cfg_parallel" in unsupported, "the fixture model now supports everything"

    disabled = set(report["after_model_switch"]["disabled_widgets"])
    assert disabled, "no option was greyed out for a model that cannot use them all"
    assert (
        disabled <= unsupported
    ), f"greyed out an option the model supports: {disabled - unsupported}"
    assert "use_cfg_parallel" in disabled
    assert report["after_model_switch"]["cfg_parallel_after_rejected_edit"] is False


def test_an_option_greyed_out_for_one_model_comes_back_for_another(harness):
    report, payloads = harness
    gates = payloads["loader_preview_multi_transformer"]["widget_gates"]
    assert gates["use_cfg_parallel"], "the fixture model no longer supports CFG parallel"
    assert "use_cfg_parallel" not in report["after_multi_transformer"]["disabled_widgets"]


def test_sample_only_offers_the_frame_count_to_a_model_that_makes_video(harness):
    report, _ = harness
    image_model = report["after_model_switch"]["sample_video_group"]
    video_model = report["after_multi_transformer"]["sample_video_group"]
    assert image_model["heading_hidden"], "a still-image model still offers video settings"
    assert image_model["num_frames"] == 1, "a stale frame count outlived the model that used it"
    assert not video_model["heading_hidden"], "a video model cannot reach its frame count"


def test_a_user_edit_survives_the_following_server_refresh(harness):
    report, _ = harness
    assert report["user_edit_preserved"] == 7


def test_saving_writes_names_and_leaves_the_positional_array_empty(harness):
    report, _ = harness
    serialized = report["serialized"]
    assert serialized["model_positional"] == []
    assert serialized["model_named_count"] > 40
    assert serialized["sample_named_count"] > 10
    assert serialized["preset_named"]["preset"] == "flux.1gpu.rdna4"


def test_loading_restores_a_value_by_name(harness):
    report, _ = harness
    assert report["reloaded_vsa_top_k"] == 7
