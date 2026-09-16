import json
import subprocess
from pathlib import Path

HELPER = Path(__file__).parents[2] / "web" / "xdit_widget_serialization.js"


def _run_js(script: str):
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_widgets_serialize_by_name_and_leave_the_positional_array_empty():
    payload = _run_js(f"""
        import {{ serializeWidgetsByName }} from {json.dumps(HELPER.as_uri())};
        const node = {{ widgets: [
            {{ name: "model", value: "flux" }},
            {{ name: "ulysses_degree", value: 2 }},
            {{ name: "Unload Model (Free VRAM)", value: null, serialize: false }},
        ] }};
        const data = {{ widgets_values: ["stale", "positional", "junk"] }};
        serializeWidgetsByName(node, data);
        console.log(JSON.stringify(data));
        """)
    assert payload["xdit_widget_values"] == {"model": "flux", "ulysses_degree": 2}
    assert payload["widgets_values"] == []


def test_restore_by_name_ignores_reordered_widgets():
    payload = _run_js(f"""
        import {{ restoreWidgetsByName }} from {json.dumps(HELPER.as_uri())};
        const node = {{ widgets: [{{ name: "ulysses_degree" }}, {{ name: "model" }}] }};
        const applied = [];
        const names = restoreWidgetsByName(
            node,
            {{ xdit_widget_values: {{ model: "flux", ring_degree: 4 }} }},
            (name, value) => {{
                if (name === "ring_degree") return false;
                applied.push([name, value]);
                return true;
            }},
        );
        console.log(JSON.stringify({{ names, applied }}));
        """)
    assert payload["names"] == ["model"]
    assert payload["applied"] == [["model", "flux"]]


def test_restore_by_name_tolerates_a_graph_saved_without_the_named_map():
    payload = _run_js(f"""
        import {{ restoreWidgetsByName }} from {json.dumps(HELPER.as_uri())};
        console.log(JSON.stringify([
            restoreWidgetsByName({{}}, {{ widgets_values: [1, 2, 3] }}, () => true),
            restoreWidgetsByName({{}}, null, () => true),
        ]));
        """)
    assert payload == [[], []]


def test_clear_vram_button_is_non_serializable():
    payload = _run_js(f"""
        import {{ markWidgetNonSerializable }} from {json.dumps(HELPER.as_uri())};
        const widget = markWidgetNonSerializable({{ options: {{ disabled: false }} }});
        console.log(JSON.stringify(widget));
        """)
    assert payload["serialize"] is False
    assert payload["options"]["serialize"] is False
