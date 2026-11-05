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


def test_restore_by_name_tolerates_a_graph_saved_without_the_named_map():
    payload = _run_js(f"""
        import {{ restoreWidgetsByName }} from {json.dumps(HELPER.as_uri())};
        console.log(JSON.stringify([
            restoreWidgetsByName({{}}, {{ widgets_values: [1, 2, 3] }}, () => true),
            restoreWidgetsByName({{}}, null, () => true),
        ]));
        """)
    assert payload == [[], []]
