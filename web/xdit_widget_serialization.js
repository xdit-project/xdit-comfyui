/** Key under which xDiT nodes store their widget values by name. */
export const NAMED_VALUES_KEY = "xdit_widget_values";

/**
 * Save widget values by name instead of by position.
 *
 * These nodes reorder and group their widgets for display, so the position of a
 * value in litegraph's `widgets_values` array does not identify which widget it
 * belongs to. Names do.
 */
export function serializeWidgetsByName(node, data) {
  const values = {};
  for (const widget of node.widgets ?? []) {
    if (!widget?.name || widget.serialize === false) continue;
    values[widget.name] = widget.value;
  }
  data[NAMED_VALUES_KEY] = values;
  // Drop the positional array so there is only one source of truth on load.
  data.widgets_values = [];
  return values;
}

/** Restore values saved by `serializeWidgetsByName`. Returns the names applied. */
export function restoreWidgetsByName(node, data, applyValue) {
  const values = data?.[NAMED_VALUES_KEY];
  if (!values || typeof values !== "object" || Array.isArray(values)) return [];
  const applied = [];
  for (const [name, value] of Object.entries(values)) {
    if (value === undefined) continue;
    if (applyValue(name, value)) applied.push(name);
  }
  return applied;
}

/** Install the named-value save/restore contract once on a node instance. */
export function hookNamedWidgetSerialization(
  node,
  applyValue,
  { trackUserEdits = false } = {},
) {
  if (node._xditSerializationHooked) return;
  node._xditSerializationHooked = true;

  const previousSerialize = node.onSerialize;
  node.onSerialize = function (data) {
    previousSerialize?.call(this, data);
    serializeWidgetsByName(this, data);
  };

  const previousConfigure = node.onConfigure;
  node.onConfigure = function (data) {
    previousConfigure?.call(this, data);
    const restored = restoreWidgetsByName(this, data, (name, value) =>
      applyValue(this, name, value),
    );
    if (trackUserEdits) this._xditUserEdited = new Set(restored);
  };
}
