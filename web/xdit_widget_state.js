import { app } from "../../scripts/app.js";

/**
 * Widget access for the xDiT nodes.
 *
 * `widget.value` is the only source of truth. In the Vue node UI a widget's
 * `value` getter delegates to the same state object the `widgetValue` Pinia store
 * holds, so reading or writing the store separately aliased the same storage --
 * it never carried anything `widget.value` did not already have.
 */

export function litegraphWidget(node, name) {
  return node?.widgets?.find((widget) => widget.name === name) ?? null;
}

export function widgetValue(node, name, fallback = undefined) {
  const widget = litegraphWidget(node, name);
  if (!widget || widget.value === undefined) return fallback;
  return widget.value;
}

export function comboChoices(widget) {
  if (!widget?.options) return null;
  return (
    widget.options.values ??
    (Array.isArray(widget.options) ? widget.options : null)
  );
}

export function setComboChoices(node, name, choices) {
  const widget = litegraphWidget(node, name);
  if (!widget?.options || !choices?.length) return;
  if (Array.isArray(widget.options.values)) {
    widget.options.values = [...choices];
  } else if (Array.isArray(widget.options)) {
    widget.options.length = 0;
    widget.options.push(...choices);
  }
  app.canvas?.setDirty?.(true, true);
}

/**
 * Write a widget value.
 *
 * `notify` runs the widget's own callback, which is how litegraph and the Vue
 * widget layer both announce a change. Server-driven writes pass `notify: false`
 * so applying a preset does not look like a user edit and re-trigger a sync.
 */
export function setWidgetValue(
  node,
  name,
  value,
  { notify = false, comboGuard = true } = {},
) {
  const widget = litegraphWidget(node, name);
  if (!widget || value === undefined) return false;
  if (comboGuard) {
    const choices = widget.type === "combo" ? comboChoices(widget) : null;
    if (choices && !choices.includes(value)) return false;
  }
  if (widget.value === value) return false;
  widget.value = value;
  if (notify) widget.callback?.(value);
  app.canvas?.setDirty?.(true, true);
  return true;
}

export function setWidgetTooltip(node, name, tooltip) {
  const widget = litegraphWidget(node, name);
  if (!widget) return;
  if (widget.options) widget.options.tooltip = tooltip;
  else widget.options = { tooltip };
}

/**
 * Grey a widget out, keeping it visible so the option is still discoverable.
 *
 * `reason` replaces the tooltip while disabled and the widget's own tooltip comes
 * back when it is enabled again.
 */
export function setWidgetDisabled(node, name, disabled, reason = "") {
  const widget = litegraphWidget(node, name);
  if (!widget) return;

  widget.options ??= {};
  if (disabled && widget._xditEnabledTooltip === undefined) {
    widget._xditEnabledTooltip = widget.options.tooltip ?? "";
  }
  const tooltip = disabled
    ? reason || widget.options.tooltip
    : widget._xditEnabledTooltip;
  if (!disabled) delete widget._xditEnabledTooltip;
  if (tooltip !== undefined) widget.options.tooltip = tooltip;
  if (widget.disabled === disabled && widget.options.disabled === disabled)
    return;
  widget.disabled = disabled;
  widget.options.disabled = disabled;
}

/**
 * Hide or show a widget.
 *
 * `canvasOnly` widgets are the disclosure headings: the Vue panel would render
 * them as an empty row, so they are collapsed to zero height and drawn by the
 * DOM header instead.
 */
export function setWidgetHidden(
  node,
  widgetOrName,
  hidden,
  { canvasOnly = false } = {},
) {
  const widget =
    typeof widgetOrName === "string"
      ? litegraphWidget(node, widgetOrName)
      : widgetOrName;
  if (!widget) return;

  widget.options ??= {};
  if (hidden && canvasOnly) {
    widget.options.canvasOnly = true;
    widget.computeLayoutSize = () => ({
      minHeight: 0,
      maxHeight: 0,
      minWidth: 0,
    });
  }
  // Visibility is re-applied over every widget on each toggle, and in the Vue node UI
  // each write is a reactive one; only the widgets that actually changed should cost
  // a re-render.
  if (widget.hidden === hidden && widget.options.hidden === hidden) return;
  widget.hidden = hidden;
  widget.options.hidden = hidden;
}

/** Names the user has edited on this node, which server-driven defaults must not clobber. */
export function userEdited(node) {
  node._xditUserEdited ??= new Set();
  return node._xditUserEdited;
}

/** Nodes reached by `outputName`, i.e. everything downstream of this output. */
export function targetNodesFromOutput(node, outputName) {
  const output = node.outputs?.find((entry) => entry.name === outputName);
  if (!output?.links?.length) return [];
  const targets = [];
  for (const linkId of output.links) {
    const link = app.graph?.links?.[linkId];
    if (!link) continue;
    const target = app.graph?.getNodeById?.(link.target_id);
    if (target) targets.push(target);
  }
  return targets;
}

/** The node feeding `inputName`, or null when the input is unconnected. */
export function originNodeFromInput(node, inputName) {
  const input = node.inputs?.find((entry) => entry.name === inputName);
  if (!input?.link) return null;
  const link = app.graph?.links?.[input.link];
  if (!link) return null;
  return app.graph?.getNodeById?.(link.origin_id) ?? null;
}

/**
 * Discard the result of a superseded request.
 *
 * Preview fetches are issued per user action, so a slow earlier response must not
 * overwrite the widgets a later one already updated.
 */
export function requestSequence(node, key) {
  const field = `_xditSeq_${key}`;
  const ticket = (node[field] || 0) + 1;
  node[field] = ticket;
  return () => node[field] === ticket;
}
