import { app } from "../../scripts/app.js";

/**
 * Lay `node.widgets` out in the given order.
 *
 * Widgets a node adds for itself -- disclosure headings, a seed's control widget --
 * land wherever they were appended, which leaves a group's heading below the widgets
 * it collapses. Names the caller does not list keep their relative order at the end,
 * so an unknown widget is never dropped.
 */
export function applyWidgetOrder(node, orderedNames) {
  if (!node.widgets?.length || !orderedNames?.length) return false;

  const byName = new Map(node.widgets.map((widget) => [widget.name, widget]));
  const reordered = [];
  const seen = new Set();

  for (const name of orderedNames) {
    const widget = byName.get(name);
    if (!widget || seen.has(name)) continue;
    reordered.push(widget);
    seen.add(name);
  }

  for (const widget of node.widgets) {
    if (seen.has(widget.name)) continue;
    reordered.push(widget);
    seen.add(widget.name);
  }

  if (reordered.length !== node.widgets.length) return false;

  let changed = false;
  for (let index = 0; index < reordered.length; index += 1) {
    if (reordered[index] !== node.widgets[index]) {
      changed = true;
      break;
    }
  }
  if (!changed) return true;

  node.widgets = reordered;
  app.canvas?.setDirty?.(true, true);
  return true;
}
