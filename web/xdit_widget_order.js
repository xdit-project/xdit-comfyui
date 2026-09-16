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

/**
 * Nodes 2.0 caches widget rows in `y` / `last_y`. Reordering the array alone leaves
 * widgets painted in their old rows, so newly revealed group members can appear above
 * their heading. Reassign the existing row coordinates in the requested order; using
 * the frontend's own slots preserves multiline and other non-uniform row spacing.
 */
export function reflowWidgetPositions(node, orderedNames) {
  if (!node?.widgets?.length || !orderedNames?.length) return false;
  const byName = new Map(node.widgets.map((widget) => [widget.name, widget]));
  const ordered = orderedNames
    .map((name) => byName.get(name))
    .filter((widget) => widget && Number.isFinite(widget.y));
  const slots = ordered
    .map((widget) => ({ y: widget.y, last_y: widget.last_y }))
    .sort((left, right) => left.y - right.y);
  if (ordered.length !== slots.length || !ordered.length) return false;
  for (let index = 0; index < ordered.length; index += 1) {
    ordered[index].y = slots[index].y;
    ordered[index].last_y = slots[index].last_y ?? slots[index].y;
  }
  app.canvas?.setDirty?.(true, true);
  return true;
}
