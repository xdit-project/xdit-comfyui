import { t } from "./xdit_i18n.js";

export function usesModernNodeWidgets() {
  return !!globalThis.LiteGraph?.vueNodesMode;
}

function expandWidgetName(group) {
  return group.expand_widget || group.label;
}

export function refreshVueWidgetSnapshot(node) {
  if (!usesModernNodeWidgets()) return;
  const widgets = node?.widgets;
  if (!Array.isArray(widgets) || !widgets.length) return;
  widgets.push(widgets[widgets.length - 1]);
  widgets.pop();
}

function setGroupExpanded(node, widget, expanded) {
  widget.value = expanded;
  if (widget._xditExpandProperty) {
    node.properties ??= {};
    node.properties[widget._xditExpandProperty] = expanded;
  }
}

function isGroupExpanded(widget, group) {
  return widget.value ?? !group?.collapsed;
}

export function groupExpanded(node, group) {
  const widget = node.widgets?.find(
    (entry) => entry.name === expandWidgetName(group),
  );
  return widget ? !!isGroupExpanded(widget, group) : !group?.collapsed;
}

export function expandGroup(node, group, expanded) {
  const widget = node.widgets?.find(
    (entry) => entry.name === expandWidgetName(group),
  );
  if (!widget) return;
  setGroupExpanded(node, widget, !!expanded);
}

export function installDisclosureGroups(node, groups, { onToggle } = {}) {
  for (const group of groups || []) {
    const name = expandWidgetName(group);
    const widget = node.widgets?.find((entry) => entry.name === name);
    if (!widget || widget._xditDisclosureHeading) continue;
    widget._xditDisclosureHeading = true;
    widget.label = group.label;
    widget.options ??= {};
    widget.options.tooltip =
      group.description || t("xdit.tooltip.disclosure", { group: group.label });
    const original = widget.callback;
    widget.callback = function (value, ...args) {
      const result = original?.apply(this, [value, ...args]);
      node.properties ??= {};
      node.properties[`_xdit_expand_${group.id ?? name}`] = !!value;
      onToggle?.(node, group);
      return result;
    };
  }
  refreshVueWidgetSnapshot(node);
}
