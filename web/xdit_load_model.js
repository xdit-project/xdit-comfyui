import { app } from "../../scripts/app.js";
import {
  groupExpanded,
  installDisclosureGroups,
  refreshVueWidgetSnapshot,
  usesModernNodeWidgets,
} from "./xdit_disclosure.js";
import {
  restoreWidgetsByName,
  serializeWidgetsByName,
} from "./xdit_widget_serialization.js";
import { applyWidgetOrder } from "./xdit_widget_order.js";
import { t } from "./xdit_i18n.js";
import {
  describeDenoiserGroups,
  isDenoiserGroup,
} from "./xdit_cache_transformers.js";
import { connectedPresetNode, presetSelection } from "./xdit_preset_bus.js";
import {
  comboChoices,
  litegraphWidget,
  requestSequence,
  setComboChoices,
  setWidgetDisabled,
  setWidgetHidden as setWidgetHiddenState,
  setWidgetTooltip,
  setWidgetValue,
  targetNodesFromOutput,
  userEdited,
  widgetValue,
} from "./xdit_widget_state.js";
import { notifyResidencyChange } from "./xdit_residency_info.js";

const NODE_NAME = "xDiT.Model";
const MIN_NODE_WIDTH = 380;

// The widget lists all come from /xdit/loader/schema, which derives them from the xdit
// CLI. Nothing here restates them, so a new runner arg needs no change in this file.
const PINNED_WIDGETS = [];
const COMBO_WIDGETS = new Set();

const TRIGGER_WIDGETS = ["model", "task", "gpu_device_ids"];
const TOP_LEVEL_WIDGETS = ["model", "task", "residency", "use_torch_compile"];

let configWidgets = null;
let payloadWidgets = null;
let widgetGroups = null;

async function ensureLoaderSchema() {
  if (configWidgets && widgetGroups) {
    return { config_widgets: configWidgets, widget_groups: widgetGroups };
  }
  const response = await fetch("/xdit/loader/schema");
  if (!response.ok) {
    throw new Error(`/xdit/loader/schema returned ${response.status}`);
  }
  const schema = await response.json();
  configWidgets = schema.config_widgets ?? [];
  widgetGroups = schema.widget_groups ?? [];
  PINNED_WIDGETS.splice(
    0,
    PINNED_WIDGETS.length,
    ...(schema.pinned_widgets ?? []),
  );
  COMBO_WIDGETS.clear();
  for (const name of schema.combo_widgets ?? []) COMBO_WIDGETS.add(name);
  ensureLoaderSchema._widgetDefaults = schema.widget_defaults ?? {};
  ensureLoaderSchema._comboOptions = schema.combo_options ?? {};
  ensureLoaderSchema._widgetConstraints = schema.widget_constraints ?? {};
  payloadWidgets = [...TRIGGER_WIDGETS, ...configWidgets];
  return { config_widgets: configWidgets, widget_groups: widgetGroups };
}

function getConfigWidgets() {
  return configWidgets || [];
}

function getPayloadWidgets() {
  return payloadWidgets || [...TRIGGER_WIDGETS, ...getConfigWidgets()];
}

/**
 * Write a widget value on behalf of the server.
 *
 * The widget's own callback is deliberately not run: a preset or capability update
 * is not a user edit, and firing it would schedule another sync.
 */
function setNodeWidgetValue(node, name, value) {
  return setWidgetValue(node, name, value, {
    comboGuard: COMBO_WIDGETS.has(name),
  });
}

function setWidgetHidden(node, widgetOrName, hidden) {
  const name =
    typeof widgetOrName === "string" ? widgetOrName : widgetOrName?.name;
  setWidgetHiddenState(node, widgetOrName, hidden, {
    canvasOnly: usesModernNodeWidgets() && expandWidgetNames(node).has(name),
  });
}

function ensureLoaderMinWidth(node) {
  const width = Array.isArray(node.size) ? node.size[0] : 0;
  const height = Array.isArray(node.size) ? node.size[1] : 0;
  if (width >= MIN_NODE_WIDTH) return;
  node.setSize([MIN_NODE_WIDTH, height]);
}

function hookLoaderSerialization(node) {
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
      setWidgetValue(this, name, value, { comboGuard: false }),
    );
    // A restored value is the user's, so a later capability refresh must not
    // replace it with the model's default.
    this._xditUserEdited = new Set(restored);
  };
}

function loaderWidgetGroups() {
  return widgetGroups || [];
}

/** What the selected model supports, keyed by widget name and by group label. */
function widgetGates(node) {
  return node?._xditWidgetGates ?? null;
}

/** Groups this model has nothing to fill, such as a second denoiser it does not have. */
function inactiveGroups(node) {
  const gates = widgetGates(node);
  if (!gates) {
    // Until a preview says which denoisers the model caches, assume the common case
    // of one; otherwise the node opens showing a group it may have nothing for.
    return new Set(
      loaderWidgetGroups()
        .filter(isDenoiserGroup)
        .map((group) => group.id),
    );
  }
  const inactive = new Set();
  for (const group of loaderWidgetGroups()) {
    if (gates[group.label] === false) inactive.add(group.id);
  }
  return inactive;
}

function desiredWidgetOrder() {
  const order = [...TOP_LEVEL_WIDGETS];
  for (const group of loaderWidgetGroups()) {
    // The clickable header comes first: it is a separate DOM widget in the Vue node
    // UI, and reordering without it leaves a group's widgets above their own heading.
    order.push(expandWidgetName(group));
    order.push(...(group.widgets || []));
  }
  return order;
}

function reorderLoaderWidgets(node) {
  return applyWidgetOrder(node, desiredWidgetOrder());
}

function expandWidgetName(group) {
  return group.expand_widget || group.label;
}

function findWidgetGroup(id) {
  return (widgetGroups || []).find((group) => group.id === id) || null;
}

function hookExpandWidgets(node) {
  installDisclosureGroups(node, widgetGroups, {
    setWidgetHidden,
    onToggle: () => applyConditionalVisibility(node),
  });
}

function setupWidgetGroups(node) {
  if (!node.widgets?.length) return false;
  ensureLoaderMinWidth(node);
  // Headings are created here, appended after every widget they label, so the order
  // pass has to come afterwards.
  hookExpandWidgets(node);
  if (!reorderLoaderWidgets(node)) return false;
  sanitizeLoaderWidgets(node);
  applyConditionalVisibility(node);
  node._xditGroupsReady = true;
  return true;
}

function sanitizePinnedWidgets(node) {
  const modelWidget = litegraphWidget(node, "model");
  const modelChoices = comboChoices(modelWidget);
  const model = widgetValue(node, "model");
  const invalidModel =
    model == null || typeof model !== "string" || !String(model).trim();
  if (invalidModel || (modelChoices && !modelChoices.includes(model))) {
    const fallback = modelChoices?.includes("black-forest-labs/FLUX.1-dev")
      ? "black-forest-labs/FLUX.1-dev"
      : modelChoices?.[0];
    if (fallback !== undefined) {
      setNodeWidgetValue(node, "model", fallback);
    }
  }
}

function sanitizeConfigWidgets(node) {
  const defaults = ensureLoaderSchema._widgetDefaults;
  const constraints = ensureLoaderSchema._widgetConstraints;
  if (!defaults) return;

  for (const name of getConfigWidgets()) {
    const widget = litegraphWidget(node, name);
    if (!widget) continue;

    const raw = widgetValue(node, name);
    const spec = constraints?.[name];
    if (spec?.type === "int") {
      const min = Number.isFinite(Number(spec.min)) ? Number(spec.min) : 0;
      const fallback = spec.default ?? defaults[name] ?? min;
      const parsed = Number(raw);
      if (
        raw === "" ||
        raw == null ||
        !Number.isFinite(parsed) ||
        parsed < min
      ) {
        setNodeWidgetValue(node, name, fallback);
      }
      continue;
    }

    if (COMBO_WIDGETS.has(name)) {
      const choices = comboChoices(widget);
      if (choices && raw != null && !choices.includes(raw)) {
        const fallback = defaults[name];
        if (fallback !== undefined && choices.includes(fallback)) {
          setNodeWidgetValue(node, name, fallback);
        }
      }
      continue;
    }

    // A null reaches the prompt as JSON null, which ComfyUI refuses to convert to the
    // declared type, so the whole graph fails to validate.
    if (raw == null && defaults[name] !== undefined) {
      setNodeWidgetValue(node, name, defaults[name]);
      continue;
    }

    if (raw === "" && defaults[name] !== undefined && defaults[name] !== "") {
      setNodeWidgetValue(node, name, defaults[name]);
    }
  }
}

function sanitizeLoaderWidgets(node) {
  sanitizePinnedWidgets(node);
  sanitizeConfigWidgets(node);
}

function connectedPresetTrigger(node) {
  const presetNode = connectedPresetNode(node);
  return presetNode ? presetSelection(presetNode).trigger : null;
}

function persistPresetTrigger(node, trigger) {
  node.properties ??= {};
  if (trigger == null) {
    delete node.properties._xdit_model_preset_trigger;
  } else {
    node.properties._xdit_model_preset_trigger = trigger;
  }
}

function stepCacheWidgetNames() {
  const names = [];
  for (const group of widgetGroups || []) {
    if (group.id !== "cache" && !isDenoiserGroup(group)) continue;
    names.push(...(group.widgets || []));
  }
  if (!names.includes("cache_method")) {
    names.unshift("cache_method");
  }
  return names;
}

/**
 * Fill the cache widgets with the model's own values.
 *
 * These come from `cache_defaults`, which the server derives from the model alone.
 * `display_widgets` cannot serve here: a widget value that differs from the model's is
 * an override, so the response would echo the widget back and the model's numbers would
 * never appear.
 */
function applyStepCacheDisplayWidgets(node, widgets) {
  if (!widgets || !Object.keys(widgets).length) return;
  const edited = userEdited(node);
  for (const name of stepCacheWidgetNames()) {
    if (!(name in widgets) || edited.has(name)) continue;
    setNodeWidgetValue(node, name, widgets[name]);
  }
}

function shouldHydrateStepCacheWidgets(
  node,
  { modelChanged, presetChanged, cacheMethodChanged },
) {
  const method = String(
    widgetValue(node, "cache_method", "none") || "none",
  ).toLowerCase();
  if (method === "none") return false;
  return (
    presetChanged ||
    modelChanged ||
    cacheMethodChanged ||
    !node._xditStepCacheHydrated
  );
}

/**
 * Apply the defaults a model implies, leaving anything the user typed alone.
 *
 * Picking a preset is an explicit request for its values, so that path forces the
 * write; a model change only fills in what the user has not touched.
 */
function applyDisplayWidgets(node, widgets, { force = false } = {}) {
  if (!widgets || !Object.keys(widgets).length) return;
  const edited = userEdited(node);
  for (const [name, value] of Object.entries(widgets)) {
    if (!force && edited.has(name)) continue;
    setNodeWidgetValue(node, name, value);
  }
  sanitizeLoaderWidgets(node);
}

function collectPreviewPayload(node, { presetApplied = false } = {}) {
  const payload = {};
  for (const name of getPayloadWidgets()) {
    payload[name] = widgetValue(node, name);
  }
  payload.custom_model_id = widgetValue(node, "custom_model_id", "");
  const presetNode = connectedPresetNode(node);
  if (presetNode) {
    const selection = presetSelection(presetNode);
    payload.preset_gpu_tag = selection.gpuTag;
    payload.preset_gpu_count = selection.gpuCount;
    payload.preset_choice = selection.preset;
    // The widgets still hold the outgoing model, so the answer has to be about
    // the preset's model instead: capabilities drive which fields survive.
    payload.preset_applied = presetApplied;
  }
  return payload;
}

async function fetchPreview(node, options) {
  const response = await fetch("/xdit/loader/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(collectPreviewPayload(node, options)),
  });
  if (!response.ok) throw new Error(`preview failed (${response.status})`);
  return await response.json();
}

function updateComboChoices(node, name, choices, fallback) {
  if (!choices?.length) return;
  setComboChoices(node, name, choices);
  const current = widgetValue(node, name);
  if (choices.includes(current)) return;
  setNodeWidgetValue(node, name, fallback ?? choices[0]);
}

function allLoaderWidgetNames() {
  const names = new Set([...PINNED_WIDGETS, ...getConfigWidgets()]);
  for (const group of loaderWidgetGroups()) {
    names.add(expandWidgetName(group));
    for (const name of group.widgets || []) {
      names.add(name);
    }
  }
  return [...names];
}

function expandWidgetNames() {
  return new Set(loaderWidgetGroups().map((group) => expandWidgetName(group)));
}

function revealAllLoaderWidgets(node) {
  const expandNames = expandWidgetNames();
  for (const name of allLoaderWidgetNames()) {
    if (expandNames.has(name)) continue;
    setWidgetHidden(node, name, false);
  }
}

/**
 * Narrow the combos to what the selected model and this hardware support.
 *
 * The per-model lists come from `/xdit/loader/preview` (ROCm has no int8 GEMM, for
 * instance); the schema's full list is only the starting point before the first
 * preview arrives.
 */
function refreshLoaderWidgetChoices(node, payload) {
  const combos = ensureLoaderSchema._comboOptions || {};
  updateComboChoices(
    node,
    "gemm_precision",
    payload?.gemm_precision_choices ?? combos.gemm_precision,
    "native",
  );
  updateComboChoices(
    node,
    "cache_method",
    payload?.cache_method_choices ?? combos.cache_method,
    "none",
  );
  updateComboChoices(
    node,
    "residency",
    payload?.residency_choices ?? combos.residency,
    "keep_gpu",
  );
  const residencyReason = payload?.residency_unavailable_reason;
  setWidgetTooltip(
    node,
    "residency",
    residencyReason
      ? t("xdit.tooltip.residencyUnavailable", { reason: residencyReason })
      : t("xdit.tooltip.residency"),
  );
}

function applyDistilledWeightsSupport(node, _payload) {
  const group = findWidgetGroup("distilled_weights");
  const widgetNames = group?.widgets || [
    "distilled_transformer_path",
    "distilled_transformer_2_path",
  ];
  const headingName = group ? expandWidgetName(group) : "DISTILLED WEIGHTS";
  const tooltip = group?.description || t("xdit.tooltip.distilledWeights");

  setWidgetTooltip(node, headingName, tooltip);
  for (const name of widgetNames) {
    setWidgetTooltip(node, name, tooltip);
  }
}

function configureTaskWidget(node, payload) {
  const widget = litegraphWidget(node, "task");
  if (!widget) return;
  widget._xditOriginalType ??= widget.type;
  setWidgetHidden(node, widget, false);

  const validTasks = payload?.capabilities?.valid_tasks;
  // Single-pipeline models (FLUX, FLUX.2, Z-Image) reject any task, so there is
  // nothing to choose: hide the field instead of showing a blank one. Unknown
  // capabilities leave it visible.
  node._xditTaskUnsupported = Array.isArray(validTasks) && !validTasks.length;
  if (!Array.isArray(validTasks)) return;

  const current = widgetValue(node, "task");
  if (node._xditTaskUnsupported) {
    widget.type = widget._xditOriginalType ?? "STRING";
    delete widget.options?.values;
    if (current !== "") setNodeWidgetValue(node, "task", "");
    setWidgetHidden(node, widget, true);
    return;
  }

  if (validTasks.length > 1) {
    widget.type = "combo";
    widget.options ??= {};
    widget.options.values = [...validTasks];
    if (!validTasks.includes(current)) {
      setNodeWidgetValue(node, "task", validTasks[0]);
    }
    return;
  }

  widget.type = widget._xditOriginalType ?? "STRING";
  // Drop stale combo choices so the value below is not rejected as out-of-list.
  delete widget.options?.values;
  setNodeWidgetValue(node, "task", validTasks[0] ?? "");
}

/**
 * Shrink the node to the widgets it is actually showing.
 *
 * `computeSize()` skips hidden widgets, but nothing recomputes the size on its own:
 * the node is built with every group's widgets visible, so without this it opens at
 * the height of all ~50 of them and collapsing a group leaves the empty space behind.
 */
function fitLoaderHeight(node) {
  const computed = node.computeSize?.();
  if (!computed) return;
  const width = Math.max(node.size?.[0] ?? 0, MIN_NODE_WIDTH);
  if (node.size?.[1] === computed[1] && node.size?.[0] === width) return;
  node.setSize([width, computed[1]]);
}

function applyConditionalVisibility(node) {
  revealAllLoaderWidgets(node);
  if (node._xditTaskUnsupported) setWidgetHidden(node, "task", true);
  const inactive = inactiveGroups(node);
  for (const group of loaderWidgetGroups()) {
    const dropped = inactive.has(group.id);
    for (const name of group.widgets || []) {
      setWidgetHidden(node, name, dropped || !groupExpanded(node, group));
    }
    // A group this model has nothing for loses its heading too, and gets it back when
    // a model that needs it is selected. Both headings are set: the canvas UI clicks
    // the toggle, the Vue UI clicks the disclosure widget that replaces it.
    setWidgetHidden(node, expandWidgetName(group), dropped);
  }
  if (widgetValue(node, "hf_cache_mode") !== "custom_path") {
    setWidgetHidden(node, "hf_cache_dir", true);
  }
  if (node._xditTaskUnsupported) setWidgetHidden(node, "task", true);
  applyCapabilityGates(node);
  refreshVueWidgetSnapshot(node);
  fitLoaderHeight(node);
  app.canvas?.setDirty?.(true, true);
}

/**
 * Grey out the options the selected model cannot use.
 *
 * The queue sanitizer resets them on the way to the worker, so leaving them editable
 * only invites edits that silently vanish; the value is snapped to the one the run
 * will actually use, so the widget shows the truth rather than a wish.
 */
function applyCapabilityGates(node) {
  const gates = widgetGates(node);
  if (!gates) return;
  const defaults = ensureLoaderSchema._widgetDefaults || {};
  const model = node._xditPreviewPayload?.runtime?.model;
  const reason = model ? `${model} does not support this option.` : "";
  for (const name of getConfigWidgets()) {
    const unsupported = gates[name] === false;
    setWidgetDisabled(node, name, unsupported, reason);
    if (!unsupported || defaults[name] === undefined) continue;
    if (widgetValue(node, name) !== defaults[name]) {
      setNodeWidgetValue(node, name, defaults[name]);
    }
  }
}

function refreshLoaderUi(node, payload) {
  node._xditPreviewPayload = payload;
  node._xditWidgetGates = payload?.widget_gates ?? null;
  refreshLoaderWidgetChoices(node, payload);
  applyDistilledWeightsSupport(node, payload);
  applyDenoiserCacheGroups(node, payload);
  configureTaskWidget(node, payload);
  applyConditionalVisibility(node);
}

function applyDenoiserCacheGroups(node, payload) {
  describeDenoiserGroups(
    node,
    loaderWidgetGroups(),
    payload?.cache_transformers,
    {
      setWidgetTooltip,
      headingName: expandWidgetName,
    },
  );
}

/**
 * Hand the selected model's input defaults to the Sample nodes it feeds.
 *
 * The Sample node's own definition can only carry one model's numbers, so the model
 * that is actually selected pushes its own (Qwen-Image wants 928x1664/50 steps where
 * FLUX wants 1024x1024/50).
 */
function pushGenerationConstraints(
  node,
  generation,
  { modelChanged = false } = {},
) {
  if (!generation) return;
  for (const target of targetNodesFromOutput(node, "model")) {
    target._xditApplyModelConstraints?.(generation, { modelChanged });
  }
}

function modelIdentity(node) {
  return `${widgetValue(node, "model", "")}\0${widgetValue(node, "custom_model_id", "")}`;
}

/**
 * Refresh this node from the server: per-model choices, capabilities, and the
 * defaults that a new model or a newly picked preset implies.
 *
 * `presetPicked` marks the run as "the user asked for this preset", which is the
 * only case allowed to overwrite widgets the user has edited.
 */
async function syncLoader(node, { presetPicked = false } = {}) {
  if (!node.widgets?.length || app.configuringGraph) return;

  const isCurrent = requestSequence(node, "loaderPreview");
  const identity = modelIdentity(node);
  const trigger = connectedPresetTrigger(node);
  const modelChanged =
    node._xditLastModelIdentity != null &&
    identity !== node._xditLastModelIdentity;
  const cacheMethod = String(
    widgetValue(node, "cache_method", "none") || "none",
  );
  const cacheMethodChanged =
    node._xditLastCacheMethod != null &&
    cacheMethod !== node._xditLastCacheMethod;
  const presetChanged =
    presetPicked ||
    (trigger != null &&
      (trigger !== node._xditLastPresetTrigger ||
        !node._xditLastPresetApplied));

  let payload;
  try {
    payload = await fetchPreview(node, { presetApplied: presetChanged });
  } catch (error) {
    console.warn("[xdit] loader preview failed", error);
    return;
  }
  if (!isCurrent()) return;

  try {
    if (presetChanged && trigger != null) {
      applyDisplayWidgets(node, payload.preset_widgets, { force: true });
      userEdited(node).clear();
      node._xditLastPresetApplied = true;
      node._xditStepCacheHydrated = true;
      persistPresetTrigger(node, trigger);
    } else {
      if (modelChanged) {
        node._xditStepCacheHydrated = false;
        applyDisplayWidgets(node, payload.display_widgets);
      }
      if (
        shouldHydrateStepCacheWidgets(node, {
          modelChanged,
          presetChanged,
          cacheMethodChanged,
        })
      ) {
        applyStepCacheDisplayWidgets(node, payload.cache_defaults);
        node._xditStepCacheHydrated = true;
      }
    }
    sanitizeLoaderWidgets(node);
    refreshLoaderUi(node, payload);
    // A preset-driven model switch keeps the preset's numbers; only a model the user
    // picked replaces them.
    pushGenerationConstraints(node, payload.generation, {
      modelChanged: modelChanged && !presetChanged,
    });
  } catch (error) {
    console.error("[xdit] loader sync failed", error);
  } finally {
    node._xditLastModelIdentity = modelIdentity(node);
    node._xditLastCacheMethod = String(
      widgetValue(node, "cache_method", "none") || "none",
    );
    node._xditLastPresetTrigger = connectedPresetTrigger(node);
    if (node._xditLastPresetTrigger == null) {
      node._xditLastPresetApplied = false;
      persistPresetTrigger(node, null);
    }
  }
}

/**
 * Coalesce a burst of edits into one request.
 *
 * Text widgets report every keystroke, so this waits for typing to stop; the
 * sequence guard in `syncLoader` covers responses that still arrive out of order.
 */
function scheduleSync(node, options) {
  clearTimeout(node._xditSyncTimer);
  node._xditSyncTimer = setTimeout(() => syncLoader(node, options), 150);
}

function hookTriggerWidgets(node) {
  for (const name of TRIGGER_WIDGETS) {
    const widget = litegraphWidget(node, name);
    if (!widget || widget._xditHooked) continue;
    widget._xditHooked = true;
    const original = widget.callback;
    widget.callback = function (value, ...args) {
      const result = original?.apply(this, [value, ...args]);
      userEdited(node).add(name);
      scheduleSync(node);
      return result;
    };
  }
}

function rejectUnsupportedEdit(node, name) {
  if (widgetGates(node)?.[name] !== false) return false;
  const fallback = ensureLoaderSchema._widgetDefaults?.[name];
  if (fallback !== undefined) setNodeWidgetValue(node, name, fallback);
  applyCapabilityGates(node);
  return true;
}

function hookConfigWidgetCallbacks(node) {
  for (const name of getConfigWidgets()) {
    const widget = litegraphWidget(node, name);
    if (!widget || widget._xditConfigHooked) continue;
    widget._xditConfigHooked = true;
    const original = widget.callback;
    widget.callback = function (value, ...args) {
      const result = original?.apply(this, [value, ...args]);
      if (rejectUnsupportedEdit(node, name)) return result;
      userEdited(node).add(name);
      applyConditionalVisibility(node);
      if (name === "cache_method") scheduleSync(node);
      return result;
    };
  }
}

/** Nodes 2.0 may deliver edits through the node lifecycle without invoking the
 * LiteGraph callback. Enforce capability gates on that path as well. */
function hookNodeWidgetChanges(node) {
  if (node._xditLoaderWidgetChangesHooked) return;
  node._xditLoaderWidgetChangesHooked = true;
  const previous = node.onWidgetChanged;
  node.onWidgetChanged = function (name, value, oldValue, widget) {
    const result = previous?.call(this, name, value, oldValue, widget);
    if (rejectUnsupportedEdit(this, name)) return result;
    if (TRIGGER_WIDGETS.includes(name)) {
      userEdited(this).add(name);
      scheduleSync(this);
    } else if (getConfigWidgets().includes(name)) {
      userEdited(this).add(name);
      applyConditionalVisibility(this);
      if (name === "cache_method") scheduleSync(this);
    }
    return result;
  };
}

function showXditToast(severity, detail, life = 4000) {
  const toast =
    app.extensionManager?.toast?.add ?? app.extensionManager?.toasts?.add;
  toast?.({
    severity,
    summary: "xDiT",
    detail,
    life,
  });
}

export async function releaseAllLoaders() {
  showXditToast(
    "info",
    "Stopping all xDiT workers and releasing GPU memory…",
    12000,
  );
  try {
    const response = await fetch("/xdit/loader/reap", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ all: true }),
    });
    const payload = await response.json();
    const released = payload?.released ?? [];
    showXditToast(
      released.length ? "success" : "info",
      released.length
        ? t("xdit.notice.released", { nodes: released.join(", ") })
        : t("xdit.notice.noneToRelease"),
      5000,
    );
    notifyResidencyChange();
  } catch (error) {
    console.warn("[xdit] releasing all models failed", error);
    showXditToast("error", t("xdit.error.releaseFailed"), 6000);
  }
}

/**
 * Every config widget is a declared node input, so litegraph has already created
 * them by the time an extension sees the node. Group setup therefore succeeds on
 * the first attempt, and a failure is a bug worth reporting rather than retrying.
 */
function ensureGroupSetup(node) {
  if (node._xditGroupsReady) {
    refreshLoaderUi(node, node._xditPreviewPayload || {});
    return;
  }
  if (setupWidgetGroups(node)) return;
  console.error(
    "[xdit] Model node widgets did not match the loader schema; groups are not set up.",
    { widgets: node.widgets?.map((widget) => widget.name) },
  );
}

async function setupLoaderNode(node) {
  if (node.comfyClass !== NODE_NAME && node.type !== NODE_NAME) return;

  await ensureLoaderSchema();
  hookLoaderSerialization(node);
  hookTriggerWidgets(node);
  hookNodeWidgetChanges(node);

  if (!node._xditOnRemovedHooked) {
    node._xditOnRemovedHooked = true;
    const previousRemoved = node.onRemoved;
    node.onRemoved = function (...args) {
      clearTimeout(this._xditSyncTimer);
      previousRemoved?.apply(this, args);
    };
  }

  ensureGroupSetup(node);

  hookConfigWidgetCallbacks(node);
  // Called by the Preset node when the user picks a preset.
  node._xditApplyPreset = () => scheduleSync(node, { presetPicked: true });
  node._xditLastModelIdentity = modelIdentity(node);
  node._xditLastPresetTrigger =
    node.properties?._xdit_model_preset_trigger ?? null;
  node._xditLastPresetApplied = node._xditLastPresetTrigger != null;
  scheduleSync(node);
}

app.registerExtension({
  name: "xdit.runtime_loader",
  commands: [
    {
      id: "xdit.unloadAllModels",
      label: t("xdit.menu.unloadAll"),
      function: () => releaseAllLoaders(),
    },
  ],
  menuCommands: [
    {
      path: [t("xdit.menu.title")],
      commands: ["xdit.unloadAllModels"],
    },
  ],
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const registeredName =
      nodeData?.name ?? nodeData?.class_type ?? nodeType?.comfyClass;
    if (registeredName !== NODE_NAME) return;
    await ensureLoaderSchema();

    // Nodes 2.0 no longer reliably dispatches the extension-level `nodeCreated`
    // callback. Hook the node class itself so every newly constructed Model node
    // initializes its groups regardless of which frontend lifecycle created it.
    if (!nodeType.prototype._xditNodeCreatedHooked) {
      nodeType.prototype._xditNodeCreatedHooked = true;
      const originalNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function (...args) {
        const result = originalNodeCreated?.apply(this, args);
        setupLoaderNode(this);
        return result;
      };
    }

    const origConfigure = nodeType.prototype.configure;
    if (
      typeof origConfigure === "function" &&
      !nodeType.prototype._xditConfigureHooked
    ) {
      nodeType.prototype._xditConfigureHooked = true;
      nodeType.prototype.configure = function (data, ...rest) {
        // Values arrive by name via onConfigure, which litegraph calls at the
        // end of configure().
        hookLoaderSerialization(this);
        const result = origConfigure.apply(this, [data, ...rest]);
        this._xditGroupsReady = false;
        setupLoaderNode(this);
        return result;
      };
    }
  },
  getCanvasMenuItems() {
    return [
      {
        content: t("xdit.menu.title"),
        submenu: {
          options: [
            {
              content: t("xdit.menu.unloadAll"),
              callback: () => releaseAllLoaders(),
            },
          ],
        },
      },
    ];
  },
  nodeCreated(node) {
    setupLoaderNode(node);
  },
  loadedGraphNode(node) {
    setupLoaderNode(node);
  },
});
