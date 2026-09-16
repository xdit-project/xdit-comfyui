import { app } from "../../scripts/app.js";
import {
  PRESET_NONE,
  fetchPresetPreview,
  presetSelection,
} from "./xdit_preset_bus.js";
import { resolvePresetFilters } from "./xdit_preset_filters.js";
import { notifyResidencyChange } from "./xdit_residency_info.js";
import {
  restoreWidgetsByName,
  serializeWidgetsByName,
} from "./xdit_widget_serialization.js";
import {
  litegraphWidget,
  requestSequence,
  setComboChoices,
  setWidgetValue,
} from "./xdit_widget_state.js";

const NODE_NAME = "xDiT.Preset";
const GPU_INFO_WIDGET = "gpu_detection_info";
const TRIGGER_WIDGETS = ["gpu_tag", "gpu_count", "preset"];
let presetFilterSchema = null;
let presetFilterSchemaPromise = null;

async function ensurePresetFilterSchema() {
  if (presetFilterSchema) return presetFilterSchema;
  if (!presetFilterSchemaPromise) {
    presetFilterSchemaPromise = fetch("/xdit/preset/schema")
      .then((response) => {
        if (!response.ok)
          throw new Error(`preset schema failed (${response.status})`);
        return response.json();
      })
      .then((schema) => (presetFilterSchema = schema))
      .finally(() => (presetFilterSchemaPromise = null));
  }
  return presetFilterSchemaPromise;
}

function applyLocalFilters(node, selection) {
  if (!presetFilterSchema) return;
  const resolved = resolvePresetFilters(
    presetFilterSchema,
    selection.gpuTag,
    selection.gpuCount,
    selection.preset,
  );
  node._xditApplyingPresetChoices = true;
  try {
    setComboChoices(node, "gpu_count", resolved.counts);
    setWidgetValue(node, "gpu_count", resolved.gpuCount);
    setComboChoices(node, "preset", [PRESET_NONE, ...resolved.presets]);
    setWidgetValue(node, "preset", resolved.preset);
  } finally {
    node._xditApplyingPresetChoices = false;
  }
}

function styleGpuDetectionWidget(node) {
  const widget = litegraphWidget(node, GPU_INFO_WIDGET);
  if (!widget) return;
  widget.disabled = true;
  widget.read_only = true;
  if (widget.options) {
    widget.options.disabled = true;
    widget.options.read_only = true;
  }
}

function hookPresetSerialization(node) {
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
    restoreWidgetsByName(this, data, (name, value) =>
      setWidgetValue(this, name, value, { comboGuard: false }),
    );
  };
}

function reorderPresetWidgets(node) {
  const widget = litegraphWidget(node, GPU_INFO_WIDGET);
  if (!widget || !node.widgets?.length) return;
  const rest = node.widgets.filter((entry) => entry !== widget);
  node.widgets = [widget, ...rest];
  app.canvas?.setDirty?.(true, true);
}

/**
 * Narrow the combos to the selection the server resolved.
 *
 * Which GPU counts a hardware tag has, and which presets exist for that pairing, is
 * decided by the benchmark configs; `/xdit/preset/preview` reports the resolved
 * selection so the browser does not re-derive it from a second copy of the tables.
 */
function applyResolvedChoices(node, payload) {
  node._xditApplyingPresetChoices = true;
  try {
    setComboChoices(node, "gpu_tag", payload.gpu_tag_choices);
    setWidgetValue(node, "gpu_tag", payload.gpu_tag);
    setComboChoices(
      node,
      "gpu_count",
      (payload.gpu_count_choices || []).map(String),
    );
    setWidgetValue(node, "gpu_count", String(payload.gpu_count));
    setComboChoices(node, "preset", payload.choices);
    setWidgetValue(node, "preset", payload.preset?.selected ?? PRESET_NONE);
    if (payload.gpu_detection_summary) {
      setWidgetValue(node, GPU_INFO_WIDGET, payload.gpu_detection_summary, {
        comboGuard: false,
      });
    }
  } finally {
    node._xditApplyingPresetChoices = false;
  }
}

function connectedConsumers(presetNode, className) {
  const graph = app.graph;
  if (!graph?.links) return [];
  const targets = [];
  for (const link of Object.values(graph.links)) {
    if (!link || link.origin_id !== presetNode.id) continue;
    const target = graph.getNodeById(link.target_id);
    if (
      !target ||
      (target.comfyClass !== className && target.type !== className)
    )
      continue;
    if (target.inputs?.[link.target_slot]?.name !== "preset") continue;
    targets.push(target);
  }
  return targets;
}

/**
 * Resolve the current selection and push it downstream in one pass.
 *
 * `push` is false during setup, where the saved graph already holds the values this
 * preset produced and only the node's own display needs refreshing. Model nodes
 * resolve their own preview because it depends on their widget values; that request
 * goes out alongside this one rather than after it.
 */
function selectionAfterWidgetChange(node, name, value) {
  const selection = presetSelection(node);
  if (name === "gpu_tag") selection.gpuTag = value;
  if (name === "gpu_count") selection.gpuCount = Number(value);
  if (name === "preset") selection.preset = value;
  selection.trigger = `${selection.gpuTag}:${selection.gpuCount}:${selection.preset}`;
  return selection;
}

async function refreshPreset(node, { push = true, selection = null } = {}) {
  if (app.configuringGraph) return;
  selection ??= presetSelection(node);
  const isCurrent = requestSequence(node, "presetPreview");

  const loaders = push ? connectedConsumers(node, "xDiT.Model") : [];
  const samples = push ? connectedConsumers(node, "xDiT.Sample") : [];
  for (const loader of loaders) loader._xditApplyPreset?.(selection.trigger);

  let payload;
  try {
    payload = await fetchPresetPreview(selection);
  } catch (error) {
    console.warn("[xdit] preset preview failed", error);
    return;
  }
  if (!isCurrent()) return;

  if (!node._xditHardwareTagInitialized) {
    node._xditHardwareTagInitialized = true;
    const suggested = payload.gpu_tag_suggested;
    if (suggested && suggested !== selection.gpuTag) {
      setWidgetValue(node, "gpu_tag", suggested);
      await refreshPreset(node, { push: false });
      return;
    }
  }

  applyResolvedChoices(node, payload);
  renderPresetImagePreviews(node, payload.image_previews || []);
  node._xditPresetPreviewPayload = payload;
  notifyResidencyChange();
  const trigger = presetSelection(node).trigger;
  for (const sample of samples) sample._xditApplyPreset?.(trigger, payload);
}

function ensurePreviewHost(node) {
  if (node._xditPresetPreviewHost) return node._xditPresetPreviewHost;
  if (node._xditPresetBaseHeight == null) {
    const naturalSize = node.computeSize?.();
    node._xditPresetBaseHeight = naturalSize?.[1] ?? 180;
  }
  const host = document.createElement("div");
  host.className = "xdit-preset-image-previews";
  host.style.marginTop = "6px";
  host.style.display = "none";
  host.style.width = "100%";
  host.style.boxSizing = "border-box";
  host.style.gap = "4px";
  node._xditPresetPreviewHost = host;
  const widget = node.addDOMWidget?.(
    "xdit_preset_image_previews",
    "xditPresetImagePreviews",
    host,
    {
      serialize: false,
    },
  );
  if (widget) {
    widget.computeSize = (width) => {
      const blockHeight = host.offsetHeight || host.scrollHeight || 0;
      return [width, blockHeight > 0 ? blockHeight + 8 : 0];
    };
    widget.computeLayoutSize = () => {
      if (host.style.display === "none") {
        return { minHeight: 0, maxHeight: 0, minWidth: 0 };
      }
      const blockHeight = host.offsetHeight || host.scrollHeight || 0;
      const height = Math.max(blockHeight + 8, 32);
      return { minHeight: height, maxHeight: height, minWidth: 0 };
    };
  }
  return host;
}

function previewGridTemplate(count) {
  if (count <= 1) return "minmax(0, 1fr)";
  return `repeat(${count}, minmax(0, 1fr))`;
}

function previewImageMaxHeight(count) {
  if (count <= 1) return 320;
  if (count === 2) return 220;
  if (count === 3) return 180;
  return 140;
}

function resizePreviewNode(node, host) {
  const blockHeight = host.offsetHeight || host.scrollHeight || 0;
  const baseHeight = node._xditPresetBaseHeight ?? node.size[1];
  const targetHeight =
    blockHeight > 0 ? baseHeight + blockHeight + 10 : baseHeight;
  if (Math.abs(node.size[1] - targetHeight) > 2) {
    node.setSize([node.size[0], targetHeight]);
  }
  app.canvas?.setDirty?.(true, true);
}

function renderPresetImagePreviews(node, previews) {
  const host = ensurePreviewHost(node);
  host.replaceChildren();
  const count = previews?.length ?? 0;
  if (!count) {
    host.style.display = "none";
    resizePreviewNode(node, host);
    return;
  }

  host.style.display = "grid";
  host.style.gridTemplateColumns = previewGridTemplate(count);
  const maxHeight = previewImageMaxHeight(count);

  let pending = count;
  const onImageReady = () => {
    pending -= 1;
    if (pending <= 0) {
      resizePreviewNode(node, host);
    }
  };

  for (const preview of previews) {
    const card = document.createElement("div");
    card.style.minWidth = "0";
    card.style.width = "100%";
    card.style.display = "flex";
    card.style.flexDirection = "column";
    card.style.gap = "2px";

    const img = document.createElement("img");
    img.src = preview.url;
    img.alt = preview.name || "preset reference";
    img.title = preview.name || "preset reference";
    img.style.width = "100%";
    img.style.height = "auto";
    img.style.maxHeight = `${maxHeight}px`;
    img.style.objectFit = "contain";
    img.style.display = "block";
    img.style.borderRadius = "4px";
    img.style.background = "rgba(255,255,255,0.06)";
    img.onload = () => {
      if (count === 1 && img.naturalWidth > 0 && img.naturalHeight > 0) {
        const nodeWidth = Math.max((node.size?.[0] ?? 420) - 24, 120);
        const aspect = img.naturalWidth / img.naturalHeight;
        const height = Math.min(Math.max(nodeWidth / aspect, 120), maxHeight);
        img.style.height = `${height}px`;
        img.style.maxHeight = `${height}px`;
      }
      onImageReady();
    };
    img.onerror = onImageReady;

    card.appendChild(img);
    if (count > 1 && preview.name) {
      const label = document.createElement("span");
      label.textContent = preview.name;
      label.style.fontSize = "10px";
      label.style.opacity = "0.75";
      label.style.textAlign = "center";
      label.style.wordBreak = "break-all";
      label.style.lineHeight = "1.2";
      card.appendChild(label);
    }
    host.appendChild(card);
  }
  requestAnimationFrame(() => resizePreviewNode(node, host));
}

function hookTriggers(node) {
  for (const name of TRIGGER_WIDGETS) {
    const widget = litegraphWidget(node, name);
    if (!widget || widget._xditPresetHooked) continue;
    widget._xditPresetHooked = true;
    const original = widget.callback;
    widget.callback = function (value, ...args) {
      const result = original?.apply(this, [value, ...args]);
      const selection = selectionAfterWidgetChange(node, name, value);
      applyLocalFilters(node, selection);
      refreshPreset(node, {
        selection,
      });
      return result;
    };
  }
}

/** Nodes 2.0 announces editor changes on the node rather than reliably invoking
 * the LiteGraph widget callback. Keep both paths because Nodes 1.0 still uses the
 * callback directly. Concurrent duplicate refreshes share the same request. */
function hookNodeWidgetChanges(node) {
  if (node._xditPresetWidgetChangesHooked) return;
  node._xditPresetWidgetChangesHooked = true;
  const previous = node.onWidgetChanged;
  node.onWidgetChanged = function (name, value, oldValue, widget) {
    const result = previous?.call(this, name, value, oldValue, widget);
    if (!this._xditApplyingPresetChoices && TRIGGER_WIDGETS.includes(name)) {
      const selection = selectionAfterWidgetChange(this, name, value);
      applyLocalFilters(this, selection);
      refreshPreset(this, {
        selection,
      });
    }
    return result;
  };
}

/** A newly connected consumer needs the current selection right away. */
function hookConnectionChanges(node) {
  if (node._xditConnectionsHooked) return;
  node._xditConnectionsHooked = true;
  const previous = node.onConnectionsChange;
  node.onConnectionsChange = function (...args) {
    const result = previous?.apply(this, args);
    if (!app.configuringGraph) refreshPreset(this);
    return result;
  };
}

async function setupPresetNode(node) {
  if (node.comfyClass !== NODE_NAME && node.type !== NODE_NAME) return;
  hookPresetSerialization(node);
  if (!node.widgets?.length) return;

  styleGpuDetectionWidget(node);
  hookTriggers(node);
  hookNodeWidgetChanges(node);
  hookConnectionChanges(node);
  reorderPresetWidgets(node);
  try {
    await ensurePresetFilterSchema();
  } catch (error) {
    console.warn("[xdit] preset schema failed", error);
  }
  refreshPreset(node, { push: false });
}

app.registerExtension({
  name: "xdit.preset",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;
    const origConfigure = nodeType.prototype.configure;
    if (
      typeof origConfigure === "function" &&
      !nodeType.prototype._xditPresetConfigureHooked
    ) {
      nodeType.prototype._xditPresetConfigureHooked = true;
      nodeType.prototype.configure = function (...args) {
        const result = origConfigure.apply(this, args);
        setupPresetNode(this);
        return result;
      };
    }
  },
  nodeCreated(node) {
    setupPresetNode(node);
  },
  loadedGraphNode(node) {
    setupPresetNode(node);
  },
});
