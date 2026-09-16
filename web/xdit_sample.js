import { app } from "../../scripts/app.js";
import {
  expandGroup,
  groupExpanded,
  installDisclosureGroups,
  refreshVueWidgetSnapshot,
  usesModernNodeWidgets,
} from "./xdit_disclosure.js";
import {
  connectedPresetNode,
  fetchPresetPreview,
  presetSelection,
} from "./xdit_preset_bus.js";
import {
  restoreWidgetsByName,
  serializeWidgetsByName,
} from "./xdit_widget_serialization.js";
import {
  litegraphWidget,
  originNodeFromInput,
  requestSequence,
  setWidgetHidden as setWidgetHiddenState,
  setWidgetValue,
  userEdited,
  widgetValue,
} from "./xdit_widget_state.js";
import { applyWidgetOrder } from "./xdit_widget_order.js";
import { t } from "./xdit_i18n.js";

const GENERATE_NODES = new Set(["xDiT.Sample"]);

const PRESET_WIDGETS = [
  "prompt",
  "negative_prompt",
  "seed",
  "num_inference_steps",
  "guidance_scale",
  "max_sequence_length",
  "height",
  "width",
  "num_frames",
  "flow_shift",
  "guidance_scale_2",
  "resize_input_images",
  "enable_tiling",
  "enable_slicing",
  "vae_tile_size_height",
  "vae_tile_size_width",
  "vae_tile_overlap_height",
  "vae_tile_overlap_width",
];

const VIDEO_CLEAR_DEFAULTS = {
  num_frames: 1,
  flow_shift: 0.0,
  guidance_scale_2: 0.0,
};

const VIDEO_GROUP = {
  id: "video",
  label: t("xdit.group.video.label"),
  expand_widget: "Video",
  collapsed: true,
  description: t("xdit.group.video.description"),
  widgets: ["num_frames", "output_fps", "flow_shift", "guidance_scale_2"],
};

const VAE_GROUP = {
  id: "vae",
  label: t("xdit.group.vae.label"),
  expand_widget: "VAE",
  collapsed: true,
  description: t("xdit.group.vae.description"),
  widgets: [
    "enable_tiling",
    "enable_slicing",
    "vae_tile_size_height",
    "vae_tile_size_width",
    "vae_tile_overlap_height",
    "vae_tile_overlap_width",
  ],
};

const WIDGET_GROUPS = [VIDEO_GROUP, VAE_GROUP];

const EXPAND_WIDGETS = WIDGET_GROUPS.map((group) => group.expand_widget);

function setWidgetHidden(node, widgetOrName, hidden) {
  const name =
    typeof widgetOrName === "string" ? widgetOrName : widgetOrName?.name;
  setWidgetHiddenState(node, widgetOrName, hidden, {
    canvasOnly: usesModernNodeWidgets() && EXPAND_WIDGETS.includes(name),
  });
}

function findSeedControlWidget(node, seedWidget) {
  if (seedWidget?.linkedWidgets?.length) {
    return seedWidget.linkedWidgets[0];
  }
  const named = litegraphWidget(node, "control_after_generate");
  if (named) return named;
  return (
    node.widgets?.find(
      (widget) =>
        widget.type === "combo" &&
        widget.options?.values?.includes("randomize") &&
        widget.options?.serialize === false,
    ) ?? null
  );
}

function seedControlMode(node) {
  const seedWidget = litegraphWidget(node, "seed");
  const controlWidget = findSeedControlWidget(node, seedWidget);
  if (controlWidget?.value !== undefined) return controlWidget.value;
  return widgetValue(node, "control_after_generate", "randomize");
}

function ensureDefaultSeedControl(node) {
  const seedWidget = litegraphWidget(node, "seed");
  const controlWidget = findSeedControlWidget(node, seedWidget);
  if (!seedWidget || !controlWidget || controlWidget.value !== undefined)
    return;
  setWidgetValue(node, "control_after_generate", "randomize", {
    comboGuard: false,
  });
}

/** Groups the connected model has no use for, such as frame count on a still-image model. */
function inactiveGroups(node) {
  const kind = node._xditModelConstraints?.output_kind;
  return kind && kind !== "video" ? new Set([VIDEO_GROUP.id]) : new Set();
}

function applyGroupVisibility(node) {
  const inactive = inactiveGroups(node);
  for (const group of WIDGET_GROUPS) {
    const dropped = inactive.has(group.id);
    const expanded = groupExpanded(node, group);
    for (const name of group.widgets) {
      setWidgetHidden(node, name, dropped || !expanded);
    }
    // Both headings are set: the canvas UI clicks the toggle, the Vue UI clicks the
    // disclosure widget that replaces it.
    setWidgetHidden(node, group.expand_widget, dropped);
  }
  refreshVueWidgetSnapshot(node);
  fitSampleHeight(node);
  applySchemaWidgetOrder(node);
  app.canvas?.setDirty?.(true, true);
}

function applySchemaWidgetOrder(node) {
  const inputOrder = (node.inputs || [])
    .map((input) => input.widget?.name)
    .filter(Boolean);
  if (!inputOrder.length) return false;
  if (!usesModernNodeWidgets()) {
    const order = [];
    for (const name of inputOrder) {
      order.push(name);
      const widget = litegraphWidget(node, name);
      for (const linked of widget?.linkedWidgets || []) {
        if (linked?.name) order.push(linked.name);
      }
    }
    const known = new Set(order);
    return applyWidgetOrder(node, [
      ...order,
      ...(node.widgets || [])
        .map((widget) => widget.name)
        .filter((name) => !known.has(name)),
    ]);
  }
  if (typeof document.querySelector !== "function") return false;
  const grid = document.querySelector(
    `[data-widgets-grid-node-id="${node.id}"]`,
  );
  if (!grid) return false;
  const styleId = `xdit-sample-order-${node.id}`;
  let style = document.getElementById?.(styleId);
  if (!style) {
    style = document.createElement("style");
    style.id = styleId;
    document.head.appendChild(style);
  }
  style.textContent = (node.inputs || [])
    .map(
      (_input, index) =>
        `[data-widgets-grid-node-id="${node.id}"] > ` +
        `:has([data-slot-key="${node.id}-in-${index}"]) { order: ${index} !important; }`,
    )
    .concat([
      `[data-widgets-grid-node-id="${node.id}"] > :has(.xdit-info) { order: -1 !important; }`,
    ])
    .join("\n");
  const nodeElement = document.querySelector(`[data-node-id="${node.id}"]`);
  if (nodeElement) {
    const nodeBox = nodeElement.getBoundingClientRect();
    const gridBox = grid.getBoundingClientRect();
    const requiredHeight = Math.ceil(
      Math.max(grid.scrollHeight, gridBox.height) +
        (gridBox.top - nodeBox.top) +
        24,
    );
    if (requiredHeight > (node.size?.[1] || 0)) {
      node.setSize?.([node.size?.[0] || 320, requiredHeight]);
    }
  }
  return true;
}

function scheduleSchemaWidgetOrder(node) {
  applySchemaWidgetOrder(node);
  for (const delay of [0, 50, 250, 1000]) {
    setTimeout(() => {
      applySchemaWidgetOrder(node);
      fitSampleHeight(node);
    }, delay);
  }
}

function fitSampleHeight(node) {
  const computed = node.computeSize?.();
  if (!computed) return;
  const width = Math.max(node.size?.[0] ?? 0, 320);
  if (node.size?.[0] === width && node.size?.[1] === computed[1]) return;
  node.setSize([width, computed[1]]);
}

/**
 * `control_after_generate` has to stay with the seed it controls.
 *
 * ComfyUI attaches it as a linked widget, so read the link rather than the name: the
 * name has changed across ComfyUI versions and a missed one would strand the combo at
 * the bottom of the node.
 */
function hookExpandWidgets(node) {
  installDisclosureGroups(node, WIDGET_GROUPS, {
    setWidgetHidden,
    onToggle: () => applyGroupVisibility(node),
  });
  applyGroupVisibility(node);
}

function persistPresetTrigger(node, trigger) {
  node.properties ??= {};
  node.properties._xdit_sample_preset_trigger = trigger;
}

function applyGenerationDefaults(
  node,
  defaults,
  { presetChanged = false } = {},
) {
  if (!defaults) return;

  const preserveSeed = !presetChanged && seedControlMode(node) !== "fixed";
  const authoritative = presetApplied(node);
  for (const name of PRESET_WIDGETS) {
    if (!litegraphWidget(node, name)) continue;
    if (preserveSeed && name === "seed") continue;
    if (name in defaults) {
      setWidgetValue(node, name, defaults[name], { comboGuard: false });
    } else if (name in VIDEO_CLEAR_DEFAULTS) {
      setWidgetValue(node, name, VIDEO_CLEAR_DEFAULTS[name], {
        comboGuard: false,
      });
    } else {
      continue;
    }
    // Picking a preset is a deliberate choice of these values, so the model's own
    // defaults do not overwrite them until the user picks another model.
    authoritative.add(name);
  }
  const numFrames = Number(
    defaults.num_frames ??
      widgetValue(node, "num_frames", VIDEO_CLEAR_DEFAULTS.num_frames),
  );
  const task = String(defaults.task ?? "").trim();
  // A video preset should not hide the frame count it just set.
  expandGroup(
    node,
    VIDEO_GROUP,
    (Number.isFinite(numFrames) && numFrames > 1) || !!task,
  );
  applyGroupVisibility(node);
}

/**
 * Put height and width on the grid the selected model accepts.
 *
 * LTX only accepts multiples of 64 and the runner raises on anything else, so the
 * widget should not offer a value that cannot run.
 */
function applyResolutionGrid(node, step) {
  const grid = Number(step);
  if (!Number.isFinite(grid) || grid < 1) return;
  for (const name of ["height", "width"]) {
    const widget = litegraphWidget(node, name);
    if (!widget) continue;
    widget.options ??= {};
    widget.options.min = grid;
    // Legacy litegraph reads `step` as a tenth of the increment; the Vue widgets
    // read `step2`.
    widget.options.step = grid * 10;
    widget.options.step2 = grid;
    widget.options.round = grid;
  }
}

/** Widgets whose value came from a preset rather than from the user or a model. */
function presetApplied(node) {
  node._xditPresetApplied ??= new Set();
  return node._xditPresetApplied;
}

/**
 * Adopt the defaults the selected model declares, leaving edited widgets alone.
 *
 * These come from the Model node's preview response, which reads the runner's own
 * `DefaultInputValues`; the node definition can only carry one model's numbers.
 * Choosing a different model discards values a preset filled in, since a preset's
 * resolution and step count belong to the model it was measured on.
 */
function applyModelConstraints(
  node,
  generation,
  { modelChanged = false } = {},
) {
  if (!generation) return;
  node._xditModelConstraints = generation;
  applyResolutionGrid(node, generation.resolution_step);

  const defaults = generation.defaults || {};
  const edited = userEdited(node);
  const fromPreset = presetApplied(node);
  let changed = false;
  for (const [name, value] of Object.entries(defaults)) {
    if (edited.has(name) || !litegraphWidget(node, name)) continue;
    if (fromPreset.has(name)) {
      if (!modelChanged) continue;
      fromPreset.delete(name);
    }
    changed =
      setWidgetValue(node, name, value, { comboGuard: false }) || changed;
  }
  clearVideoValuesForStillImageModel(node);
  if (changed) {
    const numFrames = Number(
      widgetValue(node, "num_frames", VIDEO_CLEAR_DEFAULTS.num_frames),
    );
    if (Number.isFinite(numFrames) && numFrames > 1)
      expandGroup(node, VIDEO_GROUP, true);
  }
  applyGroupVisibility(node);
}

/** A still-image model renders one frame whatever the widget says, so say so. */
function clearVideoValuesForStillImageModel(node) {
  if (!inactiveGroups(node).has(VIDEO_GROUP.id)) return;
  for (const [name, value] of Object.entries(VIDEO_CLEAR_DEFAULTS)) {
    setWidgetValue(node, name, value, { comboGuard: false });
  }
}

/** A Sample node connected to an already-synced Model node adopts what it knows. */
function adoptConnectedModelConstraints(node) {
  const modelNode = originNodeFromInput(node, "model");
  applyModelConstraints(node, modelNode?._xditPreviewPayload?.generation, {
    modelChanged: false,
  });
}

/**
 * Take generation defaults from a preset selection.
 *
 * The Preset node pushes the payload it already fetched; a connection change here
 * has no payload, so this node fetches (deduplicated against the Preset node's own
 * request for the same selection).
 */
async function applyPreset(node, trigger, payload) {
  if (app.configuringGraph) return;
  if (trigger === node._xditLastPresetTrigger && node._xditLastPresetApplied)
    return;

  const isCurrent = requestSequence(node, "presetPreview");
  let resolved = payload;
  if (!resolved) {
    const presetNode = connectedPresetNode(node);
    if (!presetNode) return;
    try {
      resolved = await fetchPresetPreview(presetSelection(presetNode));
    } catch (error) {
      console.warn("[xdit] generate preset sync failed", error);
      return;
    }
    if (!isCurrent()) return;
  }

  const spec = resolved.preset;
  if (!spec?.matched) {
    node._xditLastPresetTrigger = trigger;
    node._xditLastPresetApplied = false;
    return;
  }
  const presetChanged = trigger !== node._xditLastPresetTrigger;
  const defaults = { ...(spec.generation_defaults || {}) };
  Object.assign(defaults, spec.vae_defaults || {});
  applyGenerationDefaults(node, defaults, { presetChanged });
  node._xditLastPresetTrigger = trigger;
  node._xditLastPresetApplied = true;
  persistPresetTrigger(node, trigger);
}

function hookGenerateSerialization(node) {
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
    // A saved graph holds the values the user ran with, so a model's defaults must
    // not replace them when the node reconnects.
    this._xditUserEdited = new Set(restored);
  };
}

/** Record edits so a model change fills in only what the user left alone. */
function hookGenerationWidgets(node) {
  for (const widget of node.widgets ?? []) {
    if (widget._xditEditHooked) continue;
    widget._xditEditHooked = true;
    const original = widget.callback;
    widget.callback = function (value, ...args) {
      userEdited(node).add(widget.name);
      return original?.apply(this, [value, ...args]);
    };
  }
}

/** A freshly connected Preset or Model node has to reach this node without a poll. */
function hookConnectionChanges(node) {
  if (node._xditConnectionsHooked) return;
  node._xditConnectionsHooked = true;
  const previous = node.onConnectionsChange;
  node.onConnectionsChange = function (...args) {
    const result = previous?.apply(this, args);
    if (!app.configuringGraph) {
      const presetNode = connectedPresetNode(this);
      if (presetNode) applyPreset(this, presetSelection(presetNode).trigger);
      adoptConnectedModelConstraints(this);
    }
    return result;
  };
}

function setupGenerateNode(node) {
  if (!GENERATE_NODES.has(node.comfyClass) && !GENERATE_NODES.has(node.type))
    return;

  hookGenerateSerialization(node);
  hookConnectionChanges(node);
  hookGenerationWidgets(node);
  hookExpandWidgets(node);
  ensureDefaultSeedControl(node);
  scheduleSchemaWidgetOrder(node);
  node._xditApplyPreset = (trigger, payload) =>
    applyPreset(node, trigger, payload);
  node._xditApplyModelConstraints = (generation, options) =>
    applyModelConstraints(node, generation, options);
  adoptConnectedModelConstraints(node);
  node._xditGenerateGroupsReady = true;
}

function ensureGenerateSetup(node) {
  if (node._xditGenerateGroupsReady) {
    applyGroupVisibility(node);
    return;
  }
  setupGenerateNode(node);
}

/**
 * A restored graph already holds the values the preset produced, so record the
 * trigger as applied instead of re-fetching and overwriting edited widgets.
 */
function adoptRestoredPresetTrigger(node) {
  const trigger =
    node.properties?._xdit_sample_preset_trigger ??
    (connectedPresetNode(node)
      ? presetSelection(connectedPresetNode(node)).trigger
      : null);
  if (trigger == null) return;
  node._xditLastPresetTrigger = trigger;
  node._xditLastPresetApplied = true;
}

app.registerExtension({
  name: "xdit.sample",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const registeredName =
      nodeData?.name ?? nodeData?.class_type ?? nodeType?.comfyClass;
    if (!GENERATE_NODES.has(registeredName)) return;

    if (!nodeType.prototype._xditGenerateNodeCreatedHooked) {
      nodeType.prototype._xditGenerateNodeCreatedHooked = true;
      const originalNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function (...args) {
        const result = originalNodeCreated?.apply(this, args);
        ensureGenerateSetup(this);
        return result;
      };
    }

    const origConfigure = nodeType.prototype.configure;
    if (
      typeof origConfigure === "function" &&
      !nodeType.prototype._xditGenerateConfigureHooked
    ) {
      nodeType.prototype._xditGenerateConfigureHooked = true;
      nodeType.prototype.configure = function (...args) {
        const result = origConfigure.apply(this, args);
        this._xditGenerateGroupsReady = false;
        adoptRestoredPresetTrigger(this);
        ensureGenerateSetup(this);
        return result;
      };
    }
  },
  nodeCreated(node) {
    ensureGenerateSetup(node);
  },
  loadedGraphNode(node) {
    adoptRestoredPresetTrigger(node);
    ensureGenerateSetup(node);
  },
});
