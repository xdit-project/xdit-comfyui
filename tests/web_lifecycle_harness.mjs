/**
 * Run the xDiT ComfyUI extensions outside a browser.
 *
 * The node objects and the ComfyUI globals are faked only as far as the extensions
 * actually touch them. Fixtures are the real server payloads, written by the Python
 * test, so a widget renamed on the server shows up here as a failure.
 *
 * Usage: node web_lifecycle_harness.mjs <fixture-dir> <extension-dir> [vue|canvas]
 * Prints one JSON object describing what happened.
 */
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const [fixtureDir, extensionDir, nodeUi = "vue"] = process.argv.slice(2);
const fixture = (name) =>
  JSON.parse(readFileSync(`${fixtureDir}/${name}.json`, "utf8"));

const loaderSchema = fixture("loader_schema");
const presetSchema = fixture("preset_schema");
const presetPreview = fixture("preset_preview");
const presetPreviewMultiGpu = fixture("preset_preview_multi_gpu");
const loaderPreview = fixture("loader_preview");
const loaderPreviewSwitched = fixture("loader_preview_switched");
const loaderPreviewMultiTransformer = fixture(
  "loader_preview_multi_transformer",
);
const loaderPreviewVideo = fixture("loader_preview_video");
const objectInfo = fixture("object_info");

const errors = [];
const warnings = [];
const requests = [];
const intervals = [];

console.error = (...args) => errors.push(args.map(String).join(" "));
console.warn = (...args) => warnings.push(args.map(String).join(" "));
console.info = () => {};
console.debug = () => {};

globalThis.setInterval = (..._args) => {
  intervals.push(new Error("setInterval").stack);
  return 0;
};
globalThis.clearInterval = () => {};

// The extensions debounce their server sync, so "wait a fixed number of ticks" is a race
// the slower machine loses. Count what is outstanding instead and settle on quiescence.
const pending = { timers: new Set(), requests: 0 };
const realSetTimeout = globalThis.setTimeout;
const realClearTimeout = globalThis.clearTimeout;

globalThis.setTimeout = (fn, delay, ...args) => {
  const handle = realSetTimeout(
    (...called) => {
      pending.timers.delete(handle);
      fn?.(...called);
    },
    delay,
    ...args,
  );
  pending.timers.add(handle);
  return handle;
};
globalThis.clearTimeout = (handle) => {
  pending.timers.delete(handle);
  return realClearTimeout(handle);
};
globalThis.requestAnimationFrame = (fn) => globalThis.setTimeout(fn, 0);

const switchedModel = loaderPreviewSwitched.generation.model;
const multiTransformerModel = loaderPreviewMultiTransformer.generation.model;
const videoModel = loaderPreviewVideo.generation.model;

function loaderPreviewFor(requestBody) {
  // A preset being applied is answered for the preset's own model: the node still
  // holds the outgoing one, and the server merges the preset over it.
  if (requestBody.preset_applied) return loaderPreview;
  if (requestBody.model === switchedModel) return loaderPreviewSwitched;
  if (requestBody.model === multiTransformerModel)
    return loaderPreviewMultiTransformer;
  if (requestBody.model === videoModel) return loaderPreviewVideo;
  return loaderPreview;
}

globalThis.fetch = async (url, init) => {
  requests.push({ url: String(url), method: init?.method ?? "GET" });
  pending.requests += 1;
  try {
    const requestBody = init?.body ? JSON.parse(init.body) : {};
    const routes = {
      "/xdit/loader/schema": loaderSchema,
      "/xdit/preset/schema": presetSchema,
      "/xdit/preset/preview":
        Number(requestBody.gpu_count) === 4
          ? presetPreviewMultiGpu
          : presetPreview,
      "/xdit/loader/preview": loaderPreviewFor(requestBody),
      "/xdit/loader/reap": { released: [] },
    };
    const body = routes[String(url)];
    if (body === undefined) {
      pending.requests -= 1;
      return { ok: false, status: 404, json: async () => ({}) };
    }
    // The body resolves lazily, so the caller's `await response.json()` still counts
    // as work in flight while the harness decides whether the graph has settled.
    return {
      ok: true,
      status: 200,
      json: async () => {
        pending.requests -= 1;
        return body;
      },
    };
  } catch (error) {
    pending.requests -= 1;
    throw error;
  }
};

function fakeElement() {
  const element = {
    style: {},
    classList: { add() {}, remove() {} },
    children: [],
    offsetHeight: 0,
    scrollHeight: 0,
    setAttribute() {},
    replaceChildren() {
      this.children = [];
    },
    append(...nodes) {
      this.children.push(...nodes);
    },
    appendChild(child) {
      this.children.push(child);
    },
    addEventListener() {},
  };
  return element;
}

globalThis.document = {
  createElement: fakeElement,
  head: { appendChild() {} },
  documentElement: {},
};
globalThis.getComputedStyle = () => ({ getPropertyValue: () => "" });

// The Vue node UI draws each disclosure heading as its own DOM widget; the canvas UI
// clicks the toggle widget itself. Widget order and node height can only go wrong in the
// Vue path, but a heading can go missing in either.
const WIDGET_ROW_HEIGHT = 24;
const NODE_BASE_HEIGHT = 40;
const vueNodesMode = nodeUi !== "canvas";
globalThis.LiteGraph = { vueNodesMode, NODE_WIDGET_HEIGHT: WIDGET_ROW_HEIGHT };

/** A litegraph widget: `value` is the single storage the extensions read and write. */
function makeWidget(name, type, value, options) {
  const row = makeWidget.nextRow++ * WIDGET_ROW_HEIGHT;
  return {
    name,
    type,
    value,
    y: row,
    last_y: row,
    options: { ...(options || {}) },
    callback: undefined,
  };
}
makeWidget.nextRow = 0;

function widgetSpecToWidget(name, spec) {
  const [typ, opts = {}] = spec;
  if (Array.isArray(typ))
    return makeWidget(name, "combo", opts.default ?? typ[0], {
      values: [...typ],
    });
  if (typ === "INT" || typ === "FLOAT")
    return makeWidget(name, "number", opts.default ?? 0, opts);
  if (typ === "BOOLEAN")
    return makeWidget(name, "toggle", opts.default ?? false, opts);
  if (typ === "STRING")
    return makeWidget(name, "text", opts.default ?? "", opts);
  return null;
}

// "*" is the wildcard type the Model / Preset / Sample links use.
const SOCKET_TYPES = new Set(["*", "IMAGE", "VIDEO", "MASK", "LATENT"]);

function makeNode(classType, id) {
  const spec = objectInfo[classType];
  const widgets = [];
  const inputs = [];
  for (const section of ["required", "optional"]) {
    for (const [name, entry] of Object.entries(spec.input?.[section] ?? {})) {
      const typ = entry[0];
      if (typeof typ === "string" && SOCKET_TYPES.has(typ)) {
        inputs.push({ name, type: typ, link: null });
        continue;
      }
      const widget = widgetSpecToWidget(name, entry);
      if (widget) {
        widgets.push(widget);
        inputs.push({ name, type: typ, link: null, widget: { name } });
        // ComfyUI appends a linked combo to any seed-like widget and expects the
        // two to stay adjacent, so anything reordering widgets has to see it.
        if (entry[1]?.control_after_generate) {
          const control = makeWidget(
            "control_after_generate",
            "combo",
            "randomize",
            {
              values: ["fixed", "increment", "decrement", "randomize"],
              serialize: false,
            },
          );
          widget.linkedWidgets = [control];
          widgets.push(control);
        }
      }
    }
  }

  const node = {
    id,
    type: classType,
    comfyClass: classType,
    widgets,
    inputs,
    outputs: (spec.output_name ?? []).map((name) => ({
      name,
      type: "*",
      links: [],
    })),
    properties: {},
    size: [400, 300],
    graph: null,
    setSize(size) {
      this.size = size;
    },
    // Mirrors litegraph: hidden widgets take no room, so the height reports what
    // the node is actually showing.
    computeSize() {
      const rows = this.widgets.filter((widget) => !widget.hidden).length;
      return [this.size[0], NODE_BASE_HEIGHT + rows * WIDGET_ROW_HEIGHT];
    },
    addWidget(type, name, value, callback, options) {
      const widget = makeWidget(name, type, value, options);
      widget.callback = callback;
      this.widgets.push(widget);
      return widget;
    },
    addDOMWidget(name, type, element, options) {
      const widget = makeWidget(name, type, null, options);
      widget.element = element;
      this.widgets.push(widget);
      return widget;
    },
  };
  return node;
}

const graph = {
  id: "test-graph",
  _nodes: [],
  links: {},
  getNodeById(nodeId) {
    return (
      this._nodes.find((node) => String(node.id) === String(nodeId)) ?? null
    );
  },
};

const extensions = [];
const sidebarTabs = [];
const appStub = {
  graph,
  rootGraph: graph,
  configuringGraph: false,
  canvas: { setDirty() {} },
  extensionManager: {
    toast: { add() {} },
    registerSidebarTab(tab) {
      sidebarTabs.push(tab);
    },
  },
  registerExtension(extension) {
    extensions.push(extension);
  },
};

const apiListeners = new Map();
const apiStub = {
  fetchApi: async () => ({ ok: false }),
  addEventListener(type, handler) {
    if (!apiListeners.has(type)) apiListeners.set(type, []);
    apiListeners.get(type).push(handler);
  },
};

const { register } = await import("node:module");
register(
  new URL(
    "data:text/javascript," +
      encodeURIComponent(`
        export async function resolve(specifier, context, next) {
            if (specifier === "../../scripts/app.js") return { url: "xdit-stub:app", shortCircuit: true };
            if (specifier === "../../scripts/api.js") return { url: "xdit-stub:api", shortCircuit: true };
            return next(specifier, context);
        }
        export async function load(url, context, next) {
            if (url === "xdit-stub:app") {
                return { format: "module", shortCircuit: true, source: "export const app = globalThis.__xditApp;" };
            }
            if (url === "xdit-stub:api") {
                return { format: "module", shortCircuit: true, source: "export const api = globalThis.__xditApi;" };
            }
            return next(url, context);
        }
    `),
  ),
  import.meta.url,
);

globalThis.__xditApp = appStub;
globalThis.__xditApi = apiStub;

for (const name of [
  "xdit_load_model.js",
  "xdit_sample.js",
  "xdit_benchmark_preset.js",
  "xdit_residency_sidebar.js",
]) {
  await import(pathToFileURL(`${extensionDir}/${name}`).href);
}

for (const extension of extensions) await extension.setup?.();

// Nodes 2.0 constructs nodes through the registered class hook and does not reliably
// dispatch the extension-level nodeCreated callback. Exercise that lifecycle directly.
const modelNodeType = { prototype: {} };
for (const extension of extensions) {
  await extension.beforeRegisterNodeDef?.(modelNodeType, {
    name: "xDiT.Model",
  });
}

/**
 * Wait until the extensions have nothing left to do: no timer they scheduled is still
 * armed and no request is still in flight. Times out rather than hanging a CI job.
 */
async function settle({ timeoutMs = 30000 } = {}) {
  const deadline = Date.now() + timeoutMs;
  let quiet = 0;
  while (quiet < 3) {
    await new Promise((resolve) => realSetTimeout(resolve, 5));
    quiet = pending.timers.size === 0 && pending.requests === 0 ? quiet + 1 : 0;
    if (Date.now() > deadline) {
      throw new Error(
        `extensions never settled: ${pending.timers.size} timers, ` +
          `${pending.requests} requests in flight`,
      );
    }
  }
}

function link(originNode, originSlot, targetNode, inputName) {
  const id = Object.keys(graph.links).length + 1;
  const targetSlot = targetNode.inputs.findIndex(
    (entry) => entry.name === inputName,
  );
  graph.links[id] = {
    id,
    origin_id: originNode.id,
    origin_slot: originSlot,
    target_id: targetNode.id,
    target_slot: targetSlot,
  };
  targetNode.inputs[targetSlot].link = id;
  originNode.outputs[originSlot]?.links.push(id);
  return id;
}

const presetNode = makeNode("xDiT.Preset", 1);
const modelNode = makeNode("xDiT.Model", 2);
Object.setPrototypeOf(modelNode, modelNodeType.prototype);
const sampleNode = makeNode("xDiT.Sample", 3);
for (const node of [presetNode, modelNode, sampleNode]) {
  node.graph = graph;
  graph._nodes.push(node);
}

link(presetNode, 0, modelNode, "preset");
link(presetNode, 2, sampleNode, "preset");
link(modelNode, 0, sampleNode, "model");

modelNode.onNodeCreated?.();
for (const extension of extensions) {
  for (const node of graph._nodes) {
    if (node !== modelNode) extension.nodeCreated?.(node);
  }
}
await settle();

const widgetByName = (node, name) =>
  node.widgets.find((widget) => widget.name === name);
const visibleWidgetNames = (node) =>
  node.widgets.filter((widget) => !widget.hidden).map((widget) => widget.name);

/** Options greyed out because the selected model cannot use them. */
const disabledWidgetNames = (node) =>
  node.widgets
    .filter((widget) => widget.disabled === true)
    .map((widget) => widget.name)
    .filter((name) => loaderSchema.config_widgets.includes(name));

/** The Sample node's video section, which a still-image model has no use for. */
const sampleVideoGroup = () => ({
  heading_hidden:
    widgetByName(
      sampleNode,
      widgetByName(sampleNode, "Video__disclosure")
        ? "Video__disclosure"
        : "Video",
    )?.hidden === true,
  num_frames: widgetByName(sampleNode, "num_frames")?.value ?? null,
});

/** The Sample node's own sections, which are declared in the extension, not the schema. */
const SAMPLE_GROUPS = [
  {
    id: "video",
    expand_widget: "Video",
    widgets: ["num_frames", "output_fps", "flow_shift", "guidance_scale_2"],
  },
  {
    id: "vae",
    expand_widget: "VAE",
    widgets: [
      "enable_tiling",
      "enable_slicing",
      "vae_tile_size_height",
      "vae_tile_size_width",
      "vae_tile_overlap_height",
      "vae_tile_overlap_width",
    ],
  },
];

function sampleLayout() {
  const ordered = sampleNode.widgets.map((widget) => widget.name);
  return {
    order: ordered,
    groups: SAMPLE_GROUPS.map((group) => ({
      id: group.id,
      heading: ordered.indexOf(
        widgetByName(sampleNode, `${group.expand_widget}__disclosure`)
          ? `${group.expand_widget}__disclosure`
          : group.expand_widget,
      ),
      first_widget: Math.min(
        ...group.widgets
          .map((name) => ordered.indexOf(name))
          .filter((index) => index >= 0),
      ),
    })),
  };
}

/** Where each group's heading sits relative to the widgets it labels. */
function groupLayout(node) {
  const ordered = node.widgets.map((widget) => widget.name);
  return loaderSchema.widget_groups.map((group) => ({
    id: group.id,
    heading: ordered.indexOf(
      widgetByName(node, `${group.expand_widget}__disclosure`)
        ? `${group.expand_widget}__disclosure`
        : group.expand_widget,
    ),
    first_widget: Math.min(
      ...(group.widgets || [])
        .map((name) => ordered.indexOf(name))
        .filter((index) => index >= 0),
    ),
  }));
}
const afterSetup = {
  model_widget_count: modelNode.widgets.length,
  model_first_widgets: modelNode.widgets
    .slice(0, 4)
    .map((widget) => widget.name),
  sample_first_widgets: sampleNode.widgets
    .slice(0, 4)
    .map((widget) => widget.name),
  preset_choices: widgetByName(presetNode, "preset")?.options?.values ?? [],
  hidden_cache_widgets: (
    loaderSchema.widget_groups.find((group) => group.id === "cache")?.widgets ??
    []
  ).filter((name) => widgetByName(modelNode, name)?.options?.hidden === true),
  model_height: modelNode.size[1],
  model_visible_widgets: visibleWidgetNames(modelNode).length,
  group_layout: groupLayout(modelNode),
};

// A user picking a preset: litegraph sets the value, then runs the callback.
const chosen = presetPreview.preset.selected;
const presetWidget = widgetByName(presetNode, "preset");
presetWidget.value = chosen;
presetWidget.callback?.(chosen);
await settle();

// A user editing a Model widget must not be undone by the next server refresh.
const editable = widgetByName(modelNode, "vsa_top_k");
if (editable) {
  editable.value = 7;
  editable.callback?.(7);
}
await settle();

const afterPreset = {
  sample_widget_values: Object.fromEntries(
    sampleNode.widgets.map((widget) => [widget.name, widget.value]),
  ),
};

// Nodes 2.0 reports combo edits through the node-level lifecycle hook. Changing the
// GPU count must replace the preset list with choices for the new layout.
const countTestNode = makeNode("xDiT.Preset", 99);
countTestNode.graph = graph;
for (const extension of extensions) extension.nodeCreated?.(countTestNode);
await settle();
const gpuCountWidget = widgetByName(countTestNode, "gpu_count");
// Some ComfyUI combo implementations call the callback before committing `value`.
// The extension must use the callback argument, not read the previous widget state.
gpuCountWidget.callback?.("4");
await settle();
const afterGpuCountChange = {
  gpu_count: gpuCountWidget.value,
  preset_choices: widgetByName(countTestNode, "preset")?.options?.values ?? [],
};

// A user picking another model on the Model node: its own defaults replace the ones
// the preset filled in, because the preset was measured on the previous model.
const modelWidget = widgetByName(modelNode, "model");
modelWidget.value = switchedModel;
modelWidget.callback?.(switchedModel);
await settle();

const afterModelSwitch = {
  model: switchedModel,
  sample_widget_values: Object.fromEntries(
    sampleNode.widgets.map((widget) => [widget.name, widget.value]),
  ),
  disabled_widgets: disabledWidgetNames(modelNode),
  sample_video_group: sampleVideoGroup(),
};

// Even if a Nodes 2.0 control emits a change while disabled, an unsupported value
// must never remain selected in the node state.
const cfgParallelWidget = widgetByName(modelNode, "use_cfg_parallel");
const previousCfgParallel = cfgParallelWidget.value;
cfgParallelWidget.value = true;
modelNode.onWidgetChanged?.(
  "use_cfg_parallel",
  true,
  previousCfgParallel,
  cfgParallelWidget,
);
afterModelSwitch.cfg_parallel_after_rejected_edit = cfgParallelWidget.value;

// A model that caches two transformers with different presets.
modelWidget.value = multiTransformerModel;
modelWidget.callback?.(multiTransformerModel);
await settle();

/** One entry per denoiser group the schema declares: is it shown, and what does it hold? */
function denoiserGroups(node) {
  return loaderSchema.widget_groups
    .filter((group) => String(group.id).startsWith("cache_denoiser_"))
    .map((group) => ({
      id: group.id,
      // Whatever the user clicks in this UI: the disclosure widget under Vue nodes,
      // the toggle itself on the canvas.
      heading_hidden:
        widgetByName(
          node,
          widgetByName(node, `${group.expand_widget}__disclosure`)
            ? `${group.expand_widget}__disclosure`
            : group.expand_widget,
        )?.hidden === true,
      warmup: widgetByName(node, "t2_max_warmup_steps")?.value ?? null,
      widgets_hidden: (group.widgets || []).every(
        (name) => widgetByName(node, name)?.hidden === true,
      ),
    }));
}

const afterMultiTransformer = {
  model: multiTransformerModel,
  denoiser_groups: denoiserGroups(modelNode),
  max_warmup_steps_widget:
    widgetByName(modelNode, "max_warmup_steps")?.value ?? null,
  disabled_widgets: disabledWidgetNames(modelNode),
  sample_video_group: sampleVideoGroup(),
};

// Back to a single-transformer model: the second denoiser's group has nothing to drive.
modelWidget.value = switchedModel;
modelWidget.callback?.(switchedModel);
await settle();
const denoiserGroupsAfterSwitchBack = denoiserGroups(modelNode);

// The user opening a group: its widgets belong under that group's own heading, and the
// node grows to fit them.
const attentionGroup = loaderSchema.widget_groups.find(
  (group) => group.id === "attention",
);
const attentionHeading = widgetByName(modelNode, attentionGroup.expand_widget);
attentionHeading.value = true;
attentionHeading.callback?.(true);
await settle();

// A widget left holding null is rejected by ComfyUI's type conversion at queue time.
const nullValuedWidgets = modelNode.widgets
  .filter((widget) => widget.value === null || widget.value === undefined)
  .map((widget) => widget.name)
  .filter((name) => loaderSchema.config_widgets.includes(name));

const orderedNames = modelNode.widgets.map((widget) => widget.name);
const afterExpand = {
  height: modelNode.size[1],
  visible_widgets: visibleWidgetNames(modelNode).length,
  group_layout: groupLayout(modelNode),
  visible_after_heading: (() => {
    const heading = orderedNames.indexOf(
      `${attentionGroup.expand_widget}__disclosure`,
    );
    const visible = visibleWidgetNames(modelNode);
    const first = attentionGroup.widgets.find((name) => visible.includes(name));
    return { heading, first_visible_group_widget: orderedNames.indexOf(first) };
  })(),
};

attentionHeading.value = false;
attentionHeading.callback?.(false);
await settle();
const heightAfterCollapse = modelNode.size[1];

// Round-trip the graph through save/load.
const saved = {};
for (const node of graph._nodes) {
  const data = { widgets_values: node.widgets.map((widget) => widget.value) };
  node.onSerialize?.(data);
  saved[node.type] = data;
}
const reloaded = makeNode("xDiT.Model", 2);
reloaded.graph = graph;
for (const extension of extensions) extension.nodeCreated?.(reloaded);
await settle();
reloaded.onConfigure?.(saved["xDiT.Model"]);

// Leaving a video model for an image preset. This runs on a Model node of its own, at
// the end, so the measurements above see the graph they were written for: `task` belongs
// to the model being left, and the node still holds that model while the preset applies.
const videoModelNode = makeNode("xDiT.Model", 5);
videoModelNode.graph = graph;
graph._nodes.push(videoModelNode);
for (const extension of extensions) extension.nodeCreated?.(videoModelNode);
link(presetNode, 0, videoModelNode, "preset");
await settle();

const videoModelWidget = widgetByName(videoModelNode, "model");
videoModelWidget.value = videoModel;
videoModelWidget.callback?.(videoModel);
await settle();

const taskState = () => ({
  value: widgetByName(videoModelNode, "task")?.value ?? null,
  hidden: widgetByName(videoModelNode, "task")?.hidden === true,
});
const taskOnVideoModel = taskState();

// What the Preset node does to every Model node connected to it.
videoModelNode._xditApplyPreset?.();
await settle();
const taskAfterLeavingVideoModel = taskState();

const domLayout = (node, name) =>
  node.widgets.find((widget) => widget.name === name)?.computeLayoutSize?.();

console.log(
  JSON.stringify(
    {
      errors,
      warnings,
      intervals: intervals.length,
      sidebar_tabs: sidebarTabs.map((tab) => ({
        id: tab.id,
        title: tab.title,
        hasRender: typeof tab.render === "function",
      })),
      canvas_menus: extensions
        .flatMap((extension) => extension.getCanvasMenuItems?.() ?? [])
        .filter(Boolean)
        .map((item) => ({
          content: item.content,
          submenuItems:
            item.submenu?.options?.map((option) => option.content) ?? [],
        })),
      requests,
      after_setup: afterSetup,
      chosen_preset: chosen,
      preset_gpu_count: widgetByName(presetNode, "gpu_count")?.value ?? null,
      preset_applied_to_sample: sampleNode._xditLastPresetApplied === true,
      sample_prompt: widgetByName(sampleNode, "prompt")?.value ?? null,
      model_preset_trigger:
        modelNode.properties._xdit_model_preset_trigger ?? null,
      sample_model_constraints: sampleNode._xditModelConstraints ?? null,
      sample_resolution_step:
        widgetByName(sampleNode, "height")?.options?.step2 ?? null,
      sample_layout: sampleLayout(),
      dom_layout: {
        preset_previews: domLayout(presetNode, "xdit_preset_image_previews"),
      },
      after_preset: afterPreset,
      after_gpu_count_change: afterGpuCountChange,
      after_model_switch: afterModelSwitch,
      after_multi_transformer: afterMultiTransformer,
      task_on_video_model: taskOnVideoModel,
      task_after_leaving_video_model: taskAfterLeavingVideoModel,
      denoiser_groups_after_switch_back: denoiserGroupsAfterSwitchBack,
      null_valued_widgets: nullValuedWidgets,
      after_expand: afterExpand,
      height_after_collapse: heightAfterCollapse,
      user_edit_preserved: widgetByName(modelNode, "vsa_top_k")?.value ?? null,
      serialized: {
        model_positional: saved["xDiT.Model"].widgets_values,
        model_named_count: Object.keys(
          saved["xDiT.Model"].xdit_widget_values ?? {},
        ).length,
        sample_named_count: Object.keys(
          saved["xDiT.Sample"].xdit_widget_values ?? {},
        ).length,
        preset_named: saved["xDiT.Preset"].xdit_widget_values ?? {},
      },
      reloaded_vsa_top_k: widgetByName(reloaded, "vsa_top_k")?.value ?? null,
    },
    null,
    1,
  ),
);
