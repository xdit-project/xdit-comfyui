import { app } from "../../scripts/app.js";
import { t } from "./xdit_i18n.js";
import {
  ensureInfoStyles,
  fetchResidency,
  notifyResidencyChange,
} from "./xdit_residency_info.js";

const TAB_ID = "xdit";
const REFRESH_MS = 2500;

function gib(value) {
  return value === null || value === undefined ? "?" : Number(value).toFixed(1);
}

function gpuLabel(gpus = []) {
  return gpus.length === 1
    ? `GPU ${gpus[0]}`
    : `GPUs ${gpus.join(", ") || "?"}`;
}

async function unload(nodeId) {
  const url = nodeId ? "/xdit/loader/clear" : "/xdit/loader/reap";
  const body = nodeId ? { node_id: String(nodeId) } : { all: true };
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`Unload failed (${response.status})`);
  notifyResidencyChange();
}

function graphNodes(type) {
  return (app.graph?._nodes || []).filter(
    (node) => node.comfyClass === type || node.type === type,
  );
}

function nodeValue(node, name) {
  return node.widgets?.find((widget) => widget.name === name)?.value;
}

function sectionHeading(text) {
  const heading = document.createElement("h3");
  heading.className = "xdit-sidebar-section-heading";
  heading.textContent = text;
  return heading;
}

function presetCard(node) {
  const card = document.createElement("section");
  card.className = "xdit-sidebar-card";
  const preset = nodeValue(node, "preset") || "none";
  const gpuTag = nodeValue(node, "gpu_tag") || "unknown GPU";
  const gpuCount = nodeValue(node, "gpu_count") || "?";
  const detected = nodeValue(node, "gpu_detection_info");
  card.append(sectionHeading(`Preset · node ${node.id}`));
  const selection = document.createElement("div");
  selection.textContent = `${preset} · ${gpuCount} × ${gpuTag}`;
  card.append(selection);
  if (detected) {
    const hardware = document.createElement("div");
    hardware.textContent = detected;
    hardware.className = "xdit-sidebar-muted";
    card.append(hardware);
  }
  return card;
}

function statusLine(payload, nodeId) {
  return payload?.node_status?.[String(nodeId)]?.text?.trim() || "";
}

function loaderCard(loader, node, payload, refresh) {
  const card = document.createElement("section");
  card.className = "xdit-sidebar-card";

  const heading = document.createElement("div");
  heading.className = "xdit-sidebar-card-heading";
  const title = document.createElement("span");
  title.textContent = loader.model || `Model node ${loader.node_id}`;
  heading.append(title);
  if (loader.warm || loader.parked) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "xdit-sidebar-card-action";
    button.textContent = t("xdit.residency.unload");
    button.addEventListener("click", async () => {
      button.disabled = true;
      try {
        await unload(loader.node_id);
        await refresh();
      } finally {
        button.disabled = false;
      }
    });
    heading.append(button);
  }
  card.append(heading);

  const cache = node?._xditPreviewPayload?.model_cache;
  if (cache) {
    const storage = document.createElement("div");
    storage.textContent = `${cache.path} · ${cache.source}`;
    storage.title = "Effective Hugging Face cache";
    card.append(storage);
  }

  const state = document.createElement("div");
  state.textContent = `${loader.state || "cold"} · ${gpuLabel(loader.gpus)}`;
  card.append(state);

  const phase = statusLine(payload, loader.node_id);
  if (phase) {
    const status = document.createElement("div");
    status.className = "xdit-sidebar-active";
    status.textContent = phase;
    card.append(status);
  }

  for (const row of loader.footprint || []) {
    const memory = document.createElement("div");
    memory.className = "xdit-sidebar-memory-row";
    memory.textContent = `GPU ${row.gpu} · ${gib(row.model_gib)} GiB model · ${gib(row.other_gib)} GiB other · ${gib(row.free_gib)} GiB free`;
    card.append(memory);
  }

  if (!phase && !loader.warm && !loader.parked) {
    const empty = document.createElement("div");
    empty.className = "xdit-sidebar-muted";
    empty.textContent = t("xdit.residency.notLoaded");
    card.append(empty);
  }

  return card;
}

function sampleCard(node, run, payload) {
  const card = document.createElement("section");
  card.className = "xdit-sidebar-card";
  card.append(sectionHeading(`Sample · node ${node.id}`));
  const rows = run?.rows || [];
  const phase = statusLine(payload, node.id);
  if (phase) {
    const status = document.createElement("div");
    status.className = "xdit-sidebar-active";
    status.textContent = phase;
    card.append(status);
  }
  if (!rows.length) {
    const empty = document.createElement("div");
    empty.className = "xdit-sidebar-muted";
    empty.textContent = t("xdit.residency.noRun");
    card.append(empty);
    return card;
  }
  for (const row of rows) {
    const memory = document.createElement("div");
    memory.className = "xdit-sidebar-memory-row";
    const activation =
      row.activation_gib === null || row.activation_gib === undefined
        ? ""
        : ` · ${gib(row.activation_gib)} GiB activations`;
    memory.textContent = `GPU ${row.gpu} · ${gib(row.peak_gib)} GiB peak${activation} · ${gib(row.device_free_gib)} GiB free at end`;
    card.append(memory);
  }
  return card;
}

function renderSidebar(element) {
  element.classList.add("xdit-sidebar");
  const toolbar = document.createElement("div");
  toolbar.className = "xdit-sidebar-toolbar";
  const refreshButton = document.createElement("button");
  refreshButton.type = "button";
  refreshButton.textContent = t("xdit.residency.refresh");
  const unloadButton = document.createElement("button");
  unloadButton.type = "button";
  unloadButton.textContent = t("xdit.residency.unloadAll");
  toolbar.append(refreshButton, unloadButton);
  const content = document.createElement("div");
  content.textContent = t("xdit.residency.loading");
  element.append(toolbar, content);

  let active = true;
  const refresh = async () => {
    const payload = await fetchResidency();
    if (!active) return;
    content.replaceChildren();
    if (!payload) {
      content.textContent = t("xdit.residency.error");
      return;
    }
    const presetNodes = graphNodes("xDiT.Preset");
    const modelNodes = graphNodes("xDiT.Model");
    const sampleNodes = graphNodes("xDiT.Sample");
    const loaders = payload.loaders || [];
    if (
      !presetNodes.length &&
      !modelNodes.length &&
      !sampleNodes.length &&
      !loaders.length
    ) {
      content.textContent = t("xdit.residency.empty");
      return;
    }
    if (presetNodes.length)
      content.append(sectionHeading(t("xdit.residency.presets")));
    for (const node of presetNodes) content.append(presetCard(node));

    const loadersById = new Map(
      loaders.map((loader) => [String(loader.node_id), loader]),
    );
    if (modelNodes.length || loaders.length)
      content.append(sectionHeading(t("xdit.residency.models")));
    for (const node of modelNodes) {
      const loader = loadersById.get(String(node.id)) || {
        node_id: node.id,
        model: nodeValue(node, "model"),
        state: "cold",
        gpus: String(nodeValue(node, "gpu_device_ids") || "")
          .split(",")
          .filter(Boolean),
        footprint: [],
      };
      loadersById.delete(String(node.id));
      content.append(loaderCard(loader, node, payload, refresh));
    }
    for (const loader of loadersById.values())
      content.append(loaderCard(loader, null, payload, refresh));

    if (sampleNodes.length)
      content.append(sectionHeading(t("xdit.residency.samples")));
    for (const node of sampleNodes) {
      content.append(
        sampleCard(node, payload.sample_runs?.[String(node.id)], payload),
      );
    }
  };
  refreshButton.addEventListener("click", refresh);
  unloadButton.addEventListener("click", async () => {
    unloadButton.disabled = true;
    try {
      await unload(null);
      await refresh();
    } finally {
      unloadButton.disabled = false;
    }
  });
  const changed = () => refresh();
  window.addEventListener("xdit:residency-changed", changed);
  const timer = setInterval(refresh, REFRESH_MS);
  refresh();
  return () => {
    active = false;
    clearInterval(timer);
    window.removeEventListener("xdit:residency-changed", changed);
  };
}

app.registerExtension({
  name: "xdit.gpu_residency_sidebar",
  setup() {
    ensureInfoStyles();
    app.extensionManager.registerSidebarTab({
      id: TAB_ID,
      icon: "pi pi-chart-bar",
      title: t("xdit.menu.title"),
      tooltip: t("xdit.residency.tooltip"),
      type: "custom",
      render: renderSidebar,
    });
  },
});
