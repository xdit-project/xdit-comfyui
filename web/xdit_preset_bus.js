import { originNodeFromInput, widgetValue } from "./xdit_widget_state.js";

export const PRESET_NODE = "xDiT.Preset";
export const PRESET_NONE = "none";

/** Identifies a selection, so a repeat of the same one is not re-applied. */
export function presetTrigger(gpuTag, gpuCount, preset) {
  return `${gpuTag}:${gpuCount}:${preset}`;
}

export function isPresetNode(node) {
  return (
    !!node && (node.comfyClass === PRESET_NODE || node.type === PRESET_NODE)
  );
}

/** The Preset node feeding this node's `preset` input, if any. */
export function connectedPresetNode(node) {
  const origin = originNodeFromInput(node, "preset");
  return isPresetNode(origin) ? origin : null;
}

export function presetSelection(presetNode) {
  const gpuTag = widgetValue(presetNode, "gpu_tag", "gfx1201");
  const gpuCount = Number(widgetValue(presetNode, "gpu_count", 1));
  const preset = widgetValue(presetNode, "preset", PRESET_NONE);
  return {
    gpuTag,
    gpuCount,
    preset,
    trigger: presetTrigger(gpuTag, gpuCount, preset),
  };
}

export function connectedPresetTrigger(node) {
  const presetNode = connectedPresetNode(node);
  return presetNode ? presetSelection(presetNode).trigger : null;
}

const inFlight = new Map();

/**
 * Fetch a preset preview once per selection.
 *
 * The Preset node needs the reference images and every connected Sample node needs
 * the generation defaults out of the same response, so concurrent callers share one
 * request instead of each issuing their own.
 */
export function fetchPresetPreview({ gpuTag, gpuCount, preset, trigger }) {
  const pending = inFlight.get(trigger);
  if (pending) return pending;

  const request = fetch("/xdit/preset/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ gpu_tag: gpuTag, gpu_count: gpuCount, preset }),
  })
    .then((response) => {
      if (!response.ok)
        throw new Error(`preset preview failed (${response.status})`);
      return response.json();
    })
    .finally(() => inFlight.delete(trigger));

  inFlight.set(trigger, request);
  return request;
}
