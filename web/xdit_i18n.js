import { app } from "../../scripts/app.js";

const ENGLISH = {
  "xdit.menu.title": "xDiT",
  "xdit.menu.unloadAll": "Unload all xDiT models",
  "xdit.residency.tooltip": "Inspect xDiT workers and GPU memory usage",
  "xdit.residency.empty": "No xDiT models are loaded.",
  "xdit.residency.refresh": "Refresh",
  "xdit.residency.unload": "Unload model",
  "xdit.residency.unloadAll": "Unload all models",
  "xdit.residency.loading": "Loading residency information…",
  "xdit.residency.error": "Could not load residency information.",
  "xdit.residency.models": "Models",
  "xdit.residency.noRun": "No completed run yet.",
  "xdit.residency.notLoaded": "Not loaded.",
  "xdit.residency.presets": "Presets",
  "xdit.residency.samples": "Samples",
  "xdit.tooltip.disclosure": "Show or hide {group} settings.",
  "xdit.tooltip.residency":
    "After the last Sample using this Model: keep_gpu retains the worker; park_cpu attempts to move supported layouts to system RAM and leaves it retained if parking fails; release stops the worker.",
  "xdit.tooltip.residencyUnavailable":
    "CPU parking is unavailable for this layout: {reason}.",
  "xdit.tooltip.distilledWeights":
    "LightX2V transformer files for Wan 2.2 Distilled I2V.",
  "xdit.tooltip.denoiserCache": "Cache values for {transformer}. {description}",
  "xdit.notice.released": "Released GPU memory for Model node(s): {nodes}.",
  "xdit.notice.noneToRelease": "No resident xDiT workers to release.",
  "xdit.error.releaseFailed":
    "Release request failed. Check the ComfyUI server log.",
  "xdit.group.video.label": "Video",
  "xdit.group.video.description": "Video frame and guidance settings.",
  "xdit.group.vae.label": "VAE",
  "xdit.group.vae.description": "VAE decoding and tiling settings.",
};

export function t(key, values = {}) {
  const i18n = app.extensionManager?.i18n;
  let result;
  if (typeof i18n?.t === "function") {
    const translated = i18n.t(key);
    if (translated && translated !== key) result = translated;
  }
  result ??= ENGLISH[key] ?? key;
  return Object.entries(values).reduce(
    (text, [name, value]) => text.replaceAll(`{${name}}`, String(value)),
    result,
  );
}
