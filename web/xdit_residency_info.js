import { api } from "../../scripts/api.js";

let stylesLoaded = false;

export function ensureInfoStyles() {
  if (stylesLoaded) return;
  stylesLoaded = true;
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = new URL("./xdit_residency_info.css", import.meta.url).href;
  document.head.appendChild(link);
}

export async function fetchResidency(sampleNodeId) {
  const query = sampleNodeId
    ? `?sample_node_id=${encodeURIComponent(sampleNodeId)}`
    : "";
  try {
    const response = await api.fetchApi(`/xdit/residency${query}`);
    if (!response.ok) return null;
    return await response.json();
  } catch (error) {
    console.debug("[xdit] residency fetch failed", error);
    return null;
  }
}

export function notifyResidencyChange() {
  if (
    typeof globalThis.dispatchEvent === "function" &&
    typeof globalThis.CustomEvent === "function"
  ) {
    globalThis.dispatchEvent(new CustomEvent("xdit:residency-changed"));
  }
}
