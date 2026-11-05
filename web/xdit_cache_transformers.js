/**
 * Point the per-denoiser cache groups at the transformers the selected model has.
 *
 * Wan 2.2 denoises with a high-noise expert and a low-noise refiner and caches them
 * differently, so each denoiser gets its own group of cache widgets. The schema declares
 * one group per denoiser xDiT can cache; a model with a single transformer hides the
 * later ones, and the first group then applies to that one transformer.
 */

import { t } from "./xdit_i18n.js";

const GROUP_PREFIX = "cache_denoiser_";

export function denoiserGroupIndex(group) {
  const id = String(group?.id ?? "");
  if (!id.startsWith(GROUP_PREFIX)) return null;
  const index = Number.parseInt(id.slice(GROUP_PREFIX.length), 10);
  return Number.isInteger(index) ? index : null;
}

export function isDenoiserGroup(group) {
  return denoiserGroupIndex(group) != null;
}

/** Name the transformer each group drives, so the heading is not just a number. */
export function describeDenoiserGroups(
  node,
  groups,
  rows,
  { setWidgetTooltip, headingName },
) {
  const entries = Array.isArray(rows) ? rows : [];
  for (const group of groups || []) {
    const index = denoiserGroupIndex(group);
    if (index == null) continue;
    const row = entries[index - 1];
    if (!row) continue;
    const tooltip = t("xdit.tooltip.denoiserCache", {
      transformer: row.transformer,
      description: group.description ?? "",
    }).trim();
    setWidgetTooltip(node, headingName(group), tooltip);
  }
}
