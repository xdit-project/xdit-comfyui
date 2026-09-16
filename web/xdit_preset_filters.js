export function resolvePresetFilters(schema, gpuTag, gpuCount, preset) {
  const counts = (schema.gpu_counts_by_tag?.[gpuTag] || []).map(String);
  const requestedCount = String(gpuCount);
  const selectedCount = counts.includes(requestedCount)
    ? requestedCount
    : counts[0];
  const presets =
    schema.presets_by_tag_and_count?.[gpuTag]?.[String(selectedCount)] || [];
  return {
    counts,
    gpuCount: selectedCount,
    presets,
    preset: presets.includes(preset) ? preset : "none",
  };
}

export function presetTrigger(gpuTag, gpuCount, preset) {
  return `${gpuTag}:${gpuCount}:${preset}`;
}
