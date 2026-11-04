export function registeredNodeName(nodeType, nodeData) {
  return nodeData?.name ?? nodeData?.class_type ?? nodeType?.comfyClass;
}

/** Install Nodes 1.0/2.0-compatible construction and restore hooks once per class. */
export function installNodeLifecycle(
  nodeType,
  key,
  { onCreated, beforeConfigure, onConfigured } = {},
) {
  const createdGuard = `_xdit${key}NodeCreatedHooked`;
  if (onCreated && !nodeType.prototype[createdGuard]) {
    nodeType.prototype[createdGuard] = true;
    const previous = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = previous?.apply(this, args);
      onCreated(this);
      return result;
    };
  }

  const configure = nodeType.prototype.configure;
  const configureGuard = `_xdit${key}ConfigureHooked`;
  if (
    (beforeConfigure || onConfigured) &&
    typeof configure === "function" &&
    !nodeType.prototype[configureGuard]
  ) {
    nodeType.prototype[configureGuard] = true;
    nodeType.prototype.configure = function (...args) {
      beforeConfigure?.(this, ...args);
      const result = configure.apply(this, args);
      onConfigured?.(this, ...args);
      return result;
    };
  }
}
