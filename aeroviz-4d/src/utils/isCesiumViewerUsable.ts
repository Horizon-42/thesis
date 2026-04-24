import type * as Cesium from "cesium";

/**
 * Real Cesium viewers expose `isDestroyed()`. Some test doubles do not.
 * Treat mocks without that method as usable so runtime guards stay test-friendly.
 */
export function isCesiumViewerUsable(
  viewer: Cesium.Viewer | null | undefined,
): viewer is Cesium.Viewer {
  if (!viewer) return false;
  return typeof viewer.isDestroyed !== "function" || !viewer.isDestroyed();
}
