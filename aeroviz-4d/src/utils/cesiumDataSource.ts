/**
 * cesiumDataSource.ts
 * -------------------
 * Shared helper for the one Cesium gotcha both trajectory layers hit.
 *
 * Cesium renders a DataSource's entities the **instant** it is added to the viewer, and it
 * does NOT apply the CZML's entity-level `show: false`. So a layer that loads many entities
 * and then shows only a SAMPLED subset (the observed-trajectory layer and the optimizer-
 * comparison layer both do this) flashes EVERY entity for one frame — far more than the
 * sample, in a muddy overlap — until its React pass runs and hides the non-sampled ones.
 *
 * The systematic fix: never add such a source visible. Add it hidden via `addDataSourceHidden`,
 * and let the layer's own visibility/sampling pass set each entity's `show` and THEN reveal the
 * source (`ds.show = …`). One frame of nothing is invisible; one frame of everything is the bug.
 */

import * as Cesium from "cesium";

/**
 * Add a data source to the viewer with `show = false`, so none of its entities render until
 * the caller's visibility/sampling pass reveals it. Use this for any layer that adds a source
 * holding more entities than it intends to display at once.
 */
export function addDataSourceHidden(viewer: Cesium.Viewer, ds: Cesium.DataSource): void {
  viewer.dataSources.add(ds);
  ds.show = false;
}
