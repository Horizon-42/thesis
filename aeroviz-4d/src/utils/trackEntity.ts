/**
 * trackEntity.ts
 * --------------
 * "Select this flight" in a list: the viewer's camera follows the flight's entity (found by its id — the flight_key —
 * in whichever data source holds it). Shared by the Flights table and the multi-aircraft result table.
 */

import type * as Cesium from "cesium";

/** Track the entity `id` (undefined — no tracking — when no data source has it). */
export function trackEntityById(viewer: Cesium.Viewer, id: string): void {
  let found: Cesium.Entity | undefined;
  for (let i = 0; i < viewer.dataSources.length; i += 1) {
    const entity = viewer.dataSources.get(i).entities.getById(id);
    if (entity) {
      found = entity;
      break;
    }
  }
  viewer.trackedEntity = found;
}
