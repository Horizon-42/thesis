/**
 * trainingProcedure.ts (scene)
 * ----------------------------
 * The procedure's limits in 3D, for a prior set (`useTrainingProcedureLayer`) and a window set (`useTrainingWindowLayer`)
 * alike, per candidate runway: the outline of the region the masks rule (inside the FAF and the LPV cone) on the ground,
 * the glidepath lower edge along the course (a line at its height), the DA point where the glidepath reaches the decision
 * height and the entry point at the FAF — every coordinate the exporter's (`data/trainingProcedure.ts`), none computed
 * here. Static entities.
 */

import * as Cesium from "cesium";
import type { TrainingProcedure } from "../data/trainingProcedure";
import { TRAINING_CANDIDATE_COLOR, TRAINING_DESIGNATED_COLOR } from "../utils/trainingWordColors";
import { airLine, entityGroup, groundLine, lonLatHeights, marker, planDegrees } from "./trainingEntities";

/** The glidepath lower edge's colour. */
const EDGE_COLOR = "#c084fc";
/** The decision altitude point's and the FAF entry point's colour. */
const POINT_COLOR = "#fbbf24";

const ENTITY = {
  region: (index: number) => `training-procedure-region-${index}`,
  edge: (index: number) => `training-procedure-edge-${index}`,
  decision: (index: number) => `training-procedure-decision-${index}`,
  entry: (index: number) => `training-procedure-entry-${index}`,
};

/** The procedure's limits of every candidate runway, the one the masks act on (``designated``) in the designated colour, the
 *  others in the candidates' grey. */
export function drawProcedureLimits(viewer: Cesium.Viewer, procedure: TrainingProcedure[], designated: number) {
  const group = entityGroup(viewer);
  for (const limit of procedure) {
    const mine = limit.index === designated;
    const hue = mine ? TRAINING_DESIGNATED_COLOR : TRAINING_CANDIDATE_COLOR;
    group.add(groundLine(ENTITY.region(limit.index), `Runway ${limit.ident}: the region the procedure's masks rule (inside the FAF and the LPV cone)`,
      planDegrees(limit.region), hue, !mine));
    const edge = limit.glidepathLowerEdge;
    group.add(airLine(ENTITY.edge(limit.index), `Runway ${limit.ident}: the glidepath lower edge (${limit.glidepathBelowM} m below the glidepath)`,
      Cesium.Cartesian3.fromDegreesArrayHeights(lonLatHeights({ lon: edge.lon, lat: edge.lat, altitudeHaeM: edge.heightHaeM })),
      EDGE_COLOR, mine ? 3 : 2, mine ? 1 : 0.6));
    const da = limit.decision;
    group.add(marker(ENTITY.decision(limit.index), `Runway ${limit.ident}: where the glidepath reaches the decision height`,
      Cesium.Cartesian3.fromDegrees(da.lon, da.lat, da.heightHaeM), POINT_COLOR, 8, `DA ${limit.ident} · ${da.heightMslM.toFixed(0)} m`));
    const entry = limit.entry;
    group.add(marker(ENTITY.entry(limit.index), `Runway ${limit.ident}: the entry point at the FAF`,
      Cesium.Cartesian3.fromDegrees(entry.lon, entry.lat, entry.heightHaeM), POINT_COLOR, 8,
      `FAF ${limit.ident} · ${entry.heightMslM.toFixed(0)} m`));
  }
  return group;
}

