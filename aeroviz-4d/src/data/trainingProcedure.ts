/**
 * trainingProcedure.ts
 * --------------------
 * The procedure's limits per candidate runway, as the exporters write them in a set of stage B (prior) or of stages C and
 * D (windows): `prior_training_export.procedure_block` — the outline of the region the procedure masks rule (inside the
 * FAF and the LPV cone, on the ground), the glidepath lower edge along the course, the DA point where the glidepath
 * reaches the decision height and the entry point at the FAF. Read here once for both readers; the runway the masks act
 * on is the one in force in the sentence on screen (`runwayInForce`).
 *
 * COMPUTES NO LIMIT: every coordinate is the exporter's; the reader checks the bookkeeping and adds the candidate
 * runway's HAE − MSL once, for the 3D scene.
 */

import { Reader } from "./trainingReader";
import type { TrainingCandidate, TrainingEvent } from "./trainingSample";

/** A point of the procedure's limits: the airport frame, the globe, and its height MSL and as Cesium draws it. */
export interface TrainingProcedurePoint {
  eM: number;
  nM: number;
  lat: number;
  lon: number;
  heightMslM: number;
  heightHaeM: number;
  /** Metres before the threshold along the course. */
  beforeThresholdM: number;
}

/** One candidate runway's final as the procedure masks read it. */
export interface TrainingProcedure {
  index: number;
  ident: string;
  fafBeforeThresholdM: number;
  /** The glidepath lower edge lies this far below the glidepath. */
  glidepathBelowM: number;
  /** The closed outline of the region the masks rule (inside the FAF and the LPV cone), on the ground. */
  region: { eM: number[]; nM: number[]; lat: number[]; lon: number[] };
  /** The glidepath lower edge along the course: a height at each point. */
  glidepathLowerEdge: { lat: number[]; lon: number[]; heightMslM: number[]; heightHaeM: number[]; beforeThresholdM: number[] };
  /** Where the glidepath reaches the decision height. */
  decision: TrainingProcedurePoint;
  /** The entry point at the FAF, at the entry height. */
  entry: TrainingProcedurePoint;
}

function procedurePoint(reader: Reader, haeMinusMslM: number): TrainingProcedurePoint {
  const heightMslM = reader.number("heightMslM");
  return {
    // the exporter writes a point's place as lists of one (`placed`)
    eM: reader.numbers("eM", 1)[0], nM: reader.numbers("nM", 1)[0], lat: reader.numbers("latDeg", 1)[0],
    lon: reader.numbers("lonDeg", 1)[0], heightMslM, heightHaeM: heightMslM + haeMinusMslM,
    beforeThresholdM: reader.number("beforeThresholdM"),
  };
}

export function parseProcedure(reader: Reader, candidates: TrainingCandidate[]): TrainingProcedure[] {
  const procedure = reader.children("procedure").map((item, position) => {
    const index = item.integer("index", 0, candidates.length - 1);
    if (index !== position) item.fail(`index is ${index}, expected ${position}: the limits are in the candidates' order`);
    const ident = item.string("ident");
    if (candidates[index].ident !== ident) item.fail(`ident ${ident} is not candidate ${index}'s (${candidates[index].ident})`);
    const hae = candidates[index].haeMinusMslM;
    const region = item.child("region");
    const ring = region.numbers("eM");
    if (ring.length < 4) region.fail("the outline has fewer than four points");
    const ringN = region.numbers("nM", ring.length);
    if (ring[0] !== ring[ring.length - 1] || ringN[0] !== ringN[ringN.length - 1]) {
      region.fail("the outline is not closed: its last point is not its first");
    }
    const edge = item.child("glidepathLowerEdge");
    const points = edge.numbers("latDeg").length;
    if (points < 2) edge.fail("the glidepath lower edge has fewer than two points");
    const heightMslM = edge.numbers("heightMslM", points);
    return {
      index, ident, fafBeforeThresholdM: item.number("fafBeforeThresholdM"), glidepathBelowM: item.number("glidepathBelowM"),
      region: { eM: ring, nM: ringN, lat: region.numbers("latDeg", ring.length), lon: region.numbers("lonDeg", ring.length) },
      glidepathLowerEdge: {
        lat: edge.numbers("latDeg", points), lon: edge.numbers("lonDeg", points), heightMslM,
        heightHaeM: heightMslM.map((value) => value + hae), beforeThresholdM: edge.numbers("beforeThresholdM", points),
      },
      decision: procedurePoint(item.child("decision"), hae), entry: procedurePoint(item.child("entry"), hae),
    };
  });
  if (procedure.length !== candidates.length) reader.fail(`procedure lists ${procedure.length} runways, the set has ${candidates.length} candidates`);
  return procedure;
}

/** The runway in force at Δ row ``row`` of a sentence: the last runway word (not a go-around) said at or before it, else the
 *  first one said; the masks act on this runway, whatever the observed flight landed on. */
export function runwayInForce(events: TrainingEvent[], row: number): number {
  const runways = events.filter((event) => event.column === 0 && event.says.column === "runway" && !event.says.goAround);
  const said = runways.filter((event) => event.row <= row);
  const event = said.length > 0 ? said[said.length - 1] : runways[0];
  return (event.says as { runwayIndex: number }).runwayIndex;
}
