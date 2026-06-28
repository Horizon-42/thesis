/**
 * Per-chart calibration: the page positions (PDF.js page units, scale 1) of a
 * few named fixes, measured once against the rendered chart. The lon/lat of each
 * fix comes from the procedure detail document at runtime, so this only stores
 * ident + page position. From these the lon/lat -> page-unit projection is fit
 * (see `utils/chartProjection`).
 *
 * To (re)measure a chart: open it in the annotated view, toggle "Calibrate",
 * click each labelled fix on the chart, read the logged page coords from the
 * console, and paste an entry here. Use >= 3 well-spread, non-collinear fixes.
 */
export interface ChartReferenceMeasurement {
  ident: string;
  pageX: number;
  pageY: number;
}

export interface ChartCalibration {
  references: ChartReferenceMeasurement[];
}

export const CHART_CALIBRATIONS: Record<string, ChartCalibration> = {
  // KRDU RNAV (GPS) Y RWY 5L — measured against 00516RY5L.pdf (page 1).
  "KRDU-R05LY-RW05L": {
    references: [
      { ident: "OTTOS", pageX: 67.6, pageY: 377.3 },
      { ident: "CHWDR", pageX: 122.4, pageY: 363.8 },
      { ident: "BOULE", pageX: 145.2, pageY: 331.7 },
      { ident: "SCHOO", pageX: 160.8, pageY: 303.9 },
      { ident: "WEPAS", pageX: 187.2, pageY: 284.7 },
      { ident: "RW05L", pageX: 226.3, pageY: 239.1 },
      { ident: "DUHAM", pageX: 180.8, pageY: 189.2 },
    ],
  },
};

export function chartCalibrationFor(procedureUid: string): ChartCalibration | null {
  return CHART_CALIBRATIONS[procedureUid] ?? null;
}
