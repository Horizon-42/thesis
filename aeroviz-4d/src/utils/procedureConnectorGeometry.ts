import type { BuildDiagnostic, ProcedureSegment, SourceRef } from "../data/procedurePackage";
import type { PolylineGeometry3D } from "./procedureSegmentGeometry";
import {
  buildSampledCenterline,
  buildVariableWidthRibbon,
  sampleStationValues,
  type VariableWidthRibbonGeometry,
} from "./procedureSurfaceGeometry";
import { clamp } from "./procedureGeoMath";

export interface AlignedLnavConnectorOptions {
  beforePfafNm?: number;
  afterPfafNm?: number;
  intermediatePrimaryHalfWidthNm?: number;
  finalPrimaryHalfWidthNm?: number;
  intermediateSecondaryOuterHalfWidthNm?: number;
  finalSecondaryOuterHalfWidthNm?: number;
  samplingStepNm?: number;
}

export interface AlignedLnavConnectorGeometry {
  geometryId: string;
  segmentId: string;
  connectorType: "ALIGNED_LNAV_INTERMEDIATE_TO_FINAL";
  centerline: PolylineGeometry3D;
  primary: VariableWidthRibbonGeometry;
  secondaryOuter: VariableWidthRibbonGeometry;
  anchors: {
    pfafStationNm: number;
    beforePfafNm: number;
    afterPfafNm: number;
    startStationNm: number;
    endStationNm: number;
  };
}

const DEFAULT_ALIGNED_CONNECTOR_OPTIONS: Required<AlignedLnavConnectorOptions> = {
  beforePfafNm: 2,
  afterPfafNm: 1,
  intermediatePrimaryHalfWidthNm: 2,
  finalPrimaryHalfWidthNm: 0.6,
  intermediateSecondaryOuterHalfWidthNm: 3,
  finalSecondaryOuterHalfWidthNm: 0.9,
  samplingStepNm: 0.25,
};

function diagnostic(
  code: BuildDiagnostic["code"],
  message: string,
  severity: BuildDiagnostic["severity"],
  segmentId?: string,
  sourceRefs: SourceRef[] = [],
): BuildDiagnostic {
  return { code, message, severity, segmentId, sourceRefs };
}

function taperedWidthAtStation(
  stationNm: number,
  startStationNm: number,
  endStationNm: number,
  startHalfWidthNm: number,
  endHalfWidthNm: number,
): number {
  const ratio = clamp((stationNm - startStationNm) / (endStationNm - startStationNm), 0, 1);
  return startHalfWidthNm + (endHalfWidthNm - startHalfWidthNm) * ratio;
}

export function buildAlignedLnavConnector(
  finalSegment: ProcedureSegment,
  finalCenterline: PolylineGeometry3D,
  options: AlignedLnavConnectorOptions = {},
): { geometry: AlignedLnavConnectorGeometry | null; diagnostics: BuildDiagnostic[] } {
  const opts = { ...DEFAULT_ALIGNED_CONNECTOR_OPTIONS, ...options };
  const diagnostics: BuildDiagnostic[] = [];

  if (finalCenterline.geoPositions.length < 2 || finalCenterline.geodesicLengthNm <= 0) {
    diagnostics.push(
      diagnostic(
        "CONNECTOR_NOT_CONSTRUCTIBLE",
        `${finalSegment.segmentId}: aligned LNAV connector requires a positioned final centerline.`,
        "ERROR",
        finalSegment.segmentId,
        finalSegment.sourceRefs,
      ),
    );
    return { geometry: null, diagnostics };
  }

  const startStationNm = -opts.beforePfafNm;
  const endStationNm = opts.afterPfafNm;
  const stations = sampleStationValues(startStationNm, endStationNm, opts.samplingStepNm);
  const centerline = buildSampledCenterline(
    finalCenterline,
    startStationNm,
    endStationNm,
    opts.samplingStepNm,
  );

  const primary = buildVariableWidthRibbon(
    `${finalSegment.segmentId}:aligned-lnav-connector-primary`,
    centerline,
    stations,
    (stationNm) =>
      taperedWidthAtStation(
        stationNm,
        startStationNm,
        endStationNm,
        opts.intermediatePrimaryHalfWidthNm,
        opts.finalPrimaryHalfWidthNm,
      ),
  );
  const secondaryOuter = buildVariableWidthRibbon(
    `${finalSegment.segmentId}:aligned-lnav-connector-secondary-outer`,
    centerline,
    stations,
    (stationNm) =>
      taperedWidthAtStation(
        stationNm,
        startStationNm,
        endStationNm,
        opts.intermediateSecondaryOuterHalfWidthNm,
        opts.finalSecondaryOuterHalfWidthNm,
      ),
  );

  return {
    geometry: {
      geometryId: `${finalSegment.segmentId}:aligned-lnav-connector`,
      segmentId: finalSegment.segmentId,
      connectorType: "ALIGNED_LNAV_INTERMEDIATE_TO_FINAL",
      centerline,
      primary,
      secondaryOuter,
      anchors: {
        pfafStationNm: 0,
        beforePfafNm: opts.beforePfafNm,
        afterPfafNm: opts.afterPfafNm,
        startStationNm,
        endStationNm,
      },
    },
    diagnostics,
  };
}
