import {
  airportChartsIndexUrl,
  airportProcedureDetailUrl,
  airportProcedureDetailsIndexUrl,
} from "./airportData";

export interface ProcedureDetailsIndexProcedureSummary {
  procedureUid: string;
  procedureIdent: string;
  chartName: string;
  procedureFamily: string;
  variant: string | null;
  approachModes: string[];
  runwayIdent: string;
  defaultBranchId: string;
}

export interface ProcedureDetailsIndexRunwaySummary {
  runwayIdent: string;
  chartName: string;
  procedureUids: string[];
  procedures: ProcedureDetailsIndexProcedureSummary[];
}

export interface ProcedureDetailsIndexManifest {
  airport: string;
  airportName: string;
  sourceCycle: string | null;
  researchUseOnly: boolean;
  runways: ProcedureDetailsIndexRunwaySummary[];
}

export interface ProcedureChartManifestEntry {
  chartId: string;
  procedureUid: string | null;
  procedureIdent: string | null;
  runwayIdent: string | null;
  title: string;
  originalFileName: string;
  sourcePath: string;
  url: string;
}

export interface ProcedureChartsManifest {
  airport: string;
  researchUseOnly: boolean;
  charts: ProcedureChartManifestEntry[];
}

export interface ProcedureDetailFix {
  fixId: string;
  ident: string;
  kind: string;
  position: { lon: number; lat: number } | null;
  elevationFt: number | null;
  roleHints: string[];
  sourceRefs: string[];
}

export interface ProcedureDetailLeg {
  legId: string;
  sequence: number;
  segmentType: string;
  path: {
    pathTerminator: string;
    constructionMethod: string;
    startFixRef: string | null;
    endFixRef: string;
    courseDeg?: number;
    turnDirection?: "LEFT" | "RIGHT";
    arcRadiusNm?: number;
    centerFixRef?: string;
    centerLatDeg?: number;
    centerLonDeg?: number;
  };
  termination: {
    kind: string;
    fixRef: string | null;
  };
  constraints: {
    altitude: {
      qualifier: string;
      valueFt: number;
      rawText: string;
    } | null;
    speedKt: number | null;
    geometryAltitudeFt: number | null;
  };
  roleAtEnd: string;
  sourceRefs: string[];
  quality: {
    status: string;
    sourceLine: number;
    renderedInPlanView?: boolean;
  };
}

export interface ProcedureDetailBranch {
  branchId: string;
  branchKey?: string;
  branchIdent: string;
  procedureType?: string;
  transitionIdent?: string | null;
  branchRole: string;
  sequenceOrder: number;
  mergeFixRef: string | null;
  continuesWithBranchId: string | null;
  defaultVisible: boolean;
  warnings: string[];
  legs: ProcedureDetailLeg[];
}

export interface ProcedureDetailVerticalProfile {
  profileId: string;
  appliesToModes: string[];
  branchId: string;
  fromFixRef: string | null;
  toFixRef: string | null;
  basis: string;
  glidepathAngleDeg: number | null;
  thresholdCrossingHeightFt: number | null;
  constraintSamples: Array<{
    fixRef: string;
    ident: string;
    role: string;
    distanceFromStartM: number;
    altitudeFt: number | null;
    geometryAltitudeFt: number | null;
    sourceLine: number;
  }>;
  warnings: string[];
}

export interface ProcedureDetailDocument {
  schemaVersion: string;
  modelType: string;
  procedureUid: string;
  provenance: {
    assemblyMode: string;
    researchUseOnly: boolean;
    sources: Array<{
      sourceId: string;
      kind: string;
      cycle?: string | null;
      path?: string;
    }>;
    warnings: string[];
  };
  airport: {
    icao: string;
    faa: string;
    name: string;
  };
  runway: {
    ident: string | null;
    landingThresholdFixRef: string | null;
    threshold: {
      lon: number;
      lat: number;
      elevationFt: number | null;
    } | null;
  };
  procedure: {
    procedureType: string;
    procedureFamily: string;
    procedureIdent: string;
    chartName: string;
    variant: string | null;
    runwayIdent: string | null;
    baseBranchIdent: string;
    approachModes: string[];
  };
  fixes: ProcedureDetailFix[];
  branches: ProcedureDetailBranch[];
  verticalProfiles: ProcedureDetailVerticalProfile[];
  validation: {
    expectedRunwayIdent: string | null;
    expectedIF: string | null;
    expectedFAF: string | null;
    expectedMAPt: string | null;
    expectedMissedHoldFix: string | null;
    knownSimplifications: string[];
  };
  displayHints: {
    nominalSpeedKt: number;
    defaultVisibleBranchIds: string[];
    tunnelDefaults: {
      lateralHalfWidthNm: number;
      verticalHalfHeightFt: number;
      sampleSpacingM: number;
      mode: string;
    };
  };
}

export function procedureDetailsIndexUrl(airportCode: string): string {
  return airportProcedureDetailsIndexUrl(airportCode);
}

export function procedureDetailsDocumentUrl(airportCode: string, procedureUid: string): string {
  return airportProcedureDetailUrl(airportCode, procedureUid);
}

export function procedureChartsIndexUrl(airportCode: string): string {
  return airportChartsIndexUrl(airportCode);
}
