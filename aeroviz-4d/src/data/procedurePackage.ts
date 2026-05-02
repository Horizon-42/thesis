import type { ProcedureDetailDocument } from "./procedureDetails";

export type ProcedureFamily =
  | "RNAV_GPS"
  | "RNAV_RNP"
  | "RNP_AR_APCH"
  | "SID"
  | "STAR"
  | "UNKNOWN";

export interface ProcedurePackage {
  packageId: string;
  airportId: string;
  runwayId: string | null;
  procedureId: string;
  procedureName: string;
  procedureFamily: ProcedureFamily;
  sourceMeta: SourceMeta;
  branches: ProcedurePackageBranch[];
  segments: ProcedureSegment[];
  legs: ProcedurePackageLeg[];
  sharedFixes: ProcedurePackageFix[];
  validationConfig: ValidationConfig;
  diagnostics: BuildDiagnostic[];
  legacyDocument: {
    schemaVersion: ProcedureDetailDocument["schemaVersion"];
    modelType: ProcedureDetailDocument["modelType"];
    procedureUid: ProcedureDetailDocument["procedureUid"];
  };
}

export interface SourceMeta {
  cifpCycle: string | null;
  sourceFiles: string[];
  chartLinks: string[];
  notes: string[];
  authority: "FAA_8260_58D" | "AEROVIZ_SOURCE";
}

export interface ProcedurePackageFix {
  fixId: string;
  ident: string;
  role: FixRole[];
  latDeg: number | null;
  lonDeg: number | null;
  altFtMsl: number | null;
  isFlyOver?: boolean;
  isFlyBy?: boolean;
  annotations: string[];
  sourceRefs: SourceRef[];
}

export type FixRole =
  | "IAF"
  | "IF"
  | "PFAF"
  | "FAF"
  | "MAP"
  | "MAHF"
  | "RWY"
  | "FROP"
  | "UNKNOWN";

export interface AltitudeConstraint {
  kind: "AT" | "AT_OR_ABOVE" | "AT_OR_BELOW" | "WINDOW" | "UNKNOWN";
  minFtMsl?: number;
  maxFtMsl?: number;
  sourceText?: string;
}

export interface SpeedConstraint {
  maxKias?: number;
  minKias?: number;
  sourceText?: string;
}

export type LegType = "IF" | "TF" | "RF" | "DF" | "CA" | "HM" | "HA" | "HF" | "UNSUPPORTED";

export interface ProcedurePackageLeg {
  legId: string;
  segmentId: string;
  legType: LegType;
  rawPathTerminator: string;
  startFixId: string | null;
  endFixId: string | null;
  inboundCourseDeg?: number;
  outboundCourseDeg?: number;
  turnDirection?: "LEFT" | "RIGHT";
  arcRadiusNm?: number;
  centerLatDeg?: number;
  centerLonDeg?: number;
  requiredAltitude: AltitudeConstraint | null;
  requiredSpeed: SpeedConstraint | null;
  navSpecAtLeg: NavSpecCode;
  xttNm: number;
  attNm: number;
  secondaryEnabled: boolean;
  notes: string[];
  sourceRefs: SourceRef[];
  legacy: {
    sequence: number;
    constructionMethod: string;
    roleAtEnd: string;
    qualityStatus: string;
    renderedInPlanView: boolean;
  };
}

export type NavSpecCode =
  | "RNAV_1"
  | "RNAV_2"
  | "RNP_APCH"
  | "A_RNP_1"
  | "A_RNP_0_3"
  | "RNP_AR_0_3"
  | "RNP_AR_0_2"
  | "RNP_AR_0_1"
  | "UNKNOWN";

export type SegmentType =
  | "FEEDER"
  | "INITIAL"
  | "INTERMEDIATE"
  | "FINAL_LNAV"
  | "FINAL_LP"
  | "FINAL_LNAV_VNAV"
  | "FINAL_LPV"
  | "FINAL_GLS"
  | "FINAL_RNP_AR"
  | "MISSED_S1"
  | "MISSED_S2"
  | "HOLDING"
  | "UNKNOWN";

export interface ProcedureSegment {
  segmentId: string;
  branchId: string;
  segmentType: SegmentType;
  navSpec: NavSpecCode;
  startFixId: string | null;
  endFixId: string | null;
  legIds: string[];
  xttNm: number;
  attNm: number;
  secondaryEnabled: boolean;
  widthChangeMode: "LINEAR_TAPER" | "ABRUPT" | "SPLAY_30" | "NONE";
  transitionRule: TransitionRule | null;
  verticalRule: VerticalRule | null;
  constructionFlags: ConstructionFlags;
  sourceRefs: SourceRef[];
  legacy: {
    rawSegmentType: string;
    sequenceRange: [number, number];
  };
}

export interface TransitionRule {
  kind:
    | "INTERMEDIATE_TO_FINAL_LNAV"
    | "INTERMEDIATE_TO_FINAL_LP"
    | "INTERMEDIATE_TO_FINAL_LNAV_VNAV"
    | "INTERMEDIATE_TO_FINAL_LPV_GLS"
    | "RNP_CHANGE_ABRUPT"
    | "MODE_CHANGE_30NM"
    | "MISSED_SECTION_SPLIT";
  anchorFixId?: string;
  beforeNm?: number;
  afterNm?: number;
  notes: string[];
}

export interface VerticalRule {
  kind:
    | "NONE"
    | "LEVEL_ROC"
    | "BARO_GLIDEPATH"
    | "LPV_GLS_SURFACES"
    | "MISSED_CLIMB_SURFACE"
    | "RNP_AR_VERTICAL";
  gpaDeg?: number;
  tchFt?: number;
  mdaFtMsl?: number;
  daFtMsl?: number;
  climbGradientFtPerNm?: number;
}

export interface ConstructionFlags {
  hasOffsetConstruction?: boolean;
  hasTurnAtIf?: boolean;
  hasRfToPfaf?: boolean;
  isBasicT?: boolean;
  hasHilpt?: boolean;
  isTurningMissedApproach?: boolean;
  foTurnNotAllowed?: boolean;
  collapsedApproachModes?: string[];
}

export interface ProcedurePackageBranch {
  branchId: string;
  runwayId: string | null;
  branchName: string;
  branchRole: "LEFT_IAF" | "RIGHT_IAF" | "STRAIGHT_IN" | "MISSED" | "TRANSITION" | "HOLDING";
  segmentIds: string[];
  mergeToBranchId?: string;
  divergesFromBranchId?: string;
  legacy: {
    sourceBranchId: string;
    branchIdent: string;
    branchKey: string;
    defaultVisible: boolean;
    mergeFixRef: string | null;
    continuesWithBranchId: string | null;
  };
}

export interface ValidationConfig {
  expectedRunwayIdent: string | null;
  expectedIF: string | null;
  expectedFAF: string | null;
  expectedMAP: string | null;
  expectedMissedHoldFix: string | null;
  knownSimplifications: string[];
}

export interface SourceRef {
  docId: "FAA_ORDER_8260_58D" | "AEROVIZ_SOURCE";
  chapter?: string;
  section?: string;
  paragraph?: string;
  figure?: string;
  formula?: string;
  pdfPage?: number;
  rawRef?: string;
}

export interface BuildDiagnostic {
  severity: "INFO" | "WARN" | "ERROR";
  segmentId?: string;
  legId?: string;
  code:
    | "RF_RADIUS_MISSING"
    | "RNP_CHANGE_INSIDE_FAS"
    | "FO_NOT_ALLOWED_RNP_AR"
    | "FINAL_HAS_TURN"
    | "CONNECTOR_NOT_CONSTRUCTIBLE"
    | "SECONDARY_DISABLED_BY_RULE"
    | "SOURCE_INCOMPLETE"
    | "DEFAULT_NAV_SPEC"
    | "DEFAULT_TOLERANCE"
    | "MODE_COLLAPSED_TO_LNAV"
    | "UNSUPPORTED_LEG_TYPE"
    | "TURN_VISUAL_FILL_ONLY"
    | "TURNING_MISSED_UNIMPLEMENTED"
    | "FINAL_VERTICAL_SURFACE_UNIMPLEMENTED"
    | "CA_ENDPOINT_NOT_CONSTRUCTIBLE";
  message: string;
  sourceRefs: SourceRef[];
}
