import type {
  ProcedureDetailBranch,
  ProcedureDetailDocument,
  ProcedureDetailFix,
} from "./procedureDetails";

export interface TermReference {
  label: string;
  url: string;
}

export interface TermDetails {
  definition: string;
  references: TermReference[];
}

export interface TermGroup {
  id: string;
  title: string;
  terms: string[];
}

const FAA_AIM_PBN_REFERENCE: TermReference = {
  label: "FAA AIM 1-2-1 / 1-2-2",
  url: "https://www.faa.gov/air_traffic/publications/atpubs/aim_html/chap1_section_2.html",
};
const FAA_AIM_APPROACH_REFERENCE: TermReference = {
  label: "FAA AIM 5-4",
  url: "https://www.faa.gov/air_traffic/publications/atpubs/aim_html/chap5_section_4.html",
};
const FAA_PCG_SEGMENTS_REFERENCE: TermReference = {
  label: "FAA Pilot/Controller Glossary",
  url: "https://www.faa.gov/air_traffic/publications/atpubs/pcg_html/",
};
const FAA_CIFP_REFERENCE: TermReference = {
  label: "FAA CIFP / ARINC 424",
  url: "https://www.faa.gov/air_traffic/flight_info/aeronav/digital_products/cifp/",
};
const FAA_PBN_ORDER_REFERENCE: TermReference = {
  label: "FAA Order 8260.58D",
  url: "https://www.faa.gov/regulations_policies/orders_notices/index.cfm/go/document.information/documentID/1043458",
};
const LOCAL_ARINC_APPROACH_ROUTE_TYPE_REFERENCE: TermReference = {
  label: "Local ARINC 424-23 Table 5-8 note",
  url: "/data/reference/arinc424-approach-route-types.md",
};
const FAA_RNAV_LEG_TYPES_REFERENCE: TermReference = {
  label: "FAA RNAV leg types",
  url: "https://www.faa.gov/air_traffic/publications/atpubs/atbarc/03-5.htm",
};

const DEFAULT_IDENTIFIER_DETAILS: TermDetails = {
  definition:
    "Identifier from the selected procedure data. It names a fix, procedure, branch, or local data value rather than a glossary concept.",
  references: [],
};

const FIX_ROLE_TERMS = ["IAF", "IF", "FAF", "MAPT", "MAHF"];
const PROCEDURE_MODE_TERMS = [
  "LPV",
  "LNAV/VNAV",
  "LNAV",
  "RNAV",
  "RNP",
  "RNP AR",
  "RNAV(GPS)",
  "RNAV(RNP)",
  "OCS",
  "OEA",
  "GPA",
  "TCH",
  "XTT",
  "ATT",
];
const APPROACH_ROUTE_TYPE_TERMS = ["A", "H", "R"];

const TERM_DISPLAY_NAMES: Record<string, string> = {
  IAF: "Initial Approach Fix (IAF)",
  IF: "Intermediate Fix (IF)",
  FAF: "Final Approach Fix (FAF)",
  MAPT: "Missed Approach Point (MAPt)",
  MAHF: "Missed Approach Holding Fix (MAHF)",
  LPV: "Localizer Performance with Vertical Guidance (LPV)",
  "LNAV/VNAV": "Lateral Navigation / Vertical Navigation (LNAV/VNAV)",
  LNAV: "Lateral Navigation (LNAV)",
  RNAV: "Area Navigation (RNAV)",
  "RNAV(GPS)": "Area Navigation using GPS/WAAS (RNAV(GPS))",
  "RNAV(RNP)": "Area Navigation with Required Navigation Performance (RNAV(RNP))",
  RNP: "Required Navigation Performance (RNP)",
  "RNP AR": "Required Navigation Performance Authorization Required (RNP AR)",
  OCS: "Obstacle Clearance Surface (OCS)",
  OEA: "Obstacle Evaluation Area (OEA)",
  GPA: "Glidepath Angle (GPA)",
  TCH: "Threshold Crossing Height (TCH)",
  XTT: "Cross Track Tolerance (XTT)",
  ATT: "Along Track Tolerance (ATT)",
  IF_TERMINATOR: "Initial Fix Path Terminator (IF)",
  TF_TERMINATOR: "Track to Fix Path Terminator (TF)",
  DF_TERMINATOR: "Direct to Fix Path Terminator (DF)",
  CA_TERMINATOR: "Course to Altitude Path Terminator (CA)",
  CF_TERMINATOR: "Course to Fix Path Terminator (CF)",
  HM_TERMINATOR: "Hold to Manual Termination Path Terminator (HM)",
  HF_TERMINATOR: "Hold to Fix Path Terminator (HF)",
  A: "Approach Route Type A",
  H: "Approach Route Type H",
  R: "Approach Route Type R",
};

const TERM_DETAILS: Record<string, TermDetails> = {
  IAF: {
    definition:
      "The Initial Approach Fix is where an aircraft can join the published approach from the wider route network.",
    references: [FAA_AIM_APPROACH_REFERENCE, FAA_PCG_SEGMENTS_REFERENCE],
  },
  IF: {
    definition:
      "The Intermediate Fix is a point that lines the aircraft up and settles it before final approach.",
    references: [FAA_AIM_APPROACH_REFERENCE, FAA_PCG_SEGMENTS_REFERENCE],
  },
  FAF: {
    definition:
      "The Final Approach Fix is the point where the final descent toward the runway is established.",
    references: [FAA_AIM_APPROACH_REFERENCE, FAA_PCG_SEGMENTS_REFERENCE],
  },
  MAPT: {
    definition:
      "The Missed Approach Point is where the published missed approach begins if the runway is not safely in view.",
    references: [FAA_AIM_APPROACH_REFERENCE, FAA_PCG_SEGMENTS_REFERENCE],
  },
  MAHF: {
    definition:
      "The Missed Approach Holding Fix is the protected holding point used after a missed approach.",
    references: [FAA_AIM_APPROACH_REFERENCE, FAA_PCG_SEGMENTS_REFERENCE],
  },
  LPV: {
    definition:
      "Localizer Performance with Vertical Guidance is a GPS/WAAS approach mode with lateral and vertical guidance flown to a decision altitude.",
    references: [
      {
        label: "FAA LPV overview",
        url: "https://www.faa.gov/about/office_org/headquarters_offices/ato/service_units/techops/navservices/gnss/nas/procedures/lpv",
      },
      FAA_AIM_APPROACH_REFERENCE,
    ],
  },
  "LNAV/VNAV": {
    definition:
      "Lateral Navigation / Vertical Navigation is an area-navigation approach minimum with lateral guidance and approved vertical guidance.",
    references: [
      {
        label: "FAA LNAV/VNAV overview",
        url: "https://www.faa.gov/about/office_org/headquarters_offices/ato/service_units/techops/navservices/gnss/nas/procedures/vnav",
      },
      FAA_AIM_APPROACH_REFERENCE,
    ],
  },
  LNAV: {
    definition:
      "Lateral Navigation is a non-precision area-navigation approach minimum that provides lateral guidance.",
    references: [
      {
        label: "FAA LNAV overview",
        url: "https://www.faa.gov/about/office_org/headquarters_offices/ato/service_units/techops/navservices/gnss/nas/procedures/lnav",
      },
      FAA_AIM_APPROACH_REFERENCE,
    ],
  },
  RNAV: {
    definition:
      "Area Navigation lets aircraft fly a desired path without flying directly over ground-based navigation aids.",
    references: [FAA_AIM_PBN_REFERENCE],
  },
  "RNAV(GPS)": {
    definition:
      "Area Navigation using GPS/WAAS is a satellite-based approach procedure. It uses GPS, and when available the WAAS augmentation system, to guide the aircraft along the published lateral path; WAAS-enabled minima such as LPV can also provide approved vertical guidance.",
    references: [FAA_AIM_PBN_REFERENCE, FAA_AIM_APPROACH_REFERENCE],
  },
  "RNAV(RNP)": {
    definition:
      "Area Navigation with Required Navigation Performance is an RNAV approach built around a required navigation accuracy. In modern U.S. charting, RNAV(RNP) procedures often imply onboard performance monitoring and may include authorization-required design features.",
    references: [FAA_AIM_PBN_REFERENCE, FAA_AIM_APPROACH_REFERENCE],
  },
  RNP: {
    definition:
      "Required Navigation Performance is area navigation with onboard performance monitoring and alerting.",
    references: [FAA_AIM_PBN_REFERENCE],
  },
  "RNP AR": {
    definition:
      "Required Navigation Performance Authorization Required is an area-navigation approach that needs specific aircraft capability, crew training, and operational authorization.",
    references: [
      FAA_AIM_PBN_REFERENCE,
      {
        label: "FAA AC 90-101A",
        url: "https://www.faa.gov/regulations_policies/advisory_circulars/index.cfm/go/document.information/documentid/903610",
      },
    ],
  },
  OCS: {
    definition:
      "Obstacle Clearance Surface is a vertical or sloping reference surface used to evaluate obstacle clearance along an instrument procedure.",
    references: [FAA_PBN_ORDER_REFERENCE],
  },
  OEA: {
    definition:
      "Obstacle Evaluation Area is the lateral area used to evaluate obstacles for a segment or final approach surface.",
    references: [FAA_PBN_ORDER_REFERENCE],
  },
  GPA: {
    definition:
      "Glidepath Angle is the published or coded descent angle used by vertically guided final approach visualization. AeroViz reads it from CIFP path point metadata when available, otherwise it falls back to the final leg vertical angle.",
    references: [FAA_CIFP_REFERENCE, FAA_PBN_ORDER_REFERENCE],
  },
  TCH: {
    definition:
      "Threshold Crossing Height is the glidepath height above the runway threshold. AeroViz reads source-backed TCH from CIFP Airport Path Point records; GPA/TCH-based OCS surfaces are only constructed when this value is available.",
    references: [FAA_CIFP_REFERENCE, FAA_PBN_ORDER_REFERENCE],
  },
  XTT: {
    definition:
      "Cross Track Tolerance is the lateral tolerance used to size protected areas on either side of the nominal path.",
    references: [FAA_PBN_ORDER_REFERENCE],
  },
  ATT: {
    definition:
      "Along Track Tolerance is the along-path tolerance used in protected-area and transition construction.",
    references: [FAA_PBN_ORDER_REFERENCE],
  },
  IF_TERMINATOR: {
    definition:
      "The Initial Fix path terminator starts the published segment at that named fix.",
    references: [FAA_CIFP_REFERENCE, FAA_RNAV_LEG_TYPES_REFERENCE],
  },
  TF_TERMINATOR: {
    definition: "The Track to Fix path terminator flies a defined track between fixes.",
    references: [FAA_CIFP_REFERENCE, FAA_RNAV_LEG_TYPES_REFERENCE],
  },
  DF_TERMINATOR: {
    definition:
      "The Direct to Fix path terminator flies direct from the aircraft's present position to a fix.",
    references: [FAA_CIFP_REFERENCE, FAA_RNAV_LEG_TYPES_REFERENCE],
  },
  CA_TERMINATOR: {
    definition:
      "The Course to Altitude path terminator flies a specified course until reaching a specified altitude.",
    references: [FAA_CIFP_REFERENCE, FAA_RNAV_LEG_TYPES_REFERENCE],
  },
  CF_TERMINATOR: {
    definition: "The Course to Fix path terminator flies a specified course to a fix.",
    references: [FAA_CIFP_REFERENCE, FAA_RNAV_LEG_TYPES_REFERENCE],
  },
  HM_TERMINATOR: {
    definition: "The Hold to Manual Termination path terminator enters a hold until manual termination.",
    references: [FAA_CIFP_REFERENCE, FAA_RNAV_LEG_TYPES_REFERENCE],
  },
  HF_TERMINATOR: {
    definition: "The Hold to Fix path terminator flies a holding pattern terminating at a fix.",
    references: [FAA_CIFP_REFERENCE, FAA_RNAV_LEG_TYPES_REFERENCE],
  },
  A: {
    definition:
      "Approach Route Type A means Approach Transition for Airport Approach (PF) and Heliport Approach (HF) records. In this procedure view, it marks an entry route from a named transition fix into the shared final approach route.",
    references: [LOCAL_ARINC_APPROACH_ROUTE_TYPE_REFERENCE, FAA_CIFP_REFERENCE],
  },
  H: {
    definition:
      "Approach Route Type H means Area Navigation (RNAV) Approach with Required Navigation Performance (RNP) for Airport Approach (PF) and Heliport Approach (HF) records. In this procedure view, it identifies the RNAV(RNP)-style approach route rather than a transition branch.",
    references: [LOCAL_ARINC_APPROACH_ROUTE_TYPE_REFERENCE, FAA_CIFP_REFERENCE, FAA_AIM_PBN_REFERENCE],
  },
  R: {
    definition:
      "Approach Route Type R means Area Navigation (RNAV) Approach for Airport Approach (PF) and Heliport Approach (HF) records. In this procedure view, it identifies the RNAV approach route that carries the final approach and missed-approach coding, separate from optional transition routes.",
    references: [LOCAL_ARINC_APPROACH_ROUTE_TYPE_REFERENCE, FAA_CIFP_REFERENCE, FAA_AIM_PBN_REFERENCE],
  },
};

export function normalizeTermKey(term: string): string {
  return term.toUpperCase();
}

export function isKnownGlossaryTerm(term: string): boolean {
  const key = normalizeTermKey(term);
  return Boolean(TERM_DETAILS[term] ?? TERM_DETAILS[key]);
}

export function formatTermBrief(term: string): string {
  const key = normalizeTermKey(term);
  if (key === "MAPT") return "MAPt";
  return term.replace("_TERMINATOR", "");
}

export function formatTermMeaning(term: string): string {
  const key = normalizeTermKey(term);
  const displayName = TERM_DISPLAY_NAMES[key];
  if (!displayName) return "Name or data identifier";
  return displayName.replace(/\s\([^)]+\)$/, "");
}

export function branchIdentifierLabel(branch: ProcedureDetailBranch): string {
  if (branch.branchRole === "final") return "Final approach branch";
  return "Transition identifier";
}

function approachRouteTypeTerm(routeType: string | null | undefined): string | null {
  const normalized = normalizeTermKey(routeType ?? "");
  if (!normalized) return null;
  if (!APPROACH_ROUTE_TYPE_TERMS.includes(normalized)) return null;
  return normalized;
}

function approachRouteTypeContextMeaning(routeType: string | null | undefined): string {
  const normalized = normalizeTermKey(routeType ?? "");
  if (normalized === "A") {
    return " Approach Route Type A means Approach Transition for Airport Approach records: an entry route from an initial fix or feeder path into the shared final approach.";
  }
  if (normalized === "R") {
    return " Approach Route Type R means Area Navigation (RNAV) Approach for Airport Approach records: the RNAV route that contains the final approach and missed-approach coding.";
  }
  if (normalized === "H") {
    return " Approach Route Type H means Area Navigation (RNAV) Approach with Required Navigation Performance (RNP) for Airport Approach records.";
  }
  if (normalized) {
    return ` Approach Route Type ${normalized} is an ARINC/CIFP Airport Approach route type. Its meaning depends on the PF/HF approach-record table, not the enroute airway table.`;
  }
  return "";
}

function formatFixRef(fixRef: string | null | undefined): string {
  if (!fixRef) return "Unknown fix";
  return fixRef.replace(/^fix:/, "");
}

export function formatContextualTermMeaning(
  term: string,
  document: ProcedureDetailDocument | null,
): string {
  const key = normalizeTermKey(term);
  const glossaryMeaning = TERM_DISPLAY_NAMES[key];
  if (glossaryMeaning) return glossaryMeaning.replace(/\s\([^)]+\)$/, "");

  const branch = document?.branches.find(
    (candidate) => normalizeTermKey(candidate.branchIdent) === key,
  );
  if (branch) return branchIdentifierLabel(branch);

  const fix = document?.fixes.find((candidate) => normalizeTermKey(candidate.ident) === key);
  if (fix) return "Named procedure fix";

  if (document && normalizeTermKey(document.procedure.chartName) === key) {
    return "Published approach chart name";
  }

  return formatTermMeaning(term);
}

export function termDetails(term: string): TermDetails {
  return TERM_DETAILS[term] ?? TERM_DETAILS[normalizeTermKey(term)] ?? DEFAULT_IDENTIFIER_DETAILS;
}

export function contextualTermDetails(
  term: string,
  document: ProcedureDetailDocument | null,
): TermDetails {
  const glossaryDetails = TERM_DETAILS[term] ?? TERM_DETAILS[normalizeTermKey(term)];
  if (glossaryDetails) return glossaryDetails;

  const key = normalizeTermKey(term);
  const branch = document?.branches.find(
    (candidate) => normalizeTermKey(candidate.branchIdent) === key,
  );
  if (branch) {
    const branchType = branchIdentifierLabel(branch).toLowerCase();
    const targetFix = branch.mergeFixRef ? formatFixRef(branch.mergeFixRef) : null;
    const routeType = approachRouteTypeContextMeaning(branch.procedureType);
    return {
      definition:
        branch.branchRole === "final"
          ? `${term} is the final-branch identifier for this procedure. It contains the legs that line up with the runway, continue to the missed approach point, and then describe the missed approach.${routeType}`
          : `${term} is a ${branchType}. It names the entry route into the common final approach${targetFix ? ` near ${targetFix}` : ""}.${routeType}`,
      references: [],
    };
  }

  const fix = document?.fixes.find((candidate) => normalizeTermKey(candidate.ident) === key);
  if (fix) {
    return {
      definition: `${term} is a named fix in this procedure. A fix is a published navigation point used to define where a segment starts, changes direction, or ends.`,
      references: [],
    };
  }

  if (document && normalizeTermKey(document.procedure.chartName) === key) {
    return {
      definition: `${term} is the published procedure name shown on the approach chart.`,
      references: [],
    };
  }

  return DEFAULT_IDENTIFIER_DETAILS;
}

export function isSpecificFixRole(role: string | null | undefined): boolean {
  if (!role) return false;
  return role.toUpperCase() !== "ROUTE";
}

function procedureFamilyTerm(procedureFamily: string | null | undefined): string | null {
  if (procedureFamily === "RNAV_GPS") return "RNAV(GPS)";
  if (procedureFamily === "RNAV_RNP") return "RNAV(RNP)";
  return null;
}

export function summaryTerms(document: ProcedureDetailDocument | null): string[] {
  if (!document) return ["IAF", "IF", "FAF", "MAPT", "LPV", "LNAV/VNAV", "LNAV"];
  const terms = new Set(["IAF", "IF", "FAF", "MAPT", "LPV", "LNAV/VNAV", "LNAV"]);
  terms.add(document.procedure.chartName);
  const familyTerm = procedureFamilyTerm(document.procedure.procedureFamily);
  if (familyTerm) terms.add(familyTerm);
  document.procedure.approachModes.forEach((mode) => terms.add(mode));
  if (document.verticalProfiles.some((profile) => typeof profile.glidepathAngleDeg === "number")) {
    terms.add("GPA");
    terms.add("TCH");
  }
  document.branches.forEach((branch) => {
    terms.add(branch.branchIdent);
    const routeTypeTerm = approachRouteTypeTerm(branch.procedureType);
    if (routeTypeTerm) terms.add(routeTypeTerm);
    branch.legs.forEach((leg) => {
      if (isSpecificFixRole(leg.roleAtEnd)) terms.add(leg.roleAtEnd);
      if (leg.path.pathTerminator) terms.add(`${leg.path.pathTerminator}_TERMINATOR`);
    });
  });
  return [...terms];
}

export function contextualTerms(
  document: ProcedureDetailDocument | null,
  focusedFix: ProcedureDetailFix | null,
  focusedBranch: ProcedureDetailBranch | null,
): string[] {
  const terms = new Set(summaryTerms(document));
  if (focusedFix) terms.add(focusedFix.ident);
  if (focusedBranch) terms.add(focusedBranch.branchIdent);
  const focusedProcedureTypeTerm = approachRouteTypeTerm(focusedBranch?.procedureType);
  if (focusedProcedureTypeTerm) terms.add(focusedProcedureTypeTerm);
  focusedFix?.roleHints.forEach((role) => terms.add(role));
  focusedBranch?.legs.forEach((leg) => {
    if (isSpecificFixRole(leg.roleAtEnd)) terms.add(leg.roleAtEnd);
    if (leg.path.pathTerminator) terms.add(`${leg.path.pathTerminator}_TERMINATOR`);
  });
  return [...terms];
}

export function groupedTerms(terms: string[]): TermGroup[] {
  const uniqueTerms = Array.from(new Set(terms));
  const isFixRole = (term: string) => FIX_ROLE_TERMS.includes(normalizeTermKey(term));
  const isPathTerminator = (term: string) => normalizeTermKey(term).endsWith("_TERMINATOR");
  const isProcedureMode = (term: string) => PROCEDURE_MODE_TERMS.includes(normalizeTermKey(term));
  const isApproachRouteType = (term: string) =>
    APPROACH_ROUTE_TYPE_TERMS.includes(normalizeTermKey(term));

  const fixRoles = uniqueTerms.filter(isFixRole);
  const pathTerminators = uniqueTerms.filter(isPathTerminator);
  const procedureModes = uniqueTerms.filter(isProcedureMode);
  const approachRouteTypes = uniqueTerms.filter(isApproachRouteType);
  const identifiers = uniqueTerms.filter(
    (term) =>
      !isFixRole(term) &&
      !isPathTerminator(term) &&
      !isProcedureMode(term) &&
      !isApproachRouteType(term),
  );

  return [
    { id: "fix-roles", title: "Fix Roles", terms: fixRoles },
    { id: "procedure-modes", title: "Procedure Modes", terms: procedureModes },
    { id: "approach-route-types", title: "Approach Route Types", terms: approachRouteTypes },
    { id: "path-terminators", title: "Path Terminators", terms: pathTerminators },
    { id: "identifiers", title: "Names And Identifiers", terms: identifiers },
  ].filter((group) => group.terms.length > 0);
}
