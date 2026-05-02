import type {
  ProcedureDetailBranch,
  ProcedureDetailDocument,
  ProcedureDetailFix,
  ProcedureDetailLeg,
} from "./procedureDetails";
import type {
  AltitudeConstraint,
  BuildDiagnostic,
  FixRole,
  LegType,
  NavSpecCode,
  ProcedureFamily,
  ProcedurePackage,
  ProcedurePackageBranch,
  ProcedurePackageFix,
  ProcedurePackageLeg,
  ProcedureSegment,
  SourceRef,
  VerticalRule,
} from "./procedurePackage";

const SUPPORTED_LEG_TYPES = new Set(["IF", "TF", "RF", "DF", "CA", "HM", "HA", "HF"]);

function sourceRefs(rawRefs: string[] | undefined): SourceRef[] {
  return (rawRefs ?? []).map((rawRef) => ({
    docId: "AEROVIZ_SOURCE",
    rawRef,
  }));
}

function sourceFiles(document: ProcedureDetailDocument): string[] {
  return document.provenance.sources
    .map((source) => source.path ?? source.sourceId)
    .filter((value): value is string => Boolean(value));
}

function procedureFamily(value: string): ProcedureFamily {
  if (
    value === "RNAV_GPS" ||
    value === "RNAV_RNP" ||
    value === "RNP_AR_APCH" ||
    value === "SID" ||
    value === "STAR"
  ) {
    return value;
  }
  return "UNKNOWN";
}

function fixRole(role: string): FixRole {
  const normalized = role.toUpperCase();
  if (normalized === "MAPT" || normalized === "MAP") return "MAP";
  if (normalized === "RWY" || normalized.startsWith("RW")) return "RWY";
  if (
    normalized === "IAF" ||
    normalized === "IF" ||
    normalized === "PFAF" ||
    normalized === "FAF" ||
    normalized === "MAHF" ||
    normalized === "FROP"
  ) {
    return normalized;
  }
  return "UNKNOWN";
}

function normalizeFix(fix: ProcedureDetailFix): ProcedurePackageFix {
  const roles = fix.roleHints.map(fixRole);
  const roleSet = new Set<FixRole>(roles.length > 0 ? roles : ["UNKNOWN"]);
  if (fix.kind === "runway_threshold") roleSet.add("RWY");

  return {
    fixId: fix.fixId,
    ident: fix.ident,
    role: [...roleSet],
    latDeg: fix.position?.lat ?? null,
    lonDeg: fix.position?.lon ?? null,
    altFtMsl: fix.elevationFt,
    annotations: fix.kind ? [fix.kind] : [],
    sourceRefs: sourceRefs(fix.sourceRefs),
  };
}

function branchKey(branch: ProcedureDetailBranch): string {
  return (branch.branchKey ?? branch.branchIdent).toUpperCase();
}

function packageIdFor(document: ProcedureDetailDocument): string {
  return `${document.airport.icao.toUpperCase()}-${document.procedure.procedureIdent.toUpperCase()}-${document.runway.ident ?? "UNKNOWN"}`;
}

function scopedBranchId(document: ProcedureDetailDocument, branch: ProcedureDetailBranch): string {
  return `${packageIdFor(document)}:branch:${branchKey(branch)}`;
}

function packageBranchRole(
  branch: ProcedureDetailBranch,
): ProcedurePackageBranch["branchRole"] {
  const normalized = branch.branchRole.toLowerCase();
  if (normalized === "final") return "STRAIGHT_IN";
  if (normalized === "transition") return "TRANSITION";
  if (normalized === "missed") return "MISSED";
  if (normalized === "holding") return "HOLDING";
  return "TRANSITION";
}

function legType(pathTerminator: string): LegType {
  const normalized = pathTerminator.toUpperCase();
  return SUPPORTED_LEG_TYPES.has(normalized) ? (normalized as LegType) : "UNSUPPORTED";
}

function segmentTypeFor(
  document: ProcedureDetailDocument,
  branch: ProcedureDetailBranch,
  rawSegmentType: string,
): ProcedureSegment["segmentType"] {
  const normalized = rawSegmentType.toLowerCase();
  if (normalized.includes("missed_s2")) return "MISSED_S2";
  if (normalized.includes("missed_s1")) return "MISSED_S1";
  if (normalized.includes("missed") || branch.branchRole.toLowerCase() === "missed") {
    return "MISSED_S1";
  }
  if (normalized.includes("hold")) return "HOLDING";
  if (normalized.includes("feeder")) return "FEEDER";
  if (normalized.includes("initial")) return "INITIAL";
  if (normalized.includes("intermediate")) return "INTERMEDIATE";
  if (normalized.includes("final")) {
    if (document.procedure.procedureFamily === "RNP_AR_APCH") return "FINAL_RNP_AR";
    return "FINAL_LNAV";
  }
  return "UNKNOWN";
}

function navSpecFor(segmentType: ProcedureSegment["segmentType"]): NavSpecCode {
  if (segmentType === "FEEDER") return "RNAV_1";
  if (segmentType === "FINAL_RNP_AR") return "RNP_AR_0_3";
  if (segmentType === "UNKNOWN") return "UNKNOWN";
  return "RNP_APCH";
}

function xttFor(segmentType: ProcedureSegment["segmentType"]): number {
  if (segmentType === "FINAL_LNAV" || segmentType === "FINAL_LNAV_VNAV") return 0.3;
  if (segmentType === "FINAL_LPV" || segmentType === "FINAL_GLS") return 0.3;
  if (segmentType === "FINAL_RNP_AR") return 0.3;
  return 1;
}

function attFor(segmentType: ProcedureSegment["segmentType"]): number {
  if (segmentType.startsWith("FINAL")) return 0.3;
  return 1;
}

function verticalRuleFor(
  document: ProcedureDetailDocument,
  segmentType: ProcedureSegment["segmentType"],
): VerticalRule | null {
  if (!segmentType.startsWith("FINAL")) return { kind: "NONE" };
  const verticalProfile = document.verticalProfiles.find((profile) =>
    profile.appliesToModes.some((mode) => {
      const normalized = mode.toUpperCase();
      return normalized === "LNAV/VNAV" || normalized === "LNAV-VNAV" || normalized === "LPV";
    }),
  );
  const pathMetadata = {
    ...(typeof verticalProfile?.glidepathAngleDeg === "number"
      ? { gpaDeg: verticalProfile.glidepathAngleDeg }
      : {}),
    ...(typeof verticalProfile?.thresholdCrossingHeightFt === "number"
      ? { tchFt: verticalProfile.thresholdCrossingHeightFt }
      : {}),
  };
  if (document.procedure.approachModes.some((mode) => mode.toUpperCase() === "LPV")) {
    return { kind: "LPV_GLS_SURFACES", ...pathMetadata };
  }
  if (document.procedure.approachModes.some((mode) => mode.toUpperCase() === "LNAV/VNAV")) {
    return { kind: "BARO_GLIDEPATH", ...pathMetadata };
  }
  return { kind: "LEVEL_ROC" };
}

function altitudeConstraint(leg: ProcedureDetailLeg): AltitudeConstraint | null {
  const altitude = leg.constraints.altitude;
  if (!altitude) return null;
  const qualifier = altitude.qualifier.toLowerCase();
  if (qualifier.includes("above")) {
    return { kind: "AT_OR_ABOVE", minFtMsl: altitude.valueFt, sourceText: altitude.rawText };
  }
  if (qualifier.includes("below")) {
    return { kind: "AT_OR_BELOW", maxFtMsl: altitude.valueFt, sourceText: altitude.rawText };
  }
  return {
    kind: "AT",
    minFtMsl: altitude.valueFt,
    maxFtMsl: altitude.valueFt,
    sourceText: altitude.rawText,
  };
}

function segmentIdFor(
  branchId: string,
  segmentType: ProcedureSegment["segmentType"],
  index: number,
): string {
  return `${branchId}:segment:${segmentType.toLowerCase()}:${index + 1}`;
}

interface SegmentDraft {
  segment: ProcedureSegment;
  sourceLegs: ProcedureDetailLeg[];
}

function legStartsMissedSectionTwo(leg: ProcedureDetailLeg): boolean {
  const terminator = leg.path.pathTerminator.toUpperCase();
  const role = leg.roleAtEnd.toUpperCase();
  return terminator === "HM" || terminator === "HA" || terminator === "HF" || role === "MAHF";
}

function legTriggersTurningMissed(leg: ProcedureDetailLeg): boolean {
  const terminator = leg.path.pathTerminator.toUpperCase();
  return terminator === "HM" || terminator === "HA" || terminator === "HF" || terminator === "RF";
}

function splitMissedGroup(group: {
  rawSegmentType: string;
  legs: ProcedureDetailLeg[];
}): Array<{ rawSegmentType: string; legs: ProcedureDetailLeg[] }> {
  if (group.rawSegmentType.toLowerCase() !== "missed") return [group];

  const sectionTwoIndex = group.legs.findIndex(legStartsMissedSectionTwo);
  if (sectionTwoIndex <= 0) {
    return [
      {
        rawSegmentType: sectionTwoIndex === 0 ? "missed_s2" : "missed_s1",
        legs: group.legs,
      },
    ];
  }

  return [
    {
      rawSegmentType: "missed_s1",
      legs: group.legs.slice(0, sectionTwoIndex),
    },
    {
      rawSegmentType: "missed_s2",
      legs: group.legs.slice(sectionTwoIndex),
    },
  ];
}

function groupBranchSegments(
  document: ProcedureDetailDocument,
  branch: ProcedureDetailBranch,
  diagnostics: BuildDiagnostic[],
): SegmentDraft[] {
  const branchId = scopedBranchId(document, branch);
  const groups: Array<{ rawSegmentType: string; legs: ProcedureDetailLeg[] }> = [];

  branch.legs.forEach((leg) => {
    const current = groups[groups.length - 1];
    if (current && current.rawSegmentType === leg.segmentType) {
      current.legs.push(leg);
    } else {
      groups.push({ rawSegmentType: leg.segmentType, legs: [leg] });
    }
  });

  const sectionGroups = groups.flatMap(splitMissedGroup);

  return sectionGroups.map((group, index) => {
    const segmentType = segmentTypeFor(document, branch, group.rawSegmentType);
    const segmentId = segmentIdFor(branchId, segmentType, index);
    const navSpec = navSpecFor(segmentType);
    const xttNm = xttFor(segmentType);
    const attNm = attFor(segmentType);
    const firstLeg = group.legs[0];
    const lastLeg = group.legs[group.legs.length - 1];
    const approachModes = document.procedure.approachModes;
    const isTurningMissedApproach =
      segmentType === "MISSED_S2" && group.legs.some(legTriggersTurningMissed);

    diagnostics.push(
      {
        severity: "WARN",
        segmentId,
        code: "DEFAULT_NAV_SPEC",
        message: `${segmentId}: navSpec inferred as ${navSpec}; source data does not yet expose explicit segment navSpec.`,
        sourceRefs: sourceRefs(firstLeg.sourceRefs),
      },
      {
        severity: "WARN",
        segmentId,
        code: "DEFAULT_TOLERANCE",
        message: `${segmentId}: XTT/ATT inferred as ${xttNm}/${attNm} NM; source data does not yet expose explicit tolerances.`,
        sourceRefs: sourceRefs(firstLeg.sourceRefs),
      },
    );

    const collapsedApproachModes =
      segmentType === "FINAL_LNAV" && approachModes.length > 1 ? approachModes : undefined;
    if (collapsedApproachModes) {
      diagnostics.push({
        severity: "INFO",
        segmentId,
        code: "MODE_COLLAPSED_TO_LNAV",
        message: `${segmentId}: current adapter emits FINAL_LNAV as the constructible baseline for ${approachModes.join(
          " / ",
        )}. Mode-specific final surfaces remain future geometry work.`,
        sourceRefs: sourceRefs(firstLeg.sourceRefs),
      });
    }
    if (isTurningMissedApproach) {
      const triggerTypes = [...new Set(group.legs.map((leg) => leg.path.pathTerminator.toUpperCase()))]
        .filter((terminator) => terminator === "HM" || terminator === "HA" || terminator === "HF" || terminator === "RF")
        .join("/");
      diagnostics.push({
        severity: "WARN",
        segmentId,
        code: "TURNING_MISSED_UNIMPLEMENTED",
        message:
          `${segmentId}: missed section 2 contains ${triggerTypes || "turn-trigger"} leg semantics; ` +
          "turning missed approach TIA/wind-spiral geometry is not implemented yet.",
        sourceRefs: group.legs.flatMap((leg) => sourceRefs(leg.sourceRefs)),
      });
    }

    return {
      segment: {
        segmentId,
        branchId,
        segmentType,
        navSpec,
        startFixId: firstLeg.path.startFixRef,
        endFixId: lastLeg.path.endFixRef,
        legIds: group.legs.map((leg) => leg.legId),
        xttNm,
        attNm,
        secondaryEnabled: segmentType !== "FINAL_RNP_AR",
        widthChangeMode: segmentType.startsWith("FINAL") ? "LINEAR_TAPER" : "NONE",
        transitionRule: segmentType.startsWith("FINAL")
          ? {
              kind: "INTERMEDIATE_TO_FINAL_LNAV",
              anchorFixId: firstLeg.path.endFixRef,
              beforeNm: 2,
              afterNm: 1,
              notes: ["Baseline connector rule inferred for migration stage 1."],
            }
          : null,
        verticalRule: verticalRuleFor(document, segmentType),
        constructionFlags: {
          collapsedApproachModes,
          isTurningMissedApproach: isTurningMissedApproach || undefined,
        },
        sourceRefs: group.legs.flatMap((leg) => sourceRefs(leg.sourceRefs)),
        legacy: {
          rawSegmentType: group.rawSegmentType,
          sequenceRange: [firstLeg.sequence, lastLeg.sequence],
        },
      },
      sourceLegs: group.legs,
    };
  });
}

function normalizeLeg(
  leg: ProcedureDetailLeg,
  segment: ProcedureSegment,
  diagnostics: BuildDiagnostic[],
): ProcedurePackageLeg {
  const normalizedLegType = legType(leg.path.pathTerminator);
  if (normalizedLegType === "UNSUPPORTED") {
    diagnostics.push({
      severity: "WARN",
      segmentId: segment.segmentId,
      legId: leg.legId,
      code: "UNSUPPORTED_LEG_TYPE",
      message: `${leg.legId}: path terminator ${leg.path.pathTerminator} is preserved but not constructible by the v3 adapter yet.`,
      sourceRefs: sourceRefs(leg.sourceRefs),
    });
  }
  if (normalizedLegType === "RF") {
    const hasRfGeometry =
      leg.path.arcRadiusNm !== undefined &&
      leg.path.centerLatDeg !== undefined &&
      leg.path.centerLonDeg !== undefined;
    if (!hasRfGeometry) {
      diagnostics.push({
        severity: "ERROR",
        segmentId: segment.segmentId,
        legId: leg.legId,
        code: "RF_RADIUS_MISSING",
        message: `${leg.legId}: RF leg requires radius and center fields before geometry can be constructed.`,
        sourceRefs: sourceRefs(leg.sourceRefs),
      });
    }
  }

  return {
    legId: leg.legId,
    segmentId: segment.segmentId,
    legType: normalizedLegType,
    rawPathTerminator: leg.path.pathTerminator,
    startFixId: leg.path.startFixRef,
    endFixId: leg.path.endFixRef,
    outboundCourseDeg: leg.path.courseDeg,
    turnDirection: leg.path.turnDirection,
    arcRadiusNm: leg.path.arcRadiusNm,
    centerLatDeg: leg.path.centerLatDeg,
    centerLonDeg: leg.path.centerLonDeg,
    requiredAltitude: altitudeConstraint(leg),
    requiredSpeed: leg.constraints.speedKt
      ? { maxKias: leg.constraints.speedKt, sourceText: `${leg.constraints.speedKt} kt` }
      : null,
    navSpecAtLeg: segment.navSpec,
    xttNm: segment.xttNm,
    attNm: segment.attNm,
    secondaryEnabled: segment.secondaryEnabled,
    notes: [],
    sourceRefs: sourceRefs(leg.sourceRefs),
    legacy: {
      sequence: leg.sequence,
      constructionMethod: leg.path.constructionMethod,
      roleAtEnd: leg.roleAtEnd,
      qualityStatus: leg.quality.status,
      renderedInPlanView: leg.quality.renderedInPlanView === true,
    },
  };
}

export function normalizeProcedurePackage(document: ProcedureDetailDocument): ProcedurePackage {
  const diagnostics: BuildDiagnostic[] = [];
  const packageId = packageIdFor(document);
  const segmentDrafts = document.branches.flatMap((branch) =>
    groupBranchSegments(document, branch, diagnostics),
  );
  const segmentByLegId = new Map<string, ProcedureSegment>();
  segmentDrafts.forEach((draft) => {
    draft.sourceLegs.forEach((leg) => segmentByLegId.set(leg.legId, draft.segment));
  });

  const branchIdBySourceId = new Map(
    document.branches.map((branch) => [branch.branchId, scopedBranchId(document, branch)]),
  );

  const branches: ProcedurePackageBranch[] = document.branches.map((branch) => {
    const branchId = scopedBranchId(document, branch);
    return {
      branchId,
      runwayId: document.runway.ident,
      branchName: branch.transitionIdent ?? branch.branchIdent,
      branchRole: packageBranchRole(branch),
      segmentIds: segmentDrafts
        .map((draft) => draft.segment)
        .filter((segment) => segment.branchId === branchId)
        .map((segment) => segment.segmentId),
      mergeToBranchId: branch.continuesWithBranchId
        ? branchIdBySourceId.get(branch.continuesWithBranchId)
        : undefined,
      legacy: {
        sourceBranchId: branch.branchId,
        branchIdent: branch.branchIdent,
        branchKey: branchKey(branch),
        defaultVisible: branch.defaultVisible,
        mergeFixRef: branch.mergeFixRef,
        continuesWithBranchId: branch.continuesWithBranchId,
      },
    };
  });

  const legs = document.branches.flatMap((branch) =>
    branch.legs.map((leg) => {
      const segment = segmentByLegId.get(leg.legId);
      if (!segment) {
        throw new Error(`No segment generated for leg ${leg.legId}`);
      }
      return normalizeLeg(leg, segment, diagnostics);
    }),
  );

  document.validation.knownSimplifications.forEach((message) => {
    diagnostics.push({
      severity: "WARN",
      code: "SOURCE_INCOMPLETE",
      message,
      sourceRefs: [],
    });
  });
  document.provenance.warnings.forEach((message) => {
    diagnostics.push({
      severity: "WARN",
      code: "SOURCE_INCOMPLETE",
      message,
      sourceRefs: [],
    });
  });

  return {
    packageId,
    airportId: document.airport.icao.toUpperCase(),
    runwayId: document.runway.ident,
    procedureId: document.procedure.procedureIdent,
    procedureName: document.procedure.chartName,
    procedureFamily: procedureFamily(document.procedure.procedureFamily),
    sourceMeta: {
      cifpCycle: document.provenance.sources.find((source) => source.cycle)?.cycle ?? null,
      sourceFiles: sourceFiles(document),
      chartLinks: [],
      notes: document.provenance.warnings,
      authority: "FAA_8260_58D",
    },
    branches,
    segments: segmentDrafts.map((draft) => draft.segment),
    legs,
    sharedFixes: document.fixes.map(normalizeFix),
    validationConfig: {
      expectedRunwayIdent: document.validation.expectedRunwayIdent,
      expectedIF: document.validation.expectedIF,
      expectedFAF: document.validation.expectedFAF,
      expectedMAP: document.validation.expectedMAPt,
      expectedMissedHoldFix: document.validation.expectedMissedHoldFix,
      knownSimplifications: document.validation.knownSimplifications,
    },
    diagnostics,
    legacyDocument: {
      schemaVersion: document.schemaVersion,
      modelType: document.modelType,
      procedureUid: document.procedureUid,
    },
  };
}
