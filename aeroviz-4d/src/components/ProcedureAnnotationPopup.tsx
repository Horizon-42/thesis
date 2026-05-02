import { useApp } from "../context/AppContext";
import {
  annotationStatusLabel,
  type ProcedureEntityAnnotation,
} from "../data/procedureAnnotations";
import {
  formatTermBrief,
  formatTermMeaning,
  isKnownGlossaryTerm,
  termDetails,
} from "../data/procedureTerms";
import { navigateWithinApp } from "../utils/navigation";

function DetailRow({ label, value }: { label: string; value: string | null | undefined }) {
  if (!value) return null;
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function termsFromProcedureName(procedureName: string): string[] {
  const terms: string[] = [];
  const upper = procedureName.toUpperCase();
  if (upper.includes("RNAV(GPS)")) terms.push("RNAV(GPS)", "RNAV");
  else if (upper.includes("RNAV(RNP)")) terms.push("RNAV(RNP)", "RNAV", "RNP");
  else if (upper.includes("RNAV")) terms.push("RNAV");
  if (upper.includes("RNP AR")) terms.push("RNP AR", "RNP");
  return terms;
}

function termsFromSegmentType(segmentType: string | undefined): string[] {
  if (!segmentType) return [];
  const terms: string[] = [];
  if (segmentType.includes("LNAV_VNAV")) terms.push("LNAV/VNAV", "LNAV");
  else if (segmentType.includes("LNAV")) terms.push("LNAV");
  if (segmentType.includes("LPV")) terms.push("LPV");
  if (segmentType.includes("RNP_AR")) terms.push("RNP AR", "RNP");
  return terms;
}

function termsFromKind(annotation: ProcedureEntityAnnotation): string[] {
  if (annotation.kind === "LNAV_VNAV_OCS") return ["OCS", "LNAV/VNAV", "GPA", "TCH"];
  if (annotation.kind === "FINAL_OEA") return ["OEA", "LNAV"];
  if (annotation.kind === "PRECISION_SURFACE") return ["LPV", "GPA", "TCH"];
  if (annotation.kind === "CA_COURSE_GUIDE" || annotation.kind === "CA_CENTERLINE" || annotation.kind === "CA_ENDPOINT") {
    return ["CA_TERMINATOR"];
  }
  if (annotation.kind === "MISSED_SURFACE") return ["MAPT", "OEA"];
  if (annotation.kind === "MISSING_FINAL_SURFACE") return ["OCS", "GPA", "TCH"];
  if (annotation.kind === "TURNING_MISSED_DEBUG") return ["HM_TERMINATOR", "DF_TERMINATOR"];
  if (annotation.kind === "SEGMENT_ENVELOPE_PRIMARY" || annotation.kind === "SEGMENT_ENVELOPE_SECONDARY") {
    return ["XTT", "ATT"];
  }
  return [];
}

function termsFromParameters(annotation: ProcedureEntityAnnotation): string[] {
  const terms: string[] = [];
  annotation.parameters.forEach((parameter) => {
    const label = parameter.label.toUpperCase();
    if (label === "XTT") terms.push("XTT");
    if (label === "ATT") terms.push("ATT");
    if (label === "GPA") terms.push("GPA");
    if (label === "TCH") terms.push("TCH");
  });
  return terms;
}

function annotationKeyTerms(annotation: ProcedureEntityAnnotation): string[] {
  const legTerm = annotation.legType ? `${annotation.legType}_TERMINATOR` : null;
  const candidates = [
    ...termsFromProcedureName(annotation.procedureName),
    ...termsFromSegmentType(annotation.segmentType),
    ...termsFromKind(annotation),
    ...termsFromParameters(annotation),
    legTerm,
  ].filter((term): term is string => Boolean(term));
  return Array.from(new Set(candidates)).filter(isKnownGlossaryTerm);
}

export default function ProcedureAnnotationPopup() {
  const {
    selectedProcedureAnnotation,
    setSelectedProcedureAnnotation,
    activeAirportCode,
  } = useApp();

  if (!selectedProcedureAnnotation) return null;

  const annotation = selectedProcedureAnnotation;
  const diagnostics = annotation.diagnostics.slice(0, 3);
  const keyTerms = annotationKeyTerms(annotation);

  return (
    <section className="procedure-annotation-popup" aria-label="Procedure annotation details">
      <header>
        <div>
          <p>{annotationStatusLabel(annotation.status)}</p>
          <h3>{annotation.title}</h3>
        </div>
        <button type="button" onClick={() => setSelectedProcedureAnnotation(null)}>
          Close
        </button>
      </header>

      <p className="procedure-annotation-meaning">{annotation.meaning}</p>

      <dl className="procedure-annotation-details">
        <DetailRow label="Procedure" value={annotation.procedureName} />
        <DetailRow label="Runway" value={annotation.runwayId ?? "Unassigned"} />
        <DetailRow label="Branch" value={`${annotation.branchName} (${annotation.branchRole})`} />
        <DetailRow label="Segment" value={annotation.segmentType} />
        <DetailRow label="Leg" value={annotation.legType ? `${annotation.legType} ${annotation.legId ?? ""}` : null} />
        <DetailRow label="Kind" value={annotation.kind.replace(/_/g, " ")} />
        {annotation.parameters.map((parameter) => (
          <DetailRow key={`${parameter.label}-${parameter.value}`} label={parameter.label} value={parameter.value} />
        ))}
      </dl>

      {diagnostics.length > 0 ? (
        <div className="procedure-annotation-section">
          <strong>Diagnostics</strong>
          <ul>
            {diagnostics.map((diagnostic) => (
              <li key={diagnostic}>{diagnostic}</li>
            ))}
          </ul>
        </div>
      ) : null}

      {annotation.sourceRefs.length > 0 ? (
        <div className="procedure-annotation-source">
          Source: {annotation.sourceRefs.join(", ")}
        </div>
      ) : null}

      {keyTerms.length > 0 ? (
        <div className="procedure-annotation-section procedure-annotation-terms">
          <strong>Key Terms</strong>
          <div>
            {keyTerms.map((term) => {
              const details = termDetails(term);
              return (
                <article key={term}>
                  <h4>
                    <span>{formatTermBrief(term)}</span>
                    <small>{formatTermMeaning(term)}</small>
                  </h4>
                  <p>{details.definition}</p>
                  {details.references.length > 0 ? (
                    <div className="procedure-annotation-term-references">
                      {details.references.map((reference) => (
                        <a
                          key={`${term}-${reference.url}`}
                          href={reference.url}
                          target="_blank"
                          rel="noreferrer"
                        >
                          {reference.label}
                        </a>
                      ))}
                    </div>
                  ) : null}
                </article>
              );
            })}
          </div>
        </div>
      ) : null}

      <div className="procedure-annotation-actions">
        <button
          type="button"
          onClick={() => navigateWithinApp(`/procedure-details?airport=${activeAirportCode}`)}
        >
          Procedure Details
        </button>
      </div>
    </section>
  );
}
