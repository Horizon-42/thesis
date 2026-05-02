import { useApp } from "../context/AppContext";
import { annotationStatusLabel } from "../data/procedureAnnotations";
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

export default function ProcedureAnnotationPopup() {
  const {
    selectedProcedureAnnotation,
    setSelectedProcedureAnnotation,
    activeAirportCode,
  } = useApp();

  if (!selectedProcedureAnnotation) return null;

  const annotation = selectedProcedureAnnotation;
  const diagnostics = annotation.diagnostics.slice(0, 3);

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
