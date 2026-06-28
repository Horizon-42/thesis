import { useEffect, useMemo, useState } from "react";
import {
  procedureChartsIndexUrl,
  procedureDetailsDocumentUrl,
  type ProcedureChartsManifest,
  type ProcedureDetailDocument,
} from "../data/procedureDetails";
import { buildProcedureConstraint } from "../data/procedureConstraint";
import { altitudeConstraintFromCifp, altitudeConstraintText } from "../data/altitudeConstraints";
import { fetchJson } from "../utils/fetchJson";
import { navigateWithinApp } from "../utils/navigation";
import ChartAnnotationView, {
  type ChartMarker,
  type ChartSegment,
} from "./ChartAnnotationView";

const DEFAULT_AIRPORT = "KRDU";
const DEFAULT_PROCEDURE_UID = "KRDU-R05LY-RW05L";

const ROLE_DESCRIPTION: Record<string, string> = {
  IAF: "Initial Approach Fix — where the aircraft joins the published approach.",
  IF: "Intermediate Fix — lines up and settles the aircraft before final.",
  PFAF: "Final Approach Fix — the final descent on the glidepath begins here.",
  FAF: "Final Approach Fix — the final descent on the glidepath begins here.",
  MAP: "Missed Approach Point — decide to land or go missed by here.",
  MAPT: "Missed Approach Point — decide to land or go missed by here.",
  MAHF: "Missed Approach Holding Fix — hold here after a missed approach.",
};

function roleDescription(role: string): string {
  return ROLE_DESCRIPTION[role.trim().toUpperCase()] ?? "Published procedure fix.";
}

function readParams(): { airport: string; procedureUid: string } {
  const search = new URLSearchParams(window.location.search);
  if (window.location.hash.includes("?")) {
    const hashSearch = new URLSearchParams(window.location.hash.split("?")[1] ?? "");
    hashSearch.forEach((value, key) => {
      if (!search.has(key)) search.set(key, value);
    });
  }
  return {
    airport: search.get("airport") ?? DEFAULT_AIRPORT,
    procedureUid: search.get("procedure") ?? search.get("procedureUid") ?? DEFAULT_PROCEDURE_UID,
  };
}

function buildMarkers(doc: ProcedureDetailDocument): ChartMarker[] {
  // First leg ending at each fix gives its role + coded altitude.
  const meta = new Map<string, { role: string; altText: string | null }>();
  for (const branch of doc.branches) {
    for (const leg of branch.legs) {
      const ref = leg.path.endFixRef;
      if (meta.has(ref)) continue;
      const constraint = altitudeConstraintFromCifp(leg.constraints.altitude);
      meta.set(ref, {
        role: leg.roleAtEnd,
        altText: constraint ? altitudeConstraintText(constraint) : null,
      });
    }
  }

  return doc.fixes
    .filter((fix) => fix.position !== null)
    .map((fix) => {
      const info = meta.get(fix.fixId);
      const role = info?.role ?? fix.roleHints[0] ?? "Fix";
      const lines = [roleDescription(role)];
      if (info?.altText) lines.push(`Altitude: ${info.altText}`);
      return {
        ident: fix.ident,
        lon: fix.position!.lon,
        lat: fix.position!.lat,
        role,
        lines,
      };
    });
}

function buildSegments(doc: ProcedureDetailDocument): ChartSegment[] {
  const seen = new Set<string>();
  const segments: ChartSegment[] = [];
  for (const branch of doc.branches) {
    const constraint = buildProcedureConstraint(doc, { branchId: branch.branchId });
    if (!constraint) continue;
    const wps = constraint.waypoints;
    for (let i = 0; i < wps.length - 1; i++) {
      const a = wps[i];
      const b = wps[i + 1];
      const key = `${a.ident}->${b.ident}`;
      if (seen.has(key)) continue;
      seen.add(key);
      const distNm = (b.distanceFromStartM - a.distanceFromStartM) / 1852;
      const lines: string[] = [];
      if (distNm > 0) lines.push(`${distNm.toFixed(1)} NM`);
      if (b.altitudeRefFt !== null) lines.push(`cross ${b.ident} at ${Math.round(b.altitudeRefFt)} ft`);
      segments.push({ fromIdent: a.ident, toIdent: b.ident, title: `${a.ident} → ${b.ident}`, lines });
    }
  }
  return segments;
}

export default function ChartAnnotatedPage() {
  const { airport, procedureUid } = useMemo(readParams, []);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [error, setError] = useState<string | null>(null);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [doc, setDoc] = useState<ProcedureDetailDocument | null>(null);

  useEffect(() => {
    let cancelled = false;
    setStatus("loading");
    Promise.all([
      fetchJson<ProcedureChartsManifest>(procedureChartsIndexUrl(airport)),
      fetchJson<ProcedureDetailDocument>(procedureDetailsDocumentUrl(airport, procedureUid)),
    ])
      .then(([charts, document]) => {
        if (cancelled) return;
        const entry =
          charts.charts.find((chart) => chart.procedureUid === procedureUid) ??
          charts.charts.find((chart) => chart.runwayIdent === document.runway.ident);
        if (!entry) throw new Error(`No chart found for ${procedureUid}`);
        setPdfUrl(entry.url);
        setDoc(document);
        setStatus("ready");
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : String(err));
        setStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, [airport, procedureUid]);

  const markers = useMemo(() => (doc ? buildMarkers(doc) : []), [doc]);
  const segments = useMemo(() => (doc ? buildSegments(doc) : []), [doc]);

  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: "20px 24px", fontFamily: "system-ui, sans-serif" }}>
      <button
        onClick={() => navigateWithinApp(`#procedure-details?airport=${airport}&procedureUid=${procedureUid}`)}
        style={{ marginBottom: 12 }}
      >
        ← Procedure details
      </button>
      <h1 style={{ margin: "0 0 4px", fontSize: "1.4rem" }}>
        {doc ? doc.procedure.chartName : "Annotated chart"}
      </h1>
      <p style={{ margin: "0 0 16px", color: "#6c757d", fontSize: ".9rem" }}>
        The real {airport} chart with interactive fix/segment markers placed at their published
        positions. Hover a marker or a leg for details.
      </p>

      {status === "loading" && <div>Loading chart…</div>}
      {status === "error" && <div style={{ color: "#9c2b2b" }}>Failed to load: {error}</div>}
      {status === "ready" && pdfUrl && (
        <ChartAnnotationView
          pdfUrl={pdfUrl}
          procedureUid={procedureUid}
          markers={markers}
          segments={segments}
        />
      )}
    </div>
  );
}
