import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as pdfjsLib from "pdfjs-dist";
import pdfWorkerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import {
  fitChartProjection,
  projectLonLat,
  type ChartProjection,
  type ChartReferencePoint,
} from "../utils/chartProjection";
import { chartCalibrationFor } from "../data/chartCalibration";

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

export interface ChartMarker {
  ident: string;
  lon: number;
  lat: number;
  /** Fix role (IAF / IF / FAF / MAPt / MAHF / Route …) — drives the colour. */
  role: string;
  /** Tooltip body lines. */
  lines: string[];
}

export interface ChartSegment {
  fromIdent: string;
  toIdent: string;
  title: string;
  lines: string[];
}

interface ChartAnnotationViewProps {
  pdfUrl: string;
  procedureUid: string;
  markers: ChartMarker[];
  segments: ChartSegment[];
}

const ROLE_COLOR: Record<string, string> = {
  IAF: "#1a6eb5",
  IF: "#2d7a3a",
  PFAF: "#c05000",
  FAF: "#c05000",
  MAP: "#9c2b2b",
  MAPT: "#9c2b2b",
  MAHF: "#9c2b2b",
};

function roleColor(role: string): string {
  return ROLE_COLOR[role.trim().toUpperCase()] ?? "#5b2a8a";
}

type Hover =
  | { kind: "marker"; title: string; lines: string[]; x: number; y: number }
  | { kind: "segment"; title: string; lines: string[]; x: number; y: number }
  | null;

export default function ChartAnnotationView({
  pdfUrl,
  procedureUid,
  markers,
  segments,
}: ChartAnnotationViewProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const pdfRef = useRef<pdfjsLib.PDFDocumentProxy | null>(null);
  const renderTaskRef = useRef<pdfjsLib.RenderTask | null>(null);

  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [error, setError] = useState<string | null>(null);
  const [scale, setScale] = useState(1.4);
  const [pageSize, setPageSize] = useState<{ w: number; h: number }>({ w: 0, h: 0 });
  const [calibrate, setCalibrate] = useState(false);
  const [hover, setHover] = useState<Hover>(null);

  // ── Projection from the chart calibration (joined with marker lon/lat) ──
  const projection = useMemo<ChartProjection | null>(() => {
    const calibration = chartCalibrationFor(procedureUid);
    if (!calibration || calibration.references.length < 3) return null;
    const byIdent = new Map(markers.map((m) => [m.ident.toUpperCase(), m]));
    const refs: ChartReferencePoint[] = [];
    for (const ref of calibration.references) {
      const marker = byIdent.get(ref.ident.toUpperCase());
      if (marker) refs.push({ ident: ref.ident, lon: marker.lon, lat: marker.lat, pageX: ref.pageX, pageY: ref.pageY });
    }
    if (refs.length < 3) return null;
    try {
      return fitChartProjection(refs);
    } catch {
      return null;
    }
  }, [procedureUid, markers]);

  // ── Load the PDF once ──
  useEffect(() => {
    let cancelled = false;
    setStatus("loading");
    const task = pdfjsLib.getDocument(pdfUrl);
    task.promise
      .then((pdf) => {
        if (cancelled) return;
        pdfRef.current = pdf;
        return pdf.getPage(1).then((page) => {
          if (cancelled) return;
          const base = page.getViewport({ scale: 1 });
          setPageSize({ w: base.width, h: base.height });
          setStatus("ready");
        });
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : String(err));
        setStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, [pdfUrl]);

  // ── Render the page whenever the pdf or scale changes ──
  useEffect(() => {
    if (status !== "ready" || !pdfRef.current || !canvasRef.current) return;
    let cancelled = false;
    const canvas = canvasRef.current;
    pdfRef.current.getPage(1).then((page) => {
      if (cancelled) return;
      const viewport = page.getViewport({ scale });
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      renderTaskRef.current?.cancel();
      const task = page.render({ canvasContext: ctx, viewport });
      renderTaskRef.current = task;
      task.promise.catch(() => {/* cancelled re-render */});
    });
    return () => {
      cancelled = true;
    };
  }, [status, scale]);

  const onCanvasClick = useCallback(
    (e: React.MouseEvent) => {
      if (!calibrate || !canvasRef.current) return;
      const canvas = canvasRef.current;
      const rect = canvas.getBoundingClientRect();
      const px = (e.clientX - rect.left) * (canvas.width / rect.width);
      const py = (e.clientY - rect.top) * (canvas.height / rect.height);
      const pageX = Number((px / scale).toFixed(1));
      const pageY = Number((py / scale).toFixed(1));
      // eslint-disable-next-line no-console
      console.log(`[chart-calibrate] { ident: "?", pageX: ${pageX}, pageY: ${pageY} },`);
    },
    [calibrate, scale],
  );

  const px = (x: number) => x * scale;

  const projected = useMemo(() => {
    if (!projection) return new Map<string, { x: number; y: number }>();
    const out = new Map<string, { x: number; y: number }>();
    for (const m of markers) {
      const p = projectLonLat(projection, { lon: m.lon, lat: m.lat });
      out.set(m.ident.toUpperCase(), { x: px(p.x), y: px(p.y) });
    }
    return out;
  }, [projection, markers, scale]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
        <div style={{ display: "flex", gap: 4 }}>
          <button onClick={() => setScale((s) => Math.max(0.5, +(s - 0.2).toFixed(2)))}>−</button>
          <span style={{ minWidth: 56, textAlign: "center" }}>{Math.round(scale * 100)}%</span>
          <button onClick={() => setScale((s) => Math.min(4, +(s + 0.2).toFixed(2)))}>+</button>
        </div>
        <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: ".85rem" }}>
          <input type="checkbox" checked={calibrate} onChange={(e) => setCalibrate(e.target.checked)} />
          Calibrate (click a fix → log page coords)
        </label>
        {projection ? (
          <span style={{ fontSize: ".78rem", color: "#6c757d" }}>
            calibrated · fit RMS {projection.rmsResidual.toFixed(1)} px
          </span>
        ) : (
          <span style={{ fontSize: ".78rem", color: "#a05c00" }}>
            no calibration yet — toggle Calibrate, click each fix, paste coords into chartCalibration.ts
          </span>
        )}
      </div>

      {status === "error" && (
        <div style={{ color: "#9c2b2b" }}>Failed to load chart: {error}</div>
      )}

      <div style={{ position: "relative", display: "inline-block" }} onClick={onCanvasClick}>
        <canvas ref={canvasRef} style={{ display: "block", maxWidth: "100%", border: "1px solid #dee2e6" }} />

        {projection && (
          <svg
            width={px(pageSize.w)}
            height={px(pageSize.h)}
            viewBox={`0 0 ${px(pageSize.w)} ${px(pageSize.h)}`}
            style={{ position: "absolute", top: 0, left: 0, pointerEvents: "none", maxWidth: "100%" }}
          >
            {segments.map((seg, i) => {
              const a = projected.get(seg.fromIdent.toUpperCase());
              const b = projected.get(seg.toIdent.toUpperCase());
              if (!a || !b) return null;
              const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
              return (
                <g key={`seg-${i}`}>
                  <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="#1a6eb5" strokeOpacity={0.0} strokeWidth={16}
                    style={{ pointerEvents: "stroke", cursor: "pointer" }}
                    onMouseEnter={() => setHover({ kind: "segment", title: seg.title, lines: seg.lines, x: mid.x, y: mid.y })}
                    onMouseLeave={() => setHover(null)} />
                </g>
              );
            })}
            {markers.map((m) => {
              const p = projected.get(m.ident.toUpperCase());
              if (!p) return null;
              const color = roleColor(m.role);
              return (
                <g key={m.ident}
                  style={{ pointerEvents: "all", cursor: "pointer" }}
                  onMouseEnter={() => setHover({ kind: "marker", title: `${m.ident} — ${m.role}`, lines: m.lines, x: p.x, y: p.y })}
                  onMouseLeave={() => setHover(null)}>
                  <circle cx={p.x} cy={p.y} r={11} fill={color} fillOpacity={0.18} stroke={color} strokeWidth={2} />
                  <circle cx={p.x} cy={p.y} r={3.5} fill={color} />
                </g>
              );
            })}
          </svg>
        )}

        {hover && (
          <div
            style={{
              position: "absolute",
              left: Math.min(hover.x + 14, px(pageSize.w) - 260),
              top: hover.y + 14,
              maxWidth: 260,
              background: "#0d2233",
              color: "#eaf2fa",
              padding: "9px 12px",
              borderRadius: 8,
              fontSize: ".82rem",
              lineHeight: 1.5,
              boxShadow: "0 10px 30px rgba(0,0,0,.3)",
              pointerEvents: "none",
              zIndex: 5,
            }}
          >
            <div style={{ fontWeight: 700, color: "#7fc0ff", marginBottom: 3 }}>{hover.title}</div>
            {hover.lines.map((l, i) => (
              <div key={i}>{l}</div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
