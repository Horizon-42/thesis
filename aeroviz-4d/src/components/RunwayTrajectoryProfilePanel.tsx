import { useMemo } from "react";
import {
  useApp,
  type RunwayProfileViewMode,
} from "../context/AppContext";
import {
  useRunwayTrajectoryProfile,
  type ProfileAircraftTrack,
} from "../hooks/useRunwayTrajectoryProfile";
import type {
  HorizontalPlateRoute,
  RunwayProfilePoint,
  RunwayReferenceMark,
} from "../utils/runwayProfileGeometry";

interface PlotDomain {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

interface ProfilePlotProps {
  title: string;
  subtitle: string;
  mode: "side" | "top";
  tracks: ProfileAircraftTrack[];
  plateRoutes: HorizontalPlateRoute[];
  referenceMarks: RunwayReferenceMark[];
}

function formatIsoTime(iso: string | null): string {
  if (!iso) return "No active simulation time";
  return iso.replace("T", " ").replace(".000Z", "Z");
}

function collectViewDomain(
  mode: "side" | "top",
  tracks: ProfileAircraftTrack[],
  plateRoutes: HorizontalPlateRoute[],
  referenceMarks: RunwayReferenceMark[],
): PlotDomain {
  const xValues = [0];
  const yValues = [0];

  plateRoutes.forEach((route) => {
    route.points.forEach((point) => {
      xValues.push(point.xM);
      yValues.push(mode === "side" ? point.zM : point.yM);
      if (mode === "top") {
        yValues.push(point.yM + route.halfWidthM, point.yM - route.halfWidthM);
      }
    });
  });

  tracks.forEach((track) => {
    track.trail.forEach((point) => {
      xValues.push(point.xM);
      yValues.push(mode === "side" ? point.zM : point.yM);
    });
  });

  referenceMarks.forEach((mark) => {
    xValues.push(mark.xM);
  });

  const minX = Math.min(...xValues, -250);
  const maxX = Math.max(...xValues, 250);
  const xPadding = Math.max(120, (maxX - minX) * 0.08);

  if (mode === "side") {
    const minY = Math.min(...yValues, 0);
    const maxY = Math.max(...yValues, 120);
    const yPadding = Math.max(40, (maxY - minY) * 0.1);
    return {
      minX: minX - xPadding,
      maxX: maxX + xPadding,
      minY: Math.min(0, minY - yPadding),
      maxY: maxY + yPadding,
    };
  }

  const maxAbsY = Math.max(...yValues.map((value) => Math.abs(value)), 180);
  const yPadding = Math.max(50, maxAbsY * 0.14);
  return {
    minX: minX - xPadding,
    maxX: maxX + xPadding,
    minY: -(maxAbsY + yPadding),
    maxY: maxAbsY + yPadding,
  };
}

function formatMeters(value: number): string {
  return `${Math.round(value).toLocaleString()} m`;
}

function niceTickStep(span: number, targetTickCount: number): number {
  const rough = Math.max(1, span / Math.max(1, targetTickCount));
  const power = 10 ** Math.floor(Math.log10(rough));
  const scaled = rough / power;
  const niceScaled = scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10;
  return niceScaled * power;
}

function buildLinearTicks(min: number, max: number, targetTickCount: number): number[] {
  const step = niceTickStep(Math.max(1, max - min), targetTickCount);
  const start = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let value = start; value <= max + step * 0.25; value += step) {
    ticks.push(Number(value.toFixed(6)));
  }
  return ticks;
}

function selectVisibleReferenceMarks(
  marks: RunwayReferenceMark[],
  plotX: (value: number) => number,
): RunwayReferenceMark[] {
  const accepted: RunwayReferenceMark[] = [];

  marks.forEach((mark) => {
    const pixelX = plotX(mark.xM);
    const conflicts = accepted.some((acceptedMark) => {
      const acceptedPixelX = plotX(acceptedMark.xM);
      return Math.abs(pixelX - acceptedPixelX) < 44;
    });
    if (!conflicts) {
      accepted.push(mark);
    }
  });

  return [...accepted].sort((left, right) => right.xM - left.xM);
}

function plotPointPath(
  points: RunwayProfilePoint[],
  domain: PlotDomain,
  plotWidth: number,
  plotHeight: number,
  marginLeft: number,
  marginTop: number,
  mode: "side" | "top",
): string {
  const xSpan = Math.max(1, domain.maxX - domain.minX);
  const ySpan = Math.max(1, domain.maxY - domain.minY);
  const plotX = (value: number) =>
    marginLeft + ((domain.maxX - value) / xSpan) * plotWidth;
  const plotY = (value: number) =>
    marginTop + ((domain.maxY - value) / ySpan) * plotHeight;

  return points
    .map((point, index) => {
      const x = plotX(point.xM);
      const y = plotY(mode === "side" ? point.zM : point.yM);
      return `${index === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
}

function ProfilePlot({
  title,
  subtitle,
  mode,
  tracks,
  plateRoutes,
  referenceMarks,
}: ProfilePlotProps) {
  const width = 760;
  const height = 340;
  const marginLeft = 60;
  const marginRight = 18;
  const marginTop = 22;
  const marginBottom = 126;
  const plotWidth = width - marginLeft - marginRight;
  const plotHeight = height - marginTop - marginBottom;
  const domain = useMemo(
    () => collectViewDomain(mode, tracks, plateRoutes, referenceMarks),
    [mode, plateRoutes, referenceMarks, tracks],
  );

  const xSpan = Math.max(1, domain.maxX - domain.minX);
  const ySpan = Math.max(1, domain.maxY - domain.minY);
  const plotX = (value: number) => marginLeft + ((domain.maxX - value) / xSpan) * plotWidth;
  const plotY = (value: number) => marginTop + ((domain.maxY - value) / ySpan) * plotHeight;
  const zeroX = plotX(0);
  const zeroY = plotY(0);
  const yScale = plotHeight / ySpan;
  const xTicks = useMemo(
    () => buildLinearTicks(domain.minX, domain.maxX, 7),
    [domain.maxX, domain.minX],
  );
  const visibleReferenceMarks = useMemo(
    () => selectVisibleReferenceMarks(referenceMarks, plotX),
    [referenceMarks, plotX],
  );

  return (
    <section className="runway-profile-plot">
      <header>
        <h4>{title}</h4>
        <p>{subtitle}</p>
      </header>

      <svg
        className="runway-profile-svg"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={title}
      >
        <rect
          x={marginLeft}
          y={marginTop}
          width={plotWidth}
          height={plotHeight}
          rx={10}
          className="runway-profile-plot-frame"
        />

        <line
          x1={marginLeft}
          y1={marginTop + plotHeight}
          x2={marginLeft + plotWidth}
          y2={marginTop + plotHeight}
          className="runway-profile-axis"
        />
        <line
          x1={marginLeft}
          y1={marginTop}
          x2={marginLeft}
          y2={marginTop + plotHeight}
          className="runway-profile-axis"
        />

        {xTicks.map((tick) => {
          const tickX = plotX(tick);
          return (
            <g key={`x-tick-${mode}-${tick}`}>
              <line
                x1={tickX}
                y1={marginTop}
                x2={tickX}
                y2={marginTop + plotHeight}
                className="runway-profile-grid-line"
              />
              <line
                x1={tickX}
                y1={marginTop + plotHeight}
                x2={tickX}
                y2={marginTop + plotHeight + 6}
                className="runway-profile-axis"
              />
              <text
                x={tickX}
                y={marginTop + plotHeight + 22}
                textAnchor="middle"
                className="runway-profile-axis-tick"
              >
                {formatMeters(tick)}
              </text>
            </g>
          );
        })}

        <line
          x1={zeroX}
          y1={marginTop}
          x2={zeroX}
          y2={marginTop + plotHeight}
          className="runway-profile-threshold-line"
        />
        <text x={zeroX + 6} y={marginTop + 14} className="runway-profile-threshold-label">
          Threshold
        </text>

        {mode === "top" ? (
          <>
            <line
              x1={marginLeft}
              y1={zeroY}
              x2={marginLeft + plotWidth}
              y2={zeroY}
              className="runway-profile-centerline"
            />
            <text x={marginLeft + 8} y={zeroY - 8} className="runway-profile-centerline-label">
              Centerline y = 0
            </text>
          </>
        ) : (
          <>
            <line
              x1={marginLeft}
              y1={zeroY}
              x2={marginLeft + plotWidth}
              y2={zeroY}
              className="runway-profile-ground-line"
            />
            <text x={marginLeft + 8} y={zeroY - 8} className="runway-profile-centerline-label">
              Threshold elevation z = 0
            </text>
          </>
        )}

        {visibleReferenceMarks.map((mark) => {
          const markX = plotX(mark.xM);
          const label = `${mark.label} · ${mark.detail}`;
          return (
            <g key={`${mode}-mark-${mark.label}-${mark.detail}-${Math.round(mark.xM)}`}>
              <line
                x1={markX}
                y1={marginTop}
                x2={markX}
                y2={marginTop + plotHeight}
                className="runway-profile-reference-line"
              />
              <line
                x1={markX}
                y1={marginTop + plotHeight}
                x2={markX}
                y2={marginTop + plotHeight + 20}
                className="runway-profile-reference-tick"
              />
              <text
                x={markX + 4}
                y={marginTop + plotHeight + 56}
                transform={`rotate(-28 ${markX + 4} ${marginTop + plotHeight + 56})`}
                textAnchor="start"
                className="runway-profile-reference-label"
              >
                {label}
              </text>
            </g>
          );
        })}

        {plateRoutes.map((route) => {
          const d = plotPointPath(
            route.points,
            domain,
            plotWidth,
            plotHeight,
            marginLeft,
            marginTop,
            mode,
          );
          const bandWidth = mode === "top" ? Math.max(2, route.halfWidthM * yScale * 2) : 2;
          return (
            <g key={`${mode}-${route.routeId}`}>
              {mode === "top" ? (
                <path
                  d={d}
                  className="runway-profile-route-band"
                  style={{ strokeWidth: bandWidth }}
                />
              ) : null}
              <path d={d} className="runway-profile-route-line" />
            </g>
          );
        })}

        {tracks.map((track) => {
          const d = plotPointPath(
            track.trail,
            domain,
            plotWidth,
            plotHeight,
            marginLeft,
            marginTop,
            mode,
          );
          const currentYValue = mode === "side" ? track.current.zM : track.current.yM;
          const cx = plotX(track.current.xM);
          const cy = plotY(currentYValue);
          return (
            <g key={`${mode}-${track.flightId}`}>
              <path
                d={d}
                fill="none"
                stroke={track.color}
                strokeWidth={track.isSelected ? 3.2 : 2.2}
                strokeLinecap="round"
                strokeLinejoin="round"
                opacity={track.isSelected ? 0.96 : 0.74}
              />
              <circle
                cx={cx}
                cy={cy}
                r={track.isSelected ? 5.2 : 3.8}
                fill={track.color}
                stroke="rgba(15, 23, 42, 0.95)"
                strokeWidth={track.isSelected ? 2 : 1.4}
              />
              {track.isSelected ? (
                <text x={cx + 8} y={cy - 8} className="runway-profile-flight-label">
                  {track.flightId}
                </text>
              ) : null}
            </g>
          );
        })}

        <text
          x={marginLeft + plotWidth / 2}
          y={height - 12}
          textAnchor="middle"
          className="runway-profile-axis-label"
        >
          x: approach distance from threshold
        </text>
        <text
          x={18}
          y={marginTop + plotHeight / 2}
          textAnchor="middle"
          transform={`rotate(-90 18 ${marginTop + plotHeight / 2})`}
          className="runway-profile-axis-label"
        >
          {mode === "side" ? "z: height above threshold" : "y: lateral offset from centerline"}
        </text>

        <text x={18} y={marginTop + 10} className="runway-profile-axis-tick">
          {formatMeters(domain.maxY)}
        </text>
        <text x={18} y={marginTop + plotHeight} className="runway-profile-axis-tick">
          {formatMeters(domain.minY)}
        </text>
      </svg>
    </section>
  );
}

export default function RunwayTrajectoryProfilePanel() {
  const {
    activeAirportCode,
    isRunwayProfileOpen,
    selectedProfileRunwayIdent,
    runwayProfileViewMode,
    setRunwayProfileOpen,
    setRunwayProfileViewMode,
    trajectoryDataSource,
  } = useApp();
  const profile = useRunwayTrajectoryProfile();

  if (!isRunwayProfileOpen || !selectedProfileRunwayIdent) return null;

  const viewModes: Array<{ label: string; value: RunwayProfileViewMode }> = [
    { label: "Split", value: "split" },
    { label: "Side", value: "side-xz" },
    { label: "Top", value: "top-xy" },
  ];

  const routeCount = profile.plateRoutes.length;
  const trackCount = profile.aircraftTracks.length;

  return (
    <aside className="runway-profile-panel" aria-label="Runway trajectory profile">
      <header className="runway-profile-panel-header">
        <div>
          <h3>
            {activeAirportCode} {selectedProfileRunwayIdent} trajectory profile
          </h3>
          <p>
            Time: {formatIsoTime(profile.currentTimeIso)}{" "}
            {profile.sourceCycle ? `· CIFP ${profile.sourceCycle}` : ""}
          </p>
        </div>

        <div className="runway-profile-panel-actions">
          <div className="runway-profile-view-modes" role="tablist" aria-label="Profile view mode">
            {viewModes.map((mode) => (
              <button
                key={mode.value}
                type="button"
                className={runwayProfileViewMode === mode.value ? "active" : ""}
                onClick={() => setRunwayProfileViewMode(mode.value)}
              >
                {mode.label}
              </button>
            ))}
          </div>
          <button type="button" onClick={() => setRunwayProfileOpen(false)}>
            Close
          </button>
        </div>
      </header>

      <div className="runway-profile-summary">
        <span>{routeCount} RNAV branches in plate</span>
        <span>{trackCount} aircraft currently inside plate</span>
        <span>{trajectoryDataSource ? "CZML linked" : "CZML missing"}</span>
      </div>

      {profile.procedureNames.length > 0 ? (
        <p className="runway-profile-procedure-list">
          {profile.procedureNames.join(" · ")}
        </p>
      ) : null}

      {profile.error ? <p className="runway-profile-error">{profile.error}</p> : null}
      {!profile.error && !profile.isLoading && routeCount === 0 ? (
        <p className="runway-profile-empty">
          No RNAV arrival branches were found for this runway.
        </p>
      ) : null}
      {!profile.error && !profile.isLoading && routeCount > 0 && trackCount === 0 ? (
        <p className="runway-profile-empty">
          No aircraft are inside the RNAV horizontal plate at the current simulation time.
        </p>
      ) : null}

      {profile.isLoading ? <p className="runway-profile-empty">Loading runway profile…</p> : null}

      {!profile.isLoading && !profile.error && routeCount > 0 ? (
        <div
          className={`runway-profile-plots runway-profile-plots-${runwayProfileViewMode}`}
        >
          {runwayProfileViewMode !== "top-xy" ? (
            <ProfilePlot
              title="Side view"
              subtitle="Dynamic x-z projection driven by simulation time"
              mode="side"
              tracks={profile.aircraftTracks}
              plateRoutes={profile.plateRoutes}
              referenceMarks={profile.referenceMarks}
            />
          ) : null}
          {runwayProfileViewMode !== "side-xz" ? (
            <ProfilePlot
              title="Top view"
              subtitle="Dynamic x-y projection with RNAV horizontal plate and centerline"
              mode="top"
              tracks={profile.aircraftTracks}
              plateRoutes={profile.plateRoutes}
              referenceMarks={profile.referenceMarks}
            />
          ) : null}
        </div>
      ) : null}
    </aside>
  );
}
