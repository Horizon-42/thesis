import { useMemo } from "react";
import {
  useApp,
  type RunwayProfileViewMode,
} from "../context/AppContext";
import {
  useRunwayTrajectoryProfile,
  type ProfileAircraftTrack,
} from "../hooks/useRunwayTrajectoryProfile";
import type { HorizontalPlateRoute, RunwayProfilePoint } from "../utils/runwayProfileGeometry";

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
}

function formatIsoTime(iso: string | null): string {
  if (!iso) return "No active simulation time";
  return iso.replace("T", " ").replace(".000Z", "Z");
}

function collectViewDomain(
  mode: "side" | "top",
  tracks: ProfileAircraftTrack[],
  plateRoutes: HorizontalPlateRoute[],
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

function ProfilePlot({ title, subtitle, mode, tracks, plateRoutes }: ProfilePlotProps) {
  const width = 760;
  const height = 280;
  const marginLeft = 60;
  const marginRight = 18;
  const marginTop = 22;
  const marginBottom = 42;
  const plotWidth = width - marginLeft - marginRight;
  const plotHeight = height - marginTop - marginBottom;
  const domain = useMemo(
    () => collectViewDomain(mode, tracks, plateRoutes),
    [mode, plateRoutes, tracks],
  );

  const xSpan = Math.max(1, domain.maxX - domain.minX);
  const ySpan = Math.max(1, domain.maxY - domain.minY);
  const plotX = (value: number) => marginLeft + ((domain.maxX - value) / xSpan) * plotWidth;
  const plotY = (value: number) => marginTop + ((domain.maxY - value) / ySpan) * plotHeight;
  const zeroX = plotX(0);
  const zeroY = plotY(0);
  const yScale = plotHeight / ySpan;

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
          y={height - 10}
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

        <text x={marginLeft} y={height - 24} className="runway-profile-axis-tick">
          {formatMeters(domain.maxX)}
        </text>
        <text
          x={marginLeft + plotWidth}
          y={height - 24}
          textAnchor="end"
          className="runway-profile-axis-tick"
        >
          {formatMeters(domain.minX)}
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
            />
          ) : null}
          {runwayProfileViewMode !== "side-xz" ? (
            <ProfilePlot
              title="Top view"
              subtitle="Dynamic x-y projection with RNAV horizontal plate and centerline"
              mode="top"
              tracks={profile.aircraftTracks}
              plateRoutes={profile.plateRoutes}
            />
          ) : null}
        </div>
      ) : null}
    </aside>
  );
}
