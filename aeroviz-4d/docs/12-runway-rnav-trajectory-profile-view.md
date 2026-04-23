# Runway RNAV Trajectory Profile View

## Purpose

Add a runway-scoped 2D trajectory profile panel that opens for a selected
runway and shows aircraft motion in a runway-centered 4D frame:

```text
t = simulation time
x = distance along the selected runway centerline / inbound approach axis
y = signed perpendicular offset from the selected runway centerline
z = altitude / vertical position
```

The panel contains two synchronized 2D spatial views:

- side view: `x-z` dynamics over `t`
- top view: `x-y` dynamics over `t`

The views answer one combined operational/research question:

> For the selected runway, once aircraft enter the RNAV horizontal plate, how do
> their trajectories evolve over time along the runway centerline, vertically,
> and laterally relative to that centerline?

This is not a replacement for the 3D Cesium scene. It is a compact analytical
view that makes runway-aligned approach behavior easier to compare.

## User Demand

The requested feature is:

- Select a runway.
- Pop open a 2D view for that runway.
- Show all qualifying aircraft as single points that update with simulation
  time `t`.
- Use each aircraft's 3D trajectory data.
- Use the runway centerline as the `x` axis.
- Use altitude as the `z` axis.
- Use the perpendicular-to-centerline offset as the `y` axis.
- Provide two 2D views:
  - side view: `x-z` over `t`
  - top view: `x-y` over `t`
- Only include an aircraft in these views while it is inside the horizontal
  plate of an RNAV procedure for that runway.

## Quick Terms

These are the few terms that matter most for this feature.

- `Selected runway`
  The runway end the user chooses, such as `RW05L`. We treat `RW05L` and
  `RW23R` as different because approaches come from opposite directions.

- `Trajectory`
  The path an aircraft follows through space as time changes. In this feature,
  we read each aircraft's trajectory from the loaded CZML and sample its current
  position during playback.

- `Runway centerline`
  The straight line down the middle of the runway. This is the main reference
  axis for the profile views.

- `Threshold`
  The landing end of the runway. We use it as the `x = 0` reference point, so
  aircraft farther out on approach usually have larger positive `x` values.

- `RNAV procedure`
  The published arrival/approach path for a runway. In AeroViz, this comes from
  `procedures.geojson`.

- `Horizontal plate`
  A simple way to say "the sideways area around the RNAV route." If an aircraft
  is inside this area, we include it in the profile panel. If it is outside, we
  leave it out.

- `Included aircraft`
  An aircraft that is currently inside the selected runway's RNAV horizontal
  plate. Only these aircraft appear in the panel.

- `Side view`
  A 2D view of `x` and `z`. In plain words: how far the aircraft is along the
  runway approach, and how high it is.

- `Top view`
  A 2D view of `x` and `y`. In plain words: how far the aircraft is along the
  runway approach, and how far left/right it is from the centerline.

- `Time t`
  The current simulation time from the Cesium clock. It drives the moving point
  positions and any short history trail.

## Terminology

### Selected Runway

The runway chosen by the user, represented with a normalized runway end ident:

```text
RW05L
RW23R
RW32
```

The selected runway end matters. `RW05L` and `RW23R` share the same physical
runway polygon, but they have opposite approach directions and opposite
threshold references.

### Runway 4D Frame

The profile panel uses a runway-centered frame:

- `t`: Cesium simulation time.
- `x`: distance along the selected runway / inbound approach centerline.
- `y`: signed perpendicular offset from that centerline.
- `z`: altitude.

Recommended convention:

```text
t = current Cesium clock time
x = distance to threshold along the inbound final approach course, metres
y = lateral offset from centerline, metres
z = altitude, metres MSL
```

In this convention:

- threshold is `x = 0`
- final approach fixes are generally positive `x`
- an inbound aircraft generally moves from larger `x` toward `0`
- aircraft past the threshold can become negative `x`
- `y = 0` is the runway centerline
- positive `y` is to the right of the inbound approach direction
- negative `y` is to the left of the inbound approach direction

This convention is more useful than using arbitrary east/north axes because it
aligns directly with runway and approach geometry. Time `t` is not a spatial
screen axis; it drives animation and optional trails in both views.

Mathematically, each aircraft trajectory becomes:

```text
x = x(t)
y = y(t)
z = z(t)
```

The side view renders the dynamic projection:

```text
(x(t), z(t))
```

The top view renders the dynamic projection:

```text
(x(t), y(t))
```

### RNAV Horizontal Plate

The user phrase "horizon palate" is interpreted here as **RNAV horizontal
plate**: the 2D plan-view/lateral footprint of the RNAV procedure tunnel.

The existing RNAV visualization already builds a 3D tunnel from
`procedures.geojson` using:

```ts
buildTunnelSections(routePoints, {
  halfWidthM,
  halfHeightM,
  sampleSpacingM,
  nominalSpeedKt,
})
```

For this feature, the inclusion test should use only the horizontal/lateral
part of that tunnel:

```text
inside horizontal plate =
  aircraft lon/lat projects onto one procedure segment
  AND along-segment projection is within that segment
  AND absolute cross-track distance <= procedure lateral half width
```

Altitude should not be used for inclusion. Altitude is the `z` value inspected
in the side view.

### Side View: `x-z` Over `t`

The side view is the first 2D projection:

```text
screen horizontal = x, distance along runway centerline
screen vertical   = z, altitude
time              = t, animation / trail dimension
```

It answers:

- Is the aircraft high or low as it approaches the runway?
- How does altitude evolve as `x` changes over time?
- Where is the aircraft relative to FAF/MAPt/threshold markers?

### Top View: `x-y` Over `t`

The top view is the second 2D projection:

```text
screen horizontal = x, distance along runway centerline
screen vertical   = y, lateral offset from runway centerline
time              = t, animation / trail dimension
```

It answers:

- Is the aircraft left or right of the runway centerline?
- Is it converging toward the centerline as it enters the RNAV horizontal plate?
- Where does it enter, exit, or drift inside the RNAV lateral boundary?
- How do lateral centerline dynamics evolve over time?

In this chart:

- `y = 0` is the selected runway centerline.
- positive `y` is to the right of the inbound approach direction.
- negative `y` is to the left of the inbound approach direction.
- RNAV horizontal plate boundaries should be drawn as lateral envelopes when
  possible.

## Existing Inputs

### `runway.geojson`

Path:

```text
public/data/airports/<ICAO>/runway.geojson
```

Current geometry:

- `FeatureCollection`
- runway surfaces are `Polygon`
- each runway has `zone_type: "runway_surface"`
- landing zones are duplicate/supplemental polygons with
  `zone_type: "landing_zone"`

Important properties:

```ts
airport_ident: string
runway_ident: string       // "05L/23R"
le_ident: string           // "05L"
he_ident: string           // "23R"
length_ft: number
width_ft: number
le_elevation_ft: number
he_elevation_ft: number
```

Runway centerline can be derived from the runway surface polygon:

1. Use only `zone_type === "runway_surface"`.
2. Remove the closing polygon coordinate.
3. Identify the two short edges as threshold edges.
4. The midpoint of each short edge is a runway-end center point.
5. Assign those two endpoints to `le_ident` and `he_ident`.

For the currently generated runway polygons, the first two coordinates form one
threshold edge and the next two form the opposite threshold edge. The
implementation should not rely only on order; it should validate edge lengths
and use the short-edge pair.

### `procedures.geojson`

Path:

```text
public/data/airports/<ICAO>/procedures.geojson
```

Important route feature fields:

```ts
featureType: "procedure-route"
procedureFamily: "RNAV_GPS" | "RNAV_RNP" | ...
procedureIdent: string
procedureName: string
routeId: string
runwayIdent: string | null
branchType: "final" | "transition" | "missed" | string
nominalSpeedKt: number
tunnel: {
  lateralHalfWidthNm: number
  verticalHalfHeightFt: number
  sampleSpacingM: number
}
geometry: LineString<[lon, lat, altM]>
```

For this feature, only route features should be used. Fix point features are
useful for annotation, but not required for filtering aircraft.

Recommended first implementation scope:

- include `procedureFamily` values that start with `RNAV`
- include routes whose `runwayIdent` equals the selected runway end
- prefer `branchType === "final"` for the default plate
- if there are multiple RNAV final routes for the runway, include aircraft that
  are inside any matching route plate

### `trajectories.czml`

Path:

```text
public/data/airports/<ICAO>/trajectories.czml
```

The current app loads this into a `Cesium.CzmlDataSource`.

Aircraft entities:

- have `id !== "document"`
- expose sampled positions through `entity.position`
- can be sampled at `viewer.clock.currentTime`

The profile view should reuse loaded Cesium entities instead of fetching and
parsing CZML again for the first implementation. This keeps the profile view
synchronized with the main timeline.

## Proposed UX

### Where The Control Lives

Add a runway-level action in `ProcedurePanel`:

```text
RW05L [On/Off checkbox] [Profile]
```

Clicking `Profile` should:

- set `selectedProfileRunwayIdent = "RW05L"`
- open `RunwayTrajectoryProfilePanel`
- keep existing procedure visibility unchanged

Why `ProcedurePanel`:

- it already groups RNAV procedures by runway
- the feature is tied to RNAV runway procedures
- it avoids overloading `ControlPanel`, which is currently airport/playback/layer focused

Optional later enhancement:

- clicking a runway polygon in Cesium can set the same selected runway and open
  the same panel

### Panel Shape

The panel should be a floating overlay, similar to other AeroViz panels:

```text
+----------------------------------------------------------+
| Runway RW05L RNAV Profile                           [x]  |
| Time 2026-...Z | 3 aircraft inside plate                 |
| Procedure plate: RNAV(GPS) Y RW05L + ...                 |
| [Side x-z] [Top x-y] [Split]                             |
|                                                          |
| Side view: x-z over t                                    |
| z altitude (m)                                           |
|   ^                                                      |
|   |       UAL123 --> trail over t                        |
|   |             DAL456                                   |
|   |                                                      |
|   |  AAL789                                              |
|   +-------------------------------------------------> x  |
|      15 NM          10 NM          5 NM          THR      |
|                                                          |
| Top view: x-y over t                                     |
| y lateral offset (m)                                     |
|   ^ right                                                |
|   |      RNAV boundary                                   |
|   |        UAL123 --> trail over t                       |
| 0 +---------------- centerline ---------------------> x  |
|   |    AAL789                                            |
|   |      RNAV boundary                                   |
|   v left                                                 |
+----------------------------------------------------------+
```

Required visible information:

- selected runway ident
- current simulation time
- count of aircraft currently inside RNAV horizontal plate
- side view screen x-axis: runway-frame `x`
- side view screen y-axis: runway-frame `z`
- top view screen x-axis: runway-frame `x`
- top view screen y-axis: runway-frame `y`
- top view centerline axis at `y = 0`
- current time `t` shown in the header and represented through animation /
  optional trails
- one point per aircraft
- flight ID label or tooltip

Recommended visible information:

- procedure route name(s) used as the horizontal plate
- threshold marker at `x = 0`
- FAF/MAPt markers if available from `procedures.geojson`
- optional short historical trail for each aircraft in both 2D projections
- horizontal RNAV lateral boundary / plate edges
- current aircraft direction vector or trail fade to show centerline-axis dynamics
- `x`, `y`, `z`, and `t` in tooltip

### View Modes

The panel should support one of these layouts:

- `Side x-z`: runway side view with `x` and `z` spatial axes.
- `Top x-y`: runway top view with `x` and `y` spatial axes.
- `Split`: both views stacked or side-by-side, depending on available panel
  width.

Version 1 can default to `Split` on desktop and `Side x-z` on narrow screens.
The two views must share the same filtering, current time `t`, runway frame,
and aircraft point list.

## Projection Model

### Coordinate Frame

For selected runway end `RW05L`:

1. Resolve the physical runway surface polygon where either:
   - `le_ident === "05L"`, or
   - `he_ident === "05L"`
2. Derive `selectedThreshold` from the matching runway end.
3. Derive `oppositeThreshold` from the other runway end.
4. Build a local tangent-plane approximation around the selected threshold.
5. Define the inbound approach unit vector:

```text
runwayOutVector = normalized(oppositeThreshold - selectedThreshold)
inboundApproachVector = -runwayOutVector
```

6. For each aircraft position:

```text
delta = aircraftXY - selectedThresholdXY
xM = dot(delta, inboundApproachVector)
yM = cross(delta, inboundApproachVector)
zM = aircraft altitude
t = aircraft sample time
```

`xM`, `yM`, and `zM` are the spatial values shown in the side/top projections.
`t` is the simulation-time value used to animate current points and draw
history trails.

### Why Use Threshold Instead Of Runway Center

Runway-centered profiles are easier to read when the landing threshold is `0`.
Aircraft position relative to runway center alone would hide the most important
approach reference: distance remaining to threshold.

## RNAV Horizontal Plate Inclusion

### Build Procedure Plates

For every matching RNAV route:

1. Convert route coordinates into local XY metres.
2. Read `route.properties.tunnel.lateralHalfWidthNm`.
3. Convert half width to metres:

```ts
halfWidthM = lateralHalfWidthNm * 1852
```

4. Store each route segment:

```ts
interface ProcedurePlateSegment {
  routeId: string;
  procedureName: string;
  start: XY;
  end: XY;
  halfWidthM: number;
}
```

### Test Aircraft Against Plate

For each current aircraft point and each plate segment:

```ts
segment = end - start
t = dot(point - start, segment) / dot(segment, segment)
closest = start + clamp(t, 0, 1) * segment
crossTrackM = signedDistance(point, closest, segment)
inside =
  t >= 0 &&
  t <= 1 &&
  Math.abs(crossTrackM) <= halfWidthM
```

Use unclamped `t` for the inside test so an aircraft just beyond either segment
end does not incorrectly count as inside the rectangle cap.

If an aircraft is inside multiple route plates:

- choose the route with the smallest absolute cross-track distance
- store all matching route IDs for debug/tooltip

### Important Boundary

This filtering is intentionally horizontal-only:

- no vertical half-height test
- no glidepath gate
- no OCS clearance check

The side and top views are meant to reveal `x-z-t` and `x-y-t` behavior after
the lateral/RNAV inclusion gate has selected relevant aircraft.

## Data Model

Recommended derived types:

```ts
interface RunwayEndFrame {
  runwayIdent: string;          // "RW05L"
  physicalRunwayIdent: string;  // "05L/23R"
  thresholdLonLat: [number, number];
  oppositeLonLat: [number, number];
  thresholdElevationM: number;
  inboundUnit: { x: number; y: number };
  rightUnit: { x: number; y: number };
}

interface ProcedurePlate {
  runwayIdent: string;
  routeId: string;
  procedureName: string;
  procedureIdent: string;
  branchType: string;
  halfWidthM: number;
  segments: ProcedurePlateSegment[];
}

interface ProfileAircraftPoint {
  flightId: string;
  timeIso: string;
  timeSecondsFromEpoch: number;
  xM: number;
  yM: number;
  zM: number;
  routeId: string;
  procedureName: string;
  groundspeedMps?: number;
  verticalSpeedMps?: number;
  xVelocityMps?: number;
  yVelocityMps?: number;
  zVelocityMps?: number;
  positionLonLatAlt: [number, number, number];
}
```

`xVelocityMps`, `yVelocityMps`, and `zVelocityMps` are optional derived
dynamics:

- `xVelocityMps`: rate of movement along the inbound centerline axis.
- `yVelocityMps`: rate of movement left/right of the centerline.
- `zVelocityMps`: rate of climb/descent.

They are useful because a point alone shows where an aircraft is at the current
time `t`, while a short trail or velocity vector shows how the aircraft is
moving through the `x-z` side view and `x-y` top view.

Recommended view-mode type:

```ts
type RunwayProfileViewMode = "side-xz" | "top-xy" | "split";
```

Optional trail type:

```ts
interface ProfileAircraftTrail {
  flightId: string;
  points: ProfileAircraftPoint[];
}
```

## Implementation Plan

### 1. Add Context State

Extend `AppContext`:

```ts
selectedProfileRunwayIdent: string | null;
setSelectedProfileRunwayIdent: (runwayIdent: string | null) => void;
isRunwayProfileOpen: boolean;
setRunwayProfileOpen: (open: boolean) => void;
runwayProfileViewMode: RunwayProfileViewMode;
setRunwayProfileViewMode: (mode: RunwayProfileViewMode) => void;
```

On airport switch:

- clear selected runway profile
- close profile panel
- reset profile view mode to the default
- keep normal layer toggles and playback speed unchanged

### 2. Add Geometry Utilities

New file:

```text
src/utils/runwayProfileGeometry.ts
```

Responsibilities:

- parse runway surface features
- derive threshold center points
- normalize runway end idents (`05L` -> `RW05L`)
- build `RunwayEndFrame`
- project lon/lat to local XY metres
- project aircraft points into runway frame
- compute centerline and lateral motion deltas from adjacent trajectory samples
- build RNAV horizontal plate segments
- test point-in-procedure-plate

Keep these utilities pure and independent of React/Cesium where possible.

### 3. Expose Loaded Trajectory Data

The current `useCzmlLoader` keeps the `CzmlDataSource` in a private ref. The
profile view needs read access to aircraft positions.

Recommended change:

- add `trajectoryDataSource` to `AppContext`, or
- return it from `useCzmlLoader` and pass it to `RunwayTrajectoryProfilePanel`

Context is cleaner because `FlightTable`, profile view, and future analytics
can share the same source.

Suggested context field:

```ts
trajectoryDataSource: Cesium.CzmlDataSource | null;
setTrajectoryDataSource: (ds: Cesium.CzmlDataSource | null) => void;
```

`useCzmlLoader` should set it after successful load and clear it on cleanup.

### 4. Add Data Hook

New hook:

```text
src/hooks/useRunwayTrajectoryProfile.ts
```

Inputs:

```ts
runwayIdent: string | null
```

Reads from context:

- `viewer`
- `activeAirportCode`
- `trajectoryDataSource`

Fetches:

- `runway.geojson`
- `procedures.geojson`

Outputs:

```ts
{
  status: "idle" | "loading" | "active" | "error";
  runwayFrame: RunwayEndFrame | null;
  plates: ProcedurePlate[];
  aircraftPoints: ProfileAircraftPoint[];
  aircraftTrails: ProfileAircraftTrail[];
  currentTimeIso: string | null;
  error: string | null;
}
```

Clock synchronization:

- subscribe to `viewer.clock.onTick`
- sample positions at `viewer.clock.currentTime`
- throttle React state updates to roughly 5-10 Hz
- update once immediately when panel opens

Sampling aircraft:

```ts
for each entity in trajectoryDataSource.entities.values:
  if entity.id === "document": continue
  if !entity.position: continue
  cartesian = entity.position.getValue(viewer.clock.currentTime)
  if !cartesian: continue
  cartographic = Cesium.Cartographic.fromCartesian(cartesian)
  project to runway frame
  include only if inside at least one RNAV horizontal plate
```

For `t` dynamics:

- sample a short look-back position, for example current time minus 5 seconds
- project both current and previous positions into the same runway frame
- derive `xVelocityMps`, `yVelocityMps`, and `zVelocityMps`
- optionally build a 60-120 second trail sampled at fixed intervals
- keep only trail points that are inside the same RNAV horizontal plate, or draw
  outside-plate trail segments with lower opacity if debug mode is enabled

### 5. Add Panel Component

New component:

```text
src/components/RunwayTrajectoryProfilePanel.tsx
```

Responsibilities:

- render only when `isRunwayProfileOpen` and `selectedProfileRunwayIdent`
- call `useRunwayTrajectoryProfile`
- draw SVG chart(s)
- provide close button
- show loading/missing-data states softly

Rendering approach:

- Use plain SVG for the first version.
- No charting dependency is needed.
- Map x/y with linear scales.
- Reverse x-axis so larger distance appears left and threshold appears right.
- In `side-xz` mode, screen x is runway-frame `x` and screen y is `z`.
- In `top-xy` mode, screen x is runway-frame `x` and screen y is `y`.
- In `top-xy` mode, draw the runway centerline at `y = 0`.
- In `top-xy` mode, draw RNAV lateral plate boundaries when available.
- In both modes, current points are sampled at current time `t`.
- In both modes, optional trails show previous samples over time.
- Use point color by route/procedure or aircraft ID.

### 6. Add ProcedurePanel Entry Point

In each runway group row:

```tsx
<button onClick={() => openRunwayProfile(group.runwayIdent)}>
  Profile
</button>
```

This button should not alter procedure visibility. It only opens the analytical
profile view.

### 7. Add App Overlay

In `App.tsx`, render:

```tsx
<RunwayTrajectoryProfilePanel />
```

inside the overlay container near the other panels.

CSS should place it as a modal/floating analytical panel without blocking the
whole globe unless the user hovers over it:

```css
.runway-profile-panel {
  pointer-events: auto;
  position: absolute;
  right: 16px;
  bottom: 96px;
  width: min(720px, calc(100vw - 32px));
  height: min(420px, calc(100vh - 180px));
}
```

## Missing Data Behavior

The feature should fail softly.

| Missing item | User behavior |
| --- | --- |
| no runway.geojson | panel says runway geometry unavailable |
| selected runway not in runway.geojson | panel says runway not found |
| no procedures.geojson | panel says no RNAV procedures for airport |
| no RNAV route for selected runway | panel says no RNAV horizontal plate for runway |
| no trajectories.czml loaded | panel says no trajectory data loaded |
| no aircraft inside plate | chart stays open with "0 aircraft inside RNAV plate" |

## Tests

### Unit Tests

Add tests for `runwayProfileGeometry.ts`:

- derive runway endpoints from a rectangle polygon
- normalize runway idents
- project an aircraft ahead of threshold to positive `xM`
- project an aircraft past threshold to negative `xM`
- compute lateral `yM` sign consistently
- preserve altitude as `zM`
- include a point inside an RNAV plate
- exclude a point outside lateral half width
- exclude a point beyond segment endpoints

### Hook Tests

Add tests for `useRunwayTrajectoryProfile`:

- loads runway/procedure data from active airport URLs
- samples entities at current Cesium clock time
- includes only aircraft inside selected runway RNAV plate
- updates on clock tick
- resets on airport/runway change

### Component Tests

Add tests for `RunwayTrajectoryProfilePanel`:

- hidden when no runway selected
- shows selected runway in title
- shows current time
- renders one point per included aircraft
- switches between `side-xz`, `top-xy`, and `split` modes
- renders `x-z` side projection in side mode
- renders `x-y` top projection in top mode
- renders a centerline axis in top mode
- renders RNAV lateral boundary guides in top mode when plate data exists
- shows empty state when no aircraft are inside plate

## Acceptance Criteria

The feature is complete when:

- A user can open a profile view for a runway from `ProcedurePanel`.
- The panel defines and uses the runway-centered 4D frame: `t`, `x`, `y`, `z`.
- The side view shows `x-z` dynamics over time `t`.
- The top view shows `x-y` dynamics over time `t`.
- The top view shows the runway centerline at `y = 0`.
- The top view shows RNAV lateral plate boundaries when they can
  be derived from `procedures.geojson`.
- Aircraft are plotted as single points at the current simulation time.
- Aircraft are included only while inside a matching RNAV horizontal plate.
- Side-view vertical axis shows `z` altitude.
- Top-view vertical axis shows `y` lateral offset.
- Screen horizontal axis in both views shows runway-frame `x`.
- Time dynamics are visible through animation, current time display, and
  optional short trails or velocity cues.
- The view shows current simulation time.
- The panel updates during playback without refreshing the page.
- Switching airports closes or resets the profile view.
- Missing data produces readable messages, not console-only failures.

## Open Questions

1. Should `z` be altitude MSL or height above runway threshold?
   Recommendation: show MSL first because CZML positions are MSL/ellipsoid-like
   metres; add an AGL/threshold-relative toggle later.

2. If multiple RNAV procedures exist for the same runway, should the user choose
   one procedure or should the filter union all visible/default RNAV final
   routes?
   Recommendation: union all matching RNAV final routes for version 1, then add
   a route selector if visual clutter becomes a problem.

3. Should each view display only the current point at time `t`, or include a
   short trail?
   Recommendation: current point is required; add a 60-120 second fading trail
   so users can see `x-z-t` and `x-y-t` dynamics without adding a third screen
   axis.

4. Should time `t` also be available as a literal chart axis in a separate
   strip chart?
   Recommendation: not in version 1. Keep side/top views as spatial projections
   animated over `t`; add `t-x`, `t-y`, or `t-z` strip charts later if needed.

5. Should dynamics be displayed as trails or velocity vectors?
   Recommendation: use short fading trails first because they are visually
   simple in both side and top views. Add velocity arrows later if the trails
   are not readable enough.
