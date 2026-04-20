# RNAV Procedure Layer Plan(Already Done)

## Summary

Add a new AeroViz-4D procedure layer that reads FAA CIFP data, extracts an RNAV procedure for a selected airport, and visualizes it in Cesium as both a plan-view route and a 3D/4D approach tunnel.

Initial implementation target: **KRDU RNAV/GPS procedure `R05LY`**, using local CIFP cycle data from `data/CIFP/CIFP_260319`.

This layer is for research visualization, not certified navigation. Geometry should be clearly marked as derived/visualized from CIFP records with simplified tunnel assumptions.

## Key Changes

- Add a Python preprocessing pipeline that parses CIFP procedure records and outputs a browser-ready GeoJSON file:
  - Input: `data/CIFP/CIFP_260319`
  - Airport: `KRDU`
  - Procedure: `R05LY`
  - Output: `aeroviz-4d/public/data/procedures.geojson`
- Add a frontend `procedures` layer:
  - Add `procedures` to the app layer state and ControlPanel toggle.
  - Add a Cesium hook to load `procedures.geojson`.
  - Render the RNAV route as a colored plan-view polyline.
  - Render fix points with labels.
  - Render a translucent 3D tunnel around the final/intermediate approach path.
- Define a minimal procedure GeoJSON schema:
  - `LineString` features for procedure legs with 3D coordinates `[lon, lat, altitudeMeters]`.
  - `Point` features for fixes.
  - Properties include airport, procedure ident, runway/variant, leg type, sequence, altitude constraints, source cycle, and warnings.
- Implement simplified tunnel geometry:
  - Default lateral half-width: `0.3 NM`.
  - Default vertical half-height: `300 ft`.
  - Sample spacing: about `250 m`.
  - Generate connected cross-sections along the procedure path.
  - Use nominal speed, default `140 kt`, to assign optional 4D time gates along track.
- Support procedure profile data in the data model, but defer a polished 2D profile panel until after the first 3D layer works.

## Implementation Notes

- CIFP extraction should start from `IN_CIFP.txt` to identify KRDU procedures, then read matching records from `FAACIFP18`.
- First parser support should cover common straight-leg RNAV records such as `IF` and `TF`.
- `RF`, `CF`, and missed-approach legs may be included with warnings if simplified.
- If a fix coordinate cannot be resolved, skip that leg and add a warning instead of crashing.
- Generated GeoJSON should remain deterministic so it can be committed and reviewed when desired.

## Test Plan

- Python tests:
  - Verify KRDU `R05LY` can be found in CIFP.
  - Verify coordinate decoding.
  - Verify generated GeoJSON has route and fix features.
  - Verify unresolved fixes produce warnings instead of exceptions.
- TypeScript tests:
  - Verify tunnel cross-sections are generated from 3D route points.
  - Verify empty or malformed procedure data does not crash the layer.
  - Verify layer cleanup removes Cesium entities.
- Manual validation:
  - Run the preprocessing script for KRDU.
  - Start AeroViz-4D.
  - Toggle the procedure layer on/off.
  - Confirm route, fixes, and tunnel appear near KRDU runway 05L.
  - Confirm existing trajectory/CZML loading still works.

## Assumptions

- Target document path: `aeroviz-4d/docs/08-rnav-procedure-layer-plan.md`.
- Initial airport/procedure: `KRDU R05LY`.
- First version prioritizes visualization and thesis/research usefulness over complete ARINC 424 procedure semantics.
- The 3D tunnel is an approximate protected corridor, not an authoritative TERPS/PANS-OPS containment surface.
