# Procedure Details Page Plan

## Purpose

Add a new frontend page called **Procedure Details** that explains the RNAV
instrument approach procedures for the **selected airport** in a way that is:

- organized by **runway**
- backed by the AeroViz intermediate RNAV procedure model
- visually rich enough to cover the information a normal RNAV IAP chart carries
- much more readable for users who are **not** already familiar with aviation

This page is not just a viewer for
[11-rnav-procedure-intermediate-data-layer.example.json](/Users/liudongxu/Desktop/studys/thesis/aeroviz-4d/docs/11-rnav-procedure-intermediate-data-layer.example.json).
It should scale to **all generated runway procedures** for the currently
selected airport.

## User Need

The user wants a page that can answer questions like:

- What RNAV procedures exist for each runway of this airport?
- Which fixes belong to each branch?
- What is the final approach path?
- Where is the FAF, MAPt, and missed approach hold?
- What do the chart terms mean in plain language?
- How can I compare the AeroViz interpretation with the real FAA chart?

The page should help a beginner understand the procedure without requiring them
to decode a raw FAA chart first.

## Key Design Principle

This page should behave like a **guided, user-friendly RNAV chart companion**.

It should preserve the important content of a real IAP RNAV chart, but present
it in a more readable structure:

- visual first
- plain-language explanations beside technical data
- runway-by-runway organization
- clear distinction between:
  - published procedure facts
  - AeroViz-derived geometry
  - source/provenance/reference material

It should **not** pretend to be a legal navigation chart.

## Scope

### In Scope

- new Procedure Details page
- airport-aware content using the active airport selection
- runway-grouped procedure navigation
- chart-like visualizations
- beginner-friendly explanation of technical terms
- real chart reference links:
  - FAA link
  - local chart PDF when available
- use of the intermediate RNAV procedure model as the main content source

### Out of Scope

- replacing the 3D Cesium scene
- replacing the existing Procedure Panel
- certified flight-planning or legal navigation use
- editing procedure data in the browser
- rendering every possible ARINC nuance in phase 1

## Proposed Route

Add a dedicated route:

```text
/procedure-details
```

Optional later extension:

```text
/procedure-details?airport=KRDU
/procedure-details?airport=KRDU&runway=RW05L
/procedure-details?airport=KRDU&procedureUid=KRDU-R05LY-RW05L
```

Recommended behavior:

- default airport comes from `AppContext.activeAirportCode`
- if query params exist, they override the initial selection
- switching airports in the main app should also affect this page when opened

## Core Page Structure

Recommended page layout:

### 1. Header

Show:

- airport name
- ICAO / FAA code
- page purpose in one sentence
- research-use-only note

Example:

```text
Procedure Details
KRDU · Raleigh-Durham International
User-friendly RNAV approach procedure explorer for each runway
```

### 2. Runway Navigator

Top-level runway chooser for the selected airport:

- `RW05L`
- `RW05R`
- `RW23L`
- `RW23R`
- `RW32`

Each runway card shows:

- number of RNAV procedures
- available approach modes
- runway threshold elevation
- quick status badges like `LPV`, `LNAV/VNAV`, `LNAV`

### 3. Procedure Overview Panel

For the selected runway/procedure:

- chart name
- procedure family
- procedure variant
- runway
- base branch
- approach modes
- threshold elevation
- source cycle

This should be a short, plain-language summary before the user sees the deeper
technical details.

### 4. Chart-Like Visual Section

This is the heart of the page.

It should include multiple coordinated visualizations instead of one overloaded
diagram.

### 5. Friendly Explanations Section

A compact glossary that explains only the terms shown on the page, such as:

- IAF
- IF
- FAF
- MAPt
- MAHF
- LPV
- LNAV/VNAV
- LNAV
- Missed approach
- Transition
- Branch

Each explanation should be:

- short
- plain-language
- visible near the relevant chart/table

### 6. Source and Reference Section

Show:

- intermediate JSON provenance
- data warnings
- source cycle
- FAA chart link
- local PDF link if available

## Visualizations

The page should contain enough information to cover the main content people look
for in a normal RNAV IAP chart, but split into cleaner views.

### A. Plan View Map

Purpose:

- show the horizontal shape of the procedure
- show all branches, transitions, final approach, and missed approach

Content:

- runway threshold
- centerline
- all fixes
- branch color coding
- merge points
- hold location if present
- north-up orientation
- scale bar

Good additions:

- branch-role legend
- click fix to open fix detail popover
- click branch to highlight matching leg table rows

### B. Vertical Profile Chart

Purpose:

- show altitude constraints along the approach path

Content:

- x-axis: along-track distance to threshold
- y-axis: altitude
- key fixes plotted and labeled
- published altitude constraints
- threshold marker
- missed approach start marker

Important:

- this should use the corrected profile logic from the runway-profile work so
  placeholder zero altitudes do not create fake dips

### C. Segment / Leg Ladder

Purpose:

- explain the procedure step-by-step

Content:

- ordered legs
- branch membership
- path terminator
- start fix
- end fix
- altitude constraint
- speed constraint
- segment type

But the user-facing wording should not stop at the raw code.

Example:

```text
TF: track directly from SCHOO to WEPAS
IF: begin this branch at CHWDR
```

So each leg row should have:

- technical form
- plain-language explanation

### D. Branch Topology Diagram

Purpose:

- help users understand how transitions feed into the final branch

Content:

- transition branches
- merge fix
- base final branch
- missed approach continuation

This can be a simple left-to-right node-link diagram.

### E. Approach Modes / Minimums Comparison

Purpose:

- explain what the listed approach modes mean
- compare available minima or supported modes if the intermediate data contains
  them

Phase 1 options:

- if detailed minimums data exists, show a small comparison chart/table
- if not, show a mode explainer card based on `approachModes`

### F. Fix Detail Cards

Purpose:

- make fixes understandable without forcing the user into raw JSON

Each fix card can show:

- ident
- role
- kind
- coordinates
- elevation if known
- branch usage
- source reference
- plain-language description

## User-Friendly Explanation Strategy

This page must assume the user is not an aviation specialist.

So every technical term shown in the UI should use a two-layer pattern:

### Layer 1: Plain label

Example:

- `FAF` -> `Final Approach Fix`
- `MAPt` -> `Missed Approach Point`

### Layer 2: Short explanation

Example:

- `Final Approach Fix: the point where the aircraft is established on the final descent toward the runway`

Good implementation patterns:

- inline help text under titles
- tooltip or popover on term badges
- glossary side panel
- “What does this mean?” expandable help section

The page should never dump ARINC/CIFP codes on the user without explanation.

## Data Source Strategy

### Primary Source

Use the **intermediate RNAV procedure JSON** contract described in
[11-rnav-procedure-intermediate-data-layer.md](/Users/liudongxu/Desktop/studys/thesis/aeroviz-4d/docs/11-rnav-procedure-intermediate-data-layer.md).

The example file is only a model:

- [11-rnav-procedure-intermediate-data-layer.example.json](/Users/liudongxu/Desktop/studys/thesis/aeroviz-4d/docs/11-rnav-procedure-intermediate-data-layer.example.json)

The new page should consume generated versions of that same structure for real
runways.

### Recommended Generated Frontend Layout

Add a new airport-scoped output family:

```text
public/data/airports/<ICAO>/procedure-details/index.json
public/data/airports/<ICAO>/procedure-details/<procedureUid>.json
```

Where:

- `index.json` contains a summary manifest for the airport
- each `<procedureUid>.json` contains one full runway-specific intermediate
  procedure document

Recommended `index.json` shape:

```json
{
  "airport": "KRDU",
  "runways": [
    {
      "runwayIdent": "RW05L",
      "procedureUids": ["KRDU-R05LY-RW05L"]
    }
  ],
  "procedures": [
    {
      "procedureUid": "KRDU-R05LY-RW05L",
      "chartName": "RNAV(GPS) Y RW05L",
      "runwayIdent": "RW05L",
      "procedureFamily": "RNAV_GPS",
      "approachModes": ["LPV", "LNAV_VNAV", "LNAV"],
      "sourceCycle": "2603"
    }
  ]
}
```

This keeps page bootstrap light and allows lazy-loading the full detail JSON.

## Chart Reference Strategy

The user explicitly wants a reference link to the real RNAV chart.

Two sources are available:

### 1. FAA Reference Link

Always provide a fallback FAA link.

For KRDU, the requested reference is:

```text
https://www.faa.gov/air_traffic/flight_info/aeronav/procedures/application/?event=procedure.results&nasrId=RDU#searchResultsTop
```

Recommended UI copy:

- `Open FAA procedure search`

### 2. Local PDF Chart Data

The repo also has local chart files in:

```text
data/RNAV_CHARTS/KRDU/
```

Important implementation note:

- the browser cannot directly read files from `data/` because they are outside
  Vite `public/`

So for frontend use, local charts must be either:

- copied into `public/` during preprocessing, or
- exposed through a generated manifest that points to a browser-accessible copy

Recommended generated layout:

```text
public/data/airports/<ICAO>/charts/index.json
public/data/airports/<ICAO>/charts/<pdf-file>.pdf
```

Recommended behavior:

- if local chart exists for the selected procedure, show:
  - `Open local chart PDF`
- always also show:
  - `Open FAA reference`

## Visual Mapping From Intermediate JSON

This page should organize the intermediate model into user-facing sections:

### `airport`

Use for:

- page title
- airport identity
- runway navigation context

### `runway`

Use for:

- runway summary
- threshold marker
- threshold elevation
- runway explainer card

### `procedure`

Use for:

- chart title
- procedure family
- approach mode badges
- variant summary

### `fixes`

Use for:

- plan view labels
- fix detail cards
- fix glossary mapping

### `branches`

Use for:

- plan-view geometry grouping
- branch topology diagram
- leg ladder table

### `verticalProfiles`

Use for:

- vertical profile chart
- minima/mode explanations
- approach path description

### `validation`

Use for:

- consistency indicators
- warnings
- “what is certain vs inferred” messaging

### `displayHints`

Use only for:

- defaults
- styling
- non-authoritative UI behavior

It should never override published facts.

## Proposed Frontend Components

Recommended initial component split:

```text
src/pages/ProcedureDetailsPage.tsx
src/components/procedure-details/ProcedureAirportNavigator.tsx
src/components/procedure-details/ProcedureRunwaySidebar.tsx
src/components/procedure-details/ProcedureOverviewCard.tsx
src/components/procedure-details/ProcedurePlanView.tsx
src/components/procedure-details/ProcedureVerticalProfile.tsx
src/components/procedure-details/ProcedureBranchGraph.tsx
src/components/procedure-details/ProcedureLegTable.tsx
src/components/procedure-details/ProcedureFixInspector.tsx
src/components/procedure-details/ProcedureGlossary.tsx
src/components/procedure-details/ProcedureReferencePanel.tsx
```

Utility layer:

```text
src/data/procedureDetailsData.ts
src/utils/procedureDetailsTransforms.ts
```

## Interaction Model

Recommended behavior:

- airport selector at page level
- runway list in left sidebar or horizontal tab strip
- procedure picker within the selected runway
- click fix in any chart -> highlight same fix everywhere
- click branch -> filter leg table and emphasize matching geometry
- click help icon on term -> show short plain-language explanation

## Accessibility and Readability

Because this page is partly educational, readability matters more than dense
expert compression.

Requirements:

- strong visual hierarchy
- high-contrast charts
- no unexplained acronyms
- keyboard-accessible fix selection
- mobile fallback layout
- printable summary mode later if needed

## Data and Pipeline Work Needed

This page likely requires new generation outputs, because today the app has:

- `procedures.geojson`

but not yet a browser-ready directory of full intermediate JSON documents.

Recommended pipeline additions:

### Python / Preprocess Side

- generate one intermediate JSON file per runway-specific procedure
- generate airport-level procedure-details manifest
- generate optional chart manifest for local PDFs

### Frontend Side

- load airport procedure-details index
- lazy-load selected procedure detail file
- transform semantic model into:
  - plan-view geometry
  - profile chart data
  - fix/leg/branch tables
  - glossary explanations

## Test Plan

### Data Tests

- manifest loads for airport
- each procedure file validates against expected shape
- runway grouping is correct
- chart references resolve correctly

### UI Tests

- airport switch reloads runway list
- selecting runway updates overview and charts
- selecting fix highlights corresponding chart/table entries
- glossary help is visible for technical terms
- FAA link always appears
- local chart link appears only when available

### Visualization Tests

- plan view renders all branches
- vertical profile shows key fixes and altitude constraints
- branch graph matches branch relationships from intermediate JSON
- leg table ordering matches source sequence order

## Risks

### 1. Intermediate JSON not yet fully generated for all runways

Mitigation:

- phase 1 can start with KRDU if needed
- but the page contract should be airport-general from day one

### 2. Too much raw aviation jargon

Mitigation:

- force every chart term to have plain-language help text

### 3. Real chart PDF availability differs by airport

Mitigation:

- FAA link is mandatory fallback
- local PDF is optional enhancement

### 4. Page becomes too dense

Mitigation:

- default to one runway + one procedure at a time
- use progressive disclosure for advanced sections

## Recommended Phase Order

### Phase 1

- create page route
- load airport/runway/procedure manifests
- render overview cards
- render plan view
- render vertical profile
- add FAA reference link

### Phase 2

- add branch graph
- add fix inspector
- add glossary help
- add local PDF chart support

### Phase 3

- richer minima comparison
- cross-link with 3D Cesium scene
- deep-linking to runway/procedure/fix state

## Acceptance Criteria

This page is successful when:

- a user can select an airport and see all RNAV procedure data organized by runway
- a non-expert can understand the meaning of major chart elements without
  external aviation knowledge
- the page contains the main informational content of a normal RNAV IAP chart
  in a more readable form
- the page links to a real chart reference for validation
- the page uses the intermediate RNAV procedure model as its primary semantic
  data source

## Recommendation

Proceed with this page as a **separate analysis/education route**, not as a
small extension of the existing `ProcedurePanel`.

The existing panel is a compact layer-control tool.
This new page should be a full runway/procedure explainer surface with room for:

- charts
- tables
- explanations
- references

That separation will keep both experiences cleaner.
