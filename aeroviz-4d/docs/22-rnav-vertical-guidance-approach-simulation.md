# RNAV Vertical Guidance Approach Simulation

## Status

Date: 2026-05-04

This document is an educational scenario for AeroViz-4D procedure visualization.
It is not operational guidance, pilot training material, ATC phraseology
training, or a substitute for aircraft manuals, operator procedures, chart
notes, NOTAMs, weather minima, or ATC instructions.

## Purpose

Modern RNAV (GPS) / RNP APCH procedures often publish multiple minima on the
same chart, for example:

- `LPV`
- `LNAV/VNAV`
- `LP`
- `LNAV`

When the aircraft, avionics, procedure, weather, temperature limits, database,
and operator authorization all support it, crews usually prefer a minima line
with vertical guidance, such as `LPV` or `LNAV/VNAV`, because it supports a
stabilized descent to a DA rather than a stepdown or level-off style descent to
an MDA.

ATC normally clears the aircraft for the approach procedure. ATC does not
usually choose the minima line for the crew. The crew decides which published
line of minima is legal and appropriate for the aircraft and operation.

## Roles

### Aircraft

The aircraft role includes the flight crew and avionics:

- Loads the RNAV procedure from the current navigation database.
- Confirms runway, transition, fixes, final approach course, missed approach,
  minima, altimeter setting, and temperature restrictions.
- Checks whether the avionics annunciation supports `LPV`, `LNAV/VNAV`, `LP`,
  or only `LNAV`.
- Arms lateral and vertical approach modes when appropriate.
- Flies to DA/MDA and either lands or executes the missed approach.

### Approach Control

Approach control sequences traffic and issues vectors, altitude assignments, and
the approach clearance. Example role:

- "Maintain 3000 until established."
- "Cleared RNAV (GPS) runway 05L approach."
- Handoff to tower near final.

Approach control does not normally say "fly the LPV minima" or "use LNAV/VNAV
minima." The clearance is for the procedure. The aircraft's certified equipment
and crew procedures determine the minima line.

### Tower

Tower manages runway use and landing clearance:

- Confirms landing sequence.
- Issues wind and landing clearance.
- Cancels or changes clearance if runway conditions require it.

Tower also does not normally select `LPV` versus `LNAV/VNAV` versus `LNAV`.

## Technical Distinction

### LNAV

`LNAV` is lateral navigation only in the minima sense. The final OEA is a
lateral obstacle evaluation area. It can say whether an obstacle lies inside the
protected lateral area, but it is not by itself a sloping vertical clearance
surface.

In AeroViz terms:

- `FINAL_LNAV_OEA` is a lateral footprint.
- It should not be presented as vertical guidance.
- It should not be treated as a glidepath.

### LNAV/VNAV

`LNAV/VNAV` adds approved vertical guidance, commonly using baro-VNAV or
WAAS-derived VNAV depending on the installation and procedure. It usually
supports a DA.

In AeroViz terms:

- The lateral footprint may use LNAV OEA dimensions.
- A vertical OCS or path-related surface must be represented separately.
- The visualization should distinguish lateral OEA from vertical OCS.

### LPV

`LPV` uses SBAS/WAAS-style lateral and vertical guidance and can behave
operationally like a precision-like glidepath to a DA, though it is still an
RNAV approach category rather than an ILS localizer signal.

In AeroViz terms:

- LPV should not be collapsed visually into plain LNAV if the goal is to study
  certified obstacle surfaces.
- LPV final geometry has its own W/X/Y surface concepts and should be modeled
  separately from LNAV OEA when source data and rules are available.

### LP

`LP` has more precise lateral performance than LNAV, but no approved vertical
guidance. It normally uses an MDA.

In AeroViz terms:

- LP final OEA is not the same as LNAV final OEA.
- LP lateral geometry narrows toward the runway threshold according to its own
  formulas and dimensions.

## Scenario: Same RNAV Chart, Different Minima

The following is a fictional training-style narrative. Names, weather, fixes,
and values are illustrative.

### Setup

The aircraft is a business jet inbound to Raleigh-Durham. The chart is an
RNAV (GPS) approach to runway 05L. The chart publishes:

- `LPV DA`
- `LNAV/VNAV DA`
- `LNAV MDA`

The aircraft has a current navigation database and WAAS-capable avionics. The
weather is above LPV minima. The operator authorizes LPV approaches.

### Cockpit Briefing

Captain Lin looked down at the approach page.

"RNAV GPS runway zero-five-left loaded. Final approach fix DUHAM, runway
threshold RW05L, missed approach climbs straight ahead then to the hold."

First Officer Chen traced the magenta line on the display.

"Minima available: LPV, LNAV/VNAV, LNAV. WAAS is available, approach
annunciation currently shows LPV armed. We'll brief LPV. If we lose LPV but keep
LNAV/VNAV, we can continue only if we're still before the required point and the
annunciation and minima remain valid. Otherwise we revert to LNAV or go missed
according to company procedure."

The captain nodded.

"Set LPV DA. Cross-check altimeter, temperature note, missed approach altitude.
The important part for us: lateral course is the same charted final, but the
vertical decision changes. We don't invent a glidepath for LNAV."

### Approach Control

Approach control called:

> "AeroViz 452, descend and maintain three thousand. Proceed direct DUHAM."

First Officer Chen answered:

> "Descend and maintain three thousand, direct DUHAM, AeroViz 452."

Inside the cockpit, the FMS drew the turn toward the final approach fix. The
crew checked the approach mode again.

"LPV still armed," Chen said.

Lin replied, "Good. If this were only LNAV, we would treat the vertical path
differently. Here the box is giving approved vertical guidance."

Approach called again:

> "AeroViz 452, maintain three thousand until established on the final approach
> course, cleared RNAV GPS runway zero-five-left approach."

Chen transmitted:

> "Maintain three thousand until established, cleared RNAV GPS runway
> zero-five-left approach, AeroViz 452."

Notice what ATC did not say. ATC did not say:

- "Cleared LPV."
- "Use LNAV/VNAV."
- "Use LNAV minima."

The aircraft was cleared for the RNAV approach. The crew used the chart,
aircraft capability, avionics annunciation, weather, and operator rules to
select LPV minima.

### Intercept

The aircraft approached the final course. Lateral guidance captured first.

Chen called:

"Final approach course alive. LPV still annunciated. Glidepath armed."

Lin watched the vertical path indicator descend toward center.

"Established. Continue."

In an AeroViz visualization, this is where two concepts must not be confused:

- The lateral protected area is the footprint around the final course.
- The vertical guidance path or OCS is a separate vertical object.

The airplane does not fly the colored OEA polygon. It flies the coded path and
approved guidance. The OEA is for obstacle evaluation and design protection.

### Final Approach

At the final approach fix:

"FAF," Chen called. "Glidepath captured. Landing checklist complete."

Lin kept the descent stable. The autopilot followed lateral and vertical
guidance. The crew monitored speed, descent rate, course deviation, and vertical
deviation.

Chen said:

"This is why we prefer LPV when it is available. We get a managed descent to DA
instead of descending to an MDA and leveling."

Lin answered:

"Exactly. But the preference is conditional. If the avionics downgrade, we don't
pretend LPV still exists. The annunciation drives what minima we can use."

### Tower

Approach handed the flight to tower:

> "AeroViz 452, contact tower one two zero point seven."

Chen replied:

> "Tower one two zero point seven, AeroViz 452."

Then:

> "Raleigh Tower, AeroViz 452, RNAV GPS zero-five-left, five miles final."

Tower answered:

> "AeroViz 452, Raleigh Tower, wind zero six zero at eight, runway zero-five-left,
> cleared to land."

Chen read back:

> "Cleared to land runway zero-five-left, AeroViz 452."

Again, tower cleared the runway. Tower did not choose LPV or LNAV/VNAV. The
crew continued using the selected and annunciated minima.

### Decision Altitude

Near DA, Chen called:

"Approaching minimums."

At DA:

"Minimums."

Lin saw the runway environment.

"Landing."

Had the runway not been visible, the response would have been:

"Go around."

The missed approach would then follow the published missed procedure, not an
ad-hoc continuation of the final descent.

## What If LPV Is Not Available?

The same approach can unfold differently.

Before the FAF, the avionics annunciation changes from `LPV` to `LNAV/VNAV`.

Chen says:

"Downgrade. LPV no longer available. LNAV/VNAV available."

Lin replies:

"Check minima. Weather still above LNAV/VNAV DA. We can continue with
LNAV/VNAV if all restrictions are satisfied."

If the avionics downgrade again to `LNAV`, the crew may need to:

- Use LNAV MDA if legal, briefed, and permitted.
- Change vertical mode and descent technique.
- Go missed if the downgrade occurs too late or company procedure requires it.

The main point: the chart may show multiple minima, but the active minima is a
crew and avionics decision, not a tower decision.

## Visualization Consequences For AeroViz

For this project, the simulation implies the following rendering model:

1. Show `FINAL_LNAV_OEA` as lateral-only.
2. Show `LNAV/VNAV OCS` as a distinct vertical surface when GPA/TCH data exists.
3. Show `LPV` / `GLS` W/X/Y surfaces separately when implemented.
4. Do not let a flat plan-view OEA polygon imply vertical guidance.
5. In annotations, label whether an object is:
   - lateral OEA;
   - vertical OCS;
   - display aid;
   - debug estimate;
   - missing source.

## Short Version

The crew usually prefers `LPV` or `LNAV/VNAV` when available because vertical
guidance supports a more stable descent to DA. ATC clears the approach and the
runway; ATC usually does not select the minima. The aircraft, crew, avionics,
published chart, weather, and operator authorization determine whether the
flight may use `LPV`, `LNAV/VNAV`, `LP`, or `LNAV`.
