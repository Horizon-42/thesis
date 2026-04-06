# Detailed Note: Regulation-Aware 4D Approach Trajectory Research Plan

## 1. Purpose
This note expands the scratch ideas in note3.md into a structured research and implementation plan for the thesis project.

Core intent:
- Build and evaluate a regulation-aware 4D trajectory model for terminal approach.
- Keep trajectory prediction physically feasible and operationally safe.
- Make algorithm output interpretable through 3D/4D visualization (Cesium + CZML).


## 2. Problem Statement
Design a trajectory planning and prediction framework for approach operations that satisfies:
- Airspace and procedure geometry (approach plate structure).
- Obstacle and terrain clearance constraints.
- Aircraft separation constraints.
- Aircraft performance and attitude limits.
- Time coordination constraints (CTA-oriented 4D trajectory).

The framework should select one feasible path from candidate approach paths, optimize along that path, and compare results to baseline methods.


## 3. Multi-Layer Optimization View
Treat the problem as stacked constraint and decision layers.

### Layer A: Geometry and Procedure Layer
- Inputs: approach plate, runway threshold, FAF/IF/IAF, waypoint graph.
- Output: one selected path (or ranked candidate paths).
- Key question: which route geometry is both operationally valid and optimization-friendly?

### Layer B: Regulation Layer
- Encode aviation regulations into machine-checkable constraints.
- Include obstacle clearance (OCS/OCA-H), separation minima, and procedure envelopes.

### Layer C: Aircraft Performance Layer
- Encode aircraft-type dependent limits (speed, climb/descent, bank, configuration).
- Add weight-dependent corrections (especially stall speed scaling).

### Layer D: 4D Scheduling Layer
- Add time as a state/constraint dimension.
- Include CTA consistency, sequencing spacing, and timeline feasibility.

### Layer E: Objective Layer
Potential objective terms:
- Minimize delay and path inefficiency.
- Minimize regulation-violation risk penalties.
- Minimize control effort/speed profile aggressiveness.
- Improve schedule predictability.


## 4. Regulation Layer (Detailed)

### 4.1 Obstacle and Terrain Clearance
Use PANS-OPS style protection surfaces as hard or soft constraints.

Key geometric concepts already aligned with the project docs:
- Primary protection area.
- Secondary protection area.
- Secondary area gradient behavior (7:1 lateral degradation concept).
- Final segment OCS slope toward threshold (example value in docs: 2.5 percent).

Constraint expression concept:
- For each trajectory point p(t), required clearance margin should remain non-negative.
- Safety condition: height_above_surface(p(t)) >= 0 for all t in approach window.

### 4.2 Separation Constraints
For multi-aircraft operations, enforce pairwise minima:
- Radar case: d_ij(t) >= d_min where d_min is typically 3 NM.
- Non-radar procedural case: time spacing constraints (for example 5-10 min depending on context).

Can be written as:
- Distance form: d_ij(t) >= d_min
- Time-over-fix form: t_j(fix_k) - t_i(fix_k) >= delta_t_min

### 4.3 Rule Prioritization
When constraints conflict, prioritize in this order:
1. Collision and terrain safety (must satisfy)
2. Procedure/regulation compliance
3. Aircraft envelope feasibility
4. Efficiency and scheduling quality


## 5. Geometry and Approach Plate Integration

### 5.1 Path Representation
Represent an approach as a sequence of waypoints and segments:
- IAF -> IF -> FAF -> threshold/runway
- Segment type tags: straight, turn, final

### 5.2 Select One Path
Candidate path set can come from:
- Published procedure branches.
- Geometry variants generated around nominal centerline.

Selection strategy options:
- Feasibility-first filter (regulation + aircraft limits).
- Then optimize cost and choose argmin J(path).

### 5.3 Coordinate and Surface Handling
- Use consistent geodetic model (WGS84) and local ENU for local geometry operations.
- Keep altitude reference consistent (MSL versus AGL handling explicitly documented).


## 6. Aircraft Model and Basic Corrections

### 6.1 Aircraft Type Limits
Define per-aircraft (or per-class) limits:
- Minimum and maximum speed by phase.
- Descent/climb rate bounds.
- Maximum bank angle.
- Optional thrust or acceleration limits.

### 6.2 Weight Effect and Stall Speed Correction
Include first-order weight correction for stall speed:

$$
V_{S,new} = V_{S,ref} \sqrt{\frac{W_{new}}{W_{ref}}}
$$

Implication:
- Heavier aircraft implies higher safe minimum speed.
- This correction should propagate to approach speed floor and maneuvering margins.

### 6.3 Velocity Profile Construction
Construct a feasible speed profile along the selected path:
- Assign target speed windows per segment.
- Enforce acceleration and deceleration limits between waypoints.
- Ensure speed stays above corrected stall margin and below operational limits.

### 6.4 Attitude Variables (Roll, Pitch, Yaw)
Use attitude mainly as consistency checks (or constraints for dynamic model):
- Roll linked to turning geometry and bank limits.
- Pitch linked to vertical speed and flight path angle.
- Yaw aligned with heading/track behavior.

Possible checks:
- |roll| <= roll_max
- |pitch| <= pitch_max
- Yaw/heading change rate within realistic bounds


## 7. Evaluation Design and Performance Measures

### 7.1 Evaluation Questions
- Safety: does the trajectory remain within clearance and separation constraints?
- Feasibility: can the aircraft physically fly it under modeled limits?
- Efficiency: how much delay, path extension, and control effort is introduced?
- Robustness: does performance degrade gracefully under weight and timing perturbations?

### 7.2 Suggested Metrics
Safety metrics:
- Minimum terrain/OCS margin over trajectory.
- Minimum pairwise aircraft separation.
- Count and duration of regulation violations.

Operational metrics:
- CTA error at key fixes and runway threshold.
- Total delay versus schedule.
- Path length and flight time increase versus nominal.

Control/comfort proxies:
- Peak and RMS acceleration.
- Number of high-bank events.
- Speed profile smoothness.

Visualization/interpretability metrics:
- Whether violations can be visually identified in 4D replay.
- Time to diagnose a failure case using the Cesium scene.

### 7.3 Baseline Comparison
Compare against at least one baseline model:
- Baseline A: nominal procedure-following without advanced regulation-aware optimization.
- Baseline B (optional): time-only sequencing with weaker geometry coupling.

Report relative gains and trade-offs:
- Safety margin improvement.
- Delay change.
- Violation reduction.
- Computational cost.


## 8. Literature Review Focus (Find Strong Papers)
Organize review into five buckets:
1. 4D trajectory and CTA/TBO methods.
2. Arrival sequencing/scheduling (MILP, GA, hybrid optimization).
3. Obstacle-aware approach design and PANS-OPS geometry methods.
4. Aircraft-performance-aware trajectory optimization.
5. Safety-constrained prediction and conflict detection/resolution.

Practical search keywords:
- "4D trajectory terminal area CTA"
- "PANS-OPS obstacle clearance surface optimization"
- "arrival sequencing MILP separation constraints"
- "aircraft performance constrained trajectory planning"
- "terrain aware approach path optimization"

Screening criteria for "strong papers":
- Explicit constraints and reproducible formulation.
- Real airport or realistic simulation validation.
- Clear baseline comparisons.
- Sensitivity or robustness analysis.


## 9. Project Integration with Current Codebase
Map this note directly to your implementation pipeline:
- Geometry and regulation surfaces -> OCS geometry utilities and GeoJSON layers.
- Sequencing/prediction outputs -> CZML trajectory generation.
- Evaluation and interpretation -> Cesium 4D playback and timeline validation.

Expected artifacts:
- Constraint specification document (equations + rule source).
- Scenario dataset(s) for at least one airport, then extension to more airports.
- Baseline and proposed model result tables.
- Visualization evidence (screenshots/animations + violation cases).


## 10. Near-Term Work Plan (Proposal to Implementation)

### Week 1-2
- Finalize one-airport scope and selected approach procedures.
- Freeze regulation set and parameter assumptions.
- Build baseline model and evaluation script skeleton.

### Week 3-4
- Implement regulation-aware optimization layers.
- Add aircraft performance corrections (weight and speed-floor logic).
- Run first batch of comparative experiments.

### Week 5-6
- Robustness tests (weight, timing, traffic density).
- Consolidate plots, tables, and Cesium visual evidence.
- Draft chapter sections and narrative for defense.


## 11. Open Decisions to Resolve Early
- Hard constraints versus penalty-based soft constraints for each regulation.
- Single-aircraft focus first or direct multi-aircraft joint optimization.
- Which aircraft classes to include in first evaluation (narrow-body only or mixed).
- Minimum acceptable real-time performance target for the solver.


## 12. One-Page Proposal Abstract Draft (Seed)
This thesis studies regulation-aware 4D trajectory modeling for terminal approach operations under terrain, obstacle, separation, and aircraft performance constraints. The research integrates procedure geometry from approach design with constraint-based optimization and time-coordinated scheduling, then validates outcomes through interactive 3D/4D visualization. Core contributions include: (1) a layered formulation that couples obstacle clearance and separation constraints with aircraft dynamic limits, (2) weight-aware speed-envelope corrections for practical feasibility, and (3) a baseline comparison framework using safety, efficiency, and robustness metrics. The expected result is a reproducible method that improves interpretability and operational validity of optimized approach trajectories for complex airports.
