# CIFP Parser Adapter Identifier Notes

## Purpose

AeroViz now uses `cifparse` as the primary CIFP parser, while keeping the
original local fixed-width parser as `local_*` functions in
`python/cifp_parser.py`.

The production exporter now follows the `cifparse` model for branch concepts:

- `procedure_type` and `transition_id` are separate fields in `cifparse`.
- For Airport Approach (`PF`) records, the `cifparse` `procedure_type` field
  is ARINC 424 Section 5.7 Route Type, interpreted with Table 5-8
  Airport Approach (`PF`) and Heliport Approach (`HF`) Records.
- `branchKey` is kept as a stable internal key for existing branch references.
- `branchIdent` is the transition identifier when one exists.
- `ProcedureLeg.source_line` points back to the raw `FAACIFP18` line when possible.
- `FixRecord.source_kind` keeps the old meaning: `airport-local-cifp` or
  `global-cifp-fallback`.
- runway records expose threshold elevation, not ellipsoid height.

The adapter layer in `cifp_parser.py` converts `cifparse` objects into that
AeroViz contract.

## Identifier Differences

### Procedure Identifiers

Both parsers agree on approach procedure identifiers:

| Concept | Local fixed-width parser | `cifparse` | AeroViz output |
|---|---|---|---|
| RNAV(GPS) Y RWY 05L | `R05LY` | `procedure_id = "R05LY"` | `R05LY` |
| RNAV(RNP) Z RWY 23L | `H23LZ` | `procedure_id = "H23LZ"` | `H23LZ` |
| RNAV(GPS) RWY 32 | `R32` | `procedure_id = "R32"` | `R32` |

No adapter rewrite is required for procedure IDs.

### Branch / Transition Identifiers

This is the main mismatch.

FAA fixed-width PF records encode approach route type and transition together in columns
`19:26`. For example:

```text
ACHWDR
AOTTOS
R
H
```

`cifparse` splits those into structured fields:

| Meaning | Local fixed-width parser branch | `cifparse` fields | AeroViz output |
|---|---|---|---|
| approach transition via CHWDR | `ACHWDR` | `procedure_type = "A"`, `transition_id = "CHWDR"` | `branchKey = "ACHWDR"`, `procedureType = "A"`, `transitionIdent = "CHWDR"`, `branchIdent = "CHWDR"` |
| approach transition via OTTOS | `AOTTOS` | `procedure_type = "A"`, `transition_id = "OTTOS"` | `branchKey = "AOTTOS"`, `procedureType = "A"`, `transitionIdent = "OTTOS"`, `branchIdent = "OTTOS"` |
| RNAV(GPS) approach route | `R` | `procedure_type = "R"`, `transition_id = None` or empty | `branchKey = "R"`, `procedureType = "R"`, `transitionIdent = null`, `branchIdent = "R"` |
| RNAV(RNP) approach route | `H` | `procedure_type = "H"`, `transition_id = None` or empty | `branchKey = "H"`, `procedureType = "H"`, `transitionIdent = null`, `branchIdent = "H"` |

Adapter function:

```python
cifparse_branch(primary, procedure)
```

Rules:

1. If `transition_id` is empty, use `final_branch_for_procedure(procedure)` as
   the internal `branchKey`.
2. If `transition_id` exists, combine `procedure_type + transition_id` only for
   the internal `branchKey`.
3. Preserve the `cifparse` `procedure_type` value and `transition_id`
   separately on generated branch records. In Airport Approach records, this
   value should be displayed as an Approach Route Type, not as a generic
   procedure category.
4. Return the stable branch key to `parse_available_branches()` and
   `parse_procedure_legs()`.

This keeps existing route references stable while allowing the UI and key terms
panel to explain the branch-level value as an ARINC approach route type.

Important: do not treat `procedure_type` as just an identifier prefix, and do
not use ARINC Route Type tables for other record classes. For the AeroViz
approach pipeline, these are Airport Approach (`PF`) records, so Section 5.7
Table 5-8 applies. The local reference is
`data/CIFP/ARINC424-23.pdf`, Section 5.7, Table 5-8. The browser-served
summary used by the UI is
`aeroviz-4d/public/data/reference/arinc424-approach-route-types.md`.

Examples used by this project:

| Approach Route Type | Table 5-8 Meaning |
|---|---|
| `A` | Approach Transition |
| `R` | Area Navigation (RNAV) Approach |
| `H` | Area Navigation (RNAV) Approach with Required Navigation Performance (RNP) |

### Fix Identifiers

Fix IDs are mostly the same:

| Concept | Local fixed-width parser | `cifparse` | AeroViz output |
|---|---|---|---|
| terminal waypoint | `KASLE` from raw cols `13:19` | `waypoint_id = "KASLE"` | `KASLE` |
| enroute waypoint | `DUHAM` from raw cols `13:19` | `waypoint_id = "DUHAM"` | `DUHAM` |
| runway threshold | `RW05L` from raw cols `13:19` | `runway_id = "RW05L"` | `RW05L` |

Adapter functions:

```python
build_airport_fix_index(faacifp_path, airport)
build_fix_index(faacifp_path, airport, procedure_legs)
```

Rules:

1. Airport terminal waypoints come from `cifparse.get_terminal_waypoints()`.
2. Runway thresholds come from `cifparse.get_runways()`.
3. Missing procedure fixes are filled from `cifparse.get_enroute_waypoints()`.
4. Enroute fallback still uses the leg `fix_region_code` as a region filter.

Example: `DUHAM` is not an airport-local KRDU terminal waypoint, so it is
resolved from enroute waypoint records and kept as:

```text
source_kind = "global-cifp-fallback"
region_code = "K7"
```

### Region Identifiers

The local parser reads two-character region hints from fixed columns.
`cifparse` exposes separate named fields.

| Record Type | Local fixed-width parser | `cifparse` | Adapter field |
|---|---|---|---|
| procedure leg fix region | `line[34:36]` | `fix_region` | `ProcedureLeg.fix_region_code` |
| terminal waypoint region | `line[19:21]` | `waypoint_region` | `FixRecord.region_code` |
| runway airport region | `line[10:12]` / airport context | `airport_region` | `FixRecord.region_code` |

The adapter keeps these as short region codes like `K7`, because downstream
logic uses them to disambiguate missing fixes.

### Source Line Identifiers

The local parser naturally knows the raw file line number because it reads
`FAACIFP18` line by line.

`cifparse` exposes `record_number`, but for some record families that value is
not the same as the physical file line number. To preserve auditability, the
adapter scans the raw file and builds source-line maps:

```python
build_fix_source_line_maps(faacifp_path)
```

It returns:

```python
airport_fix_source_lines[(airport, ident)] = line_number
enroute_fix_source_lines[ident] = line_number
```

Those maps are used for `FixRecord.source_line`. Procedure leg source lines
still use `record_number + header_line_count`, which matches the procedure
records used by this pipeline.

## Adapter Function Responsibilities

### `load_cifparse_data(faacifp_path_text)`

Loads and caches all `cifparse` collections needed by AeroViz:

- procedures
- terminal waypoints
- runways
- enroute waypoints

It also builds raw source-line maps for fix auditability.

### `cifparse_branch(primary, procedure)`

Builds the stable internal `branchKey` from `cifparse` fields.

This is a compatibility key only. It prevents branch references from breaking,
but the user-facing model should use separated `procedureType` and
`transitionIdent` fields.

Naming note: the generated JSON still uses `procedureType` for compatibility
with the current TypeScript model. Treat that branch-level field as
`approachRouteType` in UI copy and documentation.

### `role_from_cifparse(primary, leg_type, fix_ident, sequence)`

Converts `cifparse` procedure leg metadata into the simplified AeroViz role
labels:

- `IF`
- `FAF`
- `MAPt`
- `MAHF`
- `Route`

This mirrors the old local parser behavior.

### `parse_available_branches(...)`

Production function used by `preprocess_procedures.py`.

Now backed by `cifparse`, but returns the same branch identifiers expected by
the rest of the AeroViz code.

### `parse_procedure_legs(...)`

Production function used by `preprocess_procedures.py`.

Builds `ProcedureLeg` objects from `cifparse` procedure records while preserving:

- sequence number
- stable branch key
- approach route type, currently stored in the compatibility field
  `procedureType`
- transition identifier
- fix ID
- path terminator
- role
- altitude constraint
- fix region code
- source line

### `build_airport_fix_index(...)`

Production function used by `preprocess_procedures.py`.

Builds airport-local `FixRecord` objects from terminal waypoints and runway
records.

Important runway rule:

```text
Use threshold_elevation, not ellipsoidal_height.
```

This fixed an earlier local-parser bug where runway ellipsoid height was used
as threshold elevation.

### `build_fix_index(...)`

Production function used by `preprocess_procedures.py`.

Starts with airport-local fixes, then resolves any missing procedure fix from
enroute waypoint records using the leg region hints.

## Local Parser Retention

The local fixed-width parser is retained under names like:

```python
local_parse_procedure_legs
local_parse_available_branches
local_build_airport_fix_index
local_build_fix_index
```

It is not the production parser anymore. It exists for:

- regression checks
- source audit
- fallback experiments
- comparing future `cifparse` upgrades

The validation script uses the local parser as a baseline:

```bash
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python \
  aeroviz-4d/python/validate_cifp_parser_packages.py --airport KRDU
```

Current KRDU result:

```text
cifparse: complete match for procedure legs and required fix coordinates
arinc424: useful record-level audit decoder, but not the preferred production model
```
