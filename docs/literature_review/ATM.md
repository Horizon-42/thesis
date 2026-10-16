# ATM components
## Airspace organization and management

## Aerodrome operations
inter alia, lighting, taxiways, runways, including exits, and precise surface guidance to improve safety and maximize aerodrome capacity in all weather conditions
1. runway occupancy time will be reduced; 
2. the capability will exist to safely manoeuvre in all weather conditions while maintaining capacity;

## Demand and capacity balancing
Demand and capacity balancing will strategically evaluate system-wide traffic flows and
aerodrome capacities to allow airspace users to determine when, where and how they operate, while
mitigating conflicting needs for airspace and aerodrome capacity.
1. strategic stage，assets will be optimized in order to maximize throughput, thus providing a basis for predictable allocation and scheduling;
2. pre-tactical stage， adjustments will be made to assets, resource allocations, projected trajectories, airspace organization, and allocation of entry/exit times for aerodromes and airspace volumes to mitigate any imbalance

## Traffic synchronization
maintenance of a safe, orderly and efficient flow of air traffic
1.  (4-D) trajectory control and negotiated conflict-free trajectories
2. choke points will be eliminated
3. optimization of traffic sequencing will achieve maximization of runway throughput

## Airspace user operations

## Conflict management
airspace organization and management, demand and capacity balancing, and traffic synchronization;
separation provision; and collision avoidance.

## ATM service delivery management
seamlessly from gate to gate for all phases of
flight and across all service providers.

## NextGen and SESAR by planning stage
Items without a dagger are from `docs/literature/aviation/next_gen_history.md`, Enea & Porretta, *A Comparison of 4D-Trajectory Operations Envisioned for NextGen and SESAR*, and, for the SESAR column, the *SESAR Deployment Programme 2024* (`docs/literature/aviation/sesar/papers/`; AF = ATM functionality). Items marked † are background knowledge, not yet checked against a source in the repository.

| Stage | NextGen (US) | SESAR (Europe) |
|---|---|---|
| **Strategic** (months to the day before): capacity planning, airspace design | 2004 integrated plan and 2007 concept of operations; a NAS segment implementation plan in 5-year segments; a shift from "curb-to-curb" to "gate-to-gate" | Flexible airspace management and free route airspace (AF3, SESAR Deployment Programme 2024, Sec. 3.3, p.58) |
| **Pre-tactical** (day of operation, before departure): collaboration, flow adjustment | Collaborative traffic flow management†; multiple runway operations and surface operations with data sharing | Airspace users agree a 4D trajectory with the air navigation service provider and airports, the *Reference Business Trajectory* (RBT); network collaborative management (AF4, same, Sec. 3.4, p.89); airport collaborative decision making† |
| **Tactical** (in flight): separation, sequencing | Performance-based navigation, ADS-B surveillance, Data Comm and decision-support tools; time-based flow management metering of arrivals† | Advanced FMS for controlled time of arrival, data communications, ADS-B and decision-support tools; extended arrival management, AF1 (same, Sec. 3.1, p.16) |
| **Across stages** | System Wide Information Management (SWIM); trajectory-based operations (TBO) | SWIM; the 4D trajectory shared among all stakeholders |

Common idea: a shift from clearance-based control to trajectory-based operations, in which the aircraft and the ATM system agree on a trajectory with time, constrained at key points by a controlled time of arrival (CTA).

Difference: US airspace is homogeneous, and capacity constraints occur mainly at major airports and in terminal airspace. European airspace is fragmented, and the main constraints are en route. The US therefore stresses terminal-area arrival sequencing, Europe en-route and network-level coordination.

## Tactical-phase gap: what arrival-management tools do and do not do
Claim to support: arrival-management tools schedule and sequence aircraft upstream of the terminal area and at metering points, while the lateral and vertical manoeuvres inside the terminal area (radar vectors) remain controller-issued.

Page numbers are PDF page indices, not printed page numbers. Sources are 2023–2026. The earlier sources of this section (NASA 2015 TSAS paper, ATD-1 ConOps 2013, SESAR E-AMAN factsheet) are no longer cited; their PDFs stay in `docs/literature/runway_assignment/nasa_eurocontrol/`.

| # | Statement supported | What the source says | File, PDF page |
|---|---|---|---|
| 1 | US tools meter arrivals in time | TBFM is already active in the national airspace; the FAA's own plan still needs technology enhancements to provide efficient sequencing and to extend time-based metering to the runway | `atc_modernization/papers/GAO-24-105254_ATC_Modernization_Program_Management.pdf`, p.27 |
| 2 | The US terminal-area tool is not in service | The FAA planned to add a Terminal Sequencing and Spacing (TSAS) tool so that control towers could use time-based metering in the terminal area to merge, sequence and space aircraft. In March 2023 FAA officials said TSAS was deferred and no new timeline had been set | same, p.27 |
| 3 | US metering extends upstream | TBFM time-based metering was expanded to more locations, with extended metering, integrated departure/arrival capability and interval-management support | `tbo/FAA_NextGen_Final_Report_2025.pdf`, p.28 |
| 4 | Europe sequences upstream of the terminal area | Extended arrival management extends the AMAN horizon from the terminal area perimeter upstream into the en-route phase; upstream controllers receive arrival management advisories for earlier sequencing, and crews absorb the resulting delay with their flight management systems | `sesar/papers/SESAR_Deployment_Programme_2024.pdf`, p.16 |
| 5 | The European horizon is at least 180 NM | Common Project 1 requires extended arrival management on a horizon of 180 NM, with sequencing and metering conducted en route before top of descent | same, p.18 and p.19 |
| 6 | Radar vectors to final are a controller task | "Vector arriving aircraft to intercept the final approach course" is a controller procedure (FAA Order JO 7110.65BB, 5-9-1) | `arrival_separation/papers/7110.65BB_chap5_section_9_Radar_Arrivals.html`, paragraph 5-9-1 |

Dropped with the old sources: the speed-only delay absorption (about two minutes in the terminal area) and the description of controllers' vectoring within a scheduling tool. No current source in the repository states them.

Not covered: the 2024 SESAR report says nothing here about how European terminal-area control works below the E-AMAN horizon, and the EUROCONTROL AMAN guidelines (2010) were not read in full. `sesar/papers/SESAR_3_JU_Annual_Activity_Report_2024.pdf` is downloaded but not cited.

Draft text:

> In the tactical phase, programs such as NextGen and SESAR provide arrival sequencing and scheduling tools~\comment{citation}. These tools assign scheduled times at metering points and sequence arrivals upstream of the terminal area~\comment{citation}. Within the terminal area, however, lateral and vertical manoeuvres such as radar vectors remain controller-issued~\comment{citation}.
