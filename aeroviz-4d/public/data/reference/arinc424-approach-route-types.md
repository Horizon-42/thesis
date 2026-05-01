# ARINC 424 Approach Route Types

Local source document:

`data/CIFP/ARINC424-23.pdf`

Reference location:

ARINC 424-23, Section 5.7 Route Type, Table 5-8, Airport Approach (PF) and
Heliport Approach (HF) Records.

The AeroViz procedure-detail branch field currently named `procedureType`
contains the ARINC Route Type value from Airport Approach (`PF`) records. Use
the Table 5-8 meanings for this field:

| Value | Meaning used by AeroViz |
|---|---|
| `A` | Approach Transition |
| `H` | Area Navigation (RNAV) Approach with Required Navigation Performance (RNP) |
| `R` | Area Navigation (RNAV) Approach |

Do not interpret these values with the enroute airway route-type table, and do
not treat them as the separate ARINC procedure availability/category field.
