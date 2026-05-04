# 1. Aerodrome Model

# 2. 4D trajectory prediction for single aircraft with constrains

## 2.1 Regulations
### 2.1.1 Aerodrome Operating Restrictions - Visibility
- RVR 2600(1/2 NM)

#### Aerodrome Operating Visibility
##### With Air Traffic Control Tower
For arrivals and departures, the aerodrome operating visibility is in accordance with the following hierarchy: 
1. Runway Visual Range (RVR) for the runway of intended use 
2. Ground visibility (METAR) 
3. Tower visibility 
4. Pilot visibility

**Note** :  Tower visibility  normally just considered as advisory, only replace ground visibility when it is not available or non-representative

##### Without Air traffic Tower
outside ATC operating hours, MF, Unicom, CARS, or advisory sites, etc
For arrivals, the aerodrome operating visibility is in accordance with the following hierarchy: 
1. Runway Visual Range (RVR) for the runway of intended use 
2. Ground visibility (METAR) 
3. Pilot visibility 
For departures, the aerodrome operating visibility is the lowest of the following visibilities: 
- Ground visibility (METAR) 
- Any reported RVR 
- Pilot visibility

### 2.1.2 Obstacle clearance
Minimum altitudes meet obstacle clearance requirements under ISA conditions. The transition altitude is 18,000' within Southern Domestic Airspace. Below this altitude, the pilot must set the aircraft altimeter in accordance with CAR 602.35. In Canada, this area is known as the Altimeter Setting Region.

# 3. Multi aircrafts secedule
## Arrival Sequencing and Scheduling

## Wake Turbulence Separation

## Method
### Mixed Integer Linear Programing

### Dynamic Proraming

### Target Function: First-Come-First-Served as baseline
#### Constrained Position Shifting




# 4. Maybe merge 2 and 3?


# Archetecture
## Multi agent?
GNN?

## Signle Aircraft

## Baseline
LSTM, Transformer without ODE/PINN

## Transformer

## ODE

## PINN
BADA P