# Evaluation
Adding correct speed gate. done

# Runway Conditions
## Runway Threshold as Origin 

## Airport Reference as Origin

## Explicit Runway Threshold and Airport Reference as Origin




# Control Legs
Before: Every time step
Now: Only predict 32 legs(abliation), and final arrival time T

Given T:
Good results when give the real fly time.

Further expiriments:
How to predict T? 

# Intention prediction
Using VAE to predict procedural intention;
not good when using only 1 anchor point

Further experiments:
Do multi step prediction with multiple anchor points, then maybe the intention could be really learned.


# Constraints
Adding procedural constraints. experiments
## Control prediction
Only soft constraints with loss function. 
## State prediction
Soft and hard constraints

# Multi-agent
Not yet started;