# Motivation

# What
A Generative Model which predict aircraft control parameters to generate 4d trajectories in terminal airspace. The model will focus on Approach Procedure prediction, and will integrete relastic aero dynamic model and Air Traffic Control regulations. The archetecture of our approach are open for multiple aircrafts interaction.

# Why
Approach Procedure of aircraft in terminal area is highly critical for aiviation safty. Based on data from IATA, 20.2% of aviation accidents from 2015 to 2025 are coursed by runway excursion, which is highly related to approach procedure sechdule.

# Eixsting works
Trajectory prediction based on histrical data.

Weak Physical constraints.

# Approach
## Precise
The model take current aircraft state as input, to predict control parameters in future. 
Compare to predict aircraft state then use PINN to reduce physical error, integrete a true aero dynamic model enable the model to directly generate trajectory that fit the real aero dynamic phycics.

## Constrains
Integreate hard constraints based RNAV navigation guild and obstcle clearence regulations, the generated trajecteries are further limited to a legal subspace.

## Categories
The model take transformer as backbone, 
use ode?
PINN ?

To solve the interaction between multiple aircraft, prevent collision and improve scedule, GNN and RL could be introduced.

# Evaluation
## Baseline
LSTM and transformer model;

## Metrics
Error in lateral and vertical axes.

Regulation fitting.

Fuel using and Schedule Efficiency

## Data
ADS-B data from opensky for 3-5 airports.
Normalization and denosing.

