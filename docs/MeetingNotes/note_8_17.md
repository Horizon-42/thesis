change dynamic model to keep the bank angle paramater continue

d/dt mu = (mu_cmd - mu) / tau_mu, where mu_cmd is the commanded bank angle and tau_mu is the time constant for bank angle response. This allows the model to simulate the aircraft's response to bank angle commands more realistically, accounting for the lag in achieving the desired bank angle.  

test tau_mu with CV; 

thrust; normalize thrust to 1.0;

load factor, -6g - 9g; sesnar +3.8 ~ -1.52; negative load should be very rare; above 2g is also very rare;  0.2g-2g is resonable. should also be continusly.

tau for thrust and load factor could be more instantly.

25 m/s 
speed gate at threshold; based on stall speed; 1.3 stall; define approach speed for each type aircraft; for Cessna (1.2 stall, 1.6 stall) (58knot, 77knot);

optimizer computional expensive; small dt; experiments with different dt;
dynamic model; 

start thesis writting; 5-6 weeks; start at now; intro, data; don't go too detail in experiments; 

contstraints; 
dynamic constraints; stall;