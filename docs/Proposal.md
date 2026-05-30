# 

# Motivation
Approach procedures are safety-critical components of aircraft operations. According to IATA statistics, at least 37.7% of aviation accidents directly involve issues with flight path control, energy management, or procedural execution during the approach and landing phases. These safety concerns highlight the need for accurate and physically reliable modeling of approach trajectories, particularly under operational constraints.

To address this need, we propose a physics-informed and constraint-aware transformer-based generative framework for 4D approach trajectory prediction. By predicting aircraft control parameters and propagating them through a realistic aerodynamic model, the framework generates physically feasible trajectories while incorporating multi-level operational constraints—ranging from spatial terrain boundaries and procedural RNAV minima to dynamic multi-aircraft separation requirements.


# Eixsting works
Current AI-driven 4D trajectory prediction methods have transitioned from deterministic sequence models like LSTM and TCN to probabilistic, context-aware approaches—including GMMs, VAEs, and diffusion models that incorporate runway, weather, or traffic data. This shift has notably enhanced the capacity to model uncertainty and multimodal terminal operations. Nevertheless, the majority of methods still focus merely on directly predicting future positions, failing to explicitly model aircraft control behavior, aerodynamic feasibility, and procedure-level constraints.

# Approach
To address these limitations, our proposed framework focuses on three core parts. First, instead of predicting spatial coordinates directly, we use a Transformer to predict control parameters like thrust and lift. These parameters drive a differentiable kinematics model to generate realistic 4D trajectories. Second, we ensure safety and compliance by encoding physical boundaries and operational rules as differentiable penalty functions. Finally, to manage dense traffic, we will use Graph Neural Networks or Reinforcement Learning for multi-aircraft conflict resolution. This allows us to actively maintain safe separation and improve overall airspace efficiency.

# Evaluation
For the experimental evaluation, we use historical ADS-B flight data from the OpenSky Network, which is processed through normalization and denoising. High-precision airport terrain data is sourced from USGS digital elevation models. To validate our framework, we compare it against two types of baseline models: deterministic sequence models, including LSTM, TCN, and Transformer, and probabilistic models such as GMM or VAE.

The performance is evaluated using five key metrics. First, we measure prediction accuracy through lateral and vertical errors. Second, we assess compliance by testing how well the trajectories follow flight regulations and kinematics constraints. Finally, we evaluate the model’s ability to handle multiple aircraft by measuring its performance in maintaining safe separation.

