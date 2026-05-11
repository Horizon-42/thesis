Recent work on 4D aircraft trajectory prediction in terminal airspace can be grouped into five main methodological streams.

First, early data-driven studies mainly use deterministic sequence models, especially LSTM/GRU and seq2seq encoder-decoder networks, to map historical ADS-B or radar tracks to future trajectory points. Representative work
includes terminal-airspace LSTM and CNN-BiGRU models, which perform well for short-term multi-step prediction but usually provide point estimates and have limited ability to represent uncertainty or multiple possible
approach/departure patterns.

Second, probabilistic and generative models have become increasingly important. Clustering plus GMM methods model terminal flows as interpretable route or runway-pattern distributions, while more recent VAE/CVAE, TCN-VAE,
diffusion, Bayesian deep learning, and GAN-style models generate multiple plausible trajectories or confidence intervals. These methods better match the multimodal and uncertain nature of terminal operations, although
physical feasibility and rigorous distributional evaluation remain open issues.

Third, several studies explicitly inject aviation domain constraints into learning. Constrained LSTM models incorporate waypoints, runway direction, top-of-climb, and phase-dependent flight rules; other work learns
uncertainty parameters for new terminal-route design evaluation. This line improves procedural consistency and makes predictions more operationally credible, but often depends on handcrafted constraints and airport-specific
knowledge.

Fourth, some research targets operational regression tasks rather than full trajectory generation, especially ETA/ALT prediction in terminal maneuvering areas. These methods combine trajectory preprocessing, runway-
configuration partitioning, feature engineering, and ensemble learning such as stacking. They are useful for arrival management, but they do not fully solve spatial 4D trajectory prediction or conflict-detection needs.

Finally, recent work is moving toward context-, interaction-, and intent-aware prediction. Weather-aware Bayesian models, socially aware TrajAir-style predictors, altitude-aware CVAE models, and local-history intent
conditioning all try to capture factors beyond the target aircraft’s own past states. New multimodal datasets such as TrajAir and TartanAviation further enable the use of weather, neighboring traffic, ATC speech, and visual
context. Overall, the field is shifting from deterministic trajectory extrapolation toward probabilistic, constraint-aware, and context-conditioned prediction suitable for high-density terminal operations.


# Version 2
Recent AI-based studies on 4D aircraft trajectory prediction in terminal airspace can be grouped into several methodological directions. One line uses deterministic neural sequence models, such as LSTM, seq2seq LSTM, TCN,
  CNN-GRU/BiGRU, to learn temporal dependencies from historical trajectory points and directly regress future positions. These models are simple to train and effective for short-term prediction, but they usually output a
  single future trajectory and therefore have limited ability to represent multimodal terminal operations.

  A second line formulates trajectory prediction as probabilistic or generative modeling. Clustering-GMM methods represent terminal flows as mixtures of typical route patterns, while recent VAE/CVAE, Bayesian deep learning,
  diffusion-based, and GAN-style models generate multiple plausible future trajectories or uncertainty bounds. Compared with deterministic regressors, these models better capture uncertainty caused by runway configuration,
  vectoring, weather, and pilot behavior, but they require more careful evaluation of distribution quality and physical feasibility.

  A third line incorporates aviation-specific constraints into learning. Examples include constrained LSTM models that encode waypoints, runway direction, flight phases, or altitude behavior, and uncertainty-transfer models
  designed for new terminal-route evaluation. These methods improve procedural consistency and reduce physically implausible predictions, but their performance often depends on handcrafted constraints and airport-specific
  knowledge.

  More recent work further moves toward context- and intent-aware prediction. Instead of relying only on the target aircraft’s past positions, these models condition prediction on runway configuration, weather, neighboring
  aircraft, local historical intent, or multimodal terminal data. This direction is promising for improving cross-scenario generalization, because terminal trajectories are strongly shaped by operational context and ATC
  intent rather than by aircraft kinematics alone. Overall, the field is shifting from point-estimate sequence regression toward probabilistic, constraint-aware, and context-conditioned models.


  # Proposaled Approach

  3. Proposed Approach and Methodology

To overcome the limitations of conventional data-driven trajectory prediction methods, this research proposes a novel Physics-Informed, Constraint-Aware, and Multi-Agent Hybrid Framework. The proposed architecture shifts the paradigm from direct coordinate regression to sequence-based control prediction, integrating high-fidelity environmental modeling with cooperative decision-making. The framework consists of three meticulously coupled modules:

3.1. Transformer-Based Aerodynamic Trajectory Generation
Instead of directly predicting future spatial coordinates, which often leads to physically infeasible paths, the foundational module leverages a Transformer-based sequence-to-sequence architecture to predict future aircraft control parameters (e.g., thrust, bank angle, and lift parameters) from historical ADS-B data. The powerful self-attention mechanism of the Transformer captures long-range temporal dependencies in historical flight states. Subsequently, these predicted control sequences are fed into a differentiable aircraft kinematics model to recursively generate the 4D trajectory. This indirect generation process guarantees that the resulting trajectories strictly adhere to aerodynamic limits and realistic flight envelopes.

3.2. High-Fidelity Geospatial and Procedural Constraint Embedding
To bridge the gap between idealized airspace and complex terminal realities, the architecture incorporates a rigorous spatial-procedural modeling engine. This module constructs a precise digital representation of the target airport environments. It explicitly embeds static physical constraints, including terrain topographies and obstacles, as well as dynamic operational constraints such as RNAV (Area Navigation) minimums, standard terminal arrival routes (STAR), and required minimum separation distances between aircraft. By encoding these constraints as differentiable penalty functions or masking layers, the framework ensures that any generated trajectory is not only physically flyable but also strictly compliant with terminal procedure regulations and safety altitudes.

3.3. Multi-Aircraft Interaction and Optimization via GNN and RL
To address the highly interactive nature of dense terminal airspace, the framework introduces a cooperative multi-agent optimization module. A Graph Neural Network (GNN) is deployed to explicitly model the spatiotemporal topology of the airspace, where each aircraft serves as a node and their relative distances and closing speeds define the edges. The GNN efficiently extracts collision risks and interactive traffic context. Finally, a Reinforcement Learning (RL) agent utilizes these graph embeddings to optimize fleet-level operations. The RL reward function is designed to penalize separation infringements and reward efficient scheduling (e.g., minimizing delay and optimizing sequence). Through this synergy, the model not only predicts individual trajectories but proactively resolves multi-aircraft conflicts, thereby enhancing overall airspace capacity and safety.


This research proposes a Physics-Informed, Constraint-Aware, and Multi-Agent Hybrid Framework, consisting of three core modules:

3.1 Transformer-Driven Kinematic Generation: Instead of direct coordinate regression, a Transformer predicts future aircraft control parameters (thrust, bank angle, and lift) from historical ADS-B data. These are fed into a differentiable kinematics model to generate 4D trajectories, ensuring strict aerodynamic feasibility.

3.2 Geospatial and Procedural Constraint Embedding: The model constructs a high-fidelity digital airport environment. Static constraints (terrain, obstacles) and dynamic rules (RNAV minimums, separation standards) are encoded as differentiable penalty functions, ensuring strict operational compliance.

3.3 Multi-Agent Interaction via GNN and RL: To manage dense terminal traffic, a Graph Neural Network (GNN) extracts spatial-temporal collision risks. A Reinforcement Learning (RL) agent then utilizes these graph embeddings to proactively resolve multi-aircraft conflicts and optimize scheduling efficiency.


本研究提出了一种融合物理机理、感知运行约束与多机协同的混合架构，包含三个核心模块：

3.1 基于 Transformer 的运动学轨迹生成： 摒弃直接坐标回归，利用 Transformer 从 ADS-B 数据中预测航空器控制参数序列（推力、坡度角和升力）。随后将其输入至可微运动学模型生成 4D 轨迹，确保结果严格符合空气动力学限制。

3.2 环境与程序约束嵌入： 构建高保真的机场数字环境，将静态（地形、障碍物）与动态运行约束（RNAV 标准、安全间隔）编码为可微惩罚函数，确保轨迹严格遵循终端区安全与程序规范。

3.3 基于 GNN 与 RL 的多机协同： 引入图神经网络（GNN）建模空域时空拓扑以提取碰撞风险，并结合强化学习（RL）利用图嵌入特征进行机队级优化，主动解脱多机冲突并提升调度效率。



To overcome these specific limitations, the proposed framework is designed around three core components.

First, rather than directly regressing spatial coordinates, a Transformer model processes historical ADS-B data to predict future aircraft control parameters, such as thrust, bank angle, and lift. These parameters then drive a differentiable kinematics model to recursively generate 4D trajectories, ensuring strict adherence to realistic aerodynamic envelopes.

Second, high-fidelity geospatial and procedural constraints are explicitly embedded into the generation process. Static physical boundaries like terrain and obstacles, along with dynamic operational rules such as RNAV minimums, are encoded as differentiable penalty functions. This guarantees that the generated trajectories remain fully compliant with terminal procedures and safety altitudes.

Finally, to address the interactive nature of dense terminal operations, the framework aims to explore multi-aircraft conflict resolution. Potential extensions may integrate cooperative optimization techniques—such as Graph Neural Networks (GNNs) for modeling spatiotemporal traffic topology or Reinforcement Learning (RL) for fleet-level scheduling—to proactively maintain separation standards and improve overall airspace efficiency.