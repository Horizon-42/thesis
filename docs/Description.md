基于 ADS-B 数据的多飞机机场进近阶段 4D 航迹预测研究

项目描述：
本项目旨在利用历史 ADS-B（Automatic Dependent Surveillance–Broadcast）数据，
构建并训练多飞机场景下的 4D（经度、纬度、高度、时间）航迹预测模型。
研究重点聚焦于飞机在机场进近阶段（approach phase）的飞行轨迹建模与预测。

具体而言，本项目将选取北美地区 3–5 个具有代表性的繁忙机场，
提取飞机从接近终端空域（Terminal Area）至跑道着陆前阶段的轨迹数据，
重点分析该阶段中飞机的空间位置（latitude, longitude）、几何高度（geoaltitude / GNSS altitude）、
速度及航向等特征。

研究目标包括：
1. 构建高质量的机场进近阶段 4D 轨迹数据集；
2. 分析多飞机环境下的轨迹分布特征与交互行为；
3. 设计并训练基于机器学习/深度学习的轨迹预测模型（如序列模型或生成模型）；
4. 提升复杂空域环境下短期轨迹预测的准确性与鲁棒性。

数据需求：
本项目主要使用 OpenSky 历史数据库中的状态向量数据（state_vectors_data4），
重点字段包括：
time, icao24, callsign, lat, lon, geoaltitude, velocity, heading, vertrate, onground 等。

使用方式：
所有数据仅用于非商业性的学术研究与模型开发，不涉及任何商业用途或数据再分发。

单位/背景：
（填写你的学校 / 实验室 / 公司，如适用）
本项目为科研/课程/毕业论文相关研究。


Project Title:
4D Trajectory Prediction for Multi-Aircraft Airport Approach Using ADS-B Data

Project Description:
This project aims to develop and train machine learning models for 4D trajectory prediction 
(longitude, latitude, altitude, and time) in multi-aircraft scenarios using historical 
ADS-B (Automatic Dependent Surveillance–Broadcast) data.

The study focuses on the aircraft approach phase, specifically modeling and predicting 
flight trajectories from the terminal airspace surrounding selected airports to the 
final approach before landing.

We plan to select 3–5 major airports in North America and extract trajectories of aircraft 
operating in their terminal areas. The dataset will include spatial and kinematic features 
such as latitude, longitude, geometric altitude (geoaltitude / GNSS altitude), velocity, 
heading, and vertical rate.

The main objectives of this project are:
1. To construct a high-quality dataset of 4D trajectories during the airport approach phase;
2. To analyze trajectory patterns and interactions among multiple aircraft in congested airspace;
3. To design and train machine learning / deep learning models (e.g., sequence-based or generative models) for trajectory prediction;
4. To improve the accuracy and robustness of short-term trajectory prediction in complex airspace environments.

Data Requirements:
The project requires access to the OpenSky historical database, particularly the 
state_vectors_data4 table. The key fields include:
time, icao24, callsign, lat, lon, geoaltitude, velocity, heading, vertrate, and onground.

Usage:
All data will be used strictly for non-commercial academic research and model development. 
No redistribution or commercial use of the data is intended.

Affiliation:
(To be filled with your university / research group / organization, if applicable)
This project is conducted as part of academic research / coursework / thesis work.