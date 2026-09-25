/**
 * The Pilot aircraft catalog as the backend writes it (`aeroviz_backend/simulation_backend.aircraft_catalog`, read from
 * `GET /simulation/aircraft` on 2026-09-25): every type today is a single-valued preset, so its target speed IS the
 * lower end of its range and the upper end is 20 kt above it — the threshold speed gate's window at the Pilot mass,
 * [V_ref,lo, V_ref,hi + 20 kt]. The speeds are rounded to 0.1 kt; every other field is the backend's own value.
 */
import type { PilotAircraftConfig } from "../pilotClient";

export const A320_CONFIG: PilotAircraftConfig = {
  code: "A320",
  name: "Airbus A320-200",
  category: "narrow_body",
  massKg: 78000,
  wingAreaM2: 122.6,
  maxThrustN: 240000,
  approachThrustGuessN: 40000,
  terminalSpeedKt: 147.8,
  terminalSpeedMinKt: 147.8,
  terminalSpeedMaxKt: 167.8,
  finalApproachMinNm: 5,
  finalApproachMaxNm: 10,
  finalApproachLateralHalfWidthNm: 0.8,
  finalApproachGlideAngleDeg: 3,
  thresholdCrossingHeightM: 15,
};

export const B77W_CONFIG: PilotAircraftConfig = {
  code: "B77W",
  name: "Boeing 777-300ER",
  category: "wide_body",
  massKg: 351530,
  wingAreaM2: 436.8,
  maxThrustN: 1026000,
  approachThrustGuessN: 140000,
  terminalSpeedKt: 176.2,
  terminalSpeedMinKt: 176.2,
  terminalSpeedMaxKt: 196.2,
  finalApproachMinNm: 6,
  finalApproachMaxNm: 12,
  finalApproachLateralHalfWidthNm: 1,
  finalApproachGlideAngleDeg: 3,
  thresholdCrossingHeightM: 15,
};
