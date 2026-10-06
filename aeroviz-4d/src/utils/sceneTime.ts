/**
 * sceneTime.ts
 * ------------
 * The real (UTC) time of a traffic scene's clock.
 *
 * A comparison's CZML sits on a synthetic display epoch (the index's `epoch`): every flight's `t = 0` is its own entry, so
 * the viewer clock says nothing about the real time. A traffic scene knows what its epoch is in UTC — an M2 scene's start
 * (`scene.startUtc`: the earliest entry of its groups), an M1 window's start (its commanded flight's `entry_time_utc`) — so
 * the real time at the clock's current position is `startUtc + (clock − epoch)`.
 */

import * as Cesium from "cesium";
import { fetchTrafficArrivals } from "../data/trafficJobs";
import { isCesiumViewerUsable } from "./isCesiumViewerUsable";

/** What the display epoch is in real time. */
export interface SceneTime {
  /** The real UTC time (ISO 8601) at the display epoch. */
  startUtc: string;
  /** The display epoch (ISO 8601): the index's. */
  epoch: string;
}

/** Real UTC milliseconds at the clock's current time. */
export function sceneRealTimeMs(scene: SceneTime, clock: Cesium.JulianDate): number {
  const sinceEpochS = Cesium.JulianDate.secondsDifference(clock, Cesium.JulianDate.fromIso8601(scene.epoch));
  return Date.parse(scene.startUtc) + sinceEpochS * 1000;
}

/** `2026-05-21 17:47:18` (whole seconds, UTC). */
export function formatSceneTime(realMs: number): string {
  return new Date(Math.floor(realMs / 1000) * 1000).toISOString().slice(0, 19).replace("T", " ");
}

/** `2026-05-21` (UTC). */
function formatSceneDate(realMs: number): string {
  return new Date(realMs).toISOString().slice(0, 10);
}

/** `17:47:18 UTC` (whole seconds). */
function formatSceneClock(realMs: number): string {
  return `${formatSceneTime(realMs).slice(11)} UTC`;
}

/**
 * The parts of Cesium's Timeline widget this file reaches into (none is in its typings): `makeLabel`, which writes every tick
 * label; `zoomTo`, which draws the ticks again; `_startJulian` / `_endJulian`, the range it now shows. `resize()` would not do:
 * it returns at once while the widget's size is unchanged, so the labels would stay what they were until the user zoomed.
 */
interface LabelledTimeline {
  makeLabel: (time: Cesium.JulianDate) => string;
  zoomTo: (start: Cesium.JulianDate, stop: Cesium.JulianDate) => void;
  _startJulian: Cesium.JulianDate;
  _endJulian: Cesium.JulianDate;
}

/**
 * Make the viewer's animation widget (its date and time) and timeline (its tick labels) show the scene's REAL UTC time instead
 * of the display epoch's: real = scene start + (clock − display epoch). Returns the function that puts the widgets' own
 * formatters back — unless the viewer has been destroyed by then (the app's teardown destroys it before its components
 * unmount: its widgets are gone, there is nothing to put back, and touching them throws). The timeline is drawn again over the
 * range it shows, so its tick labels change at once.
 */
export function showSceneTimeOnClockWidgets(viewer: Cesium.Viewer, scene: SceneTime): () => void {
  const animation = viewer.animation.viewModel;
  const timeline = viewer.timeline as unknown as LabelledTimeline;
  const original = { date: animation.dateFormatter, time: animation.timeFormatter, label: timeline.makeLabel };
  const redraw = () => timeline.zoomTo(timeline._startJulian, timeline._endJulian);
  animation.dateFormatter = (date) => formatSceneDate(sceneRealTimeMs(scene, date));
  animation.timeFormatter = (date) => formatSceneClock(sceneRealTimeMs(scene, date));
  timeline.makeLabel = (time) => `${formatSceneTime(sceneRealTimeMs(scene, time))} UTC`;
  redraw();
  return () => {
    if (!isCesiumViewerUsable(viewer)) return;
    animation.dateFormatter = original.date;
    animation.timeFormatter = original.time;
    timeline.makeLabel = original.label;
    redraw();
  };
}

const LANDING_DAY = /_(\d{4})(\d{2})(\d{2})T\d{6}Z$/;

/** The UTC day a flight lands on, read off its key (`id_runway_icao24_landingTime`): `YYYY-MM-DD`. */
export function landingDay(flightKey: string): string {
  const match = LANDING_DAY.exec(flightKey);
  if (match === null) throw new Error(`flight key ${JSON.stringify(flightKey)} carries no landing time`);
  return `${match[1]}-${match[2]}-${match[3]}`;
}

const entryCache = new Map<string, Promise<string>>();

/**
 * The real UTC time a flight enters its arrival slice (the roster's `entry_time_utc`), asked of the backend's roster for
 * the day it lands on. A published M1 index carries no real time: no CZML does, and its groups hold only the offsets
 * of their neighbours. Cached by flight key; a failed lookup is not cached.
 */
export function entryUtcOf(airportCode: string, flightKey: string): Promise<string> {
  const key = `${airportCode}/${flightKey}`;
  let found = entryCache.get(key);
  if (found === undefined) {
    found = fetchTrafficArrivals(airportCode, landingDay(flightKey)).then((day) => {
      const arrival = day.arrivals.find((candidate) => candidate.flightKey === flightKey);
      if (arrival === undefined) throw new Error(`${flightKey} is not in the ${airportCode} arrivals roster`);
      return arrival.entryUtc;
    });
    entryCache.set(key, found);
    found.catch(() => entryCache.delete(key));
  }
  return found;
}

/** Forget what was looked up (tests). */
export function clearEntryCache(): void {
  entryCache.clear();
}
