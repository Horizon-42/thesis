/**
 * comparisonTraffic.ts
 * --------------------
 * The recorded aircraft around a commanded flight (a traffic window's `traffic` block, see
 * `ComparisonTraffic`), put on the commanded flight's clock.
 *
 * Every flight the backend serves in the arrival window — the group's own reference and each
 * recorded aircraft alike — has `t = 0` at its OWN terminal-ring entry. The group's own
 * reference needs no shift: the commanded record has `t = 0` at the same kind of entry, its
 * own. A recorded aircraft enters `startOffsetsS[i]` seconds after the commanded one, so it is
 * drawn that much later on the group's clock (`position.epoch`, and the orientation samples
 * with it, moved by that offset).
 *
 * A recorded aircraft is a neighbour of ONE group: its entity here is that group's own copy
 * (`trafficEntityId`), at that group's offset — a flight listed by two shown groups is drawn
 * twice, once on each group's clock — so a group that is not shown has no neighbours on screen.
 *
 * An M2 run is one scene instead (AV47): every group, and the background aircraft that are not groups,
 * are on the scene's clock; the background is drawn once (`scenePackets`), and each group's own
 * reference moves by the group's offset (`sceneReferencePackets`).
 */

import type { ComparisonGroup, ComparisonScene } from "../data/airportData";

/** A backend CZML flight packet, as far as the time shift reads it. */
export interface TimedCzmlPacket {
  id: string;
  position: { epoch: string; [field: string]: unknown };
  orientation?: { epoch: string; [field: string]: unknown };
  [field: string]: unknown;
}

const REFERENCE_ID_PREFIX = "ref-";

/** The flight key a recorded `ref-<flight_key>` id names. */
export function trafficFlightKey(recordedId: string): string {
  if (!recordedId.startsWith(REFERENCE_ID_PREFIX)) {
    throw new Error(`recorded aircraft id ${JSON.stringify(recordedId)} is not a ref- reference id`);
  }
  return recordedId.slice(REFERENCE_ID_PREFIX.length);
}

/** The entity id of one group's copy of a recorded aircraft. */
export function trafficEntityId(group: string, flightKey: string): string {
  return `traffic-${group}/${flightKey}`;
}

/** The entity id of a scene's background aircraft (one for the whole scene, no group's copy). */
export function sceneBackgroundEntityId(flightKey: string): string {
  return `traffic-scene/${flightKey}`;
}

/** The flight keys of a scene's background, each once. */
export function sceneBackgroundKeys(scene: ComparisonScene): string[] {
  return scene.background.recorded.map(trafficFlightKey);
}

/** Every recorded flight key the groups list, each once, in the order first listed. */
export function trafficFlightKeys(groups: readonly ComparisonGroup[]): string[] {
  const keys = new Set<string>();
  for (const group of groups) {
    for (const id of group.traffic?.recorded ?? []) keys.add(trafficFlightKey(id));
  }
  return [...keys];
}

/**
 * The document packet CZML requires first. It carries no clock: the recorded aircraft follow the
 * viewer's clock, which the groups' own sources set.
 */
export const TRAFFIC_DOCUMENT_PACKET = { id: "document", name: "comparison traffic", version: "1.0" } as const;

function shiftedEpoch(epoch: string, offsetS: number): string {
  return new Date(Date.parse(epoch) + Math.round(offsetS * 1000)).toISOString();
}

/** The packet on a clock `offsetS` seconds later: `position.epoch` (and the orientation's) moved by it. */
function shiftedPacket(source: TimedCzmlPacket, id: string, offsetS: number): TimedCzmlPacket {
  return {
    ...source,
    id,
    position: { ...source.position, epoch: shiftedEpoch(source.position.epoch, offsetS) },
    ...(source.orientation
      ? { orientation: { ...source.orientation, epoch: shiftedEpoch(source.orientation.epoch, offsetS) } }
      : {}),
  };
}

/**
 * Each shown group's copies of its recorded aircraft: `packetsByKey` holds the backend's
 * arrival-window packet of every key `trafficFlightKeys` named. The samples are shared with
 * the source packet, never copied (they are read, not changed).
 */
export function trafficPackets(
  groups: readonly ComparisonGroup[],
  packetsByKey: ReadonlyMap<string, TimedCzmlPacket>,
): TimedCzmlPacket[] {
  const copies: TimedCzmlPacket[] = [];
  for (const group of groups) {
    const traffic = group.traffic;
    if (!traffic) continue;
    traffic.recorded.forEach((id, i) => {
      const key = trafficFlightKey(id);
      const source = packetsByKey.get(key);
      if (!source) {
        throw new Error(`${group.group}: the backend served no arrival-window track for recorded aircraft ${key}`);
      }
      copies.push(shiftedPacket(source, trafficEntityId(group.group, key), traffic.startOffsetsS[i]));
    });
  }
  return copies;
}

/**
 * A scene's background aircraft, each once, on the scene clock: `packetsByKey` holds the backend's
 * arrival-window packet of every key `sceneBackgroundKeys` named.
 */
export function scenePackets(
  scene: ComparisonScene,
  packetsByKey: ReadonlyMap<string, TimedCzmlPacket>,
): TimedCzmlPacket[] {
  return scene.background.recorded.map((id, i) => {
    const key = trafficFlightKey(id);
    const source = packetsByKey.get(key);
    if (!source) {
      throw new Error(`the backend served no arrival-window track for background aircraft ${key}`);
    }
    return shiftedPacket(source, sceneBackgroundEntityId(key), scene.background.startOffsetsS[i]);
  });
}

/**
 * The groups' own references on the scene clock. The backend serves every reference at its OWN entry
 * (`t = 0`); a scene group's reference enters `startOffsetS` seconds after the scene start. The packets keep
 * their flight-key ids; the document packet and any packet of no scene group pass unchanged.
 */
export function sceneReferencePackets(
  czml: readonly unknown[],
  groups: readonly ComparisonGroup[],
): unknown[] {
  const offsets = new Map(groups.map((group) => [group.group, group.scene!.startOffsetS]));
  return czml.map((packet) => {
    const offsetS = offsets.get((packet as TimedCzmlPacket).id);
    return offsetS === undefined ? packet : shiftedPacket(packet as TimedCzmlPacket, (packet as TimedCzmlPacket).id, offsetS);
  });
}
