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
 */

import type { ComparisonGroup } from "../data/airportData";

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
      const offsetS = traffic.startOffsetsS[i];
      copies.push({
        ...source,
        id: trafficEntityId(group.group, key),
        position: { ...source.position, epoch: shiftedEpoch(source.position.epoch, offsetS) },
        ...(source.orientation
          ? { orientation: { ...source.orientation, epoch: shiftedEpoch(source.orientation.epoch, offsetS) } }
          : {}),
      });
    });
  }
  return copies;
}
