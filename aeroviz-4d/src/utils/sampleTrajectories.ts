/**
 * sampleTrajectories.ts
 * ---------------------
 * Pure helpers for "show only a random subset of trajectories" — used by both the
 * normal trajectory layer (sample flight entities) and the optimizer-comparison layer
 * (sample flight groups from the comparison index, then load only the CZML files those
 * groups live in). Kept side-effect-free so the selection logic is unit-testable.
 */

import type { ComparisonGroup, ComparisonIndex } from "../data/airportData";

export type Rng = () => number;

/**
 * Pick up to `count` items uniformly at random (a partial Fisher–Yates shuffle).
 * `count <= 0` or `count >= items.length` returns a copy of all items (order preserved),
 * i.e. "no sampling — show everything".
 */
export function sampleSubset<T>(items: T[], count: number, rng: Rng = Math.random): T[] {
  if (!(count > 0) || count >= items.length) return [...items]; // 0 / NaN / >= length ⇒ all
  const pool = [...items];
  const picked: T[] = [];
  for (let i = 0; i < count; i += 1) {
    const j = i + Math.floor(rng() * (pool.length - i));
    const swap = pool[i];
    pool[i] = pool[j];
    pool[j] = swap;
    picked.push(pool[i]);
  }
  return picked;
}

export interface ComparisonSelection {
  /** The sampled groups. */
  groups: ComparisonGroup[];
  /** Distinct CZML files that must be loaded to render the sampled groups. */
  files: string[];
  /** Entity ids to reveal (everything else stays hidden). */
  shownEntityIds: Set<string>;
}

/**
 * Filter the comparison index to the eligible (non-empty) groups for the selected runway
 * (`null` = every runway), randomly sample `count` of them, and collect the CZML files +
 * entity ids needed to render them.
 *
 * Failed-optimization groups (`status: "failed"`) are kept in the pool: they carry only a
 * dark-red `ref-` track (no opt/sim), and are sampled alongside the solved groups so the
 * failures show up in proportion to the sample count (gated by the Reference toggle).
 */
export function selectComparisonGroups(
  index: ComparisonIndex,
  selectedRunway: string | null,
  count: number,
  rng: Rng = Math.random,
): ComparisonSelection {
  const eligible = index.groups.filter(
    (group) =>
      group.entities.length > 0 &&
      (selectedRunway === null || group.runway === selectedRunway),
  );
  const groups = sampleSubset(eligible, count, rng);
  const files = [...new Set(groups.map((group) => group.czml))];
  const shownEntityIds = new Set<string>();
  for (const group of groups) {
    for (const entityId of group.entities) shownEntityIds.add(entityId);
  }
  return { groups, files, shownEntityIds };
}
