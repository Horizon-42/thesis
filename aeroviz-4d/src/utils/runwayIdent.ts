/**
 * runwayIdent.ts
 * --------------
 * The single source of truth for runway-identifier handling, so every layer
 * that keys off a runway can speak one language.
 *
 * The app sources runways in two spellings:
 *   • landings manifests / optimizer-comparison groups → bare ("05L", "32")
 *   • CIFP procedure render bundles                     → "RW"-prefixed ("RW05L")
 *
 * `normalizeRunwayIdent` collapses both to the canonical "RW"-prefixed form, and
 * `runwayMatchesSelection` is the predicate the UI uses to decide whether a layer's
 * runway-scoped feature belongs to the currently selected landing runway. This is
 * what makes the Landing-Runway selector an orthogonal axis: trajectories, the
 * procedure list, and any future runway-scoped layer all filter through this one
 * predicate without depending on one another.
 *
 * Note: `src/data/rnavInitialFixCandidates.ts` deliberately keeps its own,
 * non-prefixing normalizer (it compares values that are already in one spelling),
 * so it is intentionally NOT consolidated here.
 */

/** Canonicalise a runway identifier to the "RW"-prefixed, upper-cased form. */
export function normalizeRunwayIdent(runwayIdent: string): string {
  const trimmed = runwayIdent.trim().toUpperCase();
  return trimmed.startsWith("RW") ? trimmed : `RW${trimmed}`;
}

/**
 * Does a runway-scoped feature belong to the selected landing runway?
 *
 * @param selected   the active Landing-Runway selection ("05L" / "RW05L" / null).
 *                   `null` means "All runways" → matches everything.
 * @param candidate  the feature's runway identifier in either spelling.
 */
export function runwayMatchesSelection(selected: string | null, candidate: string): boolean {
  if (selected === null) return true;
  return normalizeRunwayIdent(selected) === normalizeRunwayIdent(candidate);
}
