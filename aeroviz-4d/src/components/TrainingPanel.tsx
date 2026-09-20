/**
 * TrainingPanel.tsx
 * -----------------
 * The Training task's left-dock panel: the intermediate results of the two-tier
 * plan's stage B — the instruction words read off a real track (the "sentence"),
 * the track those words alone describe, and later the sentence a trained prior
 * says. Design: `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md`.
 *
 * THIS STEP (T1) is the empty state only. The panel does not fetch yet: the
 * manifest reader, its per-set validation and the other two empty states (§4.5 ②
 * and ③ of the design) arrive with T2, and nothing can populate them before the
 * exporter exists (T3). Until then "no export yet" is the truth on every machine —
 * `public/data/airports/<ICAO>/training/` is not written by anything today.
 *
 * The copy here is deliberately actionable rather than a bare "no data": it names
 * the exact path the reader will look at, the command that writes it, and the
 * dev-server restart that a newly created directory needs (AV5 — vite does not
 * watch `public/data`, so a running server 404s a directory created after boot).
 */

import { useApp } from "../context/AppContext";
// The path shown here and the path the reader fetches are ONE definition, in the
// data module — a second copy here would drift the moment the layout moves.
import { trainingIndexPath } from "../data/trainingSample";

export default function TrainingPanel() {
  const { activeAirportCode } = useApp();
  const airport = activeAirportCode || "—";

  return (
    <section className="training-panel" aria-label="Training">
      <header className="training-panel-header">
        <h2>Training</h2>
        <p className="training-panel-lede">
          The instruction words read off each arrival, the track those words alone
          describe, and — once the second layer is trained — the sentence it says.
        </p>
      </header>

      <div className="training-empty" role="status">
        <p className="training-empty-title">
          No Training export for {airport} yet.
        </p>

        <p className="training-empty-label">This panel reads</p>
        <code className="training-empty-path">{trainingIndexPath(airport)}</code>

        <p className="training-empty-label">Written by</p>
        <code className="training-empty-path">
          python run_ts.py instruction_sample_export --vocabulary &lt;…/vocabulary_tau10/instruction_vocabulary.json&gt; --out aeroviz-4d/public/data/airports/{airport}/training/…
        </code>

        <p className="training-empty-note">
          After the first export, restart the dev server — vite does not watch{" "}
          <code>public/data</code>, so a directory created after boot is served as
          the SPA fallback instead of JSON. Kill the <code>vite</code> node
          process, not the <code>npm run dev</code> wrapper.
        </p>
      </div>
    </section>
  );
}
