"""Runway-intent R0 readout: the five airports' R0a (`runway_intent_r0.json`) and R0b (`hypotheses.json`)
in one table set, written to the campaign folder as `readout.md` / `readout.json`.

Plan `docs/2026-09-13_runway_intent_plan.zh.md` §11. R0b is re-summarised here with the PAIRED
`runway_hypotheses.summarise` (every selector's FDE change against the assigned runway on the same
flights), from the per-flight records each run stored.

    conda run -n aeroviz python 4dTrajectory/ts_transformer/docs/runway_intent_r0_readout.py
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO_ROOT / '4dTrajectory'), str(REPO_ROOT), str(REPO_ROOT / 'geokit' / 'src')]
# The PAIRED summary (commit 9e3ac49), recomputed from each run's stored per-flight records, so
# every airport is read with one definition whichever code wrote its hypotheses.json.
from ts_transformer.experiments.runway_hypotheses import summarise as summarise_hypotheses  # noqa: E402

OUT = REPO_ROOT / '4dTrajectory' / 'outputs' / 'POOLED' / 'experiments' / 'runway_intent_r0_20260913'
AIRPORTS = ['KRDU', 'KSJC', 'KSTL', 'KSMF', 'KMSY']
RULES = ['B0_majority', 'B1_active_config', 'B2_active_config_gated', 'B3_same_sector_last', 'B4_wind']
SHORT = {'B0_majority': 'B0', 'B1_active_config': 'B1', 'B2_active_config_gated': 'B2',
         'B3_same_sector_last': 'B3', 'B4_wind': 'B4'}
lines: list[str] = []
combined: dict = {}


def pct(x):
    return 'n/a' if x is None else f'{100 * x:.1f}'


lines.append('### R0a — exact accuracy (%) by rule, at the slice entry and at 10 km remaining')
lines.append('')
lines.append('| airport | n | anchor | B0 | B1 | B2 | B3 | B4 | direction (B1) | side given direction (B1 / B3, n) |')
lines.append('|---|---:|---|---:|---:|---:|---:|---:|---:|---|')
for airport in AIRPORTS:
    path = OUT / airport / 'runway_intent_r0.json'
    if not path.is_file():
        lines.append(f'| {airport} | — | missing | | | | | | | |')
        continue
    doc = json.loads(path.read_text())
    combined[airport] = {'r0a': {k: doc[k] for k in ('validation_flights', 'direction_groups', 'summary',
                                                     'per_runway', 'direction_flips', 'context_pool')}}
    for anchor in ('entry', '10km'):
        cell = doc['summary'].get(anchor)
        if not cell:
            continue
        rules = cell['rules']
        b1, b3 = rules['B1_active_config'], rules['B3_same_sector_last']
        side = (f"{pct(b1['side_given_direction'])} / {pct(b3['side_given_direction'])} ({b1['side_flights']})"
                if b1['side_given_direction'] is not None else 'n/a (no parallel)')
        lines.append(
            f"| {airport} | {cell['flights']} | {anchor} | " +
            ' | '.join(pct(rules[r]['exact']) for r in RULES) +
            f" | {pct(b1['direction'])} | {side} |"
        )

lines += ['', '### R0a — B1 / B3 exact accuracy (%) along the approach (coverage = flights with that much path left)', '']
bins = ['entry', '30km', '20km', '15km', '10km', '6km', '3km']
lines.append('| airport | ' + ' | '.join(bins) + ' |')
lines.append('|---|' + '---:|' * len(bins))
for airport in AIRPORTS:
    doc = combined.get(airport, {}).get('r0a')
    if not doc:
        continue
    total = doc['validation_flights']
    cells = []
    for b in bins:
        cell = doc['summary'].get(b)
        if not cell:
            cells.append('—')
            continue
        r = cell['rules']
        cells.append(f"{pct(r['B1_active_config']['exact'])} / {pct(r['B3_same_sector_last']['exact'])} "
                     f"({100 * cell['flights'] / total:.0f}%)")
    lines.append(f'| {airport} | ' + ' | '.join(cells) + ' |')

lines += ['', '### R0a — per landing runway, exact accuracy (%) at the slice entry (B1 / B3)', '']
for airport in AIRPORTS:
    doc = combined.get(airport, {}).get('r0a')
    if not doc:
        continue
    entry = doc['per_runway'].get('entry', {})
    parts = [f"{runway} (n={cell['flights']}) {pct(cell['B1_active_config'])} / {pct(cell['B3_same_sector_last'])}"
             for runway, cell in entry.items()]
    lines.append(f'- **{airport}**: ' + '; '.join(parts))

lines += ['', '### R0a — landing-direction flips (15-min bins, >=2 landings, new direction held 2 bins)', '']
lines.append('| airport | days | flips | days with a flip | changes across a >3 h gap | day-blocked val: days / flips | day-blocked test: days / flips |')
lines.append('|---|---:|---:|---:|---:|---|---|')
for airport in AIRPORTS:
    doc = combined.get(airport, {}).get('r0a')
    if not doc:
        continue
    flips = doc['direction_flips']
    folds = flips['day_blocked_folds']
    per_day = flips['per_day']
    lines.append(
        f"| {airport} | {flips['days']} | {sum(per_day.values())} | {sum(v > 0 for v in per_day.values())} | "
        f"{sum(flips.get('across_gap_per_day', {}).values())} | "
        f"{folds.get('val', {}).get('days', 0)} / {folds.get('val', {}).get('flips', 0)} | "
        f"{folds.get('test', {}).get('days', 0)} / {folds.get('test', {}).get('flips', 0)} |"
    )

lines += ['', '### R0b — what a wrong runway costs the plan head (its L-1 anchor), PAIRED on the same flights', '']
lines.append('Each cell: runway accuracy % / mean FDE change vs the assigned runway on the flights that rule was scored on (m), n.')
lines.append('')
SEL = ['B0_majority', 'B1_active_config', 'B3_same_sector_last', 'B4_wind', 'oracle_same_direction']
lines.append('| airport | stratum | n | assigned FDE mean (m) | ' + ' | '.join(SHORT.get(s, 'same-dir oracle') for s in SEL) + ' |')
lines.append('|---|---|---:|---:|' + '---|' * len(SEL))
for airport in AIRPORTS:
    path = OUT / airport / 'hypotheses.json'
    if not path.is_file():
        lines.append(f'| {airport} | missing | | |' + ' |' * len(SEL))
        continue
    doc = json.loads(path.read_text())
    summary = summarise_hypotheses(doc['flights'], doc['selectors'])
    combined.setdefault(airport, {})['r0b'] = {
        'summary_paired': summary, 'unflyable_candidates': doc.get('unflyable_candidates'),
        'candidates': doc['candidates'], 'context_rules': doc.get('context_rules'),
        'flights_scored': len(doc['flights']),
    }
    for stratum_key, block in summary.items():
        if stratum_key == 'by_assigned_runway':
            continue
        s = block['selectors']
        cells = []
        for sel in SEL:
            c = s[sel]
            cells.append(f"{pct(c['runway_accuracy'])} / {c['fde_delta_mean']:+.0f} ({c['n']})")
        label = stratum_key.split(' (')[0]
        lines.append(f"| {airport} | {label} | {block['n']} | {s['assigned']['fde_mean']:.0f} | " + ' | '.join(cells) + ' |')
    if doc.get('unflyable_candidates'):
        lines.append(f"| {airport} | unflyable on the plan path: {', '.join(doc['unflyable_candidates'])} | | |" + ' |' * len(SEL))

lines += ['', '### R0b — per assigned runway: FDE median (m) under the assigned runway / under B1 (B1 runway accuracy %)', '']
for airport in AIRPORTS:
    r0b = combined.get(airport, {}).get('r0b')
    if not r0b:
        continue
    per = r0b['summary_paired']['by_assigned_runway']
    parts = [f"{runway} (n={cell['n']}) {cell['assigned']['fde_median']:.0f} / {cell['B1_active_config']['fde_median']:.0f} "
             f"({pct(cell['B1_active_config']['runway_accuracy'])})" for runway, cell in per.items()]
    lines.append(f'- **{airport}**: ' + '; '.join(parts))

text = '\n'.join(lines)
(OUT / 'readout.md').write_text(text + '\n', encoding='utf-8')
(OUT / 'readout.json').write_text(json.dumps(combined, indent=1), encoding='utf-8')
print(text)
