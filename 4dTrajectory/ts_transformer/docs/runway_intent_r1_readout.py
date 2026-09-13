"""Runway-intent R1 readout: the five airports' `runway_intent_r1.json` in one table set, and the
gates pre-registered in `docs/2026-09-13_runway_intent_plan.zh.md` §11.4 applied mechanically.

Written to the campaign folder as `readout.md` / `readout.json`.

    conda run -n aeroviz python 4dTrajectory/ts_transformer/docs/runway_intent_r1_readout.py
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT = REPO_ROOT / '4dTrajectory' / 'outputs' / 'POOLED' / 'experiments' / 'runway_intent_r1_20260913'
AIRPORTS = ['KRDU', 'KSJC', 'KSTL', 'KSMF', 'KMSY']
GATED = {'KRDU': 10.0, 'KSMF': 10.0, 'KSTL': 0.0}      # §11.4 gate 1: side gain (points) required
MINORITY_GATED = ('KRDU', 'KSMF')                       # §11.4 gate 2
PARTITIONS = ('day_a', 'day_b')
lines: list[str] = []
combined: dict = {}


def pct(x):
    return 'n/a' if x is None else f'{100 * x:.1f}'


def best(rules: dict, key: str):
    values = [cell[key] for cell in rules.values() if cell.get(key) is not None]
    return max(values) if values else None


docs = {}
for airport in AIRPORTS:
    path = OUT / airport / 'runway_intent_r1.json'
    if path.is_file():
        docs[airport] = json.loads(path.read_text())


def table(anchor: str, models: tuple[str, ...]) -> None:
    lines.append('| airport | model | val flights | exact (best rule) | direction (B1) | side given direction (best rule) | minority (best rule) | NLL (B1-prob) | ECE |')
    lines.append('|---|---|---:|---|---|---|---|---|---:|')
    for airport, doc in docs.items():
        for name in models:
            v = doc['models'][name]['validation'][anchor]
            m, r = v['model'], v['rules']
            lines.append(
                f"| {airport} | {name} | {doc['models'][name]['val_flights']} | "
                f"{pct(m['exact'])} ({pct(best(r, 'exact'))}) | {pct(m['direction'])} ({pct(r['B1_active_config']['direction'])}) | "
                f"{pct(m['side_given_direction'])} ({pct(best(r, 'side_given_direction'))}) | "
                f"{pct(m['minority_accuracy'])} ({pct(best(r, 'minority_accuracy'))}) | "
                f"{m['nll']:.3f} ({v['b1_prob_nll']:.3f}) | {m['ece']:.3f} |"
            )


lines += ['### R1 — the full model on its validation days, all anchors pooled (the gates read this)', '']
table('all', PARTITIONS)
lines += ['', '### R1 — at the arrival-slice entry (the hardest anchor)', '']
table('entry', PARTITIONS)

lines += ['', '### Along the approach — side given direction, model / best rule (%), at the entry and at each ring (first crossing)', '']
ORDER = ['entry', 'r20km', 'r15km', 'r10km', 'r6km']
lines.append('| airport | partition | ' + ' | '.join(ORDER) + ' |')
lines.append('|---|---|' + '---|' * len(ORDER))
for airport, doc in docs.items():
    if airport not in GATED:
        continue
    for part in PARTITIONS:
        v = doc['models'][part]['validation']
        cells = []
        for anchor in ORDER:
            if anchor not in v:
                cells.append('—')
                continue
            m, r = v[anchor]['model'], v[anchor]['rules']
            cells.append(f"{pct(m['side_given_direction'])} / {pct(best(r, 'side_given_direction'))}")
        lines.append(f'| {airport} | {part} | ' + ' | '.join(cells) + ' |')

lines += ['', '### Leakage — the same flights (validation under BOTH splits), day-blocked model vs per-flight model', '']
lines.append('| airport | flights | day_a exact / side / NLL | flight exact / side / NLL | B1 exact on the same flights |')
lines.append('|---|---:|---|---|---:|')
for airport, doc in docs.items():
    a, f = doc['leakage']['day_a']['all'], doc['leakage']['flight']['all']
    lines.append(
        f"| {airport} | {doc['split']['paired_leakage_flights']} | "
        f"{pct(a['model']['exact'])} / {pct(a['model']['side_given_direction'])} / {a['model']['nll']:.3f} | "
        f"{pct(f['model']['exact'])} / {pct(f['model']['side_given_direction'])} / {f['model']['nll']:.3f} | "
        f"{pct(a['rules']['B1_active_config']['exact'])} |"
    )

lines += ['', '### Secondary — without the day-level features (wind, time of day), all anchors pooled', '']
lines.append('| airport | partition | exact full → nowx | side given direction full → nowx | NLL full → nowx |')
lines.append('|---|---|---|---|---|')
for airport, doc in docs.items():
    for part in PARTITIONS:
        full = doc['models'][part]['validation']['all']['model']
        nowx = doc['models'][f'{part}_nowx']['validation']['all']['model']
        lines.append(f"| {airport} | {part} | {pct(full['exact'])} → {pct(nowx['exact'])} | "
                     f"{pct(full['side_given_direction'])} → {pct(nowx['side_given_direction'])} | "
                     f"{full['nll']:.3f} → {nowx['nll']:.3f} |")

lines += ['', '### What the head decides on — day_a, accuracy lost when a feature group is permuted (exact / side, points)', '']
groups = sorted({g for doc in docs.values() for g in doc['importance_day_a']})
lines.append('| airport | ' + ' | '.join(groups) + ' |')
lines.append('|---|' + '---|' * len(groups))
for airport, doc in docs.items():
    imp = doc['importance_day_a']
    cells = []
    for g in groups:
        c = imp.get(g)
        side = 'n/a' if c is None or c['side_drop'] is None else f"{100 * c['side_drop']:+.1f}"
        cells.append('—' if c is None else f"{100 * c['exact_drop']:+.1f} / {side}")
    lines.append(f'| {airport} | ' + ' | '.join(cells) + ' |')


def gates(variant: str) -> dict:
    verdict: dict = {}
    for airport, need in GATED.items():
        doc = docs.get(airport)
        if not doc:
            continue
        for part in PARTITIONS:
            name = part if variant == 'full' else f'{part}_nowx'
            v = doc['models'][name]['validation']['all']
            m, r = v['model'], v['rules']
            gain = 100 * (m['side_given_direction'] - best(r, 'side_given_direction'))
            cell = {
                'side_gain_points': gain,
                'g1_side': gain >= need,
                'g3_nll': m['nll'] < v['b1_prob_nll'],
                'g3_ece': m['ece'] <= 0.05,
                'g4_direction': m['direction'] >= r['B1_active_config']['direction'],
            }
            if airport in MINORITY_GATED:
                cell['g2_minority'] = m['minority_accuracy'] >= 0.60
            verdict[f'{airport}/{part}'] = cell
    veto = all(verdict[f'{a}/{p}']['side_gain_points'] < 5.0
               for a in MINORITY_GATED for p in PARTITIONS if f'{a}/{p}' in verdict)
    return {'cells': verdict, 'veto_side_irreducible': veto}


for variant in ('full', 'nowx'):
    result = gates(variant)
    combined[f'gates_{variant}'] = result
    label = 'the full model — the pre-registered reading' if variant == 'full' else 'secondary: without wind / time of day'
    lines += ['', f'### §11.4 gates — {label}', '']
    lines.append('| airport / partition | side gain (points) | G1 side | G2 minority ≥ 60 % | G3 NLL < B1-prob | G3 ECE ≤ 0.05 | G4 direction ≥ B1 |')
    lines.append('|---|---:|---|---|---|---|---|')
    for key, c in result['cells'].items():
        yn = lambda flag: 'pass' if flag else 'FAIL'  # noqa: E731
        lines.append(f"| {key} | {c['side_gain_points']:+.1f} | {yn(c['g1_side'])} | "
                     f"{yn(c['g2_minority']) if 'g2_minority' in c else '—'} | {yn(c['g3_nll'])} | "
                     f"{yn(c['g3_ece'])} | {yn(c['g4_direction'])} |")
    lines.append(f"\nVeto (side gain < 5 points at KRDU and KSMF on both partitions): "
                 f"{'FIRES' if result['veto_side_irreducible'] else 'does not fire'}")

combined['airports'] = {a: {k: d[k] for k in ('split', 'minority_runways', 'direction_groups')} for a, d in docs.items()}
text = '\n'.join(lines)
(OUT / 'readout.md').write_text(text + '\n', encoding='utf-8')
(OUT / 'readout.json').write_text(json.dumps(combined, indent=1), encoding='utf-8')
print(text)
