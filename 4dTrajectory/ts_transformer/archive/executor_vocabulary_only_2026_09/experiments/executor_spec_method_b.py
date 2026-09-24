"""Cut verbatim from `experiments/executor_spec.py` (2026-09-24): method B, the runner's delay measurement. Archived
with `autopilot/observe.py`; not importable (the names it uses lived in that runner). See README.md."""


def method_b(batch: replay.Batch, params: ExecutorParams, words: Words, device: torch.device) -> dict[str, Any]:
    spec = words.spec
    flown, verdicts = replay.fly_batch(batch, params, words, device=device)
    leads: dict[int, list[float]] = {column: [] for column in observe.MEASURED_COLUMNS}
    received: Counter = Counter()
    refused: Counter = Counter()
    for j, verdict in enumerate(verdicts):
        states = flown.states[j, : verdict.end_row + 1].cpu().numpy()
        try:
            pairs, counts = observe.flight_leads(states, flown.sentence_s[j].cpu().numpy(), params.cycle_s,
                                                 batch.series[j], batch.geometries[j], batch.readings[j], spec, words,
                                                 LEAD_WINDOW_S)
        except Refused as refusal:
            refused[refusal.reason] += 1
            continue
        for column, lead in pairs:
            leads[column].append(lead)
        received.update(counts)
    delays = observe.delays_from_leads(leads)
    by_group = {name: [lead for column in columns for lead in leads[column]] for name, columns in DELAY_GROUPS.items()}
    return {
        "delays": delays,
        "record": {
            "drawn": batch.drawn, "flown_with": {"delays": "0 s on every column"},
            "flights": replay.summary(verdicts),
            "flown_track_refused_by_the_labeller": dict(refused.most_common()),
            "words_received_and_matched": {COLUMNS[column]: {"received": received[column], "matched": len(leads[column])}
                                           for column in observe.MEASURED_COLUMNS},
            "lead_s_by_column": {COLUMNS[column]: _percentiles(values) for column, values in leads.items() if values},
            "lead_s_by_delay": {name: _percentiles(values) for name, values in by_group.items()},
            "rule": "delay = max(0, median lead of the group's columns); lead = received − re-read, seconds; "
                    f"matched within {LEAD_WINDOW_S:g} s on the same column and value; heading words are not measured "
                    "(each says the track a lead later: no delay)",
        },
    }
