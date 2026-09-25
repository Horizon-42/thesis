# Passages cut verbatim out of live modules (2026-09-25)

Each block is the text as it stood at `002998fb` (branch `dev-prior-fast`), under the file it came from; the live file now reads without it (a docstring's rewritten form is in the live file). Where a block begins or ends with lines that are still live — the context the cut sat in — its heading names them.

## `autopilot/executor.py` — the per-flight field lists `take` read

```python

    #: What `take` keeps per flight: the tensors (`PER_FLIGHT`), the flights' context (`CONTEXT`, each with its own
    #: `take`), the laws' state, and the records (`HISTORIES`, one entry per cycle).
    PER_FLIGHT = ("time_limit_s", "state", "bank", "done", "done_cycle")
    CONTEXT = ("inputs", "runways", "charts")
    HISTORIES = ("states", "commands", "wanted", "limits", "modes", "sentence_times")
```

## `autopilot/executor.py`

```python
    def take(self, index: torch.Tensor) -> None:
        """Keep the flights at ``index`` — a closed loop's branches: a flight taken twice flies on as two copies of
        itself, each copy the flight it was taken from in every respect (its laws' state and its records too)."""
        for name in self.PER_FLIGHT:
            setattr(self, name, getattr(self, name)[index])
        for name in self.CONTEXT:
            setattr(self, name, getattr(self, name).take(index))
        self.plant = Plant(self.inputs)
        for law in (self.lateral, self.vertical, self.speed):
            law.take(index)

        def taken(rows: list[torch.Tensor]) -> list[torch.Tensor]:
            return list(torch.stack(rows, dim=1)[index].unbind(dim=1)) if rows else []

        self.states, self.commands, self.wanted, self.sentence_times = map(
            taken, (self.states, self.commands, self.wanted, self.sentence_times))
        self.limits = {name: taken(rows) for name, rows in self.limits.items()}
        self.modes = {name: taken(rows) for name, rows in self.modes.items()}
```

## `autopilot/lateral.py` — `Runways.take`

```python
    def take(self, index: torch.Tensor) -> Runways:
        return Runways(self.threshold_e_m[index], self.threshold_n_m[index], self.course_deg[index],
                       self.elevation_m[index], self.crossing_height_m[index])
```

## `autopilot/lateral.py` — `Lateral.PER_FLIGHT`

```python

    #: The state kept per flight (`take`); ``runway`` and ``word_deg`` are None before the first cycle.
    PER_FLIGHT = ("captured", "tracking", "cleared", "runway", "word_deg", "word_step", "heard_s", "track_unwrapped",
                  "target_unwrapped", "last_track")
```

## `autopilot/lateral.py` — `Lateral.take`

```python
    def take(self, index: torch.Tensor) -> None:
        """Keep the flights at ``index`` (a closed loop's branches, `Executor.take`)."""
        for name in self.PER_FLIGHT:
            value = getattr(self, name)
            setattr(self, name, None if value is None else value[index])
```

## `autopilot/vertical.py` — `Vertical.PER_FLIGHT`

```python

    #: The state kept per flight (`take`).
    PER_FLIGHT = ("captured", "issued", "left_tube", "flown_m", "anchor_m", "anchor_height_m")
```

## `autopilot/vertical.py` — `Vertical.take`

```python
    def take(self, index: torch.Tensor) -> None:
        """Keep the flights at ``index`` (a closed loop's branches, `Executor.take`)."""
        for name in self.PER_FLIGHT:
            setattr(self, name, getattr(self, name)[index])
```

## `autopilot/speed.py` — `Speed.PER_FLIGHT`

```python
    #: The state kept per flight (`take`).
    PER_FLIGHT = ("approach_ias_mps",)
```

## `autopilot/speed.py` — `Speed.take`

```python
    def take(self, index: torch.Tensor) -> None:
        """Keep the flights at ``index`` (a closed loop's branches, `Executor.take`)."""
        self.approach_ias_mps = self.approach_ias_mps[index]
```

## `autopilot/flights.py` — `FlightInputs.take`

```python

    def take(self, index: torch.Tensor) -> FlightInputs:
        """The flights at ``index`` (a closed loop's branches, `Executor.take`)."""
        return FlightInputs(self.initial_state[index], self.aero_params[index], self.frame_params[index],
                            self.max_thrust_n[index])
```

## `autopilot/frame.py` — `AirportCharts.take`

```python
    def take(self, index: torch.Tensor) -> AirportCharts:
        return AirportCharts(self.lat0_deg[index], self.lon0_deg[index], self.m_per_deg_lon[index])
```

## `autopilot/sentence.py` — `Spoken.take`

```python
    def take(self, index: torch.Tensor) -> None:
        """Keep the sentences at ``index`` (a closed loop's branches)."""
        self.value, self.issued = self.value[index], self.issued[index]
        rows = index.cpu().numpy()
        self.grid = [row[rows] for row in self.grid]
```

## `prior/generate.py` — `Speaker.PER_FLIGHT`

```python

    #: What `take` keeps per flight (arrays, tensors and lists, first axis the flight); the masses the masks removed
    #: (`forbidden`, one entry per step) are taken with them.
    PER_FLIGHT = ("geometries", "contexts", "entry_s", "e", "n", "h", "features", "relative", "in_force", "since",
                  "airport", "static", "value", "said_row")
```

## `prior/generate.py` — `Speaker.take`

```python
    def take(self, index: np.ndarray) -> None:
        """Keep the flights at ``index`` — a closed loop's branches: a flight taken twice is spoken to on as two copies
        of itself, each with the rows it read and the words it said."""
        rows = torch.as_tensor(index, device=self.features.device)
        for layer, past in enumerate(self.past):                # one layer at a time: one copy alive at once
            self.past[layer] = past.take(rows)
        for name in self.PER_FLIGHT:
            value = getattr(self, name)
            if isinstance(value, list):
                setattr(self, name, [value[i] for i in index])
            else:
                setattr(self, name, value[rows] if isinstance(value, torch.Tensor) else value[index])
        self.forbidden = {column: [mass[index] for mass in masses] for column, masses in self.forbidden.items()}
```

## `prior/model.py` — `Past.take`

```python

    def take(self, index: torch.Tensor) -> Past:
        """The scenes at ``index`` (single-aircraft scenes: B·A = B)."""
        return Past(self.keys[index], self.values[index], self.present[index], self.rows)
```

## `prior/train.py` — the module docstring's sentence; "The landing reward" is context, still live

```python
Closed-loop fine-tuning (design §9.2) takes the same step on chains (`FineTuner`): one pass at a time, from the weights
it is given, at its own learning rate. The landing reward
```

## `prior/train.py` — `FineTuneConfig`, `FineTuner`

```python


@dataclass(frozen=True)
class FineTuneConfig:
    """Closed-loop fine-tuning's optimiser (design §9.2, §10): AdamW, a third of pretraining's learning rate."""

    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    clip_norm: float = 1.0
    tokens_per_batch: int = 16_384


class FineTuner:
    """The model's optimiser across closed-loop rounds: `one_pass` trains one pass over the flights it is given (in
    length buckets, shuffled), the optimiser's state and the warm-up carried from pass to pass. The seed sets the batch
    order and dropout."""

    def __init__(self, model: Prior, config: FineTuneConfig, device: torch.device, *, seed: int) -> None:
        torch.manual_seed(seed)
        self.model, self.config, self.device = model, config, device
        self.rng = np.random.default_rng(seed)
        self.optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                           weight_decay=config.weight_decay)
        self.schedule = torch.optim.lr_scheduler.LambdaLR(
            self.optimiser, lambda step: min(1.0, (step + 1) / config.warmup_steps))
        self.passes = 0

    def one_pass(self, split: Split) -> dict[str, Any]:
        """One pass over ``split``: its negative log-likelihood per aircraft-step as trained, and what it took."""
        self.passes += 1
        self.model.train()
        started = time.perf_counter()
        total, steps, count = 0.0, 0, 0
        for indices in batches(split.flights, self.config.tokens_per_batch, self.rng):
            loss, speaks = _step(self.model, to_batch(split, indices, self.device), self.optimiser, self.schedule,
                                 self.config.clip_norm, f"pass {self.passes}")
            total += loss * speaks
            steps += speaks
            count += 1
        self.model.eval()
        return {"nll_per_step": total / steps, "steps": steps, "batches": count,
                "seconds": time.perf_counter() - started}
```

## `prior/data.py` — the module docstring's sentence

```python
sentence's words as written. ``asked`` says which of them the loss counts: every column from row ``N_LOOK`` on for a
labelled flight; a closed-loop chain (design §9.2, `chain_record`) leaves out the columns `prior.relabel` could not
ask for.
```

## `prior/data.py` — `in_force_words`

```python
def in_force_words(grid: np.ndarray) -> np.ndarray:
    """``[N, 6]``: the word in force at each row of a sentence (``[N, 6]``, step 0 writing every column)."""
    grid = np.asarray(grid, dtype=np.int64)
    return np.take_along_axis(grid, issued_rows(grid), axis=0)
```

## `prior/data.py` — `chain_record`'s docstring

```python
    """One closed-loop chain (design §9.2) as the prior reads it: the rows it read — the positions ``e``, ``north``,
    ``height`` (``[N_LOOK + steps]``: observed to the first predicted step, then where the executor flew), on the
    vocabulary's step from row 0, and the words the chain said (``said``, ``[steps, 6]`` from row `N_LOOK`) as the
    words said so far — and the targets `prior.relabel` read for its steps (``classes``, ``asked``). The same rows a
    speaker (`prior.generate.Speaker`) built as it spoke on the chain."""
```

## `experiments/prior_free_generation.py` — `ClosedLoop`'s docstring

```python
    one it has cleared or captured keeps its runway. Each flight flies until the executor is done with it or its time
    limit (``limits``, seconds). `take` re-forms the batch from some of its flights (a closed loop's branches,
    `prior_closed_loop`)."""
```

## `experiments/prior_free_generation.py` — `ClosedLoop.__init__`: the executor's state per step (read only by the relabelling); `self.max_steps = …` is context, still live

```python
        self.max_steps = rows_for(max(limits), step_s) - N_LOOK
        #: the executor's state when each step was said ([B] each): cleared (since the last go-around), captured
        self.cleared: list[np.ndarray] = []
        self.captured: list[np.ndarray] = []
```

## `experiments/prior_free_generation.py` — `ClosedLoop.step`: the two recording lines; the comment and `said = …` are context, still live

```python
        cleared, captured = executor.lateral.cleared.cpu().numpy(), executor.lateral.captured.cpu().numpy()
        # a flight that is done hears nothing more; one the executor has cleared or captured keeps its runway
        said = speaker.speak(active=~done, runway_locked=executor.runway_locked.cpu().numpy())
        self.cleared.append(cleared)
        self.captured.append(captured)
```

## `experiments/prior_free_generation.py` — `ClosedLoop.take`

```python

    def take(self, index: np.ndarray) -> None:
        """Keep the flights at ``index`` (the executor's, the speaker's and the words said): a flight taken twice flies
        on as two copies of itself."""
        rows = torch.as_tensor(index, device=self.device)
        self.executor.take(rows)
        self.spoken.take(rows)
        self.speaker.take(np.asarray(index))
        self.cleared = [row[index] for row in self.cleared]
        self.captured = [row[index] for row in self.captured]
```

## After the review (2026-09-25) — wording that still named the archived step

`experiments/prior_landing_reward.py`:

```python
    python run_ts.py prior_landing_reward --prior <the §9.2 round kept, or the step-1 run> \\
```

`experiments/prior_landing_reward.py`:

```python
parser.add_argument("--prior", type=Path, required=True, help="the start: the §9.2 round kept, or the step-1 run")
```

`prior/train.py`:

```python
    """The landing reward's optimiser and weights (design §9.3, §10): a tenth of §9.2's learning rate (the reward term's
```

`prior/data.py`:

```python
        raise ValueError(f"{signals.dataset_id}: a chain of {len(said)} steps has {len(e)} rows and targets "
```

