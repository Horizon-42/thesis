# Requests from stage C to the designer

What Claude, as stage C's implementer, asks of the designer: the design text that the user's decisions still need, the
names a public interface must give, and the questions that only the user can decide. It holds only the current state:
it is rewritten in full each time, never appended to, and an item leaves when it is done. The evidence and the commits
are in the implementation log (`readouts/2026-10-05_stage_c_implementation_log.md`, cited by §).

State of 2026-10-06, 14:00 (stage C on `dev-two-tier` at `98a89871`; C10 `post_train_20261006` running, log §26).

| # | Request | For | Log |
|---|---|---|---|
| 1 | O13 decided: the user chose every setting as proposed (1,000 windows of each kind, 10 rounds, update groups 4, 64 data sentences, batch windows 64, K 8, rates 1e-5 / 1e-4, weight decay 0.01, 200 select windows an airport, traffic 64 / 4); the criterion that chooses the round (D7) later, before the validation readout. The design text of O13 is the designer's to write | The designer | §25, §26 |
| 2 | The speaking is bound by the CPU: the measurements and three proposals below ("Where a round's time goes") | The designer; proposal (a) to stage A through the user | §26 |
| 3 | An open item for the design (D131: an S2 not corrected before the run): the memory of N speaking workers is not checked before a campaign starts; round 0 is watched (a sampler records the GPU's peak). The main process's cache is released before each round's speaking (`ea2050ff`) | The designer | §26 |
| 4 | P45, a reading for the user: a window set's flown tracks are written unrounded (prior D127 followed for windows), so a set's file is larger (1.3× on the smoke set); a formal set's size is read at its export | The user | §21 |
| 5 | `requests_from_b_to_designer.md` §3 item 5 ("what stage C needs after B12 and B13") is followed on stage C's branch and can leave: `SpeakingLoop` with the start's observed rows (`post_window_loop.py:118`), the val read's lock, claim with its options and spent mark (`post_validation`, D132), the unrounded track of D127 (sample v2) | The designer | §18–§21, §23 |

## Where a round's time goes (item 2)

**Measured** (B5's base, A34's artefact v12 at Δ 4 s, executor v17, the formal census; 2026-10-06):

- The first launch of C10 spoke on one CPU core: one Python thread at 93 % (1 of 28 cores), the GPU nearly idle.
- One real speaking batch of 28 windows (D94's two passes, K 8, on the GPU), 74 s under cProfile (which inflates the
  Python-heavy parts somewhat):

  | Part | Time | Share | What it is |
  |---|---|---|---|
  | The start's rebuild of the batch's flights (stage A's `autopilot.start.start_moved`) | ~26 s | ~35 % | Once for each of the two passes; each call reads the whole split's signals file (`load_signals`, zip decompression ~7 s), the arrival records (`load_flight_dicts`, JSON ~4.5 s) and rebuilds each flight's series (`rebuild_series` ~5 s): the same files for every batch |
  | The speaker's masks (stage B's `prior.speaker._allowed`) | ~14 s | ~19 % | Each row and word: the procedure masks (`prior.procedure.permitted`, `_altitude`, `edge_m`) and the grammar's column masks (`instructions.grammar.column_mask`), in Python |
  | Inputs, edge features, traffic scene | ~8 s | ~11 % | `prior.inputs.state_inputs`, `post.edges.tokens`, the window loop's scene |
  | Compiled-kernel set-up | ~10 s | ~13 % | Once a process (torch's custom ops and inductor), not a batch |
  | The prior's forward pass | ~5 s | ~7 % | The only GPU part: the prior has 3.8 M parameters, a step takes milliseconds |
  | The executor's flight, the judge, the loop copies at the branch points | the remainder | ~15 % | Each 1 s cycle of each aircraft |

- The profile of a round (C8, log §25): 2.1 s a window at batches of 64, 2.4 s at 14 (the fixed cost of the start
  spread over more windows); the GPU at 1.3 GB while speaking; the pass (4 groups an update) 4.4 GB and 10.5 s for
  29 updates; the selection readout 0.28 s a window.
- With three speaking workers (`--speak-workers 3`, `0268e0ad`): about 1.3 batches of 64 a minute; each worker one
  core (75–80 %), about 1.4 GB of host memory of its own (2 GB counting its share of the parent's pages) and up to
  1.3 GB of the GPU while speaking (the GPU's peak 4.8 GB with three). A round: about 50 min of speaking and about
  9 min in the main process alone (the pass ~2.5 min, the selection readout ~5 min, the rest ~1 min); 10 rounds
  about 10 h. With five workers (the user's rule: switched at a round boundary when the load is below 10) about
  40 min a round.

**Why the CPU.** The post-training is mostly a closed-loop simulation: at every Δ row of every flight the prior speaks,
the executor flies, the scene and the separation are judged, and the next row's inputs are built from where the
aircraft is. Each step needs the one before it; the network is small, so the GPU finishes a step in milliseconds and
waits for the Python around it. More processes help; bigger batches give the GPU little more to do.

**Limits measured.** The GPU caps the workers at about five (7.6 GB: other processes 0.9 GB, the main process's pass
up to 4.4 GB while the idle workers keep their contexts); speaking on the CPU instead would allow about ten (the host's
31 GB, shared with other sessions), at about 25 % more time a batch and float-level different speaking (a different
campaign).

**Proposals** (Claude's, from the numbers; each is a decision of the designer and the user):

- (a) **Stage A: the start keeps what it reads.** `start_moved` reads the same split's signals and arrival records
  for every call; kept across the calls of one process (the files are written once, so a read-once cache by path
  changes no result), a batch is about a third shorter: about 63 batches × 2 passes × 10 rounds, some 4 h of CPU over
  a campaign. The code is stage A's (`autopilot/start.py`, `instructions/artefact.py` `load_signals`,
  `autopilot/flights.py`), which stage C does not change: a request to stage A.
- (b) **Stage C: the selection readout through the workers.** Its 1,000 windows (~5 min a round) run in the main
  process now; spoken by the workers, a round's serial time goes from about 9 to about 5 min. A change of
  `post_train`, at a round boundary of a running campaign or for the next one.
- (c) **Later, if more is needed: the masks and the simulation vectorised or compiled** (stage B's `speaker` and
  `procedure`, stage A's executor). About half of a batch; a larger change in other stages' code, with their behaviour
  checks.

Not proposed: speaking on the CPU with about ten workers (about 1.5–2 h saved on a campaign, but a different campaign
by its float numbers, and host memory shared with other sessions has run out once today); batches of 128 (the fixed
start cost halved a window, but each worker's GPU memory grows and the GPU already caps the workers).
