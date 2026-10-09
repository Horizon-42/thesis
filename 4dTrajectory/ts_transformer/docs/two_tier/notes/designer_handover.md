# Designer's handover (two-tier model)

Written 2026-10-09 by Claude (the designer/manager of the two-tier work) for the agent that takes over. Read this first,
then `design/outline.md` §0 and §3.2, then each stage document's §0. Paths are relative to
`4dTrajectory/ts_transformer/docs/two_tier/` unless they start with `4dTrajectory/` or name the repository root.

## 1 The role

- **You own the design documents** in `design/`: `outline.md` (principles, shared decisions, the decision index
  §3.2), `vocabulary.md` (stage A), `prior.md` (stage B), `post_training.md` (stage C), `multi_control.md` (stage D),
  `frontend.md` (the Training view, fronter). You write decisions (D numbers), open items (O numbers), milestones and
  public interfaces. **Next free numbers: D188, O22** (outline §3.2 keeps the index; update it with every new number).
- **You give orders, you do not build.** Implementers build code; you never commit code. Orders go into
  `notes/stage_c.md`, `notes/stage_d.md`, `notes/fronter.md` (in Chinese, plain words, imperative, only the changes;
  never a role preamble). The user forwards them.
- **Implementers write only** their §0.3 status lines, their logs (`readouts/2026-10-0*_*_implementation_log.md`)
  and their requests files (`design/requests_from_{c,d,fronter}_to_designer.md`, rewritten in full each time). Their
  readings are proposals until the user (or you, where the evaluation is delegated) accepts them.
- **Decisions are the user's.** Ask through AskUserQuestion with a recommended option first; plain Chinese in replies.
  A choice the user's rules already allow is yours (do not ask). When the user delegates an evaluation ("评估一下",
  "把可以做的写进设计"), you decide and say so in the source column ("the evaluation: Claude").
- **The design holds no experiment information** (no run directories, rounds, measured numbers of a campaign):
  those live in `docs/experiments/intents.json` and the implementers' logs. Design text is the final design only (no
  v1/v2 narrative); a decision's "Why" may cite a log section instead of numbers.

## 2 How you commit documents

The main checkout (`/home/supercomputing/studys/thesis`, branch `dev-two-tier`) is shared with the implementers, who
sometimes have uncommitted edits there. Commit docs through a temporary worktree and fast-forward:

```bash
cd /home/supercomputing/studys/thesis
git worktree add --detach .claude/worktrees/designer-docs dev-two-tier
cd .claude/worktrees/designer-docs && git checkout -b designer-docs-tmp
# edit (python with exact-string asserts works well), then: git add <explicit paths>; git diff --cached --stat; git commit
cd /home/supercomputing/studys/thesis && git merge --ff-only designer-docs-tmp
git worktree remove .claude/worktrees/designer-docs && git branch -d designer-docs-tmp
```

- If the fast-forward fails because dev-two-tier moved: re-add the worktree on the branch, `git rebase dev-two-tier`,
  fast-forward again. If it fails because an implementer has uncommitted edits to the same file in the main checkout:
  never touch their edits; wait for their commit (a background loop on `git diff --quiet -- <file>`), then rebase.
- Never `git add -A`; stage explicit paths; end commit messages with the Co-Authored-By line.

## 3 Notes: replace or edit in place

A note is overwritten by the next order, **but only once the implementer has read the current one.** Before rewriting a
note, check whether the implementer acted on it (its commits, log, requests). If not, edit the note in place (add the
new items, fix stale lines) so nothing unread is lost — the user checked this on 2026-10-09. Keep the order of work
explicit in the note ("顺序：…").

## 4 Rules you enforce (the user's; details in root `CLAUDE.md` and memory)

- **Code review** (root `CLAUDE.md` "Code review", 2026-10-08): the reviewer is never the author; the scope is the diff
  (no repository search); S1 fix, S2 fix if cheap, S3 one line; a second round only for a fixed S1/S2; a small change
  (merge without conflicts, tests only, one constant/name, ≤ ~30 lines in one place) needs no review agent. The agent
  is `~/.claude/agents/opus-code-reviewer.md` (xhigh, tools Read/Grep/Bash). Write "审核按项目 CLAUDE.md 的 Code
  review" in orders instead of restating it.
- **No compatibility** (兼容 is forbidden): a changed payload gets a new name; one standing permission — a new campaign
  setting may default to the old behaviour, records without it read as that default. Any other exception is the
  user's, case by case (example: frontend D178 (7), the speed field left in published sets).
- **Closing phase (outline D184, 2026-10-09):** stage D's implementer fixes small defects of stages A, B and C directly
  (bugs, waste, mirrors, wrong interfaces), with that stage's checks; names go into its report and you write them into
  the interface after. Formats/identities of artefacts, rebuilds and decided rules still go to the user first. The
  rule names stage D's implementer only.
- **Fast forms** (outline D138): a faster computation is a mode beside the readable one, checked equal on fixed inputs
  before use. Shared code changes keep stage C's behaviour bit for bit (D149). No code fingerprints anywhere (outline
  D21): behaviour checks only.
- **Runs**: formal runs from a run worktree at the merged commit (D163), in a systemd unit, intent in
  `docs/experiments/intents.json` before publication; the user merges branches; no data overwritten or deleted without
  the user's word; KAUS and the test days sealed; val read once per campaign.

## 5 State on 2026-10-09 (evening)

**Branches.** `dev-multi-control` (D), `dev-two-tier-v4-post` (C) and `dev-frontend` (fronter) are fully merged into
`dev-two-tier` (none ahead). Nothing runs on the host or the GPU.

**Stage C** (post_training.md). Stopped at `post_seg60_20261007` round 5 (the user's choice; its one val read done).
C21–C26 built and merged (D170 segments, D171/D173 value method — tried, fell below its start; D175 diagnostic
readout, read; D176 window lists; C26 the public faulty-point reader; `identity_of`). C27 (D185, a round resumed by its
batches) is assigned to stage D's implementer. `requests_from_c_to_designer.md`: no open request. The next stage C
experiment is the user's choice.

**Stage D** (multi_control.md). MC0–MC5 done; MC6 (version 1's campaign from P55 round 5) was **stopped by the user in
round 0** ("太慢而且不一定有用"). MC11's first code merged (D181 items 55, 58–60, 56 `multi_speed`). The current note
(`notes/stage_d.md`, commit `76eeecfd`, **not yet read by D** when written) orders:
1. item 69 → D183 (4): release a branch point's copies after their groups are taken (small change);
2. D183: the speaker's cache holds the rows written plus a 64-row step (no double clone, trimmed copies);
3. D186 (1): a landing inserted without re-checking the roster (exact);
4. D185 / post-training C27: a round resumed by its batches;
5. D186 (2)(3): grammar masks and row inputs over the batch (fast modes; (3) equality first — if not bit for bit,
   the user decides on a bound);
6. MC12 / D182: branch points found backward from each event (setting `branching = grid | backward`; 32 s steps, at
   most 8, avoided through the event + 60 s; the decision point, one 16 s point and point B — the latest avoiding with
   no go-around — give groups; no early point);
7. then, with the GPU free: `multi_speed` before/after, a new memory measure (D179) for the worker count, grid vs
   backward compared; report to the user, who chooses the next campaign.
D must also rewrite multi_control §0.3 (still says MC1 running, MC6 not started) and its requests file (63–70 done).
Open items: O18 (time term, readout only), O19 (a value function for D — after a C value campaign that gains; C's
value campaign fell), O20 (two devices; a shared prefix cache — only after D183's measure), O21 (cost by span and
uninformative groups — after a D campaign's readouts). D142 was kept against "other aircraft replay their first
sentence" (the user, 2026-10-09).

**Fronter** (frontend.md). F0–F5 done and merged; requests 1–15 (D177) and 1–8 (D178) decided. Pending: the two
mirrors of frontend D178 (6) can go now — `identity_of` (8baf6468) and `loop_positions` (a332466a) are merged (fronter's
note says to remove them once merged; D's note also allows D to do it). Stage D's formal Training sets wait for a D
campaign.

**Decisions waiting for the user:** the next stage D campaign (after the speed work and its measures) and the next
stage C experiment; whether MC6's line resumes at all; D186 (3)'s bound if equality fails; O19–O21 when their
evidence exists.

## 6 How requests were evaluated (patterns worth keeping)

- Check each proposal against: the decided rules (D numbers), coupling with stages A/B/C/D (shared code in
  `post/`, `post_train`, `WindowLoop`, `prior/`), the identity/format rules, the cost (time, memory, a check's own cost
  against a rerun), and whether a gradient/credit argument holds (a group's baseline must share its start state).
- Prefer the change that keeps numbers equal (a lifetime fix, an index insert, boolean vectorisation) and is cheap;
  measure before structural work (`multi_speed`, the memory measure).
- When the user objects to part of a recommendation (e.g. the early branch point would refly the whole window), drop
  it and state the remaining limit in the decision instead of re-arguing.
- Keep the user's quotes in the source column for their decisions; label your own readings as Claude's.
