// Simulation data for the paper figures, from the same JS code as the demos (so a figure and its demo cannot differ).
//   node scripts/two_tier_tutorial/figure_data.js > scripts/two_tier_tutorial/data/figure_sim.json
global.window = global;
const path = require('path');
for (const f of ['core.js', 'words.js', 'exec.js', 'demos_exec.js']) require(path.join(__dirname, 'src', f));
const TT = global.TT, out = {};
const withSpec = (over, f) => { const old = {}; Object.keys(over).forEach(k => { old[k] = TT.SPEC[k]; TT.SPEC[k] = over[k]; }); try { return f(); } finally { Object.assign(TT.SPEC, old); } };

// lateral law: one word, a turn of 90° at 80 m/s, and a series of 5° words every 4 s
const lat = (events, T) => TT.fly({ e: 0, n: 0, track: 0, V: 80, h: 0, gamma: 0 }, events, T, { level0: 0 })
  .map(s => ({ t: s.t, track: TT.wrap180(s.track), rate: Math.abs(s.lat.rate), timeLeft: s.lat.parts.timeLeft, stopping: s.lat.parts.stopping, bank: -TT.deg(s.bank), binds: s.lat.binds }));
out.lateral_one = lat([{ t: 5, heading: { abs: 90 } }], 60);
const series = []; for (let k = 1; k <= 18; k++) series.push({ t: 5 + (k - 1) * 4, heading: { abs: 5 * k } });
out.lateral_series = lat(series, 100); out.lateral_series_words = series.map(e => [e.t, e.heading.abs]);
out.lead_s = TT.SPEC.headingLead; out.rmax = TT.SPEC.turnRateMax;

// vertical law: descent from 900 m to 600 m with descent 2 (2.5°), V = 70 m/s
const vert = TT.fly({ e: 0, n: 0, track: 0, V: 70, h: 900, gamma: 0 }, [{ t: 1, level: 600, angle: 2 }], 220, { level0: 900 });
let x = 0; out.vertical = vert.map((s, i) => { if (i) x += s.V / 1000; return { t: s.t, km: x, h: s.h, gamma: -TT.deg(s.gamma), captured: s.vert.captured, levelOff: s.vert.levelOff }; });
const gaRun = TT.fly({ e: 0, n: 0, track: 0, V: 70, h: 300, gamma: 0 }, [{ t: 1, level: 900, angle: TT.W.angleClimb }], 260, { level0: 300, goAround: true, goAroundDeg: 3 });
x = 0; out.vertical_ga = gaRun.map((s, i) => { if (i) x += s.V / 1000; return { t: s.t, km: x, h: s.h, gamma: -TT.deg(s.gamma) }; });

// speed steps: 150 -> 80 m/s over 70 s, as steps and as one word
out.speed_steps = TT.speedStepsSim({ v0: 150, v1: 80, T: 70, steps: true });
out.speed_one = TT.speedStepsSim({ v0: 150, v1: 80, T: 70, steps: false });

// the closed-loop sketch, corrections on and off
const base = { rate: 2.5, lag: 4, intercept: 25, wobble: 0.6, Y: 30, seed: 5 };
const clip = (r) => ({ rows: r.rows.map(q => [q.e, q.n]), words: r.words, flown: r.flown.map(q => [q.e, q.n, q.t]), errs: r.errs, corrs: r.corrs });
out.closed_on = clip(TT.closedLoopSim({ ...base, on: true })); out.closed_off = clip(TT.closedLoopSim({ ...base, on: false }));
out.closed_params = base;

// the vocabulary grids
out.levels = TT.W.levels; out.tol = TT.W.tol;
process.stdout.write(JSON.stringify(out));
