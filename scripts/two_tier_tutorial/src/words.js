/* The vocabulary, the grammar, and the labeller's heading reading, as plain functions.
   Ports of instructions/words.py, instructions/grammar.py (_rules, column_mask) and instructions/labeller/lateral.py
   (per_step_words). tests: scripts/two_tier_tutorial/tests/ compares them with the Python on random inputs. */
(function () {
  const root = typeof window !== 'undefined' ? window : globalThis;
  const TT = root.TT = root.TT || { demos: {}, data: null };
  const UN = -1, GA = -2;
  TT.UN = UN; TT.GA = GA;
  TT.COLUMNS = ['runway', 'heading', 'altitude', 'angle', 'speed'];

  // ---------- the vocabulary spec (vocabulary §3, §8; spec.json of the formal artefact) ----------
  TT.SPEC = {
    headingStep: 5, headingLead: 4, turnRateMax: 4.7, bankMax: 32, rollRate: 5, cycle: 1,
    altSteps: [60, 120, 450], altTops: [1260, 2700, 5400], fit: 10,
    descentEdges: [-0.5, 2.0, 2.75, 3.75, 10.0], descentCentres: [1.5, 2.5, 3.0, 4.5], climbCentre: 1.25, climbMax: 15,
    speedStep: 5, speedMin: 20, speedMax: 250, speedTol: 5, aMax: 1.4, aUnspec: 0.25,
    lateralTol: 30, verticalTol: 15, verticalFinalTol: 10,
    pathTau: 2.0, pathRateFactor: 2.0,
  };

  TT.Words = function (spec) {
    const W = { spec };
    W.nHeading = Math.round(360 / spec.headingStep);
    const lv = [0];
    spec.altSteps.forEach((step, i) => { const bottom = i ? spec.altTops[i - 1] : 0, top = spec.altTops[i]; for (let k = 1; k <= Math.round((top - bottom) / step); k++) lv.push(bottom + step * k); });
    W.levels = lv; W.nLevels = lv.length; W.noLevelOff = lv.length;
    const gaps = lv.slice(1).map((v, i) => v - lv[i]);
    const left = [gaps[0], ...gaps], right = [...gaps, gaps[gaps.length - 1]];
    W.tol = lv.map((_, i) => Math.max(left[i], right[i]) / 2 + spec.fit);
    W.nDescent = spec.descentCentres.length; W.angleClimb = W.nDescent + 1; W.angleLevel = 0;
    W.nSpeed = Math.round((spec.speedMax - spec.speedMin) / spec.speedStep) + 1; W.speedUnspec = W.nSpeed;
    W.counts = { heading: W.nHeading, altitude: W.nLevels + 1, angle: W.nDescent + 2, speed: W.nSpeed + 1 };
    W.headingRel = (k) => { const r = k * spec.headingStep; return r > 180 ? r - 360 : r; };
    W.headingClass = (trackDeg, courseDeg) => { const rel = TT.wrap360(trackDeg - courseDeg); return Math.round(rel / spec.headingStep) % W.nHeading; };
    W.headingTrack = (k, course) => TT.wrap360(course + W.headingRel(k));
    W.altIndex = (h) => { let b = 0, bd = Infinity; lv.forEach((v, i) => { const d = Math.abs(v - h); if (d < bd) { bd = d; b = i; } }); return b; };
    W.altLevel = (i) => (i === W.noLevelOff ? null : lv[i]);
    W.altTol = (i) => W.tol[i === W.noLevelOff ? 0 : i];
    W.angleDeg = (k) => (k === 0 ? 0 : k === W.angleClimb ? -spec.climbCentre : spec.descentCentres[k - 1]);
    W.angleClass = (deg) => { const e = spec.descentEdges; if (deg >= e[0] && deg <= e[e.length - 1]) { let k = 0; while (k < e.length - 1 && deg >= e[k + 1]) k++; return 1 + Math.min(k, W.nDescent - 1); } return W.angleClimb; };
    W.speedIndex = (v) => Math.round((v - spec.speedMin) / spec.speedStep);
    W.speedMps = (i) => (i === W.speedUnspec ? null : spec.speedMin + i * spec.speedStep);
    W.angleName = (k) => (k === UN ? '·' : k === 0 ? 'level' : k === W.angleClimb ? 'climb' : 'descent ' + k);
    return W;
  };
  TT.W = TT.Words(TT.SPEC);

  // ---------- the grammar (vocabulary §3.7; instructions/grammar.py) ----------
  const BROKEN = [
    ['first step incomplete', (c) => `no word for ${c.missing.join(', ')}`],
    ['runway word not permitted', () => 'the first step says no candidate'],
    ['runway word not permitted', () => 'go-around while a go-around is in force'],
    ['go-around without a level above', (c) => `"no level-off" in force at ${c.height.toFixed(0)} m above the airport (rule 6)`],
    ['runway word not permitted', (c) => `${c.runway} is not a candidate (${c.candidates} candidates)`],
    ['runway word not permitted', (c) => `runway ${c.runway} is already in force`],
    ['no level-off during a go-around', () => 'a runway word ends the go-around first (rule 5)'],
    ['no level-off without a descent', (c) => `angle class ${c.angle} in force (rule 4)`],
    ['altitude and angle incompatible', (c) => `target ${Number.isFinite(c.target) ? c.target.toFixed(0) : '–'} m at ${c.height.toFixed(0)} m above the airport with angle class ${c.angle}${c.angle === 0 ? ' (level)' : ''}`],
  ];
  TT.BROKEN = BROKEN;

  TT.rules = (W, first, inRunway, inGo, inAlt, inAngle, runway, heading, altitude, angle, speed, height, candidates) => {
    let code = 0; const broken = (cond, n) => { if (code === 0 && cond) code = n; };
    const lookup = (arr, i) => (i >= 0 ? arr[i] : NaN);
    const levelAt = (i) => (i < 0 ? NaN : i === W.noLevelOff ? NaN : W.levels[i]);
    const bandAt = (i) => (i < 0 ? NaN : i === W.noLevelOff ? W.tol[0] : W.tol[i]);
    const later = !first;
    const missing = runway === UN || heading === UN || altitude === UN || angle === UN || speed === UN;
    const outside = runway < 0 || runway >= candidates;
    const go = runway === GA, candidate = runway !== UN && !go;
    const climbTo = levelAt(altitude);
    const notAbove = Number.isNaN(climbTo) || climbTo <= height + bandAt(altitude);
    broken(first && missing, 1); broken(first && outside, 2); broken(later && go && inGo, 3);
    broken(later && go && inAlt === W.noLevelOff && notAbove, 4);
    broken(later && candidate && outside, 5); broken(later && candidate && runway === inRunway && !inGo, 6);
    const afterRunway = later && !candidate ? inRunway : runway;
    const afterGo = later && (go || (!candidate && inGo));
    const afterAlt = later && altitude === UN ? inAlt : altitude;
    const afterAngle = later && angle === UN ? inAngle : angle;
    broken(altitude === W.noLevelOff && afterGo, 7);
    const said = altitude !== UN || angle !== UN;
    const descent = afterAngle >= 1 && afterAngle <= W.nDescent;
    const target = levelAt(afterAlt), band = bandAt(afterAlt);
    const agree = target < height - band ? descent : (target > height + band ? afterAngle === W.angleClimb : true);
    broken(said && afterAlt === W.noLevelOff && !descent, 8);
    broken(said && afterAlt !== W.noLevelOff && !agree, 9);
    return { code, runway: afterRunway, go: afterGo, altitude: afterAlt, angle: afterAngle, target };
  };

  // inForce: {runway, go, heading, altitude, angle, speed} or null for the first step
  TT.apply = (W, inForce, step, height, nCand) => {
    const [runway, heading, altitude, angle, speed] = step;
    const b = inForce || { runway: UN, go: false, heading: UN, altitude: UN, angle: UN, speed: UN };
    const r = TT.rules(W, inForce === null, b.runway, b.go, b.altitude, b.angle, runway, heading, altitude, angle, speed, height, nCand);
    if (r.code) {
      const [reason, f] = BROKEN[r.code - 1];
      return { ok: false, code: r.code, rule: r.code, reason, detail: f({ missing: TT.COLUMNS.filter((_, i) => step[i] === UN), runway, candidates: nCand, height, angle: r.angle, target: r.target }) };
    }
    return { ok: true, state: { runway: r.runway, go: r.go, heading: heading === UN ? b.heading : heading, altitude: r.altitude, angle: r.angle, speed: speed === UN ? b.speed : speed } };
  };

  TT.columnWords = (W, column, nCand) => {
    if (column === 0) return [UN, GA, ...Array.from({ length: nCand }, (_, i) => i)];
    const count = W.counts[TT.COLUMNS[column]]; return Array.from({ length: count + 1 }, (_, i) => i - 1);
  };

  // D62: for each word of `column`, can some words of the later columns (each among permitted[c]) make the row pass?
  // permitted: {c: Set or array of allowed words} for later columns; missing = all.
  TT.columnMask = (W, inForce, said, column, height, nCand, permitted = {}) => {
    const later = []; for (let c = column + 1; c < 5; c++) later.push(c);
    const reduced = (c) => c === 1 || c === 4;
    const allowed = (c) => { const all = TT.columnWords(W, c, nCand); const p = permitted[c]; const ok = p ? all.filter(w => (p instanceof Set ? p.has(w) : p.includes(w))) : all; return reduced(c) ? [UN, 0].filter(v => v === UN ? ok.includes(UN) : ok.some(w => w >= 0)) : ok; };
    const lists = later.map(allowed);
    const b = inForce || { runway: UN, go: false, heading: UN, altitude: UN, angle: UN, speed: UN };
    const test = (word) => {
      const row = [...said, word]; let found = false;
      const rec = (i) => { if (found) return; if (i === lists.length) { const r = TT.rules(W, inForce === null, b.runway, b.go, b.altitude, b.angle, row[0], row[1], row[2], row[3], row[4], height, nCand); if (r.code === 0) found = true; return; } for (const w of lists[i]) { row.push(w); rec(i + 1); row.pop(); if (found) return; } };
      rec(0); return found;
    };
    const words = TT.columnWords(W, column, nCand);
    if (reduced(column)) { const un = test(UN), sd = test(0); return words.map(w => (w === UN ? un : sd)); }
    return words.map(test);
  };

  // ---------- the labeller's heading reading (instructions/labeller/lateral.py per_step_words) ----------
  TT.perStepWords = (track, course, stepDeg, leadRows) => {
    const rows = track.length, led = track.map((_, i) => track[Math.min(i + leadRows, rows - 1)]);
    const snap = (v) => Math.round(v / stepDeg) * stepDeg;
    let rel = snap(led[0] - course[0]); const said = [[0, rel]]; let current = course[0] + rel;
    for (let r = 1; r < rows; r++) {
      if (Math.abs(led[r] - current) > stepDeg / 2) { rel = snap(led[r] - course[r]); said.push([r, rel]); current = course[r] + rel; }
    }
    return said;
  };


  // the speed words of a change of speed and the executor that flies them (vocabulary §4.5, §5.6; autopilot/speed.py)
  TT.speedStepsSim = (o) => { // o: {v0, v1, T, steps, t0}
    const sp = TT.SPEC, t0 = o.t0 ?? 20, ts = Array.from({ length: 181 }, (_, i) => i);
    const smooth = (t) => { const u = Math.min(1, Math.max(0, (t - t0) / o.T)); return o.v0 + (o.v1 - o.v0) * (u * u * (3 - 2 * u)); };
    const obs = ts.map(smooth), g = (v) => sp.speedMin + Math.round((v - sp.speedMin) / sp.speedStep) * sp.speedStep;
    const ev = []; let w = g(o.v0); ev.push([0, w]); const target = g(o.v1), dir = Math.sign(target - w);
    if (o.steps) { for (let t = 0; t <= 180 && w !== target; t += 2) { const nxt = w + dir * sp.speedStep; if (Math.abs(smooth(t) - nxt) < Math.abs(smooth(t) - w)) { w = nxt; ev.push([t, w]); } } }
    else if (target !== w) ev.push([t0, target]);
    let V = obs[0], pos = 0, opos = 0, k = 0, cur = ev[0][1]; const flown = [], diff = [];
    ts.forEach((t, i) => { while (k + 1 < ev.length && ev[k + 1][0] <= t) { k++; cur = ev[k][1]; } const tau = sp.speedTol / sp.aMax; const rate = Math.min(Math.max((cur - V) / tau, -sp.aMax), sp.aMax); flown.push([t, V]); diff.push([t, pos - opos]); V += rate; pos += V; opos += obs[i]; });
    return { ts, obs, ev, flown, diff };
  };

  // ---------- the executor's lateral law (autopilot/lateral.py) ----------
  TT.stoppingRate = (errDeg, V, rollRateDegS) => TT.deg(Math.sqrt(2 * TT.G * TT.rad(rollRateDegS) / V * TT.rad(Math.abs(errDeg))));
  TT.wordRate = (errDeg, toGoS, V, o) => {
    const floor = Math.max(toGoS, 2 * o.cycle);
    const a = Math.abs(errDeg / floor), b = TT.stoppingRate(errDeg, V, o.rollRate);
    const rate = Math.min(Math.min(a, b), o.rmax);
    return { rate: Math.sign(errDeg) * rate, parts: { timeLeft: a, stopping: b, max: o.rmax }, binds: a <= b && a <= o.rmax ? 'time left' : b <= o.rmax ? 'stopping rate' : 'turn-rate limit' };
  };
})();
