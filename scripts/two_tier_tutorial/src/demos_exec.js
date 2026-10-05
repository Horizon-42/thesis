/* Demos for the executor, the judge and the closed-loop reading (vocabulary.md §4.9, §5). */
(function () {
  const TT = window.TT, E = TT.el, S = TT.svg, W = () => TT.W, SP = () => TT.SPEC, D = () => TT.data;
  const withSpec = (over, f) => { const old = {}; Object.keys(over).forEach(k => { old[k] = TT.SPEC[k]; TT.SPEC[k] = over[k]; }); try { return f(); } finally { Object.assign(TT.SPEC, old); } };

  // ============================================================ lateral law
  TT.demos.lateral_law = (root, body) => {
    const st = { V: 80, turn: 90, mode: 'one', every: 4, bank: 32, lead: 4 };
    const r = TT.ui.row(body);
    TT.ui.slider(r, { label: 'Speed (m/s)', min: 50, max: 250, step: 5, value: st.V, onInput: (v) => { st.V = v; draw(); } });
    TT.ui.slider(r, { label: 'Turn (deg)', min: 10, max: 170, step: 5, value: st.turn, onInput: (v) => { st.turn = v; draw(); } });
    TT.ui.slider(r, { label: 'Bank limit (deg)', min: 15, max: 45, step: 1, value: st.bank, onInput: (v) => { st.bank = v; draw(); } });
    const r2 = TT.ui.row(body);
    TT.ui.seg(r2, [['one', 'One word'], ['series', 'A word every N s, 5° each']], 'one', (v) => { st.mode = v; draw(); });
    TT.ui.slider(r2, { label: 'N (s)', min: 2, max: 12, step: 1, value: st.every, onInput: (v) => { st.every = v; draw(); } });
    TT.ui.slider(r2, { label: 'Lead L (s)', min: 0, max: 8, step: 1, value: st.lead, onInput: (v) => { st.lead = v; draw(); } });
    const c1 = TT.chart(body, { w: 760, h: 200, x: [0, 90], y: [-10, 180], yt: [0, 45, 90, 135, 180], xl: 'time (s)', yl: 'track change (deg)', label: 'Track against time' });
    const c2 = TT.chart(body, { w: 760, h: 190, x: [0, 90], y: [0, 8], xl: 'time (s)', yl: 'turn rate (deg/s)', label: 'The three rates of the lateral law' });
    const c3 = TT.chart(body, { w: 760, h: 140, x: [0, 90], y: [-45, 45], xl: 'time (s)', yl: 'bank (deg)', yt: [-30, 0, 30], label: 'Bank angle' });
    TT.ui.legend(body, [['word', 'word target', 'dash'], ['flown', 'flown track'], ['corr', 'error ÷ time left until the lead ends'], ['ok', 'stopping rate √(2·g·p·|e|/V)'], ['obs', 'turn-rate limit 4.7°/s', 'dash'], ['flown', 'rate that flies (the smallest)']]);
    const out = TT.ui.out(body);
    function draw() {
      [c1, c2, c3].forEach(c => c.clear());
      const T = st.mode === 'one' ? 90 : Math.max(90, 5 + (st.turn / 5) * st.every + 40);
      [c1, c2, c3].forEach(c => { c.x1 = T; });
      const ev = []; const t0 = 5;
      if (st.mode === 'one') ev.push({ t: t0, heading: { abs: st.turn } });
      else for (let k = 1; k <= st.turn / 5; k++) ev.push({ t: t0 + (k - 1) * st.every, heading: { abs: k * 5 } });
      const run = withSpec({ headingLead: st.lead, bankMax: st.bank }, () => TT.fly({ e: 0, n: 0, track: 0, V: st.V, h: 0, gamma: 0 }, ev, T, { level0: 0 }));
      redraw(c1, c2, c3, T);
      c1.step(ev.map(e => [e.t, e.heading.abs]).concat([[T, ev[ev.length - 1].heading.abs]]), 'word dash');
      c1.line(run.map(s => [s.t, TT.wrap180(s.track)]), 'flown');
      if (st.mode === 'one') c1.vline(t0 + st.lead, 'word');
      c2.line(run.map(s => [s.t, s.lat.parts.timeLeft > 8 ? NaN : Math.abs(s.lat.parts.timeLeft)]), 'corr thin');
      c2.line(run.map(s => [s.t, Math.min(8, s.lat.parts.stopping)]), 'ok thin');
      c2.hline(SP().turnRateMax, 'obs dash');
      c2.line(run.map(s => [s.t, Math.abs(s.lat.rate)]), 'flown');
      c3.hline(st.bank, 'faint dash'); c3.hline(-st.bank, 'faint dash'); c3.hline(0, 'faint');
      c3.line(run.map(s => [s.t, -TT.deg(s.bank)]), 'flown');
      const done = run.find(s => s.t > t0 && Math.abs(TT.wrap180(s.track) - ev[ev.length - 1].heading.abs) < 1);
      const bind = {}; run.forEach(s => { if (Math.abs(s.lat.rate) > 0.01) bind[s.lat.binds] = (bind[s.lat.binds] || 0) + 1; });
      out(`Track within 1° of the last word at <b>${done ? done.t.toFixed(0) + ' s' : 'later than the plot'}</b>. Maximum bank <b>${TT.deg(Math.max(...run.map(s => Math.abs(s.bank)))).toFixed(0)}°</b>.\nThe law that sets the rate, in cycles: ${Object.entries(bind).map(([k, v]) => `<b>${k}</b> ${v}`).join(' · ')}`);
    }
    function redraw(...cs) { cs.slice(0, 3).forEach(c => TT.reaxes(c)); }
    draw();
    body.append(E('div', { class: 'cap', html: 'The rate that flies is the smallest of three (<code>autopilot/lateral.py</code> <code>word_rate</code>): the error divided by the time left until the lead ends (not less than 2 cycles); the stopping rate, the fastest rate whose bank the executor can still take out before the error is gone; and the turn-rate limit. The sketch uses a coordinated turn with the roll rate of 5°/s from FAA Order 8260.3G.' }));
  };

  // ============================================================ vertical law
  TT.demos.vertical_law = (root, body) => {
    const Wd = W(); const st = { V: 70, h: 900, level: 600, angle: 2, nlo: false, ga: false, gaDeg: 3 };
    const r = TT.ui.row(body);
    TT.ui.slider(r, { label: 'Speed (m/s)', min: 50, max: 150, step: 5, value: st.V, onInput: (v) => { st.V = v; draw(); } });
    TT.ui.slider(r, { label: 'Start height (m above E)', min: 100, max: 2500, step: 20, value: st.h, onInput: (v) => { st.h = v; draw(); } });
    const r2 = TT.ui.row(body);
    TT.ui.select(r2, 'Level word', Wd.levels.slice(0, 31).map(l => [String(l), l + ' m']), String(st.level), (v) => { st.level = +v; draw(); });
    TT.ui.select(r2, 'Angle word', [[1, 'descent 1 (1.5°)'], [2, 'descent 2 (2.5°)'], [3, 'descent 3 (3.0°)'], [4, 'descent 4 (4.5°)']].map(([k, t]) => [String(k), t]), String(st.angle), (v) => { st.angle = +v; draw(); });
    const r3 = TT.ui.row(body);
    TT.ui.check(r3, '"no level-off" (the final descent)', st.nlo, (v) => { st.nlo = v; draw(); });
    TT.ui.check(r3, 'Go-around state G is true (the climb word climbs at γ_GA)', st.ga, (v) => { st.ga = v; draw(); });
    TT.ui.slider(r3, { label: 'γ_GA (deg)', min: 1.885, max: 3, step: 0.005, value: st.gaDeg, fmt: (v) => v.toFixed(2), onInput: (v) => { st.gaDeg = v; draw(); } });
    const c1 = TT.chart(body, { w: 760, h: 260, x: [0, 30], y: [0, 2600], xl: 'distance flown (km)', yl: 'height above E (m)', label: 'Height against distance' });
    const c2 = TT.chart(body, { w: 760, h: 140, x: [0, 30], y: [-6, 6], xl: 'distance flown (km)', yl: 'path angle (deg)', yt: [-4, 0, 4], label: 'Path angle' });
    const out = TT.ui.out(body);
    function draw() {
      [c1, c2].forEach(c => c.clear());
      const climb = !st.nlo && st.level > st.h; const angle = climb ? Wd.angleClimb : st.angle;
      const ev = [{ t: 1, level: st.nlo ? null : st.level, angle, noLevelOff: st.nlo }];
      const run = TT.fly({ e: 0, n: 0, track: 0, V: st.V, h: st.h, gamma: 0 }, ev, 420, { level0: st.h, angle0: 0, goAround: st.ga && climb, goAroundDeg: st.gaDeg }).filter(s => s.h > 0 || s.t < 3);
      let x = 0; const pts = run.map((s, i) => { if (i) x += s.V / 1000; return [x, s.h]; });
      const xm = Math.max(5, Math.min(60, x)); [c1, c2].forEach(c => { c.x1 = xm; });
      c1.y1 = Math.max(st.h, st.level) * 1.12 + 50;
      [c1, c2].forEach(c => TT.reaxes(c));
      if (!st.nlo) c1.hline(st.level, 'word dash');
      c1.line(pts, 'flown');
      const gamma = run.map((s, i) => [pts[i][0], -TT.deg(s.gamma)]);
      c2.hline(0, 'faint'); c2.line(gamma, 'flown');
      const first = run.find(s => s.vert.captured && s.t > 2);
      const nominal = Math.abs(Wd.angleDeg(angle));
      out(st.nlo ? `<b>"no level-off"</b> with descent ${st.angle}: the executor flies the class angle ${Wd.angleDeg(st.angle)}° down. It has no law that follows the glidepath or stops the descent: the words and the judge decide.`
        : `<b>${climb ? 'Climb' : 'Descent'}</b> to ${st.level} m at the ${climb ? (st.ga ? 'go-around angle ' + st.gaDeg.toFixed(2) + '°' : 'climb-class angle ' + SP().climbCentre + '°') : 'descent-' + st.angle + ' angle ' + Wd.angleDeg(st.angle) + '°'}. Level-off starts ${first ? '<b>' + Math.abs(first.h - st.level).toFixed(0) + ' m</b> from the level' : 'later than the plot'}: V·γ²/(2·γ̇max) = ${first ? first.vert.levelOff.toFixed(0) : '–'} m.`);
    }
    draw();
    body.append(E('div', { class: 'cap', html: 'The inner loop is γ̇* = sat((γ_ref − γ)/τ_γ, ±γ̇max) with τ_γ = 2 s (<code>autopilot/vertical.py</code>). The mode chooses γ_ref: the class angle until the level-off height, then the hold law. A captured level stays captured until a new altitude or angle word.' }));
  };

  // ============================================================ one cycle: wanted rates -> controls (inverse.py)
  TT.demos.inverse_cycle = (root, body) => {
    const st = { V: 80, g: -3, tr: 3, gr: 0, vd: -0.5, prev: 0, m: 60, h: 600 };
    const grid = E('div', { class: 'grid2' }); body.append(grid); const A = E('div'), B = E('div'); grid.append(A, B);
    const rows = [['V', 'Airspeed (m/s)', 50, 250, 5], ['g', 'Path angle γ (deg)', -6, 6, 0.5], ['tr', 'Wanted track rate (°/s)', 0, 6, 0.1], ['gr', 'Wanted path-angle rate (°/s)', -1, 1, 0.05], ['vd', 'Wanted airspeed rate (m/s²)', -2, 2, 0.1], ['prev', 'Bank of the cycle before (deg)', -32, 32, 1], ['m', 'Mass (t)', 40, 80, 1], ['h', 'Height (m)', 0, 3000, 100]];
    rows.forEach(([k, label, mn, mx, step]) => TT.ui.slider(A, { label, min: mn, max: mx, step, value: st[k], onInput: (v) => { st[k] = v; draw(); } }));
    const svg = S('svg', { viewBox: '0 0 300 220', role: 'img', 'aria-label': 'Rear view of the aircraft: the bank angle and the lift vector' }); B.append(svg);
    const out = TT.ui.out(body);
    function draw() {
      const V = st.V, gam = TT.rad(st.g), Sw = 122.6, CD0 = 0.027, K = 0.04, Tmax = 210e3, mass = st.m * 1000;
      const rho = 1.225 * Math.pow(1 - 2.2558e-5 * st.h, 4.2559);
      const at = TT.attitude(V, gam, st.tr, TT.rad(st.gr), TT.rad(st.prev));
      const A_ = -TT.rad(st.tr) * V * Math.cos(gam) / TT.G, B_ = TT.rad(st.gr) * V / TT.G + Math.cos(gam);
      const wanted = Math.atan2(A_, Math.max(B_, 0));
      const n = at.load, CL = n * mass * TT.G / (0.5 * rho * V * V * Sw), CD = CD0 + K * CL * CL, drag = 0.5 * rho * V * V * Sw * CD;
      const Tw = mass * (st.vd + TT.G * Math.sin(gam)) + drag, frac = Tw / Tmax, flown = Math.max(-0.2, Math.min(1, frac));
      svg.replaceChildren(); const cx = 150, cy = 120;
      const bank = at.bank, lift = 60 * n;
      svg.append(S('line', { x1: 20, x2: 280, y1: cy + 2, y2: cy + 2, class: 'ln thin faint dash' }));
      const ar = (x, y, dx, dy, cls, lbl) => { svg.append(S('line', { x1: x, y1: y, x2: x + dx, y2: y + dy, class: 'ln ' + cls, 'marker-end': 'url(#a3)' })); if (lbl) svg.append(S('text', { x: x + dx * 1.18, y: y + dy * 1.18 + 4, class: 'lbl sm', 'text-anchor': 'middle', text: lbl })); };
      svg.append(S('defs', {}, S('marker', { id: 'a3', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto' }, S('path', { d: 'M0 0L10 5L0 10z', fill: 'currentColor' }))));
      const g = S('g', { transform: `rotate(${TT.deg(-bank)} ${cx} ${cy})` });
      g.append(S('line', { x1: cx - 70, x2: cx + 70, y1: cy, y2: cy, class: 'ln ink', 'stroke-width': 5 }), S('circle', { cx, cy, r: 7, class: 'f-flown' }));
      svg.append(g);
      ar(cx, cy, 0, TT.G * 6.1, 'ink', 'weight');
      ar(cx, cy, lift * Math.sin(-bank), -lift * Math.cos(bank), 'flown', 'lift n·W');
      svg.append(S('text', { x: cx, y: 20, 'text-anchor': 'middle', class: 'lbl', text: `bank ${TT.deg(Math.abs(bank)).toFixed(1)}°, load factor ${n.toFixed(2)}` }));
      const lim = Object.entries(at.bound).filter(([, v]) => v).map(([k]) => k); if (frac > 1) lim.push('thrust max'); if (frac < -0.2) lim.push('thrust min');
      out(`A = −χ̇·V·cosγ/g = <b>${A_.toFixed(3)}</b>   B = γ̇·V/g + cosγ = <b>${B_.toFixed(3)}</b>\nwanted bank φ = atan2(A, B) = <b>${TT.deg(wanted).toFixed(1)}°</b>   → flown bank <b>${TT.deg(bank).toFixed(1)}°</b>   load factor n = B/cosφ = <b>${n.toFixed(2)}</b>\ndrag (toy polar) ${(drag / 1000).toFixed(1)} kN   thrust T = m(V̇ + g·sinγ) + D = <b>${(Tw / 1000).toFixed(0)} kN</b> = ${(frac * 100).toFixed(0)} % of T_max → flown <b>${(flown * 100).toFixed(0)} %</b>\nlimits that bind: <b class="${lim.length ? 'bad-t' : 'ok-t'}">${lim.length ? lim.join(', ') : 'none'}</b>`);
    }
    draw();
    body.append(E('div', { class: 'cap', html: 'The three laws only want <em>rates</em>. This cycle turns them into the three controls: bank, load factor and thrust (<code>autopilot/inverse.py</code>). The limits act in order: bank (cap and roll rate), load factor [0.5, 2.0], thrust [−20 %, 100 %]. A limited bank costs turn rate, never path angle. The drag polar here is a toy (S = 122.6 m², CD0 = 0.027, k = 0.04, T_max = 210 kN); the real code uses the polar of each aircraft type.' }));
  };

  // ============================================================ the judge
  TT.demos.judge = (root, body) => {
    const c23 = D().flights.vectored.candidates.find(c => c.ident === '23R');
    const st = { dyn: false, gnd: false, cross: true, G: false, off: 20, lat: 25, h: 15, da: true, vz: 5, lz: 30, other: false };
    const g1 = E('div', { class: 'grid2' }); body.append(g1); const A = E('div', { class: 'card' }), B = E('div', { class: 'card' }); g1.append(A, B);
    A.append(E('h4', { text: 'The crossing of the runway plane' }));
    TT.ui.check(A, 'The aircraft crosses the threshold plane of R', st.cross, (v) => { st.cross = v; draw(); });
    TT.ui.check(A, 'G is true (a go-around is in force)', st.G, (v) => { st.G = v; draw(); });
    TT.ui.slider(A, { label: 'Track minus course (deg)', min: 0, max: 90, step: 1, value: st.off, onInput: (v) => { st.off = v; draw(); } });
    TT.ui.slider(A, { label: 'Offset from the centreline (m)', min: -400, max: 400, step: 5, value: st.lat, onInput: (v) => { st.lat = v; draw(); } });
    TT.ui.slider(A, { label: 'Height above the threshold (m)', min: 0, max: 250, step: 5, value: st.h, onInput: (v) => { st.h = v; draw(); } });
    B.append(E('h4', { text: 'The decision-altitude check, and the rare events' }));
    TT.ui.check(B, 'The aircraft descends through the DA on the final (a DA point exists)', st.da, (v) => { st.da = v; draw(); });
    TT.ui.slider(B, { label: 'Height above the glidepath at the DA point (m)', min: -50, max: 50, step: 1, value: st.vz, onInput: (v) => { st.vz = v; draw(); } });
    TT.ui.slider(B, { label: 'Offset at the DA point (m)', min: -300, max: 300, step: 5, value: st.lz, onInput: (v) => { st.lz = v; draw(); } });
    TT.ui.check(B, 'A state is not finite, or the stall cut-off binds (dynamics failure)', st.dyn, (v) => { st.dyn = v; draw(); });
    TT.ui.check(B, 'The aircraft is below the threshold elevation before the threshold (ground contact)', st.gnd, (v) => { st.gnd = v; draw(); });
    TT.ui.check(B, 'Instead: a lined-up crossing of ANOTHER candidate runway (G false)', st.other, (v) => { st.other = v; draw(); });
    const g2 = E('div', { class: 'grid2' }); body.append(g2); const L = E('div'), R = E('div'); g2.append(L, R);
    const cone = TT.chart(L, { w: 370, h: 230, x: [0, 6000], y: [-400, 400], xl: 'distance before the threshold (m)', yl: 'offset (m)', yt: [-300, -150, 0, 150, 300], label: 'FAS cone' });
    const vert = TT.chart(R, { w: 370, h: 230, x: [0, 6000], y: [-60, 60], xl: 'distance before the threshold (m)', yl: 'above glidepath (m)', yt: [-44, -22, 0, 22, 44], label: 'Vertical window at the DA point' });
    const out = TT.ui.out(body);
    const len = c23.length, dfpap = Math.max(len, 9023 * 0.3048), dgarp = dfpap + 1000 * 0.3048, cw = Math.max(350 * 0.3048, Math.tan(TT.rad(1.5)) * dgarp);
    const half = (d) => cw * (d + dgarp) / dgarp;
    const dDA = (c23.da - c23.tch) / Math.tan(TT.rad(c23.gp));
    cone.area([...TT.range(0, 6000, 25).map(d => [d, half(d)]), ...TT.range(6000, 0, 25).map(d => [d, -half(d)])], 'band', cone.layer);
    cone.line(TT.range(0, 6000, 25).map(d => [d, half(d)]), 'flown', cone.layer); cone.line(TT.range(0, 6000, 25).map(d => [d, -half(d)]), 'flown', cone.layer);
    vert.rect(0, -22, 6000, 22, 'band', vert.layer);
    function draw() {
      [cone, vert].forEach(c => c.clear());
      const half0 = half(dDA);
      cone.vline(dDA, 'faint'); cone.dot(dDA, st.lz, 5, Math.abs(st.lz) <= half0 ? 'f-ok' : 'f-block'); cone.text(dDA, 380, 'DA point', 'lbl sm', 'start', null, 4, 4);
      vert.vline(dDA, 'faint'); vert.dot(dDA, st.vz, 5, Math.abs(st.vz) <= 22 ? 'f-ok' : 'f-block');
      const lined = st.off <= SP().lateralTol * 0 + 30, inScreen = Math.abs(st.lat) <= 1000;
      const approach = st.cross && !st.G && lined && inScreen && !st.other;
      const daPass = st.da && Math.abs(st.lz) <= half0 && Math.abs(st.vz) <= 22;
      let outcome, why;
      if (st.dyn) { outcome = 'dynamics_failure'; why = 'a state is not finite, or the stall cut-off bound'; }
      else if (st.gnd) { outcome = 'ground_contact'; why = 'below the threshold elevation before the threshold'; }
      else if (approach && st.h > 100) { outcome = 'crossed_too_high'; why = 'an approach crossing higher than 100 m above the threshold'; }
      else if (approach && Math.abs(st.lat) > 106.7) { outcome = 'crossed_off_runway'; why = `an approach crossing at ≤ 100 m, outside the runway limit (the FAS half-width at the threshold, 106.7 m)`; }
      else if (approach) { outcome = daPass ? 'landed' : 'unstable_at_minimums'; why = daPass ? 'an approach crossing at ≤ 100 m inside the runway limit, after a DA check that passed' : (st.da ? 'the DA check failed' : 'there is no DA point: the aircraft crossed above the DA'); }
      else if (st.other && !st.G) { outcome = 'crossed_other_runway'; why = 'G is false and the plane of another candidate was crossed lined up, inside that runway’s own limit'; }
      else { outcome = 'timeout'; why = st.cross && (st.G || !lined || !inScreen) ? 'the crossing is no event (' + (st.G ? 'G is true' : !lined ? 'not lined up: the track is more than 30° from the course' : 'outside the landing screen') + '); the flight flies on to its time limit' : 'no event before the time limit'; }
      out(`DA point at ${dDA.toFixed(0)} m before the threshold (DA ${c23.da.toFixed(0)} m, TCH ${c23.tch.toFixed(1)} m, glidepath ${c23.gp}°; KRDU 23R). FAS half-width there: <b>${half0.toFixed(0)} m</b>.\n<b>Outcome: <span class="${outcome === 'landed' ? 'ok-t' : outcome === 'timeout' ? 'word-t' : 'bad-t'}">${outcome}</span></b>\nwhy: ${why}`);
    }
    draw();
    body.append(E('div', { class: 'cap', html: 'The judge gives the <b>first</b> event, in the order of the table in <span class="pill d">§5.8</span>. The DA check has no parameter: ±22 m of the published glidepath (ICAO Doc 9613) and the FAS cone of FAA Order 8260.58D, the same numbers as the evaluation module (<span class="pill d">D38</span>). A crossing that is not lined up, or one while G is true, is not an event.' }));
  };

  // ============================================================ observed path generator for the closed-loop demos
  // a pilot who flies a base leg east, turns right onto a final to the south (course 180°) and tracks the centreline
  const observedPath = (o, turnE) => {
    const rnd = TT.mulberry(o.seed), dt = 2, V = 80, course = 180, rows = [];
    let e = -12000, n = 11000, hd = 90, turning = false, tTurn = 0, wob = 0;
    for (let t = 0; t < 600 && n > -300; t += dt) {
      rows.push({ t, e, n, track: hd });
      if (!turning && e >= turnE) { turning = true; tTurn = t; }
      let cmd = 90;
      if (turning) { cmd = Math.min(180 - o.intercept, 90 + o.rate * (t - tTurn)); if (cmd >= 180 - o.intercept - 1e-6) { wob = wob * 0.95 + (rnd() - 0.5) * o.wobble; cmd = 180 + TT.clamp(0.04 * e, -o.intercept, o.intercept) + wob; } }
      const dh = TT.clamp(cmd - hd, -o.rate * dt, o.rate * dt) * (o.lag > 0 ? Math.min(1, dt / o.lag) : 1);
      hd += dh; e += V * Math.sin(TT.rad(hd)) * dt; n += V * Math.cos(TT.rad(hd)) * dt;
    }
    return { rows, course, V };
  };
  const makeObserved = (o) => { // choose the turn point that puts the aircraft on the centreline 6.5 km before the threshold
    let best = null;
    for (let turnE = -9500; turnE <= 0; turnE += 100) { const p = observedPath(o, turnE); const q = p.rows.find(r => r.n <= 6500); const err = q ? Math.abs(q.e) : 1e9; if (!best || err < best.err) best = { err, p }; }
    return best.p;
  };
  // the matched point: nearest point of the observed polyline, searched forward from the last one (vocabulary §4.9)
  const matchPoint = (rows, from, p) => {
    let best = { d: Infinity, i: from, f: 0 };
    for (let i = from; i < rows.length - 1; i++) {
      const a = rows[i], b = rows[i + 1], dx = b.e - a.e, dy = b.n - a.n, L2 = dx * dx + dy * dy;
      let f = L2 ? ((p.e - a.e) * dx + (p.n - a.n) * dy) / L2 : 0; f = TT.clamp(f, 0, 1);
      const qx = a.e + f * dx, qy = a.n + f * dy, d = Math.hypot(p.e - qx, p.n - qy);
      if (d < best.d) best = { d, i, f, qx, qy, dx, dy };
      else if (d > best.d + 400 && i > best.i + 3) break;
    }
    const L = Math.hypot(best.dx, best.dy) || 1, right = [best.dy / L, -best.dx / L];
    best.ey = (p.e - best.qx) * right[0] + (p.n - best.qy) * right[1];
    best.row = best.i + best.f; return best;
  };

  // the closed-loop reading of the lateral column, with the executor sketch (autopilot/closed_loop.py, vocabulary §4.9)
  TT.closedLoopSim = (st) => {
    const ob = makeObserved(st), rows = ob.rows, course = ob.course, V = ob.V;
    const words = TT.perStepWords(rows.map(r => r.track), rows.map(() => course), 5, 2);   // {row, rel}
    const Dt = 4, t0row = 8;
    const sim = new TT.Sim({ e: rows[t0row].e, n: rows[t0row].n, track: rows[t0row].track, V, h: 0, gamma: 0 }, { level0: 0 });
    const flown = [{ e: sim.st.e, n: sim.st.n, t: 0 }], errs = [], corrs = [];
    let last = t0row, nextWord = 0, turn = 0, rel = 0, heldIdx = -1;
    while (nextWord < words.length && words[nextWord][0] <= t0row) { rel = words[nextWord][1]; heldIdx = nextWord; nextWord++; }
    sim.say({ heading: { rel, course } }); let target = course + rel, savedKey = [heldIdx, 0];
    for (let j = 1; j < 200; j++) {
      for (let q = 0; q < Dt; q++) { sim.cycle(); if (q % 2 === 1) flown.push({ e: sim.st.e, n: sim.st.n, t: (j - 1) * Dt + q + 1 }); }
      if (sim.st.n < 0) break;
      const m = matchPoint(rows, last, { e: sim.st.e, n: sim.st.n }); last = Math.max(last, m.i);
      errs.push([j * Dt, m.ey]);
      // the observed words that the matched point has reached: the Δ row nearest to the place where they were heard (D45)
      let fresh = false; while (nextWord < words.length && words[nextWord][0] - m.row < Dt / 2 / 2) { rel = words[nextWord][1]; heldIdx = nextWord; nextWord++; fresh = true; }
      if (st.on) {
        if (fresh) turn = 0;
        else if (turn && (Math.abs(m.ey) < st.Y / 2 || Math.sign(m.ey) !== turn)) turn = Math.abs(m.ey) > st.Y ? Math.sign(m.ey) : 0;   // D53: an overshoot says the opposite correction
        else if (!turn && Math.abs(m.ey) > st.Y) turn = Math.sign(m.ey);
      }
      const key = [heldIdx, turn];
      if (key[0] !== savedKey[0] || key[1] !== savedKey[1]) {
        savedKey = key; const word = rel - turn * 5;                // right of the path (turn +1): one class to the left
        if (Math.abs(TT.wrap180(course + word - target)) > 1e-9) { sim.say({ heading: { rel: word, course } }); target = course + word; if (turn) corrs.push([j * Dt, m.ey, word]); }
      }
    }
    return { rows, words, flown, errs, corrs, course };
  };

  // ============================================================ the closed-loop reading, simulated
  TT.demos.closed_loop_sim = (root, body) => {
    const st = { rate: 2.5, lag: 4, intercept: 25, wobble: 0.6, Y: 30, on: true, seed: 5 };
    const r = TT.ui.row(body);
    TT.ui.slider(r, { label: 'Observed turn rate (°/s)', min: 1, max: 4, step: 0.25, value: st.rate, onInput: (v) => { st.rate = v; draw(); } });
    TT.ui.slider(r, { label: 'Pilot lag (s)', min: 0, max: 10, step: 1, value: st.lag, onInput: (v) => { st.lag = v; draw(); } });
    TT.ui.slider(r, { label: 'Intercept angle (deg)', min: 10, max: 45, step: 5, value: st.intercept, onInput: (v) => { st.intercept = v; draw(); } });
    const r2 = TT.ui.row(body);
    TT.ui.slider(r2, { label: 'Final wander (deg)', min: 0, max: 2, step: 0.1, value: st.wobble, onInput: (v) => { st.wobble = v; draw(); } });
    TT.ui.slider(r2, { label: 'Tolerance Y (m)', min: 10, max: 80, step: 5, value: st.Y, onInput: (v) => { st.Y = v; draw(); } });
    TT.ui.check(r2, 'Correction words ON (the closed-loop reading)', st.on, (v) => { st.on = v; draw(); });
    TT.ui.button(r2, 'New wander', () => { st.seed++; draw(); });
    const g = E('div', { class: 'grid2' }); body.append(g); const L = E('div'), R = E('div'); g.append(L, R);
    const plan = TT.chart(L, { w: 380, h: 330, x: [-12500, 2500], y: [-400, 11500], xl: 'East (m)', yl: 'North (m)', xt: [-12000, -8000, -4000, 0], yt: [0, 4000, 8000], label: 'Plan view: observed and flown path' });
    const ey = TT.chart(R, { w: 380, h: 330, x: [0, 400], y: [-250, 250], xl: 'time since the first predicted step (s)', yl: 'lateral error e_y (m)', yt: [-200, -100, 0, 100, 200], label: 'Lateral error' });
    TT.ui.legend(body, [['obs', 'observed path'], ['flown', 'flown path'], ['corr', 'correction word'], ['word', 'observed heading word']]);
    const out = TT.ui.out(body);
    function draw() {
      [plan, ey].forEach(c => c.clear());
      const { rows, words, flown, errs, corrs } = TT.closedLoopSim(st);
      plan.line(rows.map(r => [r.e, r.n]), 'obs'); plan.line(flown.map(f => [f.e, f.n]), 'flown');
      plan.line([[0, 0], [0, 11500]], 'ink dash thin'); plan.dot(0, 0, 4, 'f-ink');
      words.forEach(([row]) => plan.dot(rows[row].e, rows[row].n, 2.5, 'f-word'));
      ey.hline(st.Y, 'faint dash'); ey.hline(-st.Y, 'faint dash'); ey.hline(0, 'faint');
      ey.area([[0, -st.Y], [400, -st.Y], [400, st.Y], [0, st.Y]], 'band');
      ey.line(errs, 'flown'); corrs.forEach(([t, e]) => ey.dot(t, TT.clamp(e, -250, 250), 3.5, 'f-corr'));
      const lastE = errs[errs.length - 1][1], maxE = Math.max(...errs.map(e => Math.abs(e[1])));
      out(`Observed path: ${rows.length} rows, ${words.length} heading words. Flown path: ${flown.length} rows.\nLargest |e_y| <b class="${maxE > 106.7 ? 'bad-t' : 'ok-t'}">${maxE.toFixed(0)} m</b>. Error at the last row: <b class="${Math.abs(lastE) > 106.7 ? 'bad-t' : 'ok-t'}">${lastE.toFixed(0)} m</b>. The runway limit at the threshold is 106.7 m.\n<b>${corrs.length}</b> correction words said while a correction was in force.${st.on ? '' : '\n<span class="bad-t">Corrections are OFF: only the observed words fly the aircraft.</span>'}`);
    }
    draw();
    body.append(E('div', { class: 'cap', html: 'A <em>sketch</em> of <code>autopilot/closed_loop.py</code>: the observed path comes from a pilot model (turn at a rate, a lag, a capture of the centreline with a small wander). Its words come from the heading reading. The executor sketch flies them. At each Δ = 4 s row the matched point and the lateral error e_y decide the correction: one 5° class toward the path when |e_y| &gt; Y (<span class="pill d">D32</span>), and the end of the correction when |e_y| &lt; Y/2 or the sign changes (<span class="pill d">D53</span>). The vertical column is left out here.' }));
  };

  // ============================================================ the matched point
  TT.demos.matched_point = (root, body) => {
    // an observed path: straight, an arc of 90°, straight (arc length parameter, metres)
    const R0 = 3000, A0 = 6000, S0 = 6000;
    const total = A0 + R0 * Math.PI / 2 + S0, at = (s) => {
      if (s <= A0) return { e: -A0 + s, n: 0, h: 90 };
      const a = (s - A0) / R0; if (a <= Math.PI / 2) return { e: 0 + R0 * Math.sin(a) - 0, n: -R0 * (1 - Math.cos(a)), h: 90 + TT.deg(a) };
      return { e: R0, n: -R0 - (s - A0 - R0 * Math.PI / 2), h: 180 };
    };
    const rows = TT.range(0, total, 240).map((s) => ({ ...at(s), s }));
    const st = { time: 90, p: { e: -3000, n: 800 }, last: 0 };
    const r = TT.ui.row(body);
    TT.ui.slider(r, { label: 'Time on the observed clock (s)', min: 0, max: 200, step: 1, value: st.time, onInput: (v) => { st.time = v; draw(); } });
    TT.ui.button(r, 'Reset the search', () => { st.last = 0; draw(); });
    const c = TT.chart(body, { w: 760, h: 380, x: [-7000, 4500], y: [-9500, 2200], xl: 'East (m)', yl: 'North (m)', xt: [-6000, -3000, 0, 3000], yt: [-8000, -4000, 0], label: 'Observed path, matched point and flown position' });
    TT.ui.legend(body, [['obs', 'observed path'], ['flown', 'flown aircraft (drag it)'], ['corr', 'matched point'], ['word', 'observed aircraft at the same time', 'sq']]);
    const out = TT.ui.out(body);
    c.line(rows.map(q => [q.e, q.n]), 'obs', c.layer);
    const word = at(A0 + 300); c.dot(word.e, word.n, 5, 'f-word', c.layer); c.text(word.e, word.n, 'a turn word is said here', 'lbl w', 'end', c.layer, -9, -8);
    const ball = TT.svg('circle', { class: 'f-flown ring', r: 9, cx: 0, cy: 0 }); c.top.append(ball);
    TT.drag(c, ball, (x, y) => { st.p = { e: x, n: y }; draw(); });
    function draw() {
      c.clear(); const p = st.p;
      ball.setAttribute('cx', c.X(p.e)); ball.setAttribute('cy', c.Y(p.n));
      const m = matchPoint(rows, st.last, p); st.last = Math.max(st.last, m.i);
      const sMatched = rows[m.i].s + m.f * (rows[Math.min(rows.length - 1, m.i + 1)].s - rows[m.i].s);
      const V = 80, sObs = Math.min(total, st.time * V), obs = at(sObs);
      c.line([[p.e, p.n], [m.qx, m.qy]], 'corr thin'); c.dot(m.qx, m.qy, 5, 'f-corr');
      c.dot(obs.e, obs.n, 5, 'f-word'); c.text(obs.e, obs.n, 'observed aircraft at ' + st.time + ' s', 'lbl w sm', 'start', null, 8, -8);
      const wordReached = sMatched >= A0 + 300, timeReached = sObs >= A0 + 300;
      out(`Lateral error e_y = <b class="corr-t">${m.ey.toFixed(0)} m</b> (${m.ey > 0 ? 'right' : 'left'} of the observed path).\nAlong the path: the matched point is at ${sMatched.toFixed(0)} m, the observed aircraft at ${sObs.toFixed(0)} m. The difference, <b>${(sMatched - sObs).toFixed(0)} m</b>, is a difference of time. It is <u>not</u> corrected.\nThe turn word is said ${wordReached ? '<b class="ok-t">now</b>' : 'later'} by <b>place</b> (the matched point has ${wordReached ? '' : 'not '}reached it)  —  by <b>time</b> it would be said ${timeReached ? 'now' : 'later'}.`);
    }
    draw();
    body.append(E('div', { class: 'cap', html: 'The reference of the closed-loop reading is the <b>matched point</b>: the point of the observed path nearest to the flown position, searched forward from the last one. A turn word is said where the observed aircraft turned, not when it turned (<span class="pill d">D42</span>): an aircraft that is <em>d</em> ahead along the path before a 90° turn is <em>d</em> to the side after it. Drag the blue dot.' }));
  };

  // ============================================================ real corrections
  TT.demos.real_corrections = (root, body) => {
    let name = 'vectored'; const st = { head: true, ang: true };
    const r = TT.ui.row(body);
    TT.ui.seg(r, [['vectored', 'Vectored approach'], ['go_around', 'Approach with a go-around']], name, (v) => { name = v; draw(); });
    TT.ui.check(r, 'Heading corrections', true, (v) => { st.head = v; draw(); }); TT.ui.check(r, 'Angle corrections', true, (v) => { st.ang = v; draw(); });
    const holder = E('div'); body.append(holder);
    const out = TT.ui.out(body);
    TT.ui.legend(body, [['flown', 'error of the flown path'], ['corr', 'a correction word is said', 'sq'], ['obs', 'row with no correction possible', 'sq']]);
    function draw() {
      holder.replaceChildren(); const fl = D().flights[name], M = fl.words.grid.length, dt = fl.delta_s, t0 = fl.start_row * dt;
      const T = [...Array(M).keys()].map(k => t0 + k * dt);
      const c1 = TT.chart(holder, { w: 760, h: 230, x: [t0, T[M - 1]], y: [-150, 150], xl: 'time since the first row (s)', yl: 'lateral error e_y (m)', yt: [-100, -30, 0, 30, 100], label: 'Lateral error with the tolerance band' });
      const c2 = TT.chart(holder, { w: 760, h: 230, x: [t0, T[M - 1]], y: [-60, 60], xl: 'time since the first row (s)', yl: 'vertical error e_h (m)', yt: [-50, -15, 0, 15, 50], label: 'Vertical error with the tolerance band' });
      c1.area([[t0, -30], [T[M - 1], -30], [T[M - 1], 30], [t0, 30]], 'band', c1.layer); c1.hline(0, 'faint', c1.layer);
      // vertical tolerance in force: 15 m, 10 m once "no level-off" is in force
      let nlo = false; const tol = []; fl.words.text.forEach((row, k) => { if (row[2] === 'no level-off') nlo = true; if (row[2] && row[2] !== 'no level-off') nlo = false; tol.push(nlo ? 10 : 15); });
      c2.area([...T.map((t, k) => [t, tol[k]]), ...T.map((t, k) => [t, -tol[k]]).reverse()], 'band', c2.layer); c2.hline(0, 'faint', c2.layer);
      c1.line(T.map((t, k) => [t, fl.lateral_m[k] == null ? NaN : TT.clamp(fl.lateral_m[k], -150, 150)]), 'flown'); c2.line(T.map((t, k) => [t, fl.vertical_m[k] == null ? NaN : TT.clamp(fl.vertical_m[k], -60, 60)]), 'flown');
      let nh = 0, na = 0;
      T.forEach((t, k) => { const ch = fl.words.correction[k][1], ca = fl.words.correction[k][3];
        if (st.head && ch) { nh++; c1.dot(t, TT.clamp(fl.lateral_m[k] ?? 0, -150, 150), 3.5, 'f-corr'); }
        if (st.ang && ca) { na++; c2.dot(t, TT.clamp(fl.vertical_m[k] ?? 0, -60, 60), 3.5, 'f-corr'); }
        if (fl.uncorrectable[k][0]) c1.rect(t - 0.6, -150, t + 0.6, -142, 'f-obs'); if (fl.uncorrectable[k][1]) c2.rect(t - 0.6, -60, t + 0.6, -55, 'f-obs'); });
      const wh = fl.words.text.filter(r => r[1]).length, wa = fl.words.text.filter(r => r[3]).length;
      out(`Real flight: ${fl.airport} ${fl.runway}, ${M} rows at Δ = ${dt} s, outcome <b>${fl.outcome}</b>.\n<b>${nh}</b> heading correction words among ${wh} heading words; <b>${na}</b> angle correction words among ${wa} angle words.\nOn this artefact (train, Δ = 4 s) the share of correction words is ${(100 * D().train_counts['4'].corrections[1] / D().train_counts['4'].words[1]).toFixed(0)} % of the heading words and ${(100 * D().train_counts['4'].corrections[3] / D().train_counts['4'].words[3]).toFixed(0)} % of the angle words.`);
    }
    draw();
    body.append(E('div', { class: 'cap', html: 'The orange dots are the rows where the closed-loop reading added a word because |e_y| &gt; 30 m or |e_h| &gt; 15 m (10 m in the final descent, <span class="pill d">D66</span>). A correction is one class and needs time to work, so the error can stay outside the band for some rows (<span class="pill d">D50</span>). The small gray marks at the bottom of the charts are rows where no correction is possible (a new observed word, a level hold, the end of the observed path).' }));
  };
})();
