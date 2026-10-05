/* Demos for the prior (prior.md): inputs, candidate tokens, RoPE, the speaker with its masks, the procedure masks, the choice of design. */
(function () {
  const TT = window.TT, E = TT.el, S = TT.svg, W = () => TT.W, D = () => TT.data;

  // ---------- the procedure masks (prior/procedure.py, D64): shared by two demos ----------
  const RE = 6.371e6;
  TT.finalOf = (cand, airportE) => ({ ...cand, thr: cand.elev - airportE, faf: cand.faf_m });
  TT.glidepath = (f, d) => f.thr + f.tch + d * Math.tan(TT.rad(f.gp)) + d * d / (2 * RE);
  TT.coneHalf = (f, d) => { const dfpap = Math.max(f.length, 9023 * 0.3048), dgarp = dfpap + 1000 * 0.3048, cw = Math.max(350 * 0.3048, Math.tan(TT.rad(1.5)) * dgarp); return cw * (d + dgarp) / dgarp; };
  TT.procMask = (Wd, f, s) => { // s: {d, off, h, joinedBefore, dipped, G}
    const inside = s.d >= 0 && s.d <= f.faf && Math.abs(s.off) <= TT.coneHalf(f, s.d);
    const joined = s.joinedBefore || inside, barred = s.dipped && !joined && !s.G;
    const edge = inside ? TT.glidepath(f, s.d) - 60 : NaN, decision = f.thr + f.da;
    const ok = Wd.levels.map(() => true).concat([true]);
    Wd.levels.forEach((L, i) => { const t = Wd.tol[i]; if (inside) ok[i] = L >= edge - t; else ok[i] = L >= decision - t; if (barred && L > s.h + t) ok[i] = false; });
    if (inside) ok[Wd.noLevelOff] = s.h >= edge - Wd.tol[0];
    return { ok, inside, joined, barred, edge, decision, entry: TT.glidepath(f, f.faf), climbBlocked: barred };
  };

  // ============================================================ what the prior may see
  TT.demos.inputs_quiz = (root, body) => {
    const items = [
      ['The height of the aircraft above the airport elevation E', true, 'It is the own state. The level words are above E too, so the model sees the quantity that it must say.', 'D58'],
      ['The height of the aircraft above sea level (MSL)', false, 'With it, the airport elevation (a constant of the airport) would be an input.', 'D24, D58'],
      ['The position (E, N) of the aircraft in the airport frame', false, 'An absolute position is a code of the airport. The prior has no absolute position and no absolute direction.', 'D5'],
      ['The distance to the threshold and the offset from the final, in the frame of each candidate runway', true, 'Every position and direction is given in the frame of a candidate. The same numbers mean the same thing at every airport.', 'D23'],
      ['An embedding of the airport code (one vector for each airport)', false, 'The embeddings are an identity table. A new airport has none. Evidence: a replacement of the embedding raised the loss and lowered the landed share (R46).', 'D5, §10.1'],
      ['The length and the elevation of each candidate runway', false, 'A pair (length, elevation) is different for 24 of the 25 candidates, so it names the runway. It comes back only in the variant "constants", which cross-validation can choose.', 'D24, D39'],
      ['The number of landings on each candidate in the last 30 minutes, without the flight’s own landing', true, 'A controller knows which runways are in use. The own landing is the answer of the runway word, so it never counts.', 'D63'],
      ['The time since the aircraft entered the 25 km slice (time from row 0)', false, 'It measures a cut of the data, not a fact of the flight. A go-around adds up to 900 s that training seldom saw. RoPE gives the order of the rows.', 'D16'],
      ['The time since each column said its word in force', true, 'It is a real fact of the flight: how long a heading, level or speed word has stood. It is in seconds, so every Δ reads the same.', 'D17'],
      ['The capture row of the labeller', false, 'It uses later rows (the track must stay in the corridor to the end). Only the labeller reads it.', 'principle 7'],
      ['The ground speed and vertical rate stored in the signals for this row', false, 'They are a least-squares fit over 15 s centred on the row. They use 7.5 s of the future. The prior takes the displacement in the 2 s before the row.', 'D25, D60'],
      ['The runway on which the flight landed, as an input at the rows before the first predicted step', false, 'It is the answer of the first predicted step. In closed loop the same input does not exist.', 'D23'],
      ['The height of the aircraft above the published glidepath of each candidate', true, 'It is procedure geometry as an input; the model still decides the profile. The value exists at every row, also before R is said.', 'D13'],
      ['The outcome of the sentence ("landed") as an input', false, 'The outcome is for readouts and for the selection of training sentences. It uses later rows.', 'D74, D75'],
    ];
    let score = 0, done = 0;
    const head = E('div', { class: 'row' }); body.append(head);
    const stat = E('span', { class: 'pill' }); head.append(stat); TT.ui.button(head, 'Reset', () => { score = 0; done = 0; list.querySelectorAll('.card').forEach(c => { c.classList.remove('done'); c.querySelector('.ans').innerHTML = ''; c.querySelectorAll('button').forEach(b => { b.disabled = false; b.classList.remove('on'); }); }); upd(); });
    const upd = () => { stat.textContent = `Score ${score} / ${done} of ${items.length}`; }; upd();
    const list = E('div'); body.append(list);
    items.forEach(([q, allowed, why, d]) => {
      const card = E('div', { class: 'card', style: 'margin:8px 0' }); const ans = E('div', { class: 'ans', style: 'margin-top:6px;font-size:.9rem' });
      const row = E('div', { class: 'row', style: 'margin:4px 0 0' });
      [['Allowed input', true], ['Not allowed', false]].forEach(([t, v]) => row.append(E('button', { type: 'button', text: t, onclick: (ev) => { card.querySelectorAll('button').forEach(b => { b.disabled = true; }); ev.target.classList.add('on'); done++; const right = v === allowed; if (right) score++; ans.innerHTML = `<span class="${right ? 'ok-t' : 'bad-t'}">${right ? '✓ Correct.' : '✗ Not correct.'}</span> The prior ${allowed ? 'may' : 'must not'} see this. ${why} <span class="pill d">${d}</span>`; upd(); } })));
      card.append(E('div', { text: q }), row, ans); list.append(card);
    });
    body.append(E('div', { class: 'cap', html: 'Principle 4: airport facts are inputs, not words and not identities. Principle 7: inputs give only what a controller knows before the step.' }));
  };

  // ============================================================ candidate tokens and the pool
  TT.demos.candidate_pool = (root, body) => {
    const A = D().airports.KRDU, cands = A.candidates.map(c => ({ ...c })); const on = cands.map(() => true);
    const landings = { '05L': 0, '05R': 1, '23L': 4, '23R': 6, '32': 0 };
    const t23 = cands.find(c => c.ident === '23R'); const cr = TT.rad(t23.course);
    const st = { p: { e: t23.e - 9500 * Math.sin(cr) + 700 * Math.cos(cr), n: t23.n - 9500 * Math.cos(cr) - 700 * Math.sin(cr) }, h: 650, track: 225 };
    const r = TT.ui.row(body);
    TT.ui.slider(r, { label: 'Height above E (m)', min: 0, max: 2500, step: 10, value: st.h, onInput: (v) => { st.h = v; draw(); } });
    TT.ui.slider(r, { label: 'Track (deg)', min: 0, max: 359, step: 1, value: st.track, onInput: (v) => { st.track = v; draw(); } });
    const r2 = TT.ui.row(body); r2.append(E('span', { class: 'src', text: 'Use these candidates:' }));
    cands.forEach((c, i) => TT.ui.check(r2, c.ident, true, (v) => { on[i] = v; draw(); }));
    const c = TT.chart(body, { w: 760, h: 330, x: [-16, 10], y: [-6, 14], xl: 'East (km)', yl: 'North (km)', xt: [-15, -10, -5, 0, 5, 10], yt: [-5, 0, 5, 10], label: 'KRDU candidates and the aircraft' });
    const tab = E('div'); body.append(tab);
    cands.forEach((q) => { const a = TT.rad(q.course); c.line([[(q.e - 20000 * Math.sin(a)) / 1000, (q.n - 20000 * Math.cos(a)) / 1000], [q.e / 1000, q.n / 1000]], 'faint', c.layer); c.dot(q.e / 1000, q.n / 1000, 3, 'f-ink', c.layer); });
    const ball = S('circle', { class: 'f-flown ring', r: 9 }); c.top.append(ball); TT.drag(c, ball, (x, y) => { st.p = { e: x * 1000, n: y * 1000 }; draw(); });
    function draw() {
      c.clear(); ball.setAttribute('cx', c.X(st.p.e / 1000)); ball.setAttribute('cy', c.Y(st.p.n / 1000));
      cands.forEach((q, i) => { if (!on[i]) return; c.text(q.e / 1000, q.n / 1000, q.ident, 'lbl sm', 'start', null, 6, 12); });
      const rows = cands.map((q) => { const f = TT.finalOf(q, A.elevation_m), a = TT.rad(q.course), de = st.p.e - q.e, dn = st.p.n - q.n; const before = -(de * Math.sin(a) + dn * Math.cos(a)), right = de * Math.cos(a) - dn * Math.sin(a); const hThr = st.h - f.thr; const gp = before > 0 ? TT.glidepath(f, before) - f.thr : NaN; const above = hThr - gp; const mc = TT.rad(st.track - q.course); return { q, before, right, hThr, above, sinc: Math.sin(mc), cosc: Math.cos(mc) }; });
      const score = rows.map((r2, i) => on[i] ? -Math.pow(r2.right / 1500, 2) - (r2.before < 0 ? 2 : 0) - Math.abs(r2.before - 9000) / 20000 : -Infinity);
      const mx = Math.max(...score), ex = score.map(s => Math.exp(s - mx)), sum = ex.reduce((a, b) => a + b, 0), wts = ex.map(x => x / sum);
      tab.innerHTML = `<table><thead><tr><th>candidate</th><th class="num">before threshold</th><th class="num">right of final</th><th class="num">above its glidepath</th><th class="num">asinh(h/100 m)</th><th class="num">sin / cos(track − course)</th><th class="num">landings, 30 min*</th><th>pool weight</th></tr></thead><tbody>` + rows.map((r2, i) => `<tr style="opacity:${on[i] ? 1 : .35}"><td><b>${r2.q.ident}</b></td><td class="num">${(r2.before / 1000).toFixed(1)} km</td><td class="num">${r2.right.toFixed(0)} m</td><td class="num">${Number.isFinite(r2.above) ? r2.above.toFixed(0) + ' m' : '–'}</td><td class="num">${Number.isFinite(r2.above) ? Math.asinh(r2.above / 100).toFixed(2) : '–'}</td><td class="num">${r2.sinc.toFixed(2)} / ${r2.cosc.toFixed(2)}</td><td class="num">${landings[r2.q.ident]}</td><td><span style="display:inline-block;height:10px;width:${(wts[i] * 120).toFixed(0)}px;background:var(--c-flown);border-radius:3px"></span> ${(wts[i] * 100).toFixed(0)} %</td></tr>`).join('') + `</tbody></table><div class="src">* example counts. The pool weights are <b>toy scores</b> (nearer to a final → more weight) to show the mechanism: they always sum to 1, whatever the number of candidates.</div>`;
    }
    draw();
    body.append(E('div', { class: 'cap', html: 'Each candidate becomes one vector from <b>one shared network</b>. The values are in the frame of that candidate (<span class="pill d">D23</span>) and have a fixed physical scale (<span class="pill d">D41</span>). A row reads the vectors through one attention whose weights sum to 1, so the model works for any number of runways. Drag the aircraft; switch a candidate off.' }));
  };

  // ============================================================ RoPE
  TT.demos.rope = (root, body) => {
    const st = { delta: 20, shift: 0 };
    const rnd = TT.mulberry(11), pairs = 16, qk = Array.from({ length: pairs }, () => ({ q: [rnd() * 2 - 1, rnd() * 2 - 1], k: [rnd() * 2 - 1, rnd() * 2 - 1] }));
    const omega = Array.from({ length: pairs }, (_, j) => Math.pow(10000, -j / pairs));
    const score = (tq, tk) => qk.reduce((s, p, j) => { const aq = omega[j] * tq, ak = omega[j] * tk; const q = [p.q[0] * Math.cos(aq) - p.q[1] * Math.sin(aq), p.q[0] * Math.sin(aq) + p.q[1] * Math.cos(aq)], k = [p.k[0] * Math.cos(ak) - p.k[1] * Math.sin(ak), p.k[0] * Math.sin(ak) + p.k[1] * Math.cos(ak)]; return s + q[0] * k[0] + q[1] * k[1]; }, 0);
    const r = TT.ui.row(body);
    TT.ui.slider(r, { label: 'The key is this many seconds earlier', min: 0, max: 120, step: 1, value: st.delta, onInput: (v) => { st.delta = v; draw(); } });
    TT.ui.select(r, 'Both times shifted by', [['0', '0 s'], ['1000', '1,000 s'], ['100000', '100,000 s'], ['1760000000', '1.76 × 10⁹ s (a UTC time)']], '0', (v) => { st.shift = +v; draw(); });
    const c = TT.chart(body, { w: 760, h: 240, x: [0, 120], y: [-6, 6], xl: 'seconds between the query row and the key row', yl: 'attention score (before softmax)', label: 'RoPE score against the time difference' });
    const out = TT.ui.out(body);
    const phase = E('div', { class: 'row' }); body.append(phase);
    const curve = TT.range(0, 120, 241).map(d => [d, score(d, 0)]);
    c.line(curve, 'obs', c.layer);
    function draw() {
      c.clear(); const tq = st.shift + st.delta, tk = st.shift;
      const a = score(st.delta, 0), b = score(tq, tk);
      c.vline(st.delta, 'flown'); c.dot(st.delta, b, 5, 'f-flown');
      const f32 = (x) => Math.fround(x);
      out(`score from row times (0, ${st.delta}) s       = <b>${a.toFixed(6)}</b>\nscore from row times (${st.shift.toLocaleString('en')}, ${(st.shift + st.delta).toLocaleString('en')}) s = <b>${b.toFixed(6)}</b>   (angles in float64, as in the code)\ndifference: <b class="${Math.abs(a - b) < 1e-6 ? 'ok-t' : 'bad-t'}">${Math.abs(a - b).toExponential(1)}</b>   → the attention reads only time differences.\n\nIf the times were float32: ${st.shift.toLocaleString('en')} + ${st.delta} = ${f32(st.shift + st.delta).toLocaleString('en')}, ${st.shift.toLocaleString('en')} = ${f32(st.shift).toLocaleString('en')}: the difference becomes <b class="${f32(tq) - f32(tk) === st.delta ? 'ok-t' : 'bad-t'}">${(f32(tq) - f32(tk)).toLocaleString('en')} s</b> instead of ${st.delta} s (a float32 near 1.76 × 10⁹ has a step of 128 s).`);
      phase.replaceChildren(E('span', { class: 'src', text: 'Frequencies of the 16 pairs of a head (periods): ' }), ...[0, 4, 8, 12, 15].map(j => E('span', { class: 'chip', text: `pair ${j}: ${(2 * Math.PI / omega[j]).toLocaleString('en', { maximumFractionDigits: 0 })} s` })));
    }
    draw();
    body.append(E('div', { class: 'cap', html: 'The rotation of a row’s query and key uses its time in seconds (<span class="pill d">D16</span>, <code>prior/model.py</code> <code>rope_angles</code>). The base is 10,000; a head is 32 wide, so the periods go from 6.3 s to about 35,000 s. The query and key vectors here are random: the curve shows the mechanism, not a trained model.' }));
  };

  // ============================================================ the speaker: masks and a draw
  TT.demos.speaker = (root, body) => {
    const Wd = W(), A = D().airports.KRDU, cands = A.candidates.map(c => c.ident), nC = cands.length;
        const st = { d: 8000, h: 800, grammar: true, proc: true, caller: false, u: [0.965, 0.9, 0.97, 0.9, 0.93], seed: 1 };
    const force = { runway: 3, go: false, heading: 14, altitude: 13, angle: 2, speed: 5 };   // 23R, +70°, 780 m, descent 2, 45 m/s
    const label = (c, w) => { if (w === TT.UN) return 'unchanged'; if (c === 0) return w === TT.GA ? 'go-around' : cands[w]; if (c === 1) return (Wd.headingRel(w) > 0 ? '+' : '') + Wd.headingRel(w) + '°'; if (c === 2) return w === Wd.noLevelOff ? 'no level-off' : Wd.levels[w] + ' m'; if (c === 3) return Wd.angleName(w); return w === Wd.speedUnspec ? 'unspec.' : Wd.speedMps(w); };
    // the model's probabilities: a made-up distribution that puts some mass on words that a mask removes
    const model = (c) => {
      const words = TT.columnWords(Wd, c, nC); let w;
      if (c === 0) w = words.map(x => x === TT.UN ? 0.86 : x === TT.GA ? 0.05 : ({ 0: .005, 1: .005, 2: .03, 3: .04, 4: .01 })[x]);
      else if (c === 1) { const tail = words.map(x => x === TT.UN ? 0 : Math.exp(-Math.abs(TT.wrap180((x - force.heading) * 5)) / 8)); const s = tail.reduce((a, b) => a + b, 0); w = words.map((x, i) => x === TT.UN ? 0.8 : 0.2 * tail[i] / s); }
      else if (c === 2) { const tail = words.map(x => x === TT.UN ? 0 : x === Wd.noLevelOff ? 0.5 : Math.exp(-Math.abs(Wd.levels[x] - 500) / 260)); const s = tail.reduce((a, b) => a + b, 0); w = words.map((x, i) => x === TT.UN ? 0.84 : 0.16 * tail[i] / s); }
      else if (c === 3) w = words.map(x => ({ [-1]: .83, 0: .04, 1: .03, 2: .02, 3: .05, 4: .01, 5: .02 })[x]);
      else { const tail = words.map(x => x === TT.UN ? 0 : x === Wd.speedUnspec ? 1.2 : Math.exp(-Math.abs(Wd.speedMps(x) - 95) / 14)); const s = tail.reduce((a, b) => a + b, 0); w = words.map((x, i) => x === TT.UN ? 0.8 : 0.2 * tail[i] / s); }
      return { words, p: w };
    };
    const r = TT.ui.row(body);
    TT.ui.check(r, 'Grammar mask', st.grammar, (v) => { st.grammar = v; draw(); }); TT.ui.check(r, 'Procedure masks', st.proc, (v) => { st.proc = v; draw(); }); TT.ui.check(r, 'Mask of the caller: no go-around (the 2nd was said)', st.caller, (v) => { st.caller = v; draw(); });
    const r2 = TT.ui.row(body);
    TT.ui.slider(r2, { label: 'Distance before the threshold (km)', min: 2, max: 16, step: 0.5, value: st.d / 1000, onInput: (v) => { st.d = v * 1000; draw(); } });
    TT.ui.slider(r2, { label: 'Height above E (m)', min: 50, max: 1500, step: 10, value: st.h, onInput: (v) => { st.h = v; draw(); } });
    TT.ui.button(r2, 'Draw new random numbers', () => { const rnd = TT.mulberry(++st.seed * 977); st.u = st.u.map(() => rnd()); draw(); });
    const svg = S('svg', { viewBox: '0 0 760 470', role: 'img', 'aria-label': 'For each column, the probabilities of the model and after the masks, with the draw' }); body.append(svg);
    const out = TT.ui.out(body);
    function draw() {
      svg.replaceChildren(); const said = []; const w0 = 74, wd = 650, height = (colH) => colH;
      const inForce = { runway: force.runway, go: force.go, heading: force.heading, altitude: force.altitude, angle: force.angle, speed: force.speed };
      // the procedure masks are those of the runway in force AFTER the row's runway word, under G after it (prior §4):
      // each candidate has its own region, so the aircraft is put in the frame of that candidate
      const t23 = A.candidates.find(q => q.ident === '23R'), c23 = TT.rad(t23.course);
      const pos = { e: t23.e - st.d * Math.sin(c23), n: t23.n - st.d * Math.cos(c23) };
      const procFor = (runway, G) => { const q = A.candidates[runway], a = TT.rad(q.course), de = pos.e - q.e, dn = pos.n - q.n; return TT.procMask(Wd, TT.finalOf(q, A.elevation_m), { d: -(de * Math.sin(a) + dn * Math.cos(a)), off: de * Math.cos(a) - dn * Math.sin(a), h: st.h, joinedBefore: false, dipped: false, G }); };
      const after = (w) => ({ runway: w >= 0 ? w : force.runway, G: w === TT.GA || (w === TT.UN && force.go) });
      const altMask = (w) => { const r = after(w), pm = procFor(r.runway, r.G); return TT.columnWords(Wd, 2, nC).map(x => x === TT.UN ? true : (st.proc ? pm.ok[x] : true)); };
      const callerRunway = TT.columnWords(Wd, 0, nC).map(x => x === TT.GA ? !st.caller : true);
      const picks = [], info = [];
      for (let c = 0; c < 5; c++) {
        const m = model(c), words = m.words;
        const lookahead = (w) => { const perm = {}; if (c < 2) { const o = altMask(w); perm[2] = TT.columnWords(Wd, 2, nC).filter((_, i) => o[i]); } return perm; };
        let gm;
        if (!st.grammar) gm = words.map(() => true);
        else if (c === 0) gm = words.map((w, i) => TT.columnMask(Wd, inForce, said, 0, st.h, nC, lookahead(w))[i]);   // the later masks depend on the runway word asked
        else gm = TT.columnMask(Wd, inForce, said, c, st.h, nC, lookahead(said[0]));
        const own = c === 0 ? callerRunway : c === 2 ? altMask(said[0]) : words.map(() => true);
        const ok = words.map((x, i) => gm[i] && own[i]);
        const mass = words.reduce((s, x, i) => s + (ok[i] ? m.p[i] : 0), 0);
        const q = words.map((x, i) => ok[i] ? m.p[i] / mass : 0);
        let cum = 0, pick = 0; for (let i = 0; i < q.length; i++) { cum += q[i]; if (cum > st.u[c]) { pick = i; break; } if (i === q.length - 1) pick = i; }
        said.push(words[pick]); picks.push(pick);
        const y0 = 12 + c * 92; svg.append(S('text', { x: 0, y: y0 + 20, class: 'lbl', style: 'font-weight:600;fill:var(--ink)', text: TT.COLUMNS[c] }));
        let x = w0; words.forEach((wd2, i) => { const wdt = m.p[i] * wd; if (wdt <= 0) return; svg.append(S('rect', { x, y: y0, width: Math.max(wdt - 0.6, 0.4), height: 20, class: ok[i] ? 'band' : 'f-block', opacity: ok[i] ? 1 : .55 })); if (wdt > 46) svg.append(S('text', { x: x + 3, y: y0 + 14, class: 'lbl sm', text: label(c, wd2) })); x += wdt; });
        svg.append(S('text', { x: w0 + wd + 4, y: y0 + 14, class: 'lbl sm', text: 'model' }));
        x = w0; words.forEach((wd2, i) => { const wdt = q[i] * wd; if (wdt <= 0) return; svg.append(S('rect', { x, y: y0 + 30, width: Math.max(wdt - 0.6, 0.4), height: 20, class: i === pick ? 'f-word' : 'f-flown', opacity: i === pick ? 1 : .55 })); if (wdt > 46) svg.append(S('text', { x: x + 3, y: y0 + 44, class: 'lbl sm', style: 'fill:#fff', text: label(c, wd2) })); x += wdt; });
        svg.append(S('text', { x: w0 + wd + 4, y: y0 + 44, class: 'lbl sm', text: 'masked' }));
        const ux = w0 + st.u[c] * wd; svg.append(S('path', { d: `M${ux} ${y0 + 54} l-5 8 h10 z`, class: 'f-ink' }), S('text', { x: ux > 450 ? ux - 8 : ux + 8, y: y0 + 64, 'text-anchor': ux > 450 ? 'end' : 'start', class: 'lbl sm', text: `u = ${st.u[c].toFixed(2)} → ${label(c, words[pick])}` }));
        const blockedMass = words.reduce((s, x, i) => s + (!ok[i] ? m.p[i] : 0), 0); info.push(`${TT.COLUMNS[c]}: <b class="word-t">${label(c, words[pick])}</b>  (the masks remove ${(blockedMass * 100).toFixed(1)} % of the model’s mass)`);
        const slider = null;
      }
      const ap = TT.apply(Wd, inForce, said, st.h, nC);
      out(info.join('\n') + `\n<b>The row passes the grammar:</b> ${ap.ok ? '<span class="ok-t">yes</span>' : '<span class="bad-t">no — ' + ap.reason + ': ' + ap.detail + '</span>'}`);
    }
    // five little sliders for u
    const ur = TT.ui.row(body); TT.COLUMNS.forEach((n, c) => TT.ui.slider(ur, { label: 'u ' + n, min: 0, max: 0.999, step: 0.01, value: st.u[c], fmt: (v) => v.toFixed(2), onInput: (v) => { st.u[c] = v; draw(); } }));
    draw();
    body.append(E('div', { class: 'cap', html: 'The speaker says a row column by column (<span class="pill d">§4</span>). In each column it removes the words that the masks block (red), renormalises (blue), and takes the word where the cumulative probability <b>first passes the caller’s random number u</b>, in the class order of the heads (<span class="pill d">D96</span>). The distribution of the model here is made up. The grammar mask is the real one (a port of <code>column_mask</code>); the procedure mask is the rule of <span class="pill d">D64</span> for KRDU 23R. An earlier word changes the later masks: try the runway column and then the altitude column.' }));
  };

  // ============================================================ the procedure masks, side view
  TT.demos.procedure_masks = (root, body) => {
    const Wd = W(), A = D().airports.KRDU, f = TT.finalOf(A.candidates.find(c => c.ident === '23R'), A.elevation_m);
    const st = { d: 8000, h: 450, off: 0, joinedBefore: false, dipped: false, G: false };
    const r = TT.ui.row(body);
    TT.ui.slider(r, { label: 'Distance before the threshold (km)', min: 0.5, max: 16, step: 0.1, value: st.d / 1000, onInput: (v) => { st.d = v * 1000; draw(); } });
    TT.ui.slider(r, { label: 'Height above E (m)', min: 0, max: 1500, step: 5, value: st.h, onInput: (v) => { st.h = v; draw(); } });
    TT.ui.slider(r, { label: 'Offset from the final (m)', min: 0, max: 600, step: 10, value: st.off, onInput: (v) => { st.off = v; draw(); } });
    const r2 = TT.ui.row(body);
    TT.ui.check(r2, 'The aircraft joined the final before (and left the region)', st.joinedBefore, (v) => { st.joinedBefore = v; draw(); });
    TT.ui.check(r2, 'It was below the entry height before the join (a dip)', st.dipped, (v) => { st.dipped = v; draw(); });
    TT.ui.check(r2, 'G is true (a go-around)', st.G, (v) => { st.G = v; draw(); });
    const g = E('div', { style: 'display:grid;grid-template-columns:minmax(0,1fr) 230px;gap:8px' }); body.append(g); const L = E('div'), R = E('div'); g.append(L, R);
    const Y1 = 1500;
    const c = TT.chart(L, { w: 600, h: 360, x: [16, 0], xt: [16, 12, 8, 4, 0], y: [-50, Y1], xl: 'distance before the threshold (km)', yl: 'height above E (m)', label: 'Side view of the final with the three masks' });
    const rul = TT.chart(R, { w: 230, h: 360, x: [0, 1], y: [-50, Y1], xt: [], yt: [], xl: 'altitude words', grid: false, label: 'Altitude words, permitted or blocked' });
    rul.m.l = 6;
    const out = TT.ui.out(body);
    const dd = TT.range(0, 16000, 81);
    c.line(dd.map(d => [d / 1000, TT.glidepath(f, d)]), 'ink', c.layer); c.text(15.9, TT.glidepath(f, 15900) + 12, 'published glidepath', 'lbl sm', 'start', c.top, 0, 0);
    c.line(dd.filter(d => d <= f.faf).map(d => [d / 1000, TT.glidepath(f, d) - 60]), 'block dash', c.layer);
    c.vline(f.faf / 1000, 'faint', c.layer); c.text(f.faf / 1000, Y1 - 60, 'FAF', 'lbl sm', 'start', c.top, 4, 0);
    c.hline(f.thr + f.da, 'ok dash', c.layer); c.text(15.9, f.thr + f.da + 14, 'DA', 'lbl g sm', 'start', c.top, 0, 0);
    function draw() {
      c.clear(); rul.clear();
      const m = TT.procMask(Wd, f, st);
      c.dot(st.d / 1000, st.h, 6, m.barred ? 'f-block' : 'f-flown');
      c.hline(m.entry, 'faint', null);
      c.text(0.15, m.entry, 'entry height', 'lbl sm', 'end', null, 0, -4);
      // ruler of the words
      Wd.levels.forEach((L2, i) => { const y = rul.Y(L2); rul.dyn.append(S('line', { x1: 40, x2: 100, y1: y, y2: y, class: 'ln thin ' + (m.ok[i] ? 'ok' : 'block') })); });
      rul.dyn.append(S('text', { x: 108, y: rul.Y(1420), class: 'lbl sm', text: 'level words' }));
      [0, 500, 1000, 1500].forEach(v => rul.dyn.append(S('text', { x: 36, y: rul.Y(v) + 3, class: 'lbl sm', 'text-anchor': 'end', text: String(v) })));
      rul.dyn.append(S('text', { x: 108, y: rul.Y(1250), class: 'lbl sm ' + (m.ok[Wd.noLevelOff] ? 'g' : 'r'), text: 'no level-off: ' + (m.ok[Wd.noLevelOff] ? 'ok' : 'blocked') }), S('text', { x: 108, y: rul.Y(1130), class: 'lbl sm ' + (m.climbBlocked ? 'r' : 'g'), text: 'climb class: ' + (m.climbBlocked ? 'blocked' : 'ok') }));
      const blocked = Wd.levels.filter((_, i) => !m.ok[i]);
      out(`Inside the region (FAF ${(f.faf / 1000).toFixed(1)} km and the LPV cone): <b>${m.inside ? 'yes' : 'no'}</b>.   Joined: ${m.joined ? 'yes' : 'no'}.\n${m.inside ? `Mask 1 — glidepath lower edge: ${m.edge.toFixed(0)} m. A level T is blocked below edge − ε(T).` : `Mask 2 — the DA: ${m.decision.toFixed(0)} m. A level T is blocked below DA − ε(T) (the aircraft is not inside the region).`}\nMask 3 — no climb before the join, after a dip below the entry height (${m.entry.toFixed(0)} m): <b class="${m.barred ? 'bad-t' : 'ok-t'}">${m.barred ? 'ACTIVE: no level above the aircraft’s height + ε, no climb class' : 'not active'}</b>${st.G ? ' (G is true: this mask never applies)' : ''}.\nBlocked level words: <b>${blocked.length}</b> of ${Wd.levels.length}${blocked.length ? ' (' + blocked.filter((_, i) => i < 3).join(', ') + (blocked.length > 3 ? ', …, ' + blocked[blocked.length - 1] : '') + ' m)' : ''}.`);
    }
    draw();
    body.append(E('div', { class: 'cap', html: 'KRDU 23R: real TCH, glidepath, DA and FAF. A mask blocks a word <b>when it is said</b> and never blocks "unchanged" (<span class="pill d">D64</span>): a mask that forced the aircraft up from a low height would fly the glidepath floor that the executor does not have. Whether to correct or to go around is the model’s decision, and the judge scores it. The model decides the profile (principles 2 and 3).' }));
  };

  // ============================================================ choosing the design (D39, D40)
  TT.demos.cv_rule = (root, body) => {
    const P = D().prior_params, cfg = ['A', 'B', 'C', 'D'], air = ['KMSY', 'KRDU', 'KSJC', 'KSMF', 'KSTL'];
    const scores = { A: [0.255, 0.281, 0.301, 0.262, 0.274], B: [0.262, 0.286, 0.307, 0.268, 0.281], C: [0.251, 0.279, 0.298, 0.261, 0.272], D: [0.258, 0.283, 0.304, 0.264, 0.277] };
    const st = { seed2: 0.2745, constants: 0.2800 };
    body.append(E('div', { class: 'note warn', html: '<b>These numbers are an example, not results.</b> The formal campaign (B5) has not run. Edit the numbers to see how the rule decides. The rule is fixed before the runs (<span class="pill d">D39</span>, <span class="pill d">D40</span>).' }));
    const tbl = E('table'); body.append(tbl);
    const out = TT.ui.out(body);
    const r = TT.ui.row(body);
    TT.ui.button(r, 'Random example', () => { const rnd = TT.mulberry(Math.floor(Math.random() * 1e6)); air.forEach((_, i) => { const base = 0.25 + rnd() * 0.06; cfg.forEach(c => { scores[c][i] = +(base + (c === 'B' ? 0.006 : c === 'C' ? -0.003 : 0.001) + (rnd() - 0.5) * 0.004).toFixed(3); }); }); st.seed2 = +(mean(scores.A) + (rnd() - 0.5) * 0.006).toFixed(4); st.constants = +(mean(scores.A) + (rnd() - 0.4) * 0.01).toFixed(4); build(); });
    const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
    function build() {
      tbl.replaceChildren(); tbl.append(E('thead', {}, E('tr', {}, [E('th', { text: 'Configuration' }), E('th', { class: 'num', text: 'Parameters' }), ...air.map(a => E('th', { class: 'num', text: 'held out: ' + a })), E('th', { class: 'num', text: 'score (mean)' })])));
      const tb = E('tbody'); tbl.append(tb);
      cfg.forEach(c => { const tr = E('tr', { 'data-c': c }); tr.append(E('td', { html: `<b>${c}</b>` }), E('td', { class: 'num', text: P[c + '/full'].toLocaleString('en') })); scores[c].forEach((v, i) => { const inp = E('input', { type: 'number', step: '0.001', value: v, style: 'width:5.5em;font:inherit' }); inp.addEventListener('input', () => { scores[c][i] = +inp.value; calc(); }); tr.append(E('td', { class: 'num' }, inp)); }); tr.append(E('td', { class: 'num', id: 'm' + c })); tb.append(tr); });
      const extra = E('div', { class: 'row' }); const inp1 = E('input', { type: 'number', step: '0.0005', value: st.seed2, style: 'width:6em;font:inherit' }), inp2 = E('input', { type: 'number', step: '0.0005', value: st.constants, style: 'width:6em;font:inherit' });
      inp1.addEventListener('input', () => { st.seed2 = +inp1.value; calc(); }); inp2.addEventListener('input', () => { st.constants = +inp2.value; calc(); });
      extra.append(E('label', { class: 'ctl' }, ['Score of A with a second seed', inp1]), E('label', { class: 'ctl' }, ['Score of variant "constants" (the chosen configuration)', inp2]));
      if (!tbl.nextSibling || !tbl.nextSibling.classList || !tbl.nextSibling.classList.contains('row')) tbl.after(extra); else tbl.nextSibling.replaceWith(extra);
      calc();
    }
    function calc() {
      const m = {}; cfg.forEach(c => { m[c] = mean(scores[c]); const el = document.getElementById('m' + c); if (el) el.textContent = m[c].toFixed(4); });
      const seed = Math.abs(m.A - st.seed2), best = Math.min(...cfg.map(c => m[c])), within = cfg.filter(c => m[c] <= best + 2 * seed);
      const chosen = within.slice().sort((a, b) => P[a + '/full'] - P[b + '/full'])[0];
      const variant = (st.constants < m[chosen] - 2 * seed) ? 'constants' : 'full';
      tbl.querySelectorAll('tbody tr').forEach(tr => { const c = tr.dataset.c; tr.style.background = c === chosen ? 'var(--c-band)' : within.includes(c) ? 'var(--bg-3)' : ''; });
      out(`seed scale s = |score(A) − score(A, seed 2)| = <b>${seed.toFixed(4)}</b>\nbest score ${best.toFixed(4)}; configurations within 2·s of it: <b>${within.join(', ')}</b>\nchosen: the one with the fewest parameters → <b class="word-t">${chosen}</b> (${P[chosen + '/full'].toLocaleString('en')} parameters)\nvariant: "constants" only if its score is lower than that of "full" by more than 2·s: ${st.constants.toFixed(4)} vs ${m[chosen].toFixed(4)} − ${(2 * seed).toFixed(4)} = ${(m[chosen] - 2 * seed).toFixed(4)} → <b class="word-t">${variant}</b>\nruns: 4 configurations × 5 folds = 20, + 5 (the second seed) + 5 ("constants") + 1 (the base on all five airports) = <b>31</b>`);
    }
    build();
    body.append(E('div', { class: 'cap', html: 'Leave-one-airport-out: a fold trains on four airports and reads the fifth, because the goal is a <b>new</b> airport. The parameter counts are real: they come from building the model of <code>prior/model.py</code> for each configuration (A: d_model 192, 6 heads; B: 128, 4 heads; C: 256, 8 heads; D is A with more dropout and weight decay).' }));
  };
})();
