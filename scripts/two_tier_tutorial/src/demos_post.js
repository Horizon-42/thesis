/* Demos for the post-training (post_training.md): the reward, branch training, the loss, the traffic. */
(function () {
  const TT = window.TT, E = TT.el, S = TT.svg;

  // ============================================================ the reward (D30, D91, D112)
  TT.demos.reward = (root, body) => {
    const st = { outcome: 'landed', n: 0, present: true, lost: false, p: 0.6, q: 0.6, f: 0.28 };
    const outcomes = ['landed', 'unstable_at_minimums', 'crossed_too_high', 'crossed_off_runway', 'crossed_other_runway', 'ground_contact', 'timeout', 'dynamics_failure'];
    const g = E('div', { class: 'grid2' }); body.append(g); const A = E('div', { class: 'card' }), B = E('div', { class: 'card' }); g.append(A, B);
    A.append(E('h4', { text: 'The reward of one sentence' }));
    const row = TT.ui.row(A);
    TT.ui.select(row, 'Outcome (the judge)', outcomes, st.outcome, (v) => { st.outcome = v; draw(); });
    TT.ui.slider(A, { label: 'Go-arounds n', min: 0, max: 2, step: 1, value: 0, onInput: (v) => { st.n = v; draw(); } });
    TT.ui.check(A, 'It landed on a runway of the airport’s present landing direction', true, (v) => { st.present = v; draw(); });
    TT.ui.check(A, 'It lost separation (and the commanded aircraft answers for it)', false, (v) => { st.lost = v; draw(); });
    const rOut = TT.ui.out(A);
    B.append(E('h4', { text: 'Why no payment for a go-around' }));
    TT.ui.slider(B, { label: 'Chance of landing if the approach continues (p)', min: 0.05, max: 0.99, step: 0.01, value: st.p, fmt: (v) => v.toFixed(2), onInput: (v) => { st.p = v; draw(); } });
    TT.ui.slider(B, { label: 'Chance after the go-around (q)', min: 0.05, max: 0.99, step: 0.01, value: st.q, fmt: (v) => v.toFixed(2), onInput: (v) => { st.q = v; draw(); } });
    TT.ui.slider(B, { label: 'A payment f for a go-around that does not land', min: 0, max: 0.5, step: 0.01, value: st.f, fmt: (v) => v.toFixed(2), onInput: (v) => { st.f = v; draw(); } });
    const bOut = TT.ui.out(B);
    const c = TT.chart(body, { w: 760, h: 220, x: [0, 1], y: [0, 1], xl: 'p: chance of landing if the approach continues', yl: 'expected reward', label: 'Expected reward of continuing and of a go-around' });
    TT.ui.legend(body, [['obs', 'continue: p'], ['flown', 'go around with q = p: 0.9·p', 'dash'], ['corr', 'go around with the q above: 0.9·q'], ['word', 'with a payment f: 0.9·q + (1 − q)·f']]);
    c.line([[0, 0], [1, 1]], 'obs', c.layer); c.line([[0, 0], [1, 0.9]], 'flown dash', c.layer);
    function draw() {
      c.clear();
      const landed = st.outcome === 'landed'; let r = 0;
      if (landed && !st.lost) r = st.present ? Math.pow(0.9, st.n) : 0;
      rOut(`reward = <b class="${r > 0 ? 'ok-t' : 'bad-t'}">${r.toFixed(3)}</b>\n${st.lost ? 'a loss of separation: 0' : !landed ? 'an outcome other than landed: 0' : !st.present ? 'a landing against the present landing direction: 0' : `landed after ${st.n} go-around${st.n === 1 ? '' : 's'}: 0.9^${st.n}`}`);
      c.line([[0, 0], [1, 0.9 * st.q]], 'corr', c.dyn); c.line([[0, st.f], [1, 0.9 * st.q + (1 - st.q) * st.f]], 'word thin', c.dyn);
      c.dot(st.p, st.p, 4, 'f-obs', c.dyn);
      const better = 0.9 * st.q > st.p;
      const thr = st.f / (st.f + 0.1);
      bOut(`With the same chance before and after (q = p), a go-around gives 0.9·p, below p: it never pays.\nIt pays only when the go-around turns a probable failure into a probable landing: <b class="${better ? 'ok-t' : 'bad-t'}">0.9·q = ${(0.9 * st.q).toFixed(2)} ${better ? '>' : '≤'} p = ${st.p.toFixed(2)}</b>.\nWith a payment f = ${st.f.toFixed(2)} for a go-around without a landing, the go-around beats the approach when p &lt; f/(f+0.1) = <b>${thr.toFixed(2)}</b>. The model would learn to go around where it is not sure, not where the approach is unstable.`);
    }
    draw();
    body.append(E('div', { class: 'cap', html: 'The reward comes only from the outcome (<span class="pill d">D30</span>). The go-around bound of the loop is 2; after the second go-around the caller masks the word (<span class="pill d">D91</span>). With no landing in the 30 minutes before the window, every runway counts as the present direction (<span class="pill d">D112</span>).' }));
  };

  // ============================================================ branch training (D37, D94)
  TT.demos.branch_training = (root, body) => {
    const st = { tE: 420, q0: 0.14, seed: 1, fail: 0.15 };
    const r = TT.ui.row(body);
    TT.ui.slider(r, { label: 'The first sentence ends at (s)', min: 60, max: 720, step: 20, value: st.tE, onInput: (v) => { st.tE = v; draw(); } });
    TT.ui.slider(r, { label: 'Chance that a continuation from the start lands', min: 0.02, max: 0.4, step: 0.01, value: st.q0, fmt: (v) => v.toFixed(2), onInput: (v) => { st.q0 = v; draw(); } });
    TT.ui.slider(r, { label: 'Assumed share of windows whose first sentence fails', min: 0.05, max: 0.6, step: 0.05, value: st.fail, fmt: (v) => Math.round(v * 100) + ' %', onInput: (v) => { st.fail = v; draw(); } });
    TT.ui.button(r, 'Speak again (new random numbers)', () => { st.seed++; draw(); });
    const svg = S('svg', { viewBox: '0 0 780 420', role: 'img', 'aria-label': 'Branch groups of one window: the first sentence and eight continuations at each branch point' }); body.append(svg);
    const out = TT.ui.out(body);
    const K = 8, EVERY = 120;
    function draw() {
      svg.replaceChildren(); const rnd = TT.mulberry(st.seed * 7919), tMax = Math.max(st.tE + 40, 200);
      const x = (t) => 150 + t / tMax * 420, branch = []; for (let t = 0; t < st.tE; t += EVERY) branch.push(t);
      let y = 20; let spoken = st.tE, informative = 0, samples = 0, rowsCounted = 0; const lines = [];
      svg.append(S('text', { x: 150, y: 12, class: 'lbl sm', text: '0 s' }), S('text', { x: x(st.tE), y: 12, class: 'lbl sm r', 'text-anchor': 'middle', text: `event at ${st.tE} s` }));
      const evLine = S('line', { x1: x(st.tE), x2: x(st.tE), y1: 16, y2: 410, class: 'ln thin block dash' }); svg.append(evLine);
      branch.forEach((tb, gi) => {
        const q = st.q0 * Math.min(1, (st.tE - tb) / 250), rewards = [0];
        const conts = Array.from({ length: K }, () => { const ok = rnd() < q; const ga = ok && rnd() < 0.15; return { ok, reward: ok ? (ga ? 0.9 : 1) : 0, end: ok ? st.tE - 25 : st.tE }; });
        conts.forEach(c => rewards.push(c.reward));
        const mean = rewards.reduce((a, b) => a + b, 0) / rewards.length, inf = Math.max(...rewards) > Math.min(...rewards);
        if (inf) { informative++; samples += K + 1; }
        spoken += conts.reduce((s, c) => s + (c.end - tb), 0); if (inf) rowsCounted += (K + 1) * 0;
        svg.append(S('text', { x: 0, y: y + 10, class: 'lbl', style: 'font-weight:600;fill:var(--ink)', text: `branch point ${tb} s` }), S('text', { x: 0, y: y + 24, class: 'lbl sm', text: inf ? `group mean ${mean.toFixed(2)}` : 'rewards all equal: no sample' }));
        const lineRow = (t0, t1, cls, ry, label) => { svg.append(S('line', { x1: x(t0), x2: x(t1), y1: ry, y2: ry, class: 'ln ' + cls, 'stroke-width': 4, opacity: inf ? 1 : .35 })); if (label) svg.append(S('text', { x: x(t1) + 6, y: ry + 3.5, class: 'lbl sm', text: label })); };
        lineRow(tb, st.tE, 'word', y, '');
        conts.forEach((c, i) => lineRow(tb, c.end, c.ok ? 'ok' : 'block', y + 7 * (i + 1), ''));
        if (inf) { const ok = conts.find(c => c.ok); svg.append(S('text', { x: x(st.tE) + 8, y: y + 14, class: 'lbl sm', text: `advantage of the first sentence: ${(0 - mean).toFixed(2)}` })); svg.append(S('text', { x: x(st.tE) + 8, y: y + 28, class: 'lbl sm g', text: ok ? `of a landing continuation: +${(ok.reward - mean).toFixed(2)}` : '' })); svg.append(S('text', { x: x(st.tE) + 8, y: y + 42, class: 'lbl sm r', text: `of a failing continuation: ${(0 - mean).toFixed(2)}` })); }
        svg.append(S('line', { x1: x(tb), x2: x(tb), y1: y - 4, y2: y + 7 * (K + 1), class: 'ln thin ink' }));
        y += 7 * (K + 1) + 18;
      });
      evLine.setAttribute('y2', y - 8); svg.setAttribute('viewBox', `0 0 780 ${Math.max(60, y)}`);
      const extra = spoken - st.tE, flat = K * st.tE, branchAvg = st.tE + st.fail * extra;
      out(`Window with one failed first sentence (reward 0). Branch points every ${EVERY} s before the event: <b>${branch.length}</b>. Groups with different rewards (they give samples): <b class="${informative ? 'ok-t' : 'bad-t'}">${informative}</b>.\nSamples for the loss: ${samples} sentences. The advantage of each is its reward minus the mean of its group, and it counts only the words <b>after</b> the branch point.\n<b>Cost for each window, on average.</b> ${K} full new sentences for every window: ${(flat / 60).toFixed(1)} min. Branch training: the first sentence (${(st.tE / 60).toFixed(1)} min) plus, in the ${Math.round(st.fail * 100)} % of windows that fail, the continuations of this plot (${(extra / 60).toFixed(1)} min) = <b class="ok-t">${(branchAvg / 60).toFixed(1)} min</b>.`);
    }
    draw();
    TT.ui.legend(body, [['word', 'the first sentence (reward 0), from the branch point on'], ['ok', 'a continuation that lands (reward 1, or 0.9 after a go-around)'], ['block', 'a continuation that fails (reward 0)']]);
    body.append(E('div', { class: 'cap', html: 'Each training window is spoken once. A sentence with reward 1 gives no sample. Otherwise a second pass copies the state of the window <b>K = 8</b> times at each branch point (the first predicted step and every 120 s before the event), and each copy continues with its own random numbers (<span class="pill d">D37</span>, <span class="pill d">D94</span>). The chance of a rescue here is a toy model fitted by eye to the shape of the window-rewind readout in post-training §6.2 (an earlier design): later branch points rescue less.' }));
  };

  // ============================================================ the loss (clipped ratio, pull to the base)
  TT.demos.surrogate = (root, body) => {
    const st = { A: 0.6, eps: 0.2, r: 1.1, gap: 0.5 };
    const r1 = TT.ui.row(body);
    TT.ui.slider(r1, { label: 'Advantage A', min: -1, max: 1, step: 0.05, value: st.A, fmt: (v) => v.toFixed(2), onInput: (v) => { st.A = v; draw(); } });
    TT.ui.slider(r1, { label: 'Clip ε', min: 0.05, max: 0.4, step: 0.05, value: st.eps, fmt: (v) => v.toFixed(2), onInput: (v) => { st.eps = v; draw(); } });
    TT.ui.slider(r1, { label: 'Ratio r = p_new / p_old', min: 0.4, max: 1.8, step: 0.01, value: st.r, fmt: (v) => v.toFixed(2), onInput: (v) => { st.r = v; draw(); } });
    const g = E('div', { class: 'grid2' }); body.append(g); const L = E('div'), R = E('div'); g.append(L, R);
    const c1 = TT.chart(L, { w: 370, h: 260, x: [0.4, 1.8], y: [-1.2, 1.2], xl: 'ratio r', yl: 'objective min(r·A, clip(r)·A)', label: 'The clipped surrogate' });
    const c2 = TT.chart(R, { w: 370, h: 260, x: [-1.5, 1.5], y: [0, 2], xl: 'gap = log p_base − log p_new', yl: 'exp(gap) − gap − 1', label: 'The pull to the base model' });
    const out = TT.ui.out(body);
    c2.line(TT.range(-1.5, 1.5, 100).map(x => [x, Math.exp(x) - x - 1]), 'word', c2.layer);
    const f = (r, A, e) => Math.min(r * A, TT.clamp(r, 1 - e, 1 + e) * A);
    function draw() {
      [c1, c2].forEach(c => c.clear());
      c1.area([[1 - st.eps, -1.2], [1 + st.eps, -1.2], [1 + st.eps, 1.2], [1 - st.eps, 1.2]], 'band');
      c1.line(TT.range(0.4, 1.8, 120).map(r => [r, f(r, st.A, st.eps)]), 'flown');
      c1.line(TT.range(0.4, 1.8, 120).map(r => [r, r * st.A]), 'faint dash');
      const v = f(st.r, st.A, st.eps), clipped = Math.abs(st.r - 1) > st.eps && ((st.A > 0 && st.r > 1 + st.eps) || (st.A < 0 && st.r < 1 - st.eps));
      c1.dot(st.r, v, 5, clipped ? 'f-block' : 'f-flown');
      const k3 = Math.exp(st.gap) - st.gap - 1; c2.dot(st.gap, k3, 5, 'f-word');
      out(`objective ${v.toFixed(3)} (the loss is its negative). ${clipped ? '<span class="bad-t">The ratio is beyond the clip on the side that the advantage pushes: the gradient is 0.</span>' : '<span class="ok-t">Inside the clip, or on the side that the objective keeps: the gradient flows.</span>'}\nThe loss = surrogate + 0.04 × pull to the base + 1.0 × teacher-forced data term. The pull is computed with dropout off; the data term with the dropout of the base’s training (D107).`);
    }
    draw();
    TT.ui.row(body);
    body.append(E('h4', { text: 'Every counted row weighs the same (D115)' }));
    const wr = TT.ui.row(body); let n1 = 30, n2 = 3;
    TT.ui.slider(wr, { label: 'Counted rows of sample 1 (a whole first sentence)', min: 5, max: 60, step: 1, value: n1, onInput: (v) => { n1 = v; w(); } });
    TT.ui.slider(wr, { label: 'Counted rows of sample 2 (a late continuation)', min: 1, max: 10, step: 1, value: n2, onInput: (v) => { n2 = v; w(); } });
    const wb = E('div'); body.append(wb);
    const w = () => { const a = 1 / n1, b = 1 / n2, tot = n1 + n2; wb.innerHTML = `<table><thead><tr><th></th><th class="num">weight of one row of sample 1</th><th class="num">weight of one row of sample 2</th></tr></thead><tbody><tr><td>each sample divided by its own rows</td><td class="num">${a.toFixed(3)}</td><td class="num">${b.toFixed(3)} (<b class="bad-t">${(b / a).toFixed(1)}×</b>)</td></tr><tr><td>the batch divided by its counted rows (the design)</td><td class="num">${(1 / tot).toFixed(3)}</td><td class="num">${(1 / tot).toFixed(3)} (<b class="ok-t">1.0×</b>)</td></tr></tbody></table>`; }; w();
    body.append(E('div', { class: 'cap', html: 'The ratio of a word is its probability under the model over its probability under the frozen copy that spoke it, <b>word by word</b> (<span class="pill d">D117</span>). The log-probabilities come from the speaker’s records of the permitted words, so a masked word has no probability.' }));
  };

  // ============================================================ traffic: edge features and the speed-word mask
  TT.demos.traffic_scene = (root, body) => {
    const st = { o: { e: 4000, n: -2500 }, tc: 90, vc: 80, to: 0, vo: 70, dd: 9800, dl: 6000, vl: 70, vf: 80, req: 3 };
    body.append(E('h4', { text: 'A. Edge features of one other aircraft' }));
    const r = TT.ui.row(body);
    TT.ui.slider(r, { label: 'Commanded: track (deg)', min: 0, max: 359, step: 5, value: st.tc, onInput: (v) => { st.tc = v; draw(); } });
    TT.ui.slider(r, { label: 'Commanded: speed (m/s)', min: 40, max: 140, step: 5, value: st.vc, onInput: (v) => { st.vc = v; draw(); } });
    const r2 = TT.ui.row(body);
    TT.ui.slider(r2, { label: 'Other: track (deg)', min: 0, max: 359, step: 5, value: st.to, onInput: (v) => { st.to = v; draw(); } });
    TT.ui.slider(r2, { label: 'Other: speed (m/s)', min: 40, max: 140, step: 5, value: st.vo, onInput: (v) => { st.vo = v; draw(); } });
    const c = TT.chart(body, { w: 760, h: 300, x: [-6000, 10000], y: [-6000, 5000], xl: 'East (m)', yl: 'North (m)', xt: [-5000, 0, 5000, 10000], yt: [-5000, 0, 5000], label: 'The commanded aircraft and one other aircraft' });
    const ball = S('circle', { class: 'f-corr ring', r: 9 }); c.top.append(ball); TT.drag(c, ball, (x, y) => { st.o = { e: x, n: y }; draw(); });
    const out = TT.ui.out(body);
    function draw() {
      c.clear(); ball.setAttribute('cx', c.X(st.o.e)); ball.setAttribute('cy', c.Y(st.o.n));
      const vc = [st.vc * Math.sin(TT.rad(st.tc)), st.vc * Math.cos(TT.rad(st.tc))], vo = [st.vo * Math.sin(TT.rad(st.to)), st.vo * Math.cos(TT.rad(st.to))];
      const rel = [st.o.e, st.o.n], vr = [vo[0] - vc[0], vo[1] - vc[1]], dist = Math.hypot(...rel);
      const fwd = [vc[0] / st.vc, vc[1] / st.vc], left = [-fwd[1], fwd[0]];
      const front = rel[0] * fwd[0] + rel[1] * fwd[1], lft = rel[0] * left[0] + rel[1] * left[1];
      const closing = -(rel[0] * vr[0] + rel[1] * vr[1]) / dist, v2 = vr[0] ** 2 + vr[1] ** 2;
      const tcpa = TT.clamp(v2 > 0 ? -(rel[0] * vr[0] + rel[1] * vr[1]) / v2 : 120, 0, 120), cpa = Math.hypot(rel[0] + vr[0] * tcpa, rel[1] + vr[1] * tcpa);
      const arrow = (p, v, k, cls) => c.line([[p[0], p[1]], [p[0] + v[0] * k, p[1] + v[1] * k]], cls);
      c.dot(0, 0, 8, 'f-flown'); arrow([0, 0], vc, 40, 'flown'); c.dot(st.o.e, st.o.n, 0.1, 'f-corr'); arrow([st.o.e, st.o.n], vo, 40, 'corr');
      c.line([[0, 0], [vc[0] * tcpa, vc[1] * tcpa]], 'flown dash thin'); c.line([[st.o.e, st.o.n], [st.o.e + vo[0] * tcpa, st.o.n + vo[1] * tcpa]], 'corr dash thin');
      c.dot(vc[0] * tcpa, vc[1] * tcpa, 4, 'f-flown'); c.dot(st.o.e + vo[0] * tcpa, st.o.n + vo[1] * tcpa, 4, 'f-corr');
      c.text(0, 0, 'commanded', 'lbl f sm', 'start', null, 10, -10); c.text(st.o.e, st.o.n, 'other (drag)', 'lbl c sm', 'start', null, 12, -10);
      out(`front <b>${(front / 1000).toFixed(2)} km</b> (ahead of the commanded aircraft’s direction of motion) · left <b>${(lft / 1000).toFixed(2)} km</b>\nclosing rate <b>${closing.toFixed(1)} m/s</b> ${closing > 0 ? '(closing)' : '(opening)'}\nclosest point of approach, both flying straight on: in <b>${tcpa.toFixed(0)} s</b> (clipped to 0–120 s), at <b>${cpa.toFixed(0)} m</b>`);
    }
    draw();
    body.append(E('h4', { text: 'B. The speed-word mask on a final' }));
    const r3 = TT.ui.row(body);
    TT.ui.slider(r3, { label: 'Follower: distance to the threshold (m)', min: 6000, max: 14000, step: 100, value: st.dd, onInput: (v) => { st.dd = v; m(); } });
    TT.ui.slider(r3, { label: 'Leader (ahead): distance to the threshold (m)', min: 1500, max: 9000, step: 250, value: st.dl, onInput: (v) => { st.dl = v; m(); } });
    const r4 = TT.ui.row(body);
    TT.ui.slider(r4, { label: 'Leader speed (m/s)', min: 50, max: 100, step: 1, value: st.vl, onInput: (v) => { st.vl = v; m(); } });
    TT.ui.slider(r4, { label: 'Follower speed now (m/s)', min: 50, max: 120, step: 1, value: st.vf, onInput: (v) => { st.vf = v; m(); } });
    TT.ui.slider(r4, { label: 'Required gap (NM)', min: 2.5, max: 6, step: 0.5, value: st.req, onInput: (v) => { st.req = v; m(); } });
    const chips = E('div', { class: 'chips' }); const mo = TT.ui.out(body); body.append(chips);
    chips.before(E('div', { class: 'src', text: 'Speed words of the follower (m/s):' }));
    function m() {
      chips.replaceChildren(); const NM = 1852, a = TT.SPEC.aMax, faf = 10108, tL = st.dl / st.vl, req = st.req * NM;
      const ramp = (t, v0, tg) => { const dv = tg - v0, tr = Math.abs(dv) / a; return t <= tr ? v0 * t + 0.5 * Math.sign(dv) * a * t * t : v0 * tr + 0.5 * Math.sign(dv) * a * tr * tr + tg * (t - tr); };
      const acts = st.dd > 9260 && st.dd <= faf && st.dl < st.dd;
      let blockedN = 0; const words = []; for (let v = 40; v <= 120; v += 5) words.push(v);
      const rows = words.map(v => { const gap = st.dd - ramp(tL, st.vf, v); const bad = acts && gap < req; if (bad) blockedN++; return { v, gap, bad }; });
      const every = blockedN === rows.length;
      rows.forEach(({ v, gap, bad }) => chips.append(E('span', { class: 'chip ' + (bad && !every ? 'no' : 'ok'), 'data-tip': `predicted gap ${(gap / NM).toFixed(1)} NM when the leader crosses`, text: v })));
      mo(`The leader crosses its threshold in ${tL.toFixed(0)} s. The follower is predicted to that time along its course toward each word at 1.4 m/s² and holds the target.\nThe mask acts only when the follower is between 5 NM (9,260 m) and the FAF (${(faf / 1000).toFixed(1)} km) from the threshold (both established on the final): <b class="${acts ? 'ok-t' : 'bad-t'}">${acts ? 'it acts' : 'it does not act here'}</b>.\n${acts ? (every ? 'Every word falls short: nothing is masked.' : `${blockedN} of ${rows.length} words are blocked (red). "Unchanged" and "unspecified" are never blocked.`) : 'All words are permitted. This band is only 0.1–2 km wide at most runways (D101).'}`);
    }
    m();
    body.append(E('div', { class: 'cap', html: 'A <b>stated approximation</b> (<code>post/speed_mask.py</code>): constant rate, no turn, no descent, no wind. The required gap is the terminal radar minimum here (7110.65BB 5-5-4); the real code asks <code>Separation.distance_nm</code> for the pair. The edge features of part A are the geometric ones of <code>post/edges.py</code> (the real tokens also carry the approach clock, the runway relation and the required distance).' }));
  };
})();
