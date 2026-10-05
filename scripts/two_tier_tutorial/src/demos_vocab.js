/* Demos for the vocabulary, the grammar and the labeller (vocabulary.md §3, §4). */
(function () {
  const TT = window.TT, E = TT.el, S = TT.svg, W = () => TT.W, SP = () => TT.SPEC;
  const D = () => TT.data;

  // ============================================================ overview (clickable block diagram)
  TT.demos.overview = (root, body) => {
    const nodes = {
      obs: { x: 20, y: 30, w: 150, h: 54, t: ['Observed track', 'ADS-B, 2 s rows'], info: 'Real arrivals, from the entry of the 25 km slice to the runway. The data has no radio and no wind. The test days are sealed: no code opens them.', code: 'instructions/signals.py', doc: 'vocabulary §4.1' },
      lab: { x: 230, y: 30, w: 150, h: 54, t: ['Labeller', 'track → words'], info: 'Reads the words that a controller would say from an observed track. Pass 1 (open loop) reads the observed words. Pass 2 (closed loop) flies them with the executor and adds correction words.', code: 'instructions/labeller/, autopilot/closed_loop.py', doc: 'vocabulary §4' },
      art: { x: 440, y: 30, w: 170, h: 54, t: ['Sentence artefact', 'words + flown states'], info: 'Each flight becomes a sentence: rows of five words, each correction marked, with the states that the executor flew. It has three row intervals (2, 4 and 8 s). Δ = 4 s is the chosen one.', code: 'instructions/artefact.py', doc: 'vocabulary §6 item 3' },
      pri: { x: 440, y: 170, w: 170, h: 62, t: ['Prior', 'speaker: says rows of words'], info: 'A language model of the controller. At each row it says one row of five words for each aircraft. It trains on the sentences with teacher forcing (stage B).', code: 'prior/ (branch dev-two-tier-v4-prior)', doc: 'prior.md' },
      msk: { x: 205, y: 176, w: 140, h: 50, t: ['Masks', 'grammar + procedure'], info: 'Masks block words when the speaker says them: the grammar rules, the procedure masks (glidepath lower edge, DA, no climb) and masks of the caller. A mask never blocks "unchanged".', code: 'instructions/grammar.py, prior/procedure.py', doc: 'vocabulary §3.7, prior §4' },
      exe: { x: 700, y: 170, w: 150, h: 62, t: ['Executor', 'flies only the words'], info: 'An autopilot with point-mass dynamics. It reads the words, the aircraft data and the runway geometry. It has three laws (lateral, vertical, speed) and no law that follows the glidepath or the centreline.', code: 'autopilot/executor.py, lateral.py, vertical.py, speed.py', doc: 'vocabulary §5' },
      jdg: { x: 700, y: 320, w: 150, h: 54, t: ['Judge', 'outcome of the flight'], info: 'Classifies what the executor flew: landed, unstable_at_minimums, crossed_too_high, and others. It runs the decision-altitude check. Only the judge and the masks read the procedure data.', code: 'autopilot/judge.py', doc: 'vocabulary §5.8' },
      pst: { x: 440, y: 320, w: 170, h: 54, t: ['Post-training', 'reward from outcome'], info: 'Trains the prior further in a closed loop in windows of recorded traffic. The reward comes only from the outcome. Branch training gives the credit (stage C).', code: 'post/ (branch dev-two-tier-v4-post)', doc: 'post_training.md' },
      trf: { x: 205, y: 326, w: 140, h: 50, t: ['Recorded traffic', 'other aircraft'], info: 'In a window the other aircraft fly their records. A traffic attention module in the prior reads them. A separation judge gives loss of separation.', code: 'post/scene.py, post/traffic_attention.py', doc: 'post_training.md §3' },
    };
    const edges = [
      ['obs', 'lab', 'r', 'l'], ['lab', 'art', 'r', 'l', 'sentences'], ['art', 'pri', 'b', 't', 'teacher forcing'],
      ['msk', 'pri', 'r', 'l', 'blocks words'], ['pri', 'exe', 'r', 'l', 'rows of words'], ['exe', 'pri', 'rb', 'rb2', 'states'],
      ['exe', 'jdg', 'b', 't', 'flown track'], ['jdg', 'pst', 'l', 'r', 'reward'], ['pst', 'pri', 't', 'b', 'update'], ['trf', 'pst', 'r', 'l'],
    ];
    const svg = S('svg', { viewBox: '0 0 880 410', role: 'img', 'aria-label': 'Block diagram of the two-tier model: the labeller makes sentences from tracks; the prior says words; the executor flies them; the judge decides; the post-training updates the prior.' });
    svg.append(S('defs', {}, S('marker', { id: 'arr', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, S('path', { d: 'M0 0L10 5L0 10z', fill: 'currentColor', style: 'color:var(--ink-3)' }))));
    const anchor = (n, side) => { const q = nodes[n]; return { l: [q.x, q.y + q.h / 2], r: [q.x + q.w, q.y + q.h / 2], t: [q.x + q.w / 2, q.y], b: [q.x + q.w / 2, q.y + q.h], rb: [q.x + q.w - 20, q.y + q.h], rb2: [q.x + q.w - 20, q.y] }[side]; };
    edges.forEach(([a, b, sa, sb, label]) => {
      let p = anchor(a, sa), q = anchor(b, sb), d;
      if (a === 'exe' && b === 'pri') { p = [nodes.exe.x + 30, nodes.exe.y + nodes.exe.h]; q = [nodes.pri.x + nodes.pri.w - 30, nodes.pri.y + nodes.pri.h]; d = `M${p[0]} ${p[1]} C ${p[0]} ${p[1] + 62}, ${q[0]} ${q[1] + 62}, ${q[0]} ${q[1]}`; }
      else d = `M${p[0]} ${p[1]} L${q[0]} ${q[1]}`;
      svg.append(S('path', { d, class: 'ln thin ink', 'marker-end': 'url(#arr)' }));
      if (label) { const mx = (p[0] + q[0]) / 2, my = (p[1] + q[1]) / 2; const off = a === 'exe' && b === 'pri' ? [0, 52] : sa === 'b' || sa === 't' ? [8, 0] : [0, -6]; svg.append(S('text', { x: mx + off[0], y: my + off[1], 'text-anchor': sa === 'b' || sa === 't' ? 'start' : 'middle', class: 'lbl sm', text: label })); }
    });
    // the closed loop frame
    svg.append(S('rect', { x: 425, y: 150, width: 440, height: 110, rx: 14, fill: 'none', stroke: 'var(--c-word)', 'stroke-dasharray': '4 4', 'stroke-width': 1.3, opacity: .8 }));
    svg.append(S('text', { x: 435, y: 146, class: 'lbl w sm', text: 'closed loop: speak → fly → judge' }));
    const groups = {};
    Object.entries(nodes).forEach(([k, n]) => {
      const g = S('g', { class: 'clickable', tabindex: 0, role: 'button', 'aria-label': n.t[0] });
      g.append(S('rect', { class: 'node', x: n.x, y: n.y, width: n.w, height: n.h, rx: 10 }), S('text', { x: n.x + n.w / 2, y: n.y + n.h / 2 - 3, 'text-anchor': 'middle', class: 'lbl big', style: 'font-weight:600;fill:var(--ink)', text: n.t[0] }), S('text', { x: n.x + n.w / 2, y: n.y + n.h / 2 + 13, 'text-anchor': 'middle', class: 'lbl sm', text: n.t[1] }));
      g.addEventListener('click', () => pick(k)); g.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') pick(k); });
      svg.append(g); groups[k] = g;
    });
    body.append(svg);
    const panel = E('div', { class: 'out', style: 'font-family:var(--font);font-size:.92rem;white-space:normal;min-height:84px' }); body.append(panel);
    const pick = (k) => { Object.entries(groups).forEach(([n, g]) => g.querySelector('rect').classList.toggle('on', n === k)); const n = nodes[k]; panel.innerHTML = `<b>${n.t[0]}.</b> ${n.info}<br><span class="src">Code: <code>${n.code}</code> · Design: ${n.doc}</span>`; };
    pick('pri');
    body.append(E('div', { class: 'cap', text: 'Click a box. The dashed frame is the closed loop that the post-training uses.' }));
  };

  // ============================================================ sentence viewer (real flight)
  TT.demos.sentence = (root, body) => {
    let fl, M, k = 0, timer = null;
    const row0 = E('div', { class: 'row' }); body.append(row0);
    const flightSeg = TT.ui.seg(row0, [['vectored', 'Vectored approach'], ['go_around', 'Approach with a go-around']], 'vectored', (v) => load(v));
    const slider = TT.ui.slider(row0, { label: 'Row', min: 0, max: 10, value: 0, onInput: (v) => { k = v; draw(); } });
    const play = TT.ui.button(row0, '▶ Play', () => { if (timer) { clearInterval(timer); timer = null; play.textContent = '▶ Play'; } else { play.textContent = '❚❚ Pause'; timer = setInterval(() => { k = (k + 1) % M; slider.set(k, false); draw(); }, 140); } });
    const grid = E('div', { class: 'grid2' }); body.append(grid);
    const left = E('div'), right = E('div'); grid.append(left, right);
    const plan = TT.chart(left, { w: 540, h: 400, x: [0, 1], y: [0, 1], xl: 'East (km, airport frame)', yl: 'North (km)' , label: 'Plan view of the flight with the runway final'});
    const alt = TT.chart(right, { w: 540, h: 190, x: [0, 1], y: [0, 1], xl: 'time since the first row (s)', yl: 'height above E (m)', label: 'Height against time' });
    const spd = TT.chart(right, { w: 540, h: 190, x: [0, 1], y: [0, 1], xl: 'time since the first row (s)', yl: 'ground speed (m/s)', label: 'Ground speed against time' });
    TT.ui.legend(body, [['obs', 'observed track'], ['flown', 'flown by the executor'], ['word', 'word in force', 'dash'], ['corr', 'correction word']]);
    const table = E('div'); body.append(table);
    body.append(E('div', { class: 'cap', html: 'Real data: a KRDU arrival (train split), Δ = 4 s, closed-loop sentence of the formal artefact. The first 16 s are observed only. From the first predicted step the executor flies the words. Orange words are corrections of the closed-loop reading.' }));

    function load(name) {
      fl = D().flights[name]; M = fl.words.grid.length; k = 0; slider.input.max = M - 1; slider.set(0, false);
      const st = fl.states.rows, ob = fl.observed;
      const Ep = st.map(r => r[0] / 1000).concat(ob.e.map(v => v / 1000)), Np = st.map(r => r[1] / 1000).concat(ob.n.map(v => v / 1000));
      const c = fl.candidates[fl.runway_index];
      let e0 = Math.min(...Ep, c.e / 1000) - 1, e1 = Math.max(...Ep, c.e / 1000) + 1, n0 = Math.min(...Np, c.n / 1000) - 1, n1 = Math.max(...Np, c.n / 1000) + 1;
      const sc = Math.max((e1 - e0) / plan.iw, (n1 - n0) / plan.ih), cx = (e0 + e1) / 2, cy = (n0 + n1) / 2;
      Object.assign(plan, { x0: cx - sc * plan.iw / 2, x1: cx + sc * plan.iw / 2, y0: cy - sc * plan.ih / 2, y1: cy + sc * plan.ih / 2 });
      drawAxes(plan);
      const T = st.length * 2, hs = st.map(r => r[2] - fl.elevation_m).concat(ob.alt.map(v => v - fl.elevation_m)), vs = st.map(r => r[4]).concat(ob.gs);
      Object.assign(alt, { x0: 0, x1: T, y0: Math.min(0, ...hs) - 20, y1: Math.max(...hs) + 60 }); drawAxes(alt);
      Object.assign(spd, { x0: 0, x1: T, y0: Math.min(...vs) - 10, y1: Math.max(...vs) + 10 }); drawAxes(spd);
      draw();
    }
    const drawAxes = (c) => TT.reaxes(c);
    const inForceAt = (row) => { const out = ['', '', '', '', '']; for (let r = 0; r <= row; r++) fl.words.text[r].forEach((t, c) => { if (t) out[c] = t; }); return out; };
    function draw() {
      const st = fl.states.rows, ob = fl.observed, start = fl.start_row, ev = fl.every, dt = fl.delta_s;
      const j = start + k, i = j * ev, tNow = j * dt, now = st[i], force = inForceAt(k);
      // runway in force
      let R = fl.runway_index; for (let r = 0; r <= k; r++) { const t0 = fl.words.text[r][0]; const q = fl.candidates.findIndex(c => c.ident === t0); if (q >= 0) R = q; }
      const c = fl.candidates[R], course = c.course;
      [plan, alt, spd].forEach(x => x.clear());
      // plan
      const cr = TT.rad(course), back = 40000;
      plan.line([[(c.e - back * Math.sin(cr)) / 1000, (c.n - back * Math.cos(cr)) / 1000], [c.e / 1000, c.n / 1000]], 'ink dash thin');
      plan.dot(c.e / 1000, c.n / 1000, 5, 'f-ink'); plan.text(c.e / 1000, c.n / 1000, 'runway ' + c.ident, 'lbl', 'start', null, 8, 14);
      const right = (x) => x > (plan.x0 + plan.x1) / 2;
      plan.line(ob.e.map((e, q) => [e / 1000, ob.n[q] / 1000]), 'obs');
      plan.line(st.slice(start * ev, i + 1).map(r => [r[0] / 1000, r[1] / 1000]), 'flown');
      plan.dot(st[start * ev][0] / 1000, st[start * ev][1] / 1000, 4, 'f-ink'); plan.text(st[start * ev][0] / 1000, st[start * ev][1] / 1000, 'first predicted step', 'lbl sm', right(st[start * ev][0] / 1000) ? 'end' : 'start', null, right(st[start * ev][0] / 1000) ? -8 : 8, 16);
      const hd = fl.words.text.slice(0, k + 1).map(r => r[1]).filter(Boolean).pop();
      if (hd !== undefined) { const abs = TT.rad(course + parseFloat(hd)), L = 1.6; plan.line([[now[0] / 1000, now[1] / 1000], [now[0] / 1000 + L * Math.sin(abs), now[1] / 1000 + L * Math.cos(abs)]], 'word'); plan.text(now[0] / 1000 + L * Math.sin(abs), now[1] / 1000 + L * Math.cos(abs), 'heading word ' + hd + '°', 'lbl w sm', right(now[0] / 1000) ? 'end' : 'start', null, right(now[0] / 1000) ? -4 : 4, -6); }
      plan.dot(now[0] / 1000, now[1] / 1000, 6, 'f-flown');
      // altitude
      alt.line(ob.t.map((t, q) => [t, ob.alt[q] - fl.elevation_m]), 'obs');
      alt.line(st.slice(start * ev, i + 1).map((r, q) => [(start * ev + q) * 2, r[2] - fl.elevation_m]), 'flown');
      const level = force[2]; if (level && level !== 'no level-off') alt.hline(parseFloat(level), 'word dash'); if (level) alt.text(alt.x1, alt.y1, 'altitude word: ' + level + (level === 'no level-off' ? '' : ' m'), 'lbl w sm', 'end', null, -4, 12);
      alt.dot(tNow, now[2] - fl.elevation_m, 5, 'f-flown'); alt.vline(start * dt, 'faint');
      // speed
      spd.line(ob.t.map((t, q) => [t, ob.gs[q]]), 'obs'); spd.line(st.slice(start * ev, i + 1).map((r, q) => [(start * ev + q) * 2, r[4]]), 'flown');
      const sw = force[4]; if (sw && sw !== 'unspecified') spd.hline(parseFloat(sw), 'word dash'); if (sw) spd.text(spd.x1, spd.y1, 'speed word: ' + sw + (sw === 'unspecified' ? '' : ' m/s'), 'lbl w sm', 'end', null, -4, 12);
      spd.dot(tNow, now[4], 5, 'f-flown'); spd.vline(start * dt, 'faint');
      // table
      const from = Math.max(0, k - 7); let h = '<table class="words"><thead><tr><th>time</th>' + TT.COLUMNS.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>';
      for (let r = from; r <= k; r++) {
        h += `<tr><td>${(start + r) * dt} s</td>` + fl.words.text[r].map((t, q) => `<td class="${r === k ? 'now' : ''} ${fl.words.correction[r][q] ? 'corr' : t ? 'said' : ''}">${t || '·'}</td>`).join('') + '</tr>';
      }
      h += `<tr><td class="force">in force</td>` + force.map(t => `<td class="force">${t || '·'}</td>`).join('') + '</tr></tbody></table>';
      table.innerHTML = h;
    }
    load('vectored');
  };

  // ============================================================ heading frame (D8)
  TT.demos.heading_frame = (root, body) => {
    const A = D().airports; let code = 'KRDU', ri = 3, cls = 0;
    const r1 = TT.ui.row(body);
    const selA = TT.ui.select(r1, 'Airport', Object.keys(A), code, (v) => { code = v; ri = Math.min(ri, A[code].candidates.length - 1); fillRw(); draw(); });
    let selR; const rwBox = E('span'); r1.append(rwBox);
    const fillRw = () => { rwBox.replaceChildren(); selR = TT.ui.select(rwBox, 'Runway', A[code].candidates.map((c, i) => [String(i), c.ident]), String(ri), (v) => { ri = +v; draw(); }); };
    fillRw();
    const sl = TT.ui.slider(r1, { label: 'Heading word (relative, deg)', min: -180, max: 175, step: 5, value: 0, onInput: (v) => { cls = v; draw(); }, fmt: (v) => (v > 0 ? '+' : '') + v });
    const grid = E('div', { class: 'grid2' }); body.append(grid);
    const L = E('div'), R = E('div'); grid.append(L, R);
    const svg = S('svg', { viewBox: '0 0 340 340', role: 'img', 'aria-label': 'Compass with the runway course and the absolute track that a relative heading word says' }); L.append(svg);
    const out = E('div'); R.append(out);
    function draw() {
      const c = A[code].candidates[ri], cx = 170, cy = 170, rad = 130; svg.replaceChildren();
      svg.append(S('circle', { cx, cy, r: rad, fill: 'none', class: 'ln thin faint' }));
      for (let a = 0; a < 360; a += 5) { const r2 = a % 45 === 0 ? 14 : a % 15 === 0 ? 8 : 4, p = TT.rad(a); svg.append(S('line', { x1: cx + rad * Math.sin(p), y1: cy - rad * Math.cos(p), x2: cx + (rad - r2) * Math.sin(p), y2: cy - (rad - r2) * Math.cos(p), class: 'ln thin ink', opacity: a % 15 === 0 ? 1 : .5 })); }
      ['N', 'E', 'S', 'W'].forEach((t, i) => svg.append(S('text', { x: cx + (rad + 14) * Math.sin(i * Math.PI / 2), y: cy - (rad + 14) * Math.cos(i * Math.PI / 2) + 4, 'text-anchor': 'middle', class: 'lbl', text: t })));
      const arrow = (deg, len, cls2, label, anchor = 'middle') => { anchor = label && label.endsWith('°') && cls2 === 'word' ? anchor : anchor; const p = TT.rad(deg), x = cx + len * Math.sin(p), y = cy - len * Math.cos(p); svg.append(S('line', { x1: cx, y1: cy, x2: x, y2: y, class: 'ln ' + cls2, 'marker-end': 'url(#a2)' })); if (label) svg.append(S('text', { x: cx + (len + 12) * Math.sin(p), y: cy - (len + 12) * Math.cos(p) + 4, 'text-anchor': anchor, class: 'lbl ' + (cls2 === 'word' ? 'w' : ''), text: label })); };
      svg.insertBefore(S('defs', {}, S('marker', { id: 'a2', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto' }, S('path', { d: 'M0 0L10 5L0 10z', fill: 'currentColor' }))), svg.firstChild);
      arrow(c.course, rad - 6, 'ink', null);
      const abs = TT.wrap360(c.course + cls);
      arrow(abs, rad - 26, 'word', `${abs.toFixed(1)}°`, abs > 180 ? 'end' : 'start');
      svg.append(S('text', { x: cx, y: cy + 150, 'text-anchor': 'middle', class: 'lbl sm', text: 'course ' + c.course.toFixed(2) + '° (black arrow)' }));
      const near = Math.round(c.course / 5) * 5, err = Math.abs(c.course - near);
      const names = { 0: 'the final (class 0)', 180: 'downwind (class 36)', 90: 'base leg, right', '-90': 'base leg, left' };
      out.innerHTML = `<div class="out"><b>Absolute track</b> = course + word = ${c.course.toFixed(2)}° ${cls >= 0 ? '+' : '−'} ${Math.abs(cls)}° = <b class="word-t">${abs.toFixed(2)}°</b>\n${names[cls] ? 'This word means ' + names[cls] + '.' : 'The word says a track ' + (cls > 0 ? 'to the right of' : 'to the left of') + ' the final.'}\n\n<b>Why a relative grid?</b>\nThe nearest absolute 5° value to the course ${c.course.toFixed(2)}° is ${near}°.\nIt is ${err.toFixed(2)}° away. The capture corridor allows 2°.\n${err > 2 ? '<span class="bad-t">An absolute word cannot hold this final.</span>' : '<span class="ok-t">An absolute word could hold this final.</span>'}</div>`;
      const rows = A[code].candidates.map(q => { const n5 = Math.round(q.course / 5) * 5, e = Math.abs(q.course - n5); return `<tr><td>${q.ident}</td><td class="num">${q.course.toFixed(2)}°</td><td class="num">${e.toFixed(2)}°</td><td>${e > 2 ? '<span class="bad-t">outside 2°</span>' : '<span class="ok-t">inside</span>'}</td></tr>`; }).join('');
      out.innerHTML += `<table><thead><tr><th>${code} runway</th><th class="num">course</th><th class="num">gap to a 5° value</th><th>vs 2° corridor</th></tr></thead><tbody>${rows}</tbody></table>`;
    }
    draw();
    body.append(E('div', { class: 'cap', html: 'The heading word is relative to the course of the runway in force (<span class="pill d">D8</span>). The same word means "downwind" or "final" at every airport. KSTL has courses 62.7°, 122.3°, 242.7°, 302.3°: each is 2.3° from the absolute grid.' }));
  };

  // ============================================================ altitude grid (D22, D52, D58)
  TT.demos.altitude_grid = (root, body) => {
    const Wd = W(); let h = 780;
    const r1 = TT.ui.row(body);
    TT.ui.slider(r1, { label: 'Height above the airport (m)', min: 0, max: 5400, step: 5, value: h, onInput: (v) => { h = v; draw(); } });
    const c = TT.chart(body, { w: 760, h: 240, x: [0, 5400], y: [-250, 250], xl: 'height above the airport elevation E (m)', yl: 'rounding error (m)', label: 'Rounding error of the level words and the band of each level', yt: [-200, -100, 0, 100, 200] });
    const out = TT.ui.out(body);
    // static: band of each level, error sawtooth
    const levels = Wd.levels;
    const hs = TT.range(0, 5400, 1081), err = hs.map(x => { const i = Wd.altIndex(x); return [x, x - levels[i]]; });
    const band = []; levels.forEach((L, i) => { const t = Wd.tol[i]; band.push([L, t]); });
    c.area([...band.map(b => [b[0], b[1]]), ...band.slice().reverse().map(b => [b[0], -b[1]])], 'band', c.layer);
    c.line(err, 'obs', c.layer);
    levels.forEach(L => c.layer.append(S('line', { x1: c.X(L), x2: c.X(L), y1: c.Y(0) - 4, y2: c.Y(0) + 4, class: 'ln thin word' })));
    [[1260, '60 m steps'], [2700, '120 m steps'], [5400, '450 m steps']].forEach(([x, t], i) => c.text(x, 235, t + ' → ' + x + ' m', 'lbl sm', 'end', c.top, -4, 0));
    function draw() {
      c.clear();
      const i = Wd.altIndex(h), L = levels[i], e = h - L, t = Wd.tol[i];
      c.vline(h, 'flown'); c.dot(h, e, 5, 'f-flown');
      c.text(h, 232, `word ${L} m, error ${e.toFixed(0)} m`, 'lbl f', h > 3500 ? 'end' : 'start', null, h > 3500 ? -6 : 6, 0);
      out(`<b>Height</b> ${h} m above E  →  <b class="word-t">word: level ${L} m</b> (class ${i} of 40)\nrounding error <b>${e.toFixed(0)} m</b>; band of this level ε = <b>${t.toFixed(0)} m</b> (half the larger gap to its neighbours + 10 m)`);
    }
    draw();
    // the same assigned MSL level at different airports
    body.append(E('h4', { text: 'One round MSL level, a different word at each airport (D58)' }));
    const r2 = TT.ui.row(body); let ft = 3000;
    TT.ui.slider(r2, { label: 'Assigned level (ft MSL)', min: 1000, max: 8000, step: 500, value: ft, onInput: (v) => { ft = v; t2(); } });
    const tab = E('div'); body.append(tab);
    const t2 = () => {
      const msl = ft * 0.3048, A = D().airports;
      const rows = Object.entries({ ...A, 'KDEN (not a training airport)': { elevation_m: 1650 } }).map(([k, a]) => { const T = msl - a.elevation_m; if (T < -30 || T > 5400) return `<tr><td>${k}</td><td class="num">${a.elevation_m.toFixed(0)}</td><td class="num">${T.toFixed(0)}</td><td colspan=2>outside the grid</td></tr>`; const i = Wd.altIndex(Math.max(0, T)); return `<tr><td>${k}</td><td class="num">${a.elevation_m.toFixed(0)}</td><td class="num">${T.toFixed(0)}</td><td class="num"><b class="word-t">${levels[i]}</b></td><td class="num">${(T - levels[i]).toFixed(0)}</td></tr>`; }).join('');
      tab.innerHTML = `<table><thead><tr><th>Airport</th><th class="num">E (m)</th><th class="num">level above E (m)</th><th class="num">word (m)</th><th class="num">rounding (m)</th></tr></thead><tbody>${rows}</tbody></table>`;
    };
    t2();
    body.append(E('div', { class: 'cap', html: 'The grid has 40 levels on three segments (<span class="pill d">D22</span>, <span class="pill d">D59</span>). A word is a height above the airport, so the final approach lies in the 60 m segment at every airport (<span class="pill d">D58</span>). The blue band is the tolerance ε of each level (<span class="pill d">D52</span>): 40 m, 70 m and 235 m.' }));
  };

  // ============================================================ angle classes (D54, D56)
  TT.demos.angle_classes = (root, body) => {
    const sp = SP(), Wd = W(); let a = 3.2;
    TT.ui.slider(TT.ui.row(body), { label: 'Path angle of a piece (deg, descending positive)', min: 0.6, max: 6, step: 0.05, value: a, fmt: (v) => v.toFixed(2) + '°', onInput: (v) => { a = v; draw(); } });
    const c = TT.chart(body, { w: 760, h: 300, x: [0, 10], y: [0, 560], xl: 'distance flown (km)', yl: 'height lost (m)', yt: [0, 100, 200, 300, 400, 500], label: 'Descent classes as a fan of lines' });
    const out = TT.ui.out(body);
    const edges = sp.descentEdges;
    for (let k = 1; k < edges.length - 1; k++) c.line([[0, 0], [10, 10000 * Math.tan(TT.rad(edges[k]))]], 'faint dash', c.layer);
    sp.descentCentres.forEach((d, k) => { c.line([[0, 0], [10, 10000 * Math.tan(TT.rad(d))]], 'word thin', c.layer); c.text(9.98, 10000 * Math.tan(TT.rad(d)), `descent ${k + 1}: ${d}°`, 'lbl w sm', 'end', c.layer, 0, -5); });
    function draw() {
      c.clear();
      const k = Wd.angleClass(a), nom = Wd.angleDeg(k), end = 10000 * Math.tan(TT.rad(a)), endN = 10000 * Math.tan(TT.rad(nom));
      c.line([[0, 0], [10, end]], 'obs'); c.line([[0, 0], [10, endN]], 'flown dash');
      c.dot(10, end, 4, 'f-obs'); c.dot(10, endN, 4, 'f-flown');
      c.line([[9.7, end], [9.7, endN]], 'corr');
      out(`<b>Observed piece</b> ${a.toFixed(2)}° → <b class="word-t">class descent ${k}</b>, flown at its nominal angle ${nom}°.\nAfter 10 km the executor is <b>${Math.abs(end - endN).toFixed(0)} m</b> from the observed end (length · |tan a − tan c|).\nThe class edges: ${edges.slice(1, -1).join('°, ')}°. The nominals are a k-means fit on tan(angle), weighted by length² (D54).`);
    }
    draw();
    body.append(E('div', { class: 'cap', html: 'Orange: the height error at the end of a 10 km piece. The k-means fit makes the sum of these squared errors smallest over all train pieces. Descent 3 is the 3.0° glidepath that 24 of 25 candidate runways publish.' }));
  };

  // ============================================================ speed steps (D43)
  TT.demos.speed_steps = (root, body) => {
    const sp = SP(); const st = { v0: 150, v1: 80, T: 70, steps: true };
    const r = TT.ui.row(body);
    TT.ui.slider(r, { label: 'Start speed (m/s)', min: 100, max: 200, step: 5, value: st.v0, onInput: (v) => { st.v0 = v; draw(); } });
    TT.ui.slider(r, { label: 'End speed (m/s)', min: 60, max: 120, step: 5, value: st.v1, onInput: (v) => { st.v1 = v; draw(); } });
    TT.ui.slider(r, { label: 'Observed change takes (s)', min: 20, max: 160, step: 5, value: st.T, onInput: (v) => { st.T = v; draw(); } });
    const r2 = TT.ui.row(body);
    TT.ui.seg(r2, [['steps', 'Words in 5 m/s steps (the design)'], ['one', 'One target word']], 'steps', (v) => { st.steps = v === 'steps'; draw(); });
    const c1 = TT.chart(body, { w: 760, h: 230, x: [0, 180], y: [50, 210], xl: 'time (s)', yl: 'ground speed (m/s)', label: 'Observed speed, speed words and flown speed' });
    const c2 = TT.chart(body, { w: 760, h: 150, x: [0, 180], y: [-1500, 1500], xl: 'time (s)', yl: 'flown − observed (m)', yt: [-1000, 0, 1000], label: 'Distance difference along the path' });
    TT.ui.legend(body, [['obs', 'observed speed'], ['word', 'speed word in force', 'dash'], ['flown', 'flown speed']]);
    const out = TT.ui.out(body);
    function draw() {
      [c1, c2].forEach(c => c.clear());
      const { ts, obs, ev, flown, diff } = TT.speedStepsSim(st);
      c1.line(ts.map((t, i) => [t, obs[i]]), 'obs'); c1.step(ev.concat([[180, ev[ev.length - 1][1]]]), 'word dash'); c1.line(flown, 'flown');
      ev.forEach(([t, v]) => c1.dot(t, v, 3.5, 'f-word'));
      c2.hline(0, 'faint'); c2.line(diff, 'corr');
      const m = Math.max(...diff.map(d => Math.abs(d[1])));
      out(`<b>${ev.length - 1} speed word${ev.length === 2 ? '' : 's'}</b> after the first. ${st.steps ? 'Each word is a step of 5 m/s; the time between the words gives the rate of the change.' : 'One word with a target: the executor decides the rate, at a_max = 1.4 m/s².'}\nLargest difference along the path: <b class="${m > 1000 ? 'bad-t' : 'ok-t'}">${m.toFixed(0)} m</b>.`);
    }
    draw();
    body.append(E('div', { class: 'cap', html: 'A target word gives no rate, but the real pilot has one. With the steps (<span class="pill d">D43</span>) the speaker sets the rate through the time between its words, and the flown aircraft stays near the observed one along the path. This sketch uses one smooth observed change and the speed law of <code>autopilot/speed.py</code> without the stall floor.' }));
  };

  // ============================================================ grammar playground (D62)
  TT.demos.grammar = (root, body) => {
    const Wd = W(); const cands = D().airports.KRDU.candidates.map(c => c.ident); const nC = cands.length;
    const S0 = { first: false, R: 3, G: false, alt: 15, angle: 2, height: 800 };
    const row = [TT.UN, TT.UN, TT.UN, TT.UN, TT.UN];
    const colWords = (c) => TT.columnWords(Wd, c, nC);
    const label = (c, w) => { if (w === TT.UN) return 'unchanged'; if (c === 0) return w === TT.GA ? 'go-around' : cands[w]; if (c === 1) return (Wd.headingRel(w) > 0 ? '+' : '') + Wd.headingRel(w); if (c === 2) return w === Wd.noLevelOff ? 'no level-off' : Wd.levels[w] + ' m'; if (c === 3) return Wd.angleName(w); return w === Wd.speedUnspec ? 'unspecified' : Wd.speedMps(w) + ' m/s'; };
    const top = E('div', { class: 'grid2' }); body.append(top);
    const A = E('div', { class: 'card' }), B = E('div', { class: 'card' }); top.append(A, B);
    A.append(E('h4', { text: '1. Words in force and the aircraft' }));
    const rA = TT.ui.row(A);
    TT.ui.check(rA, 'This is the first predicted step', S0.first, (v) => { S0.first = v; update(); });
    TT.ui.select(rA, 'Runway in force', cands.map((c, i) => [String(i), c]), String(S0.R), (v) => { S0.R = +v; update(); });
    TT.ui.check(rA, 'Go-around state G is true', S0.G, (v) => { S0.G = v; update(); });
    const rA2 = TT.ui.row(A);
    TT.ui.select(rA2, 'Altitude word in force', [...Wd.levels.map((l, i) => [String(i), l + ' m']), [String(Wd.noLevelOff), 'no level-off']], String(S0.alt), (v) => { S0.alt = +v; update(); });
    TT.ui.select(rA2, 'Angle in force', [0, 1, 2, 3, 4, 5].map(i => [String(i), Wd.angleName(i)]), String(S0.angle), (v) => { S0.angle = +v; update(); });
    TT.ui.slider(A, { label: 'Height above E (m)', min: 0, max: 3000, step: 10, value: S0.height, onInput: (v) => { S0.height = v; update(); } });
    B.append(E('h4', { text: '2. The row to say' }));
    const sels = TT.COLUMNS.map((name, c) => { const w = E('div', { class: 'row' }); B.append(w); return TT.ui.select(w, name, colWords(c).map(x => [String(x), label(c, x)]), String(TT.UN), (v) => { row[c] = +v; update(); }); });
    const presets = [
      ['A level below the aircraft with no descent', { first: false, G: false, alt: 8, angle: 0, height: 800 }, [-1, -1, 8, 0, -1]],
      ['"no level-off" while G is true', { first: false, G: true, alt: 15, angle: 2, height: 800 }, [-1, -1, Wd.noLevelOff, 3, -1]],
      ['Go-around at the first step', { first: true, G: false, alt: 15, angle: 2, height: 800 }, [TT.GA, 0, 15, 0, 5]],
      ['Say the runway already in force', { first: false, G: false, alt: 15, angle: 2, height: 800 }, [3, -1, -1, -1, -1]],
      ['A good row: descend to 600 m at descent 2', { first: false, G: false, alt: 15, angle: 0, height: 800 }, [-1, -1, 10, 2, -1]],
    ];
    const pr = E('div', { class: 'row' }); body.append(E('div', { class: 'src', text: 'Presets:' }), pr);
    presets.forEach(([name, s, r]) => TT.ui.button(pr, name, () => { Object.assign(S0, s); r.forEach((v, c) => { row[c] = v; sels[c].set(String(v)); }); update(); }));
    const out = TT.ui.out(body);
    body.append(E('h4', { text: '3. Which words may a speaker say in one column? (the mask, D62)' }));
    const rc = TT.ui.row(body); let maskCol = 2;
    TT.ui.seg(rc, TT.COLUMNS.map((c, i) => [String(i), c]), '2', (v) => { maskCol = +v; update(); });
    const chips = E('div', { class: 'chips' }); body.append(chips);
    function update() {
      const inForce = S0.first ? null : { runway: S0.R, go: S0.G, heading: 0, altitude: S0.alt, angle: S0.angle, speed: 5 };
      const r = TT.apply(Wd, inForce, row, S0.height, nC);
      if (r.ok) out(`<span class="ok-t">✓ The row passes.</span>\nWords in force after it: runway ${cands[r.state.runway]}, G = ${r.state.go}, altitude ${r.state.altitude === Wd.noLevelOff ? 'no level-off' : Wd.levels[r.state.altitude] + ' m'}, angle ${Wd.angleName(r.state.angle)}`);
      else out(`<span class="bad-t">✗ Rule broken: ${r.reason}</span>\n${r.detail}`);
      const said = row.slice(0, maskCol);
      const mask = TT.columnMask(Wd, inForce, said, maskCol, S0.height, nC, {});
      const words = colWords(maskCol); chips.replaceChildren();
      words.forEach((w, i) => chips.append(E('span', { class: 'chip ' + (mask[i] ? 'ok' : 'no'), text: label(maskCol, w) })));
      chips.prepend(E('div', { class: 'src', style: 'width:100%', text: `Column "${TT.COLUMNS[maskCol]}", after the words that the earlier columns say in the row above (${said.map((v, c) => label(c, v)).join(' · ') || 'none'}). Green: some words of the later columns make the row pass. Red: none can.` }));
    }
    update();
    body.append(E('div', { class: 'cap', html: 'This is a JavaScript port of <code>instructions/grammar.py</code>: the six rules of <span class="pill d">§3.7</span> and the mask of <span class="pill d">D62</span>. It was checked against the Python code on 4,000 random rows and 120 random masks: no difference.' }));
  };

  // ============================================================ heading reading (labeller, §3.3)
  TT.demos.heading_reading = (root, body) => {
    const st = { profile: 'turn3', lead: 4, noise: 1.0, seed: 3 };
    const r = TT.ui.row(body);
    TT.ui.select(r, 'Observed track', [['turn3', 'Turn of 90° at 3°/s'], ['turn1', 'Turn of 90° at 1°/s'], ['s', 'S-turn: +40°, then −40°'], ['noise', 'Straight, with noise']], st.profile, (v) => { st.profile = v; draw(); });
    TT.ui.slider(r, { label: 'Lead L (s)', min: 0, max: 8, step: 2, value: st.lead, onInput: (v) => { st.lead = v; draw(); } });
    TT.ui.slider(r, { label: 'Noise (deg)', min: 0, max: 3, step: 0.25, value: st.noise, onInput: (v) => { st.noise = v; draw(); } });
    TT.ui.button(r, 'New noise', () => { st.seed++; draw(); });
    const c = TT.chart(body, { w: 760, h: 300, x: [0, 200], y: [-20, 110], xl: 'time (s)', yl: 'track − course (deg)', label: 'Observed track, the track a lead later and the heading words' });
    TT.ui.legend(body, [['obs', 'observed track'], ['flown', 'the track L later'], ['word', 'word in force (±2.5° band)'], ['word', 'a heading word is said', 'sq']]);
    const out = TT.ui.out(body);
    function draw() {
      c.clear(); const rnd = TT.mulberry(st.seed), dt = 2, n = 101, t = TT.range(0, 200, n);
      let tr = t.map(x => { if (st.profile === 'turn3') return TT.clamp((x - 40) * 3, 0, 90); if (st.profile === 'turn1') return TT.clamp((x - 40) * 1, 0, 90); if (st.profile === 's') return 40 * TT.clamp((x - 40) / 13, 0, 1) - 80 * TT.clamp((x - 100) / 13, 0, 1) * (x > 100 ? 1 : 0) + 40 * 0 - (x > 100 ? 0 : 0); return 0; });
      if (st.profile === 's') tr = t.map(x => x < 100 ? 40 * TT.clamp((x - 40) / 13, 0, 1) : 40 - 80 * TT.clamp((x - 100) / 27, 0, 1));
      let nz = 0; tr = tr.map(v => { nz = nz * 0.9 + (rnd() - 0.5) * st.noise * 1.2; return v + nz; });
      const lead = Math.round(st.lead / dt), words = TT.perStepWords(tr, tr.map(() => 0), 5, lead);
      const led = tr.map((_, i) => tr[Math.min(i + lead, n - 1)]);
      c.line(t.map((x, i) => [x, tr[i]]), 'obs'); c.line(t.map((x, i) => [x, led[i]]), 'flown dash');
      const wst = []; words.forEach(([row, rel], q) => { const tEnd = q + 1 < words.length ? t[words[q + 1][0]] : 200; wst.push([t[row], rel], [tEnd, rel]); c.area([[t[row], rel - 2.5], [tEnd, rel - 2.5], [tEnd, rel + 2.5], [t[row], rel + 2.5]], 'band'); });
      c.line(wst, 'word');
      words.forEach(([row, rel]) => c.dot(t[row], rel, 3.5, 'f-word'));
      out(`<b>${words.length} heading word${words.length === 1 ? '' : 's'}</b> (the first says the class at row 0). Rule: a new word is said when the track L later leaves the word in force by more than half a step (2.5°).\n` + words.slice(0, 22).map(([row, rel]) => `t=${t[row]}s: ${rel > 0 ? '+' : ''}${rel}°`).join('   ') + (words.length > 22 ? '  …' : ''));
    }
    draw();
    body.append(E('div', { class: 'cap', html: 'A continuous turn becomes a series of words, one for each 5°, and the time between them gives the turn rate. A word says the track <b>L = 4 s later</b>, so the executor starts to turn at the time of the word. Noise below half a step says nothing. Port of <code>per_step_words</code>, checked against the Python code.' }));
  };

  // ============================================================ the Δ grid (D45)
  TT.demos.delta_grid = (root, body) => {
    const fl = D().flights.vectored; let delta = 4;
    const ob = fl.observed, n = ob.t.length, c0 = fl.candidates[fl.runway_index].course;
    const un = []; let prev = ob.track[0]; ob.track.forEach((v, i) => { if (i) { prev += TT.wrap180(v - ob.track[i - 1]); } un.push(i ? prev : v); });
    const sm = un.map((_, i) => { const a = Math.max(0, i - 1), b = Math.min(n - 1, i + 1); let s = 0; for (let q = a; q <= b; q++) s += un[q]; return s / (b - a + 1); });
    const words = TT.perStepWords(sm, sm.map(() => c0), 5, 2);
    const seg = TT.ui.seg(TT.ui.row(body), [['2', 'Δ = 2 s'], ['4', 'Δ = 4 s (chosen)'], ['8', 'Δ = 8 s']], '4', (v) => { delta = +v; draw(); });
    const c = TT.chart(body, { w: 760, h: 300, x: [0, ob.t[n - 1]], y: [-10, 110], xl: 'time (s)', yl: 'heading word, relative (deg)', label: 'Heading words of a real flight on the 2 s rows and on the Δ rows' });
    TT.ui.legend(body, [['obs', 'word on the 2 s rows (open-loop reading)', 'sq'], ['word', 'word on the Δ grid', 'sq'], ['corr', 'lateness of a word']]);
    const out = TT.ui.out(body);
    function draw() {
      c.clear();
      const rel = words.map(([r, v]) => ({ t: ob.t[r], v: v })); const wrapped = rel.map(w => ({ ...w, v: TT.wrap180(w.v) }));
      c.step(wrapped.map(w => [w.t, w.v]), 'obs thin'); wrapped.forEach(w => c.dot(w.t, w.v, 3, 'f-obs'));
      // the nearest Δ row; a tie goes to the later row; the last word of a row wins
      const rows = new Map(); wrapped.forEach(w => { const j = Math.floor(w.t / delta + 0.5); rows.set(j, { ...w, row: j }); });
      const grid = [...rows.values()].sort((a, b) => a.row - b.row);
      c.step(grid.map(w => [w.row * delta, w.v]), 'word'); grid.forEach(w => { c.dot(w.row * delta, w.v, 4, 'f-word'); });
      wrapped.forEach(w => { const j = Math.floor(w.t / delta + 0.5), late = j * delta - w.t; if (Math.abs(late) > 0.01) c.line([[w.t, w.v], [j * delta, w.v]], 'corr thin'); });
      let jump = 0; for (let i = 1; i < grid.length; i++) jump = Math.max(jump, Math.abs(grid[i].v - grid[i - 1].v) / 5);
      const late = wrapped.map(w => Math.floor(w.t / delta + 0.5) * delta - w.t);
      const T = ob.t[n - 1];
      out(`<b>2 s rows:</b> ${wrapped.length} heading words, ${(T / 2).toFixed(0)} rows.   <b>Δ = ${delta} s:</b> ${grid.length} words, ${(T / delta).toFixed(0)} rows.\nLargest jump between two words: <b>${jump.toFixed(0)} class${jump === 1 ? '' : 'es'}</b>.  Mean lateness ${(late.reduce((a, b) => a + b, 0) / late.length).toFixed(2)} s (each word is within ±${delta / 2} s).`);
    }
    draw();
    body.append(E('div', { class: 'cap', html: 'Real observed track (KRDU, train). The words come from a JavaScript port of the heading reading; the paper figure uses the same rule. Each word goes to the <b>nearest</b> Δ row (<span class="pill d">D45</span>); a coarse Δ moves a word by up to Δ/2 and packs several words into one row.' }));
  };
})();
