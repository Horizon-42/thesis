/* Shared helpers: DOM, SVG charts, controls, tooltips, term glosses. Everything lives in window.TT. */
(function () {
  const root = typeof window !== 'undefined' ? window : globalThis;
  const TT = root.TT = root.TT || { demos: {}, data: null };
  const NS = 'http://www.w3.org/2000/svg';
  const hasDOM = typeof document !== 'undefined';

  // ---------- small maths ----------
  TT.clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  TT.deg = (r) => r * 180 / Math.PI;
  TT.rad = (d) => d * Math.PI / 180;
  TT.wrap180 = (a) => ((a + 180) % 360 + 360) % 360 - 180;
  TT.wrap360 = (a) => ((a % 360) + 360) % 360;
  TT.range = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));
  TT.fmt = (v, d = 1) => (Number.isFinite(v) ? v.toFixed(d) : '–');
  TT.sgn = (v) => (v > 0) - (v < 0);
  TT.G = 9.81; // aerodynamic_model.common.GRAVITY_MPS2
  TT.mulberry = (seed) => () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  TT.nice = (lo, hi, n = 5) => {
    const span = hi - lo || 1, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) || raw;
    const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  };

  if (!hasDOM) return;

  // ---------- DOM ----------
  TT.el = (tag, attrs = {}, kids = []) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') e.className = v; else if (k === 'html') e.innerHTML = v; else if (k === 'text') e.textContent = v;
      else if (k.startsWith('on')) e.addEventListener(k.slice(2), v); else if (v !== false && v != null) e.setAttribute(k, v === true ? '' : v);
    }
    (Array.isArray(kids) ? kids : [kids]).forEach(c => c != null && e.append(c.nodeType ? c : document.createTextNode(c)));
    return e;
  };
  TT.svg = (tag, attrs = {}, kids = []) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) { if (k === 'text') e.textContent = v; else if (v != null && v !== false) e.setAttribute(k, v); }
    (Array.isArray(kids) ? kids : [kids]).forEach(c => c != null && e.append(c));
    return e;
  };

  // ---------- chart helper ----------
  // c = TT.chart(parent, {w,h,m,x:[a,b],y:[a,b],xl,yl,xt,yt,grid}) ; c.dyn is a layer to clear and redraw
  TT.chart = (parent, o) => {
    const w = o.w || 640, h = o.h || 300, m = Object.assign({ l: 58, r: 14, t: 10, b: 38 }, o.m || {});
    const svg = TT.svg('svg', { viewBox: `0 0 ${w} ${h}`, role: 'img', 'aria-label': o.label || '' });
    parent.append(svg);
    const iw = w - m.l - m.r, ih = h - m.t - m.b;
    const c = { svg, w, h, m, iw, ih, x0: o.x[0], x1: o.x[1], y0: o.y[0], y1: o.y[1] };
    c.X = (v) => m.l + (v - c.x0) / (c.x1 - c.x0) * iw;
    c.Y = (v) => m.t + ih - (v - c.y0) / (c.y1 - c.y0) * ih;
    c.iX = (px) => c.x0 + (px - m.l) / iw * (c.x1 - c.x0);
    c.iY = (py) => c.y0 + (m.t + ih - py) / ih * (c.y1 - c.y0);
    const axis = TT.svg('g', { class: 'axis' }), grid = TT.svg('g', { class: 'grid' }), ticks = TT.svg('g', { class: 'tick' });
    const xt = o.xt || TT.nice(c.x0, c.x1, Math.max(3, Math.round(iw / 90))), yt = o.yt || TT.nice(c.y0, c.y1, Math.max(3, Math.round(ih / 55)));
    xt.forEach(v => { if (o.grid !== false) grid.append(TT.svg('line', { x1: c.X(v), x2: c.X(v), y1: m.t, y2: m.t + ih })); ticks.append(TT.svg('text', { x: c.X(v), y: m.t + ih + 15, 'text-anchor': 'middle', text: (o.xf || String)(v) })); });
    yt.forEach(v => { if (o.grid !== false) grid.append(TT.svg('line', { x1: m.l, x2: m.l + iw, y1: c.Y(v), y2: c.Y(v) })); ticks.append(TT.svg('text', { x: m.l - 6, y: c.Y(v) + 3.5, 'text-anchor': 'end', text: (o.yf || String)(v) })); });
    axis.append(TT.svg('line', { x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih }), TT.svg('line', { x1: m.l, x2: m.l, y1: m.t, y2: m.t + ih }));
    svg.append(grid, axis, ticks);
    if (o.xl) svg.append(TT.svg('text', { x: m.l + iw / 2, y: h - 4, 'text-anchor': 'middle', class: 'lbl', text: o.xl }));
    if (o.yl) svg.append(TT.svg('text', { transform: `translate(13 ${m.t + ih / 2}) rotate(-90)`, 'text-anchor': 'middle', class: 'lbl', text: o.yl }));
    const clip = 'clip' + Math.random().toString(36).slice(2, 8);
    svg.append(TT.svg('defs', {}, TT.svg('clipPath', { id: clip }, TT.svg('rect', { x: m.l, y: m.t - 2, width: iw, height: ih + 4 }))));
    c.layer = TT.svg('g', { 'clip-path': `url(#${clip})` }); c.dyn = TT.svg('g', { 'clip-path': `url(#${clip})` }); c.top = TT.svg('g');
    svg.append(c.layer, c.dyn, c.top);
    c.clear = () => { c.dyn.replaceChildren(); };
    const add = (e, to) => { (to || c.dyn).append(e); return e; };
    c.line = (pts, cls = 'ink', to) => add(TT.svg('polyline', { class: 'ln ' + cls, points: pts.filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1])).map(p => c.X(p[0]).toFixed(1) + ',' + c.Y(p[1]).toFixed(1)).join(' ') }), to);
    c.step = (pts, cls = 'ink', to) => { const q = []; pts.forEach((p, i) => { if (i) q.push([p[0], pts[i - 1][1]]); q.push(p); }); return c.line(q, cls, to); };
    c.area = (pts, cls = 'band', to) => add(TT.svg('polygon', { class: cls, points: pts.map(p => c.X(p[0]).toFixed(1) + ',' + c.Y(p[1]).toFixed(1)).join(' ') }), to);
    c.rect = (xa, ya, xb, yb, cls = 'band', to) => add(TT.svg('rect', { class: cls, x: Math.min(c.X(xa), c.X(xb)), y: Math.min(c.Y(ya), c.Y(yb)), width: Math.abs(c.X(xb) - c.X(xa)), height: Math.abs(c.Y(yb) - c.Y(ya)) }), to);
    c.vline = (x, cls = 'faint', to) => add(TT.svg('line', { class: 'ln thin ' + cls, x1: c.X(x), x2: c.X(x), y1: m.t, y2: m.t + ih }), to);
    c.hline = (y, cls = 'faint', to) => add(TT.svg('line', { class: 'ln thin ' + cls, x1: m.l, x2: m.l + iw, y1: c.Y(y), y2: c.Y(y) }), to);
    c.dot = (x, y, r = 4, cls = 'f-flown', to) => add(TT.svg('circle', { class: cls + ' ring', cx: c.X(x), cy: c.Y(y), r }), to);
    c.text = (x, y, s, cls = 'lbl', anchor = 'start', to, dx = 0, dy = 0) => add(TT.svg('text', { class: cls, x: c.X(x) + dx, y: c.Y(y) + dy, 'text-anchor': anchor, text: s }), to);
    c.px = (x, y, s, cls = 'lbl', anchor = 'start', to) => add(TT.svg('text', { class: cls, x, y, 'text-anchor': anchor, text: s }), to);
    return c;
  };

  // rebuild the static layers (grid, axes, ticks) of a chart after its domain changed
  TT.reaxes = (c) => {
    c.svg.querySelectorAll(':scope > .grid, :scope > .axis, :scope > .tick').forEach(n => n.remove());
    const g = TT.svg('g', { class: 'grid' }), ax = TT.svg('g', { class: 'axis' }), tk = TT.svg('g', { class: 'tick' });
    TT.nice(c.x0, c.x1, Math.max(3, Math.round(c.iw / 90))).forEach(v => { g.append(TT.svg('line', { x1: c.X(v), x2: c.X(v), y1: c.m.t, y2: c.m.t + c.ih })); tk.append(TT.svg('text', { x: c.X(v), y: c.m.t + c.ih + 15, 'text-anchor': 'middle', text: String(v) })); });
    TT.nice(c.y0, c.y1, Math.max(3, Math.round(c.ih / 55))).forEach(v => { g.append(TT.svg('line', { x1: c.m.l, x2: c.m.l + c.iw, y1: c.Y(v), y2: c.Y(v) })); tk.append(TT.svg('text', { x: c.m.l - 6, y: c.Y(v) + 3.5, 'text-anchor': 'end', text: String(v) })); });
    ax.append(TT.svg('line', { x1: c.m.l, x2: c.m.l + c.iw, y1: c.m.t + c.ih, y2: c.m.t + c.ih }), TT.svg('line', { x1: c.m.l, x2: c.m.l, y1: c.m.t, y2: c.m.t + c.ih }));
    c.svg.insertBefore(tk, c.svg.firstChild); c.svg.insertBefore(ax, c.svg.firstChild); c.svg.insertBefore(g, c.svg.firstChild);
  };

  // ---------- controls ----------
  TT.ui = {};
  TT.ui.slider = (parent, o) => {
    const v = TT.el('span', { class: 'v' });
    const input = TT.el('input', { type: 'range', min: o.min, max: o.max, step: o.step || 1, value: o.value });
    const lab = TT.el('label', { class: 'ctl' }, [o.label, input, v]);
    const show = () => { v.textContent = (o.fmt || String)(+input.value); };
    input.addEventListener('input', () => { show(); o.onInput && o.onInput(+input.value); });
    show(); parent.append(lab);
    return { el: lab, input, get: () => +input.value, set: (x, fire = true) => { input.value = x; show(); if (fire && o.onInput) o.onInput(+input.value); } };
  };
  TT.ui.seg = (parent, options, value, onChange) => {
    const box = TT.el('span', { class: 'seg' });
    const btns = options.map(opt => { const [k, label] = Array.isArray(opt) ? opt : [opt, opt]; const b = TT.el('button', { type: 'button', 'aria-pressed': String(k === value), text: label }); b.dataset.k = k; box.append(b); return b; });
    let cur = value;
    box.addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; cur = b.dataset.k; btns.forEach(x => x.setAttribute('aria-pressed', String(x === b))); onChange(isNaN(+cur) || cur === '' ? cur : +cur); });
    parent.append(box);
    return { el: box, get: () => (isNaN(+cur) ? cur : +cur), set: (k) => { cur = String(k); btns.forEach(x => x.setAttribute('aria-pressed', String(x.dataset.k === cur))); } };
  };
  TT.ui.check = (parent, label, value, onChange) => {
    const input = TT.el('input', { type: 'checkbox' }); input.checked = value;
    input.addEventListener('change', () => onChange(input.checked));
    parent.append(TT.el('label', { class: 'ctl' }, [input, label]));
    return { get: () => input.checked, set: (v) => { input.checked = v; } };
  };
  TT.ui.button = (parent, label, onClick) => { const b = TT.el('button', { type: 'button', text: label, onclick: onClick }); parent.append(b); return b; };
  TT.ui.select = (parent, label, options, value, onChange) => {
    const sel = TT.el('select', {}, options.map(o => { const [k, t] = Array.isArray(o) ? o : [o, o]; const op = TT.el('option', { value: k, text: t }); if (k === value) op.selected = true; return op; }));
    sel.addEventListener('change', () => onChange(sel.value));
    parent.append(TT.el('label', { class: 'ctl' }, [label, sel]));
    return { get: () => sel.value, set: (v) => { sel.value = v; } };
  };
  TT.ui.out = (parent) => { const d = TT.el('div', { class: 'out' }); parent.append(d); return (html) => { d.innerHTML = html; }; };
  TT.ui.legend = (parent, items) => parent.append(TT.el('div', { class: 'legend' }, items.map(([cls, text, kind]) => TT.el('span', {}, [TT.el('i', { class: kind || '', style: `border-color:var(--c-${cls});background:${kind === 'sq' ? `var(--c-${cls})` : 'none'}` }), text]))));
  TT.ui.row = (parent) => { const r = TT.el('div', { class: 'row' }); parent.append(r); return r; };

  // drag helper for SVG elements: onMove(x,y) in chart data units
  TT.drag = (c, node, onMove) => {
    node.classList.add('drag');
    const pt = (ev) => { const r = c.svg.getBoundingClientRect(); const k = c.w / r.width; return [(ev.clientX - r.left) * k, (ev.clientY - r.top) * k]; };
    node.addEventListener('pointerdown', (ev) => { node.setPointerCapture(ev.pointerId); ev.preventDefault();
      const mv = (e2) => { const [px, py] = pt(e2); onMove(c.iX(px), c.iY(py)); };
      const up = () => { node.removeEventListener('pointermove', mv); node.removeEventListener('pointerup', up); };
      node.addEventListener('pointermove', mv); node.addEventListener('pointerup', up); });
  };

  // ---------- demo shell ----------
  TT.shell = (root, title) => {
    // the demo root has data-title; the content goes in a body div
    root.classList.add('demo');
    root.replaceChildren(TT.el('div', { class: 'dh' }, [TT.el('span', { class: 'tag', text: 'Try it' }), TT.el('b', { text: title || root.dataset.title || '' })]));
    const body = TT.el('div'); root.append(body); return body;
  };

  // ---------- tooltip ----------
  const tip = TT.el('div', { class: 'tip' }); document.body.append(tip);
  document.addEventListener('pointermove', (e) => {
    const t = e.target.closest && e.target.closest('[data-tip]');
    if (!t) { tip.style.display = 'none'; return; }
    tip.textContent = t.getAttribute('data-tip'); tip.style.display = 'block';
    tip.style.left = Math.min(innerWidth - 300, e.clientX + 14) + 'px'; tip.style.top = (e.clientY + 16) + 'px';
  });

  // ---------- boot ----------
  TT.boot = () => {
    if (TT.buildFigures) TT.buildFigures();
    // Chinese gloss: the first use of each term shows it in line; later uses show it on hover
    const seen = new Set();
    document.querySelectorAll('t[zh]').forEach(t => { const z = t.getAttribute('zh'); if (!seen.has(z)) { seen.add(z); t.classList.add('first'); } });
    // table of contents
    const toc = document.querySelector('nav.toc .items');
    const heads = [...document.querySelectorAll('main h2[id], main h3[id].toc')];
    heads.forEach(h => toc.append(TT.el('a', { href: '#' + h.id, class: h.tagName === 'H3' ? 'sub' : '', text: h.dataset.toc || h.textContent })));
    const links = [...toc.querySelectorAll('a')];
    const io = new IntersectionObserver((es) => es.forEach(e => { if (e.isIntersecting) { links.forEach(l => l.classList.toggle('on', l.getAttribute('href') === '#' + e.target.id)); } }), { rootMargin: '-10% 0px -80% 0px' });
    heads.forEach(h => io.observe(h));
    // demos mount when they come near the screen
    const mount = (el) => { const f = TT.demos[el.dataset.demo]; if (!f) { el.textContent = 'missing demo: ' + el.dataset.demo; return; } try { f(el, TT.shell(el)); el.dataset.mounted = '1'; } catch (err) { console.error('demo ' + el.dataset.demo, err); el.append(TT.el('div', { class: 'out bad-t', text: 'demo error: ' + err.message })); } };
    const dio = new IntersectionObserver((es) => es.forEach(e => { if (e.isIntersecting) { dio.unobserve(e.target); mount(e.target); } }), { rootMargin: '600px 0px' });
    document.querySelectorAll('[data-demo]').forEach(el => dio.observe(el));
    TT.mountAll = () => document.querySelectorAll('[data-demo]:not([data-mounted])').forEach(mount);
    // theme and language toggles
    const body = document.body;
    document.getElementById('btn-theme').addEventListener('click', () => { const cur = document.documentElement.getAttribute('data-theme') || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'); document.documentElement.setAttribute('data-theme', cur === 'dark' ? 'light' : 'dark'); });
    document.getElementById('btn-zh').addEventListener('click', (e) => { body.classList.toggle('nozh'); e.target.setAttribute('aria-pressed', String(!body.classList.contains('nozh'))); });
  };
  document.addEventListener('DOMContentLoaded', () => { TT.data = JSON.parse(document.getElementById('tutorial-data').textContent); TT.boot(); });
})();
