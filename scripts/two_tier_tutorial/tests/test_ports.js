// Compare the JavaScript ports with the Python on the vectors that make_vectors.py wrote.
global.window = global;
const path = require('path'), fs = require('fs');
require(path.join(__dirname, '../src/core.js'));
require(path.join(__dirname, '../src/words.js'));
const v = JSON.parse(fs.readFileSync(path.join(__dirname, 'vectors.json'), 'utf8'));
const TT = global.TT, W = TT.W;
let bad = 0, n = 0;
const fail = (what, info) => { bad++; if (bad < 12) console.log('MISMATCH', what, JSON.stringify(info).slice(0, 300)); };

// vocabulary tables
n++; if (JSON.stringify(W.levels) !== JSON.stringify(v.levels)) fail('levels', [W.levels.length, v.levels.length]);
n++; if (W.tol.some((t, i) => Math.abs(t - v.tol[i]) > 1e-9)) fail('tolerances', 0);
n++; if (JSON.stringify(W.counts) !== JSON.stringify(v.counts)) fail('counts', [W.counts, v.counts]);

for (const c of v.rules) {
  n++;
  const r = TT.rules(W, c.first, c.in[0], c.in[1], c.in[2], c.in[3], c.step[0], c.step[1], c.step[2], c.step[3], c.step[4], c.height, c.n);
  if (r.code !== c.code) { fail('rules code', [c, r.code]); continue; }
  if (c.code === 0 && (r.runway !== c.after[0] || r.go !== c.after[1] || r.altitude !== c.after[2] || r.angle !== c.after[3])) fail('rules after', [c, r]);
}
for (const c of v.masks) {
  n++;
  const inForce = c.in && { runway: c.in[0], go: c.in[1], heading: c.in[2], altitude: c.in[3], angle: c.in[4], speed: c.in[5] };
  const permitted = {}; for (const [k, ws] of Object.entries(c.permitted)) permitted[+k] = ws;
  const got = TT.columnMask(W, inForce, c.said, c.column, c.height, c.n, permitted);
  if (JSON.stringify(got) !== JSON.stringify(c.mask)) fail('column mask', [c.column, c.n, got.length, c.mask.length]);
}
for (const c of v.reads) {
  n++;
  const got = TT.perStepWords(c.track, c.course, 5, c.lead);
  if (got.length !== c.words.length || got.some((g, i) => g[0] !== c.words[i][0] || Math.abs(g[1] - c.words[i][1]) > 1e-9)) fail('per-step words', [got.length, c.words.length]);
}
for (const c of v.rates) {
  n++;
  const r = TT.wordRate(c.e, c.to_go, c.v, { cycle: 1, rollRate: 5, rmax: v.rmax });
  if (Math.abs(r.rate - c.rate) > 1e-4 * Math.max(1, Math.abs(c.rate)) || Math.abs(r.parts.stopping - c.stop) > 1e-4 * Math.max(1, c.stop)) fail('word rate', [c, r.rate, r.parts.stopping]);
}
console.log(`${n} checks, ${bad} mismatches`);
process.exit(bad ? 1 : 0);
