/* A kinematic sketch of the executor (autopilot/): the same lateral, vertical and speed laws, one 1 s cycle at a time.
   The real executor integrates the point-mass equations with RK4 and a drag polar; here the aircraft is a point that
   turns, climbs and accelerates at the rates the laws ask for, within the bank limit and the roll rate. */
(function () {
  const root = typeof window !== 'undefined' ? window : globalThis;
  const TT = root.TT = root.TT || {};
  const S = () => TT.SPEC, W = () => TT.W;

  // attitude (autopilot/inverse.py): wanted track rate and path-angle rate -> bank and load factor, with the limits
  TT.attitude = (V, gamma, trackRateDegS, gammaRate, prevBank, o = {}) => {
    const cap = TT.rad(o.bankMax || S().bankMax), roll = TT.rad(o.rollRate || S().rollRate) * (o.cycle || 1);
    const A = -TT.rad(trackRateDegS) * V * Math.cos(gamma) / TT.G;
    const B = gammaRate * V / TT.G + Math.cos(gamma);
    const wanted = Math.atan2(A, Math.max(B, 0));
    const capped = Math.max(-cap, Math.min(cap, wanted));
    const bank = Math.min(Math.max(capped, prevBank - roll), prevBank + roll);
    const load = B / Math.cos(bank), flown = Math.max(0.5, Math.min(2.0, load));
    return { bank, load: flown, bound: { 'bank limit': capped !== wanted, 'roll rate': bank !== capped, 'load factor': flown !== load } };
  };

  // the vertical law (autopilot/vertical.py)
  TT.verticalLaw = (st, V, gamma, h, w, o = {}) => {
    const sp = S(), Wd = W();
    const tol = Math.min(...Wd.tol), steepestLow = TT.rad(sp.descentEdges[sp.descentEdges.length - 2]);
    const rateMax = sp.pathRateFactor * V * steepestLow ** 2 / (2 * tol);
    const climbRad = o.goAround ? TT.rad(o.goAroundDeg || 3) : TT.rad(sp.climbCentre);
    const descent = w.angle >= 1 && w.angle <= Wd.nDescent, climbing = w.angle === Wd.angleClimb;
    const newWord = st.issued !== w.id;
    if (newWord) { st.captured = false; st.issued = w.id; }
    const nominal = o.goAround && climbing ? -climbRad : TT.rad(Wd.angleDeg(w.angle)); // descending positive
    const toGo = h - (w.level == null ? NaN : w.level);
    const levelOff = V * nominal * nominal / (2 * rateMax);
    const moving = !w.noLevelOff && w.angle !== 0 && !st.captured;
    const reached = moving && ((nominal > 0 ? toGo : -toGo) <= levelOff);
    st.captured = st.captured || reached || (!w.noLevelOff && w.angle === 0);
    const hold = Math.min(Math.max(-toGo / (V * 4 * sp.pathTau), -TT.rad(Math.max(...sp.descentCentres))), climbRad);
    const ref = st.captured && !w.noLevelOff ? hold : -nominal;
    const wanted = (ref - gamma) / sp.pathTau;
    return { rate: Math.max(-rateMax, Math.min(rateMax, wanted)), wanted, rateMax, captured: st.captured, levelOff, ref };
  };

  // the speed law (autopilot/speed.py), without the stall floor
  TT.speedLaw = (V, gamma, w, o = {}) => {
    const sp = S();
    const unspec = w.speed == null;
    const wanted = unspec ? (o.approachV || 70) : w.speed / Math.cos(gamma);
    const landing = Math.max(sp.aUnspec, Math.min(sp.aMax, (V * V - wanted * wanted) / (2 * Math.max(o.straightM || 1e9, 1))));
    const rising = unspec ? sp.aUnspec : sp.aMax;
    const slowing = unspec ? landing : sp.aMax;
    const faster = wanted > V, tau = sp.speedTol / (faster ? rising : slowing);
    return { rate: Math.min(Math.max((wanted - V) / tau, -slowing), rising), wanted };
  };

  // The executor as a stepper: say(words) when a row is heard, cycle() flies one 1 s cycle.
  // init: {e,n,track,V,h,gamma}; opts: {level0, angle0, speed0, goAround, goAroundDeg, approachV, straightM}
  TT.Sim = class {
    constructor(init, o = {}) {
      this.o = o; this.dt = S().cycle || 1; this.t = 0;
      this.st = { e: init.e, n: init.n, track: init.track, V: init.V, h: init.h, gamma: init.gamma || 0, bank: 0 };
      this.lat = { word: null, target: init.track, tracku: init.track, heard: 0, last: init.track };
      this.vert = { captured: false, issued: -1 }; this.id = 0;
      this.words = { level: o.level0 ?? null, angle: o.angle0 ?? 0, noLevelOff: false, speed: o.speed0 ?? init.V, id: 0 };
    }
    say(e) {
      const { st, lat } = this;
      if (e.heading) {
        const abs = e.heading.abs != null ? TT.wrap360(e.heading.abs) : TT.wrap360(e.heading.course + e.heading.rel);
        if (lat.word == null) lat.target = st.track + TT.wrap180(abs - st.track); else lat.target = lat.target + TT.wrap180(abs - lat.word);
        lat.word = abs; lat.heard = this.t;
      }
      if ('level' in e || 'angle' in e || 'noLevelOff' in e) {
        const w = this.words;
        this.words = Object.assign({}, w, { level: 'level' in e ? e.level : w.level, angle: 'angle' in e ? e.angle : w.angle, noLevelOff: 'noLevelOff' in e ? e.noLevelOff : (w.noLevelOff && !('level' in e)), id: ++this.id });
      }
      if ('speed' in e) this.words = Object.assign({}, this.words, { speed: e.speed });
    }
    cycle() {
      const sp = S(), { st, lat, vert, words, o, dt, t } = this;
      lat.tracku += TT.wrap180(st.track - lat.last); lat.last = st.track;
      const err = lat.target - lat.tracku, toGo = lat.heard + sp.headingLead - t;
      const lr = TT.wordRate(err, toGo, st.V, { cycle: dt, rollRate: sp.rollRate, rmax: sp.turnRateMax });
      const vr = TT.verticalLaw(vert, st.V, st.gamma, st.h, words, o);
      const sr = TT.speedLaw(st.V, st.gamma, words, o);
      const at = TT.attitude(st.V, st.gamma, lr.rate, vr.rate, st.bank, { cycle: dt });
      const trackRate = TT.deg(-TT.G * at.load * Math.sin(at.bank) / (st.V * Math.cos(st.gamma)));
      const rec = { t, e: st.e, n: st.n, track: st.track, V: st.V, h: st.h, gamma: st.gamma, bank: at.bank, load: at.load, err, toGo, lat: lr, vert: vr, spd: sr, att: at, words, wordTrack: lat.word, trackRate };
      const track2 = st.track + trackRate * dt, g2 = st.gamma + vr.rate * dt, V2 = Math.max(20, st.V + sr.rate * dt);
      const tm = TT.rad((st.track + track2) / 2), gm = (st.gamma + g2) / 2, Vm = (st.V + V2) / 2;
      st.e += Vm * Math.cos(gm) * Math.sin(tm) * dt; st.n += Vm * Math.cos(gm) * Math.cos(tm) * dt; st.h += Vm * Math.sin(gm) * dt;
      st.track = TT.wrap360(track2); st.gamma = g2; st.V = V2; st.bank = at.bank; this.t += dt;
      return rec;
    }
  };

  // fly a schedule of word events: [{t, heading:{rel,course}|{abs}, level, angle, noLevelOff, speed}] -> one sample per cycle
  TT.fly = (init, events, T, o = {}) => {
    const sim = new TT.Sim(init, o), out = [];
    events = events.slice().sort((a, b) => a.t - b.t); let ev = 0;
    for (let t = 0; t <= T + 1e-9; t += sim.dt) {
      while (ev < events.length && events[ev].t <= t + 1e-9) sim.say(events[ev++]);
      out.push(sim.cycle());
    }
    return out;
  };
})();
