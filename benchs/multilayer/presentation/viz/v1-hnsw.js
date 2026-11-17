/* V1 · HNSW refresher: stacked layers, build (random levels) and search (greedy on top, beam on layer 0).
   Classic script. Real (small) HNSW on seeded toy 2-D data. Colours only via the deck's CSS variables. */
(function () {
'use strict';
var NS = 'http://www.w3.org/2000/svg';
var CFG = { N: 30, M: 3, EFC: 16, SEED: 314, K: 3, MIND: 54 };
var DW = 550, DD = 220;                    // data domain (stage px along the plane)
var X0 = 82, SH = 170, PH = 140;          // plane origin x, shear (depth to the right), projected depth
var BASE0 = 550, GAP = 200, NL = 3;        // layer 0 front edge y, distance between planes, number of planes
var QDEF = { x: 260, y: 130 };

function rng(seed) {
  return function () {
    seed = (seed + 0x6D2B79F5) | 0;
    var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ---------- model: data + HNSW construction (paper Alg. 1: each new point links to up to M (layer 0: 2M) neighbours
   chosen with the heuristic of Alg. 4, in both directions; links are kept, never pruned, so the drawing only grows) ---------- */
function makeModel(seed, N, M, efc) {
  var r = rng(seed), pts = [], tries = 0, i, l, lc;
  while (pts.length < N && tries < 50000) {
    tries++;
    var x = 20 + r() * (DW - 40), y = 16 + r() * (DD - 32), ok = true;
    for (i = 0; i < pts.length; i++) if ((pts[i].x - x) * (pts[i].x - x) + (pts[i].y - y) * (pts[i].y - y) < CFG.MIND * CFG.MIND) { ok = false; break; }
    if (ok) pts.push({ x: x, y: y });
  }
  var n = pts.length, mL = 1 / Math.log(M);
  var us = [], lvl = pts.map(function () { var u = r(); if (u <= 0) u = 1e-12; us.push(u); return Math.min(NL - 1, Math.floor(-Math.log(u) * mL)); });
  var adj = [], ep = -1, L = -1, rec = [];
  function dd(a, b) { return Math.hypot(pts[a].x - b.x, pts[a].y - b.y); }
  function searchLayer(dq, eps, ef, l) {
    var vis = new Set(eps), C = eps.map(function (e) { return [dq(e), e]; }), W = C.slice();
    var by = function (a, b) { return a[0] - b[0]; };
    while (C.length) {
      C.sort(by); var c = C.shift(); W.sort(by);
      if (c[0] > W[W.length - 1][0]) break;
      var nb = adj[l][c[1]];
      for (var j = 0; j < nb.length; j++) {
        var e = nb[j]; if (vis.has(e)) continue; vis.add(e);
        var de = dq(e); W.sort(by);
        if (W.length < ef || de < W[W.length - 1][0]) { C.push([de, e]); W.push([de, e]); W.sort(by); if (W.length > ef) W.pop(); }
      }
    }
    W.sort(by); return W.map(function (w) { return w[1]; });
  }
  // keep a candidate only if it is closer to the base point than to every neighbour kept so far
  function shrink(b, cands, mx) {
    var pb = pts[b], c = cands.slice().sort(function (x, y) { return dd(x, pb) - dd(y, pb); }), out = [];
    for (var k = 0; k < c.length && out.length < mx; k++) {
      var v = c[k], dv = dd(v, pb), good = true;
      for (var j = 0; j < out.length; j++) if (dd(v, pts[out[j]]) < dv) { good = false; break; }
      if (good) out.push(v);
    }
    return out;
  }
  // add a directed link; links are never removed, so every edge drawn stays (no shrinking of full lists)
  function addLink(lc, src, dst, R) {
    if (adj[lc][src].indexOf(dst) >= 0) return;
    adj[lc][src].push(dst); R.ops.push([lc, src, dst]);
  }
  for (i = 0; i < n; i++) {
    l = lvl[i];
    var R = { id: i, level: l, ops: [], newTop: false };
    for (lc = 0; lc <= l; lc++) { if (!adj[lc]) adj[lc] = []; adj[lc][i] = []; }
    if (ep < 0) { ep = i; L = l; R.newTop = true; rec.push(R); continue; }
    var p = pts[i], dq = function (e) { return dd(e, p); }, cur = [ep];
    for (lc = L; lc > l; lc--) cur = [searchLayer(dq, cur, 1, lc)[0]];
    for (lc = Math.min(L, l); lc >= 0; lc--) {
      var mx = lc === 0 ? 2 * M : M, Wc = searchLayer(dq, cur, efc, lc);
      // neighbours of the new point: heuristic selection (paper Alg. 4, Faiss shrink_neighbor_list), up to mx
      var nb = shrink(i, Wc, mx);
      nb.forEach(function (e) { addLink(lc, i, e, R); addLink(lc, e, i, R); });
      cur = Wc;
    }
    if (l > L) { ep = i; L = l; R.newTop = true; }
    rec.push(R);
  }
  return { pts: pts, lvl: lvl, u: us, adj: adj, ep: ep, L: L, rec: rec, n: n };
}

/* ---------- query: records every step for the animation (paper Alg. 5, Alg. 2 on layer 0) ---------- */
function runSearch(md, q, ef, K) {
  var count = 0, ev = [], adj = md.adj;
  function dq(e) { count++; return Math.hypot(md.pts[e].x - q.x, md.pts[e].y - q.y); }
  var cur = md.ep, dc = dq(cur);
  ev.push({ t: 'entry', l: md.L, id: cur, n: count });
  for (var l = md.L; l >= 1; l--) {
    var vis = new Set([cur]);
    for (;;) {
      var best = cur, bd = dc, evals = [];
      adj[l][cur].forEach(function (e) { if (vis.has(e)) return; vis.add(e); var de = dq(e); evals.push(e); if (de < bd) { bd = de; best = e; } });
      if (best === cur) { ev.push({ t: 'look', l: l, id: cur, evals: evals, n: count }); break; }
      ev.push({ t: 'hop', l: l, from: cur, to: best, evals: evals, n: count }); cur = best; dc = bd;
    }
    ev.push({ t: 'min', l: l, id: cur, n: count });
  }
  var by = function (a, b) { return a[0] - b[0]; };
  var C = [[dc, cur]], W = [[dc, cur]], v0 = new Set([cur]);
  ev.push({ t: 'start0', id: cur, n: count });
  while (C.length) {
    C.sort(by); var c = C.shift(); W.sort(by);
    if (c[0] > W[W.length - 1][0]) { ev.push({ t: 'stop', c: c[1], n: count }); break; }
    var list = [];
    adj[0][c[1]].forEach(function (e) {
      if (v0.has(e)) return; v0.add(e);
      var de = dq(e), add = false, out = null; W.sort(by);
      if (W.length < ef || de < W[W.length - 1][0]) { add = true; C.push([de, e]); W.push([de, e]); W.sort(by); if (W.length > ef) out = W.pop()[1]; }
      list.push({ e: e, add: add, out: out });
    });
    ev.push({ t: 'expand', c: c[1], list: list, n: count });
  }
  if (!C.length && ev[ev.length - 1].t !== 'stop') ev.push({ t: 'stop', c: null, n: count });
  W.sort(by);
  var res = W.slice(0, K).map(function (w) { return w[1]; });
  var truth = md.pts.map(function (p, i) { return [Math.hypot(p.x - q.x, p.y - q.y), i]; }).sort(by).slice(0, K).map(function (w) { return w[1]; });
  return { ev: ev, res: res, truth: truth, count: count };
}

/* ---------- timeline: timed events + tweens; marks split it into steps; finish() jumps to the end ---------- */
function Timeline(view) {
  this.ev = []; this.act = []; this.i = 0; this.raf = 0; this.seq = 0; this.view = view; this.dead = false;
  this.pos = 0; this.marks = []; this.sorted = false;
}
Timeline.prototype.at = function (t, fn) { this.ev.push({ t: t, s: this.seq++, fn: fn }); this.sorted = false; };
Timeline.prototype.mark = function (t) { this.marks.push(t); };
Timeline.prototype.tween = function (t, dur, fn) {
  var self = this, tw = { t0: t, dur: dur, fn: fn };
  this.at(t, function () { self.act.push(tw); fn(0); });
  this.at(t + dur, function () { fn(1); self.act.splice(self.act.indexOf(tw), 1); });
};
Timeline.prototype._sort = function () {
  if (this.sorted) return; this.sorted = true;
  this.ev.sort(function (a, b) { return a.t - b.t || a.s - b.s; });
  this.marks.sort(function (a, b) { return a - b; });
};
Timeline.prototype._adv = function (el) {
  this._sort();
  while (this.i < this.ev.length && this.ev[this.i].t <= el) this.ev[this.i++].fn();
  this.act.forEach(function (tw) { tw.fn(Math.max(0, Math.min(1, (el - tw.t0) / tw.dur))); });
  this.pos = Math.max(this.pos, Math.min(el, 1e12));
  if (this.i >= this.ev.length && !this.dead) { this.dead = true; if (this.done) this.done(); }
};
Timeline.prototype._run = function (stopAt, rate) {
  var self = this, t0 = performance.now(), p0 = this.pos;
  this.pause();
  function frame(now) {
    if (self.dead) return;
    var el = Math.min(stopAt, p0 + (now - t0) * rate);
    self._adv(el);
    if (self.dead) return;
    if (el >= stopAt) { self.raf = 0; return; }
    self.raf = requestAnimationFrame(frame);
  }
  this.raf = requestAnimationFrame(frame);
};
Timeline.prototype.play = function (done) { this.done = done; this._sort(); this._run(Infinity, 1); };
Timeline.prototype.pause = function () { cancelAnimationFrame(this.raf); this.raf = 0; };
Timeline.prototype.playing = function () { return !this.dead && this.raf !== 0; };
Timeline.prototype.stepNext = function (animate, rate) {
  if (this.dead) return;
  this.pause(); this._sort();
  var pos = this.pos, next = Infinity;
  for (var k = 0; k < this.marks.length; k++) if (this.marks[k] > pos + 0.5) { next = this.marks[k]; break; }
  if (animate) { this._run(next, rate || 1); return; }
  this.view.fast = true; this._adv(next); this.view.fast = false;
};
Timeline.prototype.finish = function () {
  if (this.dead) return;
  this.pause(); this.view.fast = true; this._adv(Infinity); this.view.fast = false;
};
Timeline.prototype.cancel = function () { this.dead = true; this.pause(); };

/* ---------- view ---------- */
function mount(el, api) {
  var md = makeModel(CFG.SEED, CFG.N, CFG.M, CFG.EFC), N = md.n, M = CFG.M;
  var view = { fast: false };
  var reduced = function () { return window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches; };
  function S(tag, at, par) {
    var e = document.createElementNS(NS, tag);
    for (var k in at) if (at[k] != null) e.setAttribute(k, at[k]);
    if (par) par.appendChild(e); return e;
  }
  function H(tag, at, par, txt) {
    var e = document.createElement(tag);
    for (var k in at) e.setAttribute(k, at[k]);
    if (txt != null) e.textContent = txt;
    if (par) par.appendChild(e); return e;
  }
  function P(i, l) { var p = md.pts[i]; return proj(p.x, p.y, l); }
  function proj(x, y, l) { var v = y / DD; return { x: X0 + x + v * SH, y: BASE0 - l * GAP - v * PH }; }
  function anim(e, kf, ms) { if (!view.fast && e.animate) e.animate(kf, { duration: ms, easing: 'ease-out' }); }
  var fadeIn = function (e, ms) { anim(e, [{ opacity: 0 }, { opacity: 1 }], ms || 300); };
  var pulse = function (e) { anim(e, [{ transform: 'scale(2.4)' }, { transform: 'scale(1)' }], 380); };

  el.innerHTML = '';
  el.classList.add('v1h');
  /* level draw (build step only): u ~ Uniform(0, 1) on a segment; u < 1/M reaches layer 1, u < 1/M^2 layer 2 */
  var UW = 340, ud = document.createElement('div');
  ud.style.cssText = 'position:absolute;left:-382px;top:410px;width:' + UW + 'px;display:none';
  el.appendChild(ud);
  var usv = document.createElementNS(NS, 'svg'); usv.setAttribute('width', UW); usv.setAttribute('height', 110);
  usv.setAttribute('viewBox', '-8 0 ' + (UW + 16) + ' 110'); usv.style.overflow = 'visible'; ud.appendChild(usv);
  function US(tag, at, txt) { var e = document.createElementNS(NS, tag); for (var k in at) e.setAttribute(k, at[k]); if (txt != null) e.textContent = txt; usv.appendChild(e); return e; }
  var LC = ['var(--l0)', 'var(--l1)', 'var(--l2)'], cut = [1 / (CFG.M * CFG.M), 1 / CFG.M];
  US('text', { x: 0, y: 14, 'font-size': 18, 'font-weight': 600, fill: 'var(--ink)' }, 'Number of neighbours: M = ' + CFG.M);
  [[0, cut[0], 2], [cut[0], cut[1], 1], [cut[1], 1, 0]].forEach(function (z) {
    US('rect', { x: z[0] * UW, y: 30, width: (z[1] - z[0]) * UW, height: 20, fill: LC[z[2]], 'fill-opacity': 0.35, stroke: LC[z[2]], 'stroke-width': 1.5 });
  });
  [[0, '0'], [cut[0], '1/M²'], [cut[1], '1/M'], [1, '1']].forEach(function (k) {
    US('text', { x: k[0] * UW, y: 68, 'font-size': 13, 'text-anchor': 'middle', fill: 'var(--mute)', 'font-family': 'var(--mono)' }, k[1]);
  });
  var uLine = US('line', { x1: 0, x2: 0, y1: 22, y2: 58, stroke: 'var(--ink)', 'stroke-width': 3, 'stroke-linecap': 'round', visibility: 'hidden' });
  var uTxt = US('text', { x: 0, y: 96, 'font-size': 18, 'font-weight': 600, 'text-anchor': 'middle', visibility: 'hidden' }, '');
  function showDraw(i, lv) {
    if (i < 0) { uLine.setAttribute('visibility', 'hidden'); uTxt.setAttribute('visibility', 'hidden'); return; }
    var x = md.u[i] * UW;
    uLine.setAttribute('x1', x); uLine.setAttribute('x2', x); uLine.setAttribute('visibility', 'visible');
    uTxt.setAttribute('x', Math.max(30, Math.min(UW - 30, x))); uTxt.setAttribute('fill', LC[lv]);
    uTxt.textContent = 'layer ' + lv; uTxt.setAttribute('visibility', 'visible');
  }
  var css = H('style', {}, el);
  css.textContent =
    '.v1h svg{position:absolute;left:0;top:0;overflow:visible}' +
    '.v1h .nd{transform-box:fill-box;transform-origin:center}' +
    '.v1h .lbl{font-size:16px;font-weight:600}' +
    '.v1h .mono{font-family:var(--mono);font-weight:500}' +
    '.v1h .row{position:absolute;left:0;bottom:16px;height:40px;display:flex;align-items:center;gap:12px;white-space:nowrap}' +
    '.v1h .row .sep{width:4px}' +
    '.v1h .row .chip{font-size:14px}' +
    '.v1h .row label{display:flex;align-items:center;gap:8px;font-size:16px;color:var(--mute)}' +
    '.v1h .row input[type=range]{width:84px}' +
    '.v1h .clk{cursor:crosshair}' +
    '.v1h svg.srch .e0{opacity:.3}' +
    '.v1h svg.srch .n0 circle{opacity:.4}' +
    '.v1h svg.srch .n0 circle.hit{opacity:1}';

  var svg = S('svg', { width: 810, height: 566, viewBox: '0 0 810 566' }, el);
  var gPl = S('g', {}, svg), gV = S('g', {}, svg), gQline = S('g', {}, svg);
  var gE = [], gN = [], gO = [];
  for (var l = 0; l < NL; l++) { gE[l] = S('g', { class: 'e' + l }, svg); }
  var gOv = S('g', {}, svg);                         // search edges/paths, under nodes
  for (l = 0; l < NL; l++) { gN[l] = S('g', { class: 'n' + l }, svg); }
  for (l = 0; l < NL; l++) { gO[l] = S('g', {}, svg); } // rings on top of nodes
  var gQ = S('g', {}, svg), gTop = S('g', {}, svg);

  /* planes */
  var layerTip = [
    'Layer 0: every point is here, with short links (up to 2·M per node).',
    'Layer 1: about 1 in M points reach it, P(level ≥ 1) = 1/M. Fewer points, longer links.',
    'Top layer: few points, long links. Search starts here.'
  ];
  var plane0;
  for (l = 0; l < NL; l++) {
    var a = proj(0, 0, l), b = proj(DW, 0, l), c = proj(DW, DD, l), d = proj(0, DD, l);
    var pg = S('polygon', { points: [a, b, c, d].map(function (p) { return p.x + ',' + p.y; }).join(' '),
      fill: 'var(--l' + l + ')', 'fill-opacity': l === 0 ? 0.08 : 0.06, stroke: 'var(--l' + l + ')', 'stroke-opacity': 0.6, 'stroke-width': 1.3 }, gPl);
    if (l === 0) { plane0 = pg; pg.setAttribute('class', 'clk'); }
    var t = S('text', { x: X0 - 8, y: a.y - 4, 'text-anchor': 'end', class: 'lbl', fill: 'var(--l' + l + ')', 'data-explain': layerTip[l] }, gPl);
    t.textContent = 'layer ' + l;
  }

  /* nodes: a circle per (node, layer it may live on) */
  var nodeEl = [], vlink = [];
  for (var i = 0; i < N; i++) {
    nodeEl[i] = [];
    for (l = 0; l <= md.lvl[i]; l++) {
      var p = P(i, l);
      nodeEl[i][l] = S('circle', { cx: p.x, cy: p.y, r: l ? 6 : 5, class: 'nd' }, gN[l]);
    }
    if (md.lvl[i] > 0) { var p0 = P(i, 0); vlink[i] = S('line', { x1: p0.x, y1: p0.y, x2: p0.x, y2: p0.y, stroke: 'var(--mute)', 'stroke-opacity': 0.45, 'stroke-dasharray': '3 4', 'stroke-width': 1 }, gV); }
  }
  function nodeStyle(i, l, on) {
    var e = nodeEl[i][l]; e.style.fill = ''; e.classList.remove('hit'); e.setAttribute('r', l ? 6 : 5); e.style.stroke = ''; e.removeAttribute('stroke-width');
    if (l === 0 && !on) { e.setAttribute('fill', 'var(--panel)'); e.setAttribute('stroke', 'var(--mute)'); e.setAttribute('stroke-width', 1.3); e.style.display = ''; return; }
    e.setAttribute('fill', 'var(--l' + l + ')'); e.setAttribute('stroke', 'var(--paper)'); e.setAttribute('stroke-width', 1);
    e.style.display = on ? '' : 'none';
  }

  /* edges: one line per undirected pair, drawn the first time either direction is added and never removed */
  var edges = {}, inserted = 0, built = false, epNode = -1;
  function key(l, a, b) { return l + ':' + Math.min(a, b) + '-' + Math.max(a, b); }
  function applyOp(op) {
    var l = op[0], k = key(l, op[1], op[2]);
    if (edges[k]) return;
    var pa = P(op[1], l), pb = P(op[2], l);
    edges[k] = S('line', { x1: pa.x, y1: pa.y, x2: pb.x, y2: pb.y, stroke: 'var(--l' + l + ')', 'stroke-opacity': l ? 0.8 : 0.6, 'stroke-width': l ? 2 : 1.5 }, gE[l]);
    fadeIn(edges[k], 350);
  }

  /* entry-point ring (persistent once built) */
  var epRing = S('circle', { r: 11, fill: 'transparent', stroke: 'var(--ink)', 'stroke-width': 1.6, 'data-explain': 'Entry point: the node with the highest level so far. Every search starts here.' }, gTop);
  function setEp(i) {
    epNode = i;
    if (i < 0) { epRing.style.display = 'none'; return; }
    var p = P(i, md.lvl[i]); epRing.setAttribute('cx', p.x); epRing.setAttribute('cy', p.y); epRing.style.display = '';
  }

  /* a rising copy per point during the build */
  function riser(i) {
    var p = P(i, 0);
    return S('circle', { cx: p.x, cy: p.y, r: 6, fill: 'var(--hot)', stroke: 'var(--paper)', 'stroke-width': 1.2,
      'data-keep': 1, 'data-explain': 'Top layer drawn at random: l = ⌊−ln(U)·m_L⌋ with m_L = 1/ln M, so P(level ≥ l) = M^-l.' }, gTop);
  }

  /* query markers on every layer + true/returned rings */
  var q = { x: QDEF.x, y: QDEF.y }, qShown = false, qEls = [];
  for (l = 0; l < NL; l++) {
    qEls[l] = S('circle', { r: l ? 6 : 7.5, fill: l ? 'var(--paper)' : 'var(--q)', stroke: 'var(--q)', 'stroke-width': 2,
      'data-explain': l ? 'The query, projected onto this layer: greedy moves toward it.' : 'Query q. Click layer 0 to move it.' }, gQ);
  }
  var qLine = S('line', { stroke: 'var(--q)', 'stroke-opacity': 0.45, 'stroke-width': 1.2, 'stroke-dasharray': '2 5' }, gQline);
  var qLab0 = S('text', { class: 'mono', 'font-size': 16, fill: 'var(--q)' }, gQ); qLab0.textContent = 'q';
  function drawQ() {
    var disp = qShown ? '' : 'none';
    for (var l = 0; l < NL; l++) { var p = proj(q.x, q.y, l); qEls[l].setAttribute('cx', p.x); qEls[l].setAttribute('cy', p.y); qEls[l].style.display = (qShown && l <= Math.max(md.L, 0)) ? '' : 'none'; }
    var a = proj(q.x, q.y, 0), b = proj(q.x, q.y, md.L);
    qLine.setAttribute('x1', a.x); qLine.setAttribute('y1', a.y); qLine.setAttribute('x2', b.x); qLine.setAttribute('y2', b.y); qLine.style.display = disp;
    qLab0.setAttribute('x', a.x + 10); qLab0.setAttribute('y', a.y - 9); qLab0.style.display = disp;
  }

  /* controls */
  var row = H('div', { class: 'row' }, el);
  var bBuild = H('button', { class: 'ctl primary', 'data-explain': 'Each point draws a random top layer and rises to it; then each links to its nearest neighbours on every layer it lives on. Press again to skip.' }, row, 'Build');
  var bSearch = H('button', { class: 'ctl', 'data-explain': 'Greedy on the upper layers, efSearch-wide beam on layer 0. Press again to skip. Click layer 0 to move q.' }, row, 'Search');
  var bStep = H('button', { class: 'ctl', 'data-explain': 'One action per press. Before the graph is built: insert the next point (draw its level, rise, link). After: one greedy hop, one drop down a layer, or one layer-0 beam step. Pauses a running animation.' }, row, 'Step');
  H('span', { class: 'sep' }, row);
  var lab = H('label', { 'data-explain': 'efSearch: on layer 0, efSearch candidates are kept instead of one path.' }, row);
  H('span', {}, lab, 'efSearch');
  var slider = H('input', { type: 'range', min: 1, max: 16, step: 1, value: 4, 'aria-label': 'efSearch' }, lab);
  var efVal = H('span', { class: 'chip' }, lab, '4');
  var cDist = H('span', { class: 'chip', style: 'display:none', 'data-explain': 'Distance computations so far in this search (all layers).' }, row, '');
  var cRec = H('span', { class: 'chip', style: 'display:none', 'data-explain': 'How many of the 3 returned points are among the true 3 nearest (brute force).' }, row, '');
  function chips(dist, rec) {
    cDist.textContent = 'distance computations ' + (dist == null ? '–' : dist);
    cRec.textContent = 'recall ' + (rec == null ? '–' : rec + '/' + CFG.K);
  }

  /* ---------- states ---------- */
  var tl = null, phase = 'idle', searchShown = false, lastStep = -1;
  function stop() { if (tl) tl.cancel(); tl = null; phase = 'idle'; }
  function clearSearch() {
    [gOv].concat(gO).forEach(function (g) { while (g.firstChild) g.removeChild(g.firstChild); });
    for (var i = 0; i < N; i++) for (var l = 0; l <= md.lvl[i]; l++) nodeStyle(i, l, built || (l === 0 && i < inserted));
    epRing.setAttribute('stroke', 'var(--ink)'); svg.classList.remove('srch');
    searchShown = false; qShown = false; drawQ(); chips(null, null);
  }
  function toUnbuilt() {
    stop();
    Object.keys(edges).forEach(function (k) { edges[k].remove(); }); edges = {};
    inserted = 0; built = false; setEp(-1); showDraw(-1); gTop.querySelectorAll('.rz').forEach(function (r) { r.remove(); });
    for (var i = 0; i < N; i++) { if (vlink[i]) { vlink[i].style.display = 'none'; } }
    clearSearch();
  }
  function buildTimeline() {
    // one point after the other: its level is drawn and it rises to it, then its links appear
    var T = new Timeline(view), t = 0, RISE = 90;
    md.rec.forEach(function (R, k) {
      var i = R.id, lv = R.level, rz = null;
      T.at(t, function () {
        inserted = k + 1; nodeStyle(i, 0, true); pulse(nodeEl[i][0]); showDraw(i, lv);
        if (lv) { rz = riser(i); rz.setAttribute('class', 'rz'); vlink[i].style.display = ''; vlink[i].setAttribute('y2', P(i, 0).y); }
      });
      for (var l = 1; l <= lv; l++) (function (l) {
        T.tween(t + 30 + (l - 1) * RISE, RISE, function (u) {
          var a = P(i, l - 1), b = P(i, l), y = a.y + (b.y - a.y) * u;
          rz.setAttribute('cy', y); vlink[i].setAttribute('y2', y);
          if (u === 1) { nodeStyle(i, l, true); if (l === lv) rz.remove(); }
        });
      })(l);
      t += lv ? 30 + lv * RISE + 20 : 20;
      T.at(t, function () { R.ops.forEach(applyOp); if (R.newTop) setEp(i); });
      t += lv ? 60 : 30;
      T.mark(t - 1);
    });
    T.at(t + 100, function () { built = true; phase = 'idle'; });
    return T;
  }
  // mode: 'anim' | 'instant' | 'step'
  function playBuild(mode) {
    toUnbuilt(); phase = 'building';
    tl = buildTimeline();
    if (mode === 'step') tl.stepNext(!reduced(), 0.3);
    else { tl.play(); if (mode === 'instant') tl.finish(); }
  }
  function toBuilt() {
    if (phase === 'building' && tl) tl.finish();
    if (!built) playBuild('instant');
    stop(); clearSearch();
  }

  /* search visuals */
  function ring(l, i, r, stroke, w, tip, dash) {
    var p = P(i, l);
    return S('circle', { class: 'nd', cx: p.x, cy: p.y, r: r, fill: 'transparent', stroke: stroke, 'stroke-width': w, 'stroke-dasharray': dash || null, 'data-explain': tip || null }, gO[l]);
  }
  function seg(l1, a, l2, b, stroke, w, dash, tip) {
    var pa = P(a, l1), pb = P(b, l2);
    return S('line', { x1: pa.x, y1: pa.y, x2: pb.x, y2: pb.y, stroke: stroke, 'stroke-width': w, 'stroke-linecap': 'round', 'stroke-dasharray': dash || null, 'data-explain': tip || null }, gOv);
  }
  function grow(T, t, dur, line) {
    var x2 = +line.getAttribute('x2'), y2 = +line.getAttribute('y2'), x1 = +line.getAttribute('x1'), y1 = +line.getAttribute('y1');
    T.tween(t, dur, function (u) { line.setAttribute('x2', x1 + (x2 - x1) * u); line.setAttribute('y2', y1 + (y2 - y1) * u); });
    line.setAttribute('x2', x1); line.setAttribute('y2', y1);
  }
  function fill(i, l, v) { var e = nodeEl[i][l]; e.style.fill = v; e.classList.add('hit'); if (!l) e.setAttribute('r', 6.5); }

  function playSearch(mode) {
    toBuilt();
    var ef = +slider.value, S0 = runSearch(md, q, ef, CFG.K), T = new Timeline(view), t = 0;
    qShown = true; drawQ(); searchShown = true; phase = 'searching'; svg.classList.add('srch'); chips(0, null);
    var inC = new Set(), inW = new Set(), seen = new Set(), cur0 = null;
    var beamTip = 'Layer-0 beam: efSearch candidates are kept instead of one path (C: candidates, W: results).';
    function paint0(i) { fill(i, 0, inW.has(i) ? 'var(--res)' : inC.has(i) ? 'var(--cand)' : seen.has(i) ? 'var(--visit)' : ''); }
    S0.ev.forEach(function (e) {
      if (e.t === 'entry') {
        T.at(t, function () { epRing.setAttribute('stroke', 'var(--hot)'); var r = ring(e.l, e.id, 13, 'var(--hot)', 2.6, 'Search starts here, on the top layer.'); pulse(r); fill(e.id, e.l, 'var(--hot)'); chips(e.n, null); });
        t += 550; T.mark(t - 1);
      } else if (e.t === 'hop' || e.t === 'look') {
        T.at(t, function () { e.evals.forEach(function (x) { if (x !== (e.to)) fill(x, e.l, 'var(--visit)'); }); chips(e.n, null); });
        if (e.t === 'hop') {
          var ln = seg(e.l, e.from, e.l, e.to, 'var(--hot)', 3.2, null, 'Greedy hop: move to the neighbour closest to q.');
          ln.style.display = 'none';
          T.at(t + 120, function () { ln.style.display = ''; });
          grow(T, t + 120, 320, ln);
          T.at(t + 440, function () { fill(e.to, e.l, 'var(--hot)'); pulse(nodeEl[e.to][e.l]); });
          t += 520; T.mark(t - 1);
        } else t += 260;
      } else if (e.t === 'min') {
        var dl = seg(e.l, e.id, e.l - 1, e.id, 'var(--hot)', 2.4, '5 4', 'Local minimum becomes the entry point of the layer below.');
        dl.style.display = 'none';
        T.at(t, function () { var r = ring(e.l, e.id, 11, 'var(--hot)', 2.2, 'No neighbour is closer: greedy stops and goes down a layer.', '3 2'); pulse(r); dl.style.display = ''; });
        grow(T, t + 80, 360, dl);
        T.at(t + 440, function () { if (e.l - 1 > 0) { fill(e.id, e.l - 1, 'var(--hot)'); pulse(nodeEl[e.id][e.l - 1]); } });
        t += 560; if (e.l > 1) T.mark(t - 1);
      } else if (e.t === 'start0') {
        T.at(t, function () { inC.add(e.id); inW.add(e.id); seen.add(e.id); paint0(e.id); pulse(nodeEl[e.id][0]); chips(e.n, null); });
        t += 300; T.mark(t - 1);
      } else if (e.t === 'expand') {
        T.at(t, function () {
          inC.delete(e.c); paint0(e.c);
          if (!cur0) cur0 = ring(0, e.c, 10.5, 'var(--hot)', 2.4, beamTip);
          var p = P(e.c, 0); cur0.setAttribute('cx', p.x); cur0.setAttribute('cy', p.y); pulse(cur0);
        });
        T.at(t + 160, function () {
          e.list.forEach(function (x) {
            var ln = seg(0, e.c, 0, x.e, x.add ? 'var(--cand)' : 'var(--visit)', x.add ? 2.4 : 1.6, null, beamTip);
            fadeIn(ln, 250);
            seen.add(x.e); if (x.add) { inC.add(x.e); inW.add(x.e); } if (x.out != null) inW.delete(x.out);
            paint0(x.e); if (x.out != null) paint0(x.out);
          });
          chips(e.n, null);
        });
        t += e.list.length ? 480 : 260; T.mark(t - 1);
      } else if (e.t === 'stop') {
        T.at(t, function () { if (cur0) { cur0.setAttribute('stroke', 'var(--mute)'); cur0.setAttribute('data-explain', 'Nearest candidate is farther than the worst result: stop.'); } });
        t += 300;
      }
    });
    T.at(t, function () {
      if (cur0) { cur0.remove(); cur0 = null; }
      S0.truth.forEach(function (i) { ring(0, i, 14.5, 'var(--q)', 1.8, 'True 3 nearest to q (brute force): a gap between these and the green rings is a miss.', '3 2'); });
      var hit = 0;
      S0.res.forEach(function (i) { if (S0.truth.indexOf(i) >= 0) hit++; var r = ring(0, i, 10, 'var(--res)', 2.6, 'Returned: the 3 nearest in W.'); pulse(r); });
      chips(S0.count, hit); phase = 'idle';
    });
    tl = T;
    if (mode === 'step') T.stepNext(!reduced(), 1);
    else { T.play(); if (mode === 'instant') T.finish(); }
  }

  /* ---------- controls ---------- */
  bBuild.addEventListener('click', function () {
    if (phase === 'building' && tl) { tl.finish(); return; }
    stop(); playBuild(reduced() ? 'instant' : 'anim');
  });
  bSearch.addEventListener('click', function () {
    if (phase === 'searching' && tl) { tl.finish(); return; }
    stop(); playSearch(reduced() ? 'instant' : 'anim');
  });
  bStep.addEventListener('click', function () {
    if (tl && !tl.dead) {
      if (tl.playing()) tl.stepNext(false);   // running: complete the current action and pause
      else tl.stepNext(!reduced(), phase === 'building' ? 0.3 : 1);
      return;
    }
    stop();
    if (!built) playBuild('step'); else playSearch('step');
  });
  slider.addEventListener('input', function () {
    efVal.textContent = slider.value;
    if (searchShown) { stop(); playSearch('instant'); }
  });
  function onClick(ev) {
    var m = plane0.getScreenCTM(); if (!m) return;
    var pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY; pt = pt.matrixTransform(m.inverse());
    var v = (BASE0 - pt.y) / PH, x = pt.x - X0 - v * SH;
    if (v < 0 || v > 1 || x < 0 || x > DW) return;
    q = { x: x, y: v * DD };
    if (phase === 'building') { qShown = true; drawQ(); return; }
    stop(); playSearch(reduced() ? 'instant' : 'anim');
  }
  svg.addEventListener('click', onClick);

  toUnbuilt();

  return {
    step: function (n) {
      var animate = !reduced() && n > lastStep;
      ud.style.display = n === 1 ? '' : 'none';
      stop();
      if (n <= 0) { toUnbuilt(); q = { x: QDEF.x, y: QDEF.y }; }
      /* no autoplay: going forward waits for Build / Search; going back shows the finished state */
      else if (n === 1) { if (!animate) toBuilt(); }
      else { q = q || { x: QDEF.x, y: QDEF.y }; if (animate) toBuilt(); else playSearch('instant'); }
      lastStep = n;
    },
    /* deck protocol: Next first plays the step's animation, then moves on */
    pending: function () { return (lastStep === 1 && !built && phase !== 'building') || (lastStep >= 2 && !searchShown); },
    busy: function () { return !!(tl && tl.playing()); },
    finish: function () { if (tl) tl.finish(); },
    play: function () { if (lastStep === 1) bBuild.click(); else if (lastStep >= 2) bSearch.click(); },
    destroy: function () { stop(); svg.removeEventListener('click', onClick); el.innerHTML = ''; el.classList.remove('v1h'); }
  };
}

VIZ.register('v1', { mount: mount, _model: makeModel, _search: runSearch, _cfg: CFG });
})();
