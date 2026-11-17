/* V2: SEARCH-LAYER on one layer (HNSW paper, Algorithm 2), step by step.
   Toy data: seeded PRNG points, proximity graph built the HNSW way on a single layer. */
(function(){
'use strict';

/* ---------- configuration (deterministic) ---------- */
var CFG = { seed: 44, N: 55, M: 2, cap: 4, minD: 42, q: [250, 400], ep: 10 };
var GW = 430, GH = 494, PAD = 28;         /* graph panel, stage px */

/* ---------- data ---------- */
function mulberry32(a){ return function(){ a|=0; a=a+0x6D2B79F5|0; var t=Math.imul(a^a>>>15,1|a);
  t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }

function buildGraph(){
  var r = mulberry32(CFG.seed), P = [], tries = 0, i, j;
  while (P.length < CFG.N && tries < 20000){
    tries++;
    var x = PAD + r()*(GW-2*PAD), y = PAD + r()*(GH-2*PAD), ok = true;
    for (i=0;i<P.length;i++){ var dx=P[i][0]-x, dy=P[i][1]-y; if (dx*dx+dy*dy < CFG.minD*CFG.minD){ ok=false; break; } }
    if (ok) P.push([x,y]);
  }
  function d(a,b){ return Math.hypot(P[a][0]-P[b][0], P[a][1]-P[b][1]); }
  var adj = P.map(function(){ return []; });
  /* insert in order; connect to the M nearest already-inserted points (exact); links both ways;
     a node over cap links keeps its cap nearest */
  for (i=1;i<P.length;i++){
    var prev = []; for (j=0;j<i;j++) prev.push(j);
    prev.sort(function(a,b){ return d(i,a)-d(i,b); });
    prev.slice(0, CFG.M).forEach(function(j){
      adj[i].push(j); adj[j].push(i);
      if (adj[j].length > CFG.cap){
        adj[j].sort(function(a,b){ return d(j,a)-d(j,b); });
        var drop = adj[j].pop();
        adj[drop] = adj[drop].filter(function(k){ return k!==j; });
      }
    });
  }
  return { P: P, adj: adj };
}

/* ---------- the algorithm, recorded as a list of displayable events ---------- */
function trace(G, q, ep, ef){
  function dq(i){ return Math.hypot(G.P[i][0]-q[0], G.P[i][1]-q[1]); }
  function byD(a,b){ return dq(a)-dq(b); }
  function furthest(W){ var f=W[0]; W.forEach(function(w){ if (dq(w)>dq(f)) f=w; }); return f; }
  var V = [ep], C = [ep], W = [ep], n = 1, hops = 0, touched = [], ev = [];
  function snap(o){
    o.V = V.slice(); o.C = C.slice(); o.W = W.slice(); o.n = n; o.hops = hops;
    o.f = W.length ? furthest(W) : null; o.touched = touched.slice(); ev.push(o);
  }
  snap({ kind:'init', lines:[1], w:1 });
  var Vs = {}; Vs[ep] = 1;
  while (C.length){
    C.sort(byD); var c = C.shift(); var f = furthest(W);
    if (dq(c) > dq(f)){ snap({ kind:'break', lines:[4], c:c, w:1.6 }); break; }
    hops++;
    snap({ kind:'pop', lines:[2,3,4], c:c, w:1.3 });
    for (var k=0;k<G.adj[c].length;k++){
      var e = G.adj[c][k];
      touched.push(c+'-'+e);
      if (Vs[e]){ snap({ kind:'seen', lines:[5], c:c, e:e, w:.35 }); continue; }
      Vs[e] = 1; V.push(e); n++;
      f = furthest(W);
      if (dq(e) < dq(f) || W.length < ef){
        C.push(e); W.push(e); var rm = null;
        if (W.length > ef){ rm = furthest(W); W = W.filter(function(x){ return x!==rm; }); }
        snap({ kind:'add', lines: rm===null?[5,6,7,8]:[5,6,7,8,9], c:c, e:e, removed:rm, w:1 });
      } else {
        snap({ kind:'reject', lines:[5,6,7], c:c, e:e, w:.8 });
      }
    }
  }
  W.sort(byD);
  snap({ kind:'return', lines:[10], w:1, done:true });
  var nn = 0; for (var i=1;i<G.P.length;i++) if (dq(i) < dq(nn)) nn = i;
  return { ev: ev, nn: nn, found: W.indexOf(nn) >= 0, dq: dq };
}

/* ---------- pseudocode ---------- */
var CODE = [
  'search_layer(q, entry, ef)',
  '  candidates = {entry};  results = {entry};  visited = {entry}',
  '  while candidates not empty',
  '    c = closest candidate to q',
  '    if c is farther than the worst result: stop',
  '    for each neighbour n of c not in visited',
  '      add n to visited',
  '      if n is closer than the worst result or |results| < ef',
  '        add n to candidates and results',
  '        if |results| > ef: drop the worst result',
  '  return results'
];
var TIP = {
  candidates: 'points found but not yet expanded (blue rings)',
  results: 'best ef points found so far, the answer (green)',
  visited: 'every point whose distance to q was computed; each is computed once (grey)',
  worst: 'the result farthest from q: the dashed circle around q has this radius',
  ef: 'search width: how many results are kept',
  entry: 'entry point handed down from the layer above (square)',
  c: 'the candidate being expanded (orange ring)',
  n: 'the neighbour being checked (orange edge)'
};
var TCL = { candidates:'tc', results:'tw', visited:'tv', worst:'tf', ef:'te', entry:'tk', c:'th', n:'th' };
function esc(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function codeHTML(line){
  return esc(line).replace(/\b(worst result|candidates?|results|visited|ef|entry|c|n)\b/g, function(m){
    var k = m === 'worst result' ? 'worst' : (m === 'candidate' ? 'candidates' : m);
    return '<span class="tok '+TCL[k]+'" data-explain="'+TIP[k]+'">'+m+'</span>';
  });
}

/* ---------- styles (scoped to .v2s) ---------- */
var CSS = [
'.v2s{position:absolute;inset:0;font-family:var(--font);color:var(--ink)}',
'.v2s .gp{position:absolute;left:0;top:0;width:'+GW+'px;height:'+GH+'px;background:var(--panel);border:1.5px solid var(--line);border-radius:14px;overflow:hidden}',
'.v2s .gp svg{display:block;width:100%;height:100%;cursor:crosshair}',
'.v2s .cp{position:absolute;left:'+(GW+14)+'px;top:0;right:0;height:'+GH+'px;background:var(--panel);border:1.5px solid var(--line);border-radius:14px;padding:14px 0 0}',
'.v2s .code{font-family:var(--mono);font-size:19px;line-height:38px;white-space:pre}',
'.v2s .ln{padding:0 8px 0 8px;border-left:6px solid transparent;color:var(--ink);transition:opacity .15s}',
'.v2s .ln:first-child{font-weight:500}',
'.v2s .code.run .ln{opacity:.35}',
'.v2s .code.run .ln.on{opacity:1;background:var(--faint);background:color-mix(in srgb,var(--hot) 18%,var(--panel));border-left-color:var(--hot);font-weight:500}',
'.v2s .tok{color:var(--ink);font-weight:600}',
'.v2s .tv{color:var(--visit)} .v2s .tc{color:var(--cand)} .v2s .tw{color:var(--res)} .v2s .te{color:var(--q)} .v2s .tf{color:var(--res);font-style:italic} .v2s .tk{color:var(--ink)} .v2s .th{color:var(--hot)}',
'.v2s .bar{position:absolute;left:0;right:150px;bottom:0;height:44px;display:flex;align-items:center;gap:10px}',
'.v2s .bar .sp{flex:1}',
'.v2s .bar .ctl{min-width:74px}',
'.v2s .bar .ctl:disabled{opacity:.4;cursor:default}',
'.v2s .ef{display:inline-flex;align-items:center;gap:8px;font-family:var(--mono);font-size:17px;margin-left:8px}',
'.v2s .ef input{width:112px}',
'.v2s .ef output{display:inline-block;min-width:22px;font-weight:500}',
'.v2s .chip b{font-weight:500}',
'.v2s .chip.ok{border-color:var(--res);color:var(--res)}',
'.v2s .chip.no{border-color:var(--q);color:var(--q)}',
'.v2s .chip.idle{color:var(--mute)}',
/* graph */
'.v2s .ed{stroke:var(--line);stroke-width:1.2}',
'.v2s .ed.t{stroke:var(--visit);stroke-width:2}',
'.v2s .fl{stroke:var(--hot);stroke-width:4.5;stroke-linecap:round;opacity:0}',
'.v2s .fl.on{opacity:1}',
'.v2s .nd{fill:var(--panel);stroke:var(--mute);stroke-width:1.4;r:4px;transition:r .2s,fill .2s}',
'.v2s .nd.v{fill:var(--visit);stroke:var(--visit);r:5px}',
'.v2s .nd.w{fill:var(--res);stroke:var(--res);r:6.5px}',
'.v2s .cr{fill:none;stroke:var(--cand);stroke-width:2.4;r:10px;opacity:0;transition:opacity .2s}',
'.v2s .cr.on{opacity:1}',
'.v2s .fc{fill:var(--res);fill-opacity:.07;stroke:var(--res);stroke-width:1.6;stroke-dasharray:6 5;transition:r .25s}',
'.v2s .hot{fill:none;stroke:var(--hot);stroke-width:3}',
'.v2s .lab{font-family:var(--mono);font-size:16px;font-weight:500;paint-order:stroke;stroke:var(--panel);stroke-width:4px;stroke-linejoin:round}',
'.v2s .lab.h{fill:var(--hot)} .v2s .lab.r{fill:var(--res)} .v2s .lab.q{fill:var(--q);font-size:17px} .v2s .lab.k{fill:var(--ink)}',
'.v2s .nn{fill:none;stroke:var(--q);stroke-width:2.2;stroke-dasharray:4 3}',
'.v2s .hide{display:none}',
'@media (prefers-reduced-motion: reduce){.v2s .nd,.v2s .cr,.v2s .fc,.v2s .ln{transition:none}}'
].join('\n');

var SVGNS = 'http://www.w3.org/2000/svg';
function S(tag, attrs, parent){
  var e = document.createElementNS(SVGNS, tag);
  for (var k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}

VIZ.register('v2', { mount: function(el, api){
  var G = buildGraph(), P = G.P;
  var reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  var st = { q: CFG.q.slice(), ef: 1, T: null, idx: -1, timer: null, deckStep: null };

  var style = document.createElement('style'); style.textContent = CSS;
  var root = document.createElement('div'); root.className = 'v2s';
  el.appendChild(style); el.appendChild(root);

  /* --- graph panel --- */
  var gp = document.createElement('div'); gp.className = 'gp'; root.appendChild(gp);
  var svg = S('svg', { viewBox: '0 0 '+GW+' '+GH, role: 'img', 'aria-label': 'Proximity graph with query point' }, gp);
  var gE = S('g', {}, svg), gT = S('g', {}, svg);
  var fc = S('circle', { 'class': 'fc hide', cx: 0, cy: 0, r: 0,
  }, svg);
  var fl = S('line', { 'class': 'fl' }, svg);
  var gN = S('g', {}, svg), gR = S('g', {}, svg), gTop = S('g', {}, svg);

  var edgeEl = {}, edgeKey = function(a,b){ return a<b ? a+'-'+b : b+'-'+a; };
  G.adj.forEach(function(ns, a){ ns.forEach(function(b){
    var k = edgeKey(a,b); if (edgeEl[k]) return;
    edgeEl[k] = S('line', { 'class':'ed', x1:P[a][0], y1:P[a][1], x2:P[b][0], y2:P[b][1] }, gE);
  }); });
  var nodeEl = P.map(function(p){ return S('circle', { 'class':'nd', cx:p[0], cy:p[1], r:4 }, gN); });
  var ringEl = P.map(function(p){ return S('circle', { 'class':'cr', cx:p[0], cy:p[1], r:10 }, gR); });

  var epP = P[CFG.ep];
  S('rect', { x: epP[0]-11, y: epP[1]-11, width: 22, height: 22, rx: 3, fill: 'none', stroke: 'var(--ink)', 'stroke-width': 2,
    'data-explain': 'entry point handed down from the layer above' }, gTop);
  var epLab = S('text', { 'class': 'lab k', x: epP[0], y: epP[1]-18, 'text-anchor': 'middle',
    'data-explain': 'entry point handed down from the layer above' }, gTop);
  epLab.textContent = 'entry';

  var addRing = S('circle', { 'class': 'hide', r: 13, fill: 'var(--res)', 'fill-opacity': .22, stroke: 'none' }, gTop);
  var cRing = S('circle', { 'class': 'hot hide', r: 12 }, gTop);
  var eRing = S('circle', { 'class': 'hot hide', r: 9, 'stroke-width': 2 }, gTop);
  var nnRing = S('circle', { 'class': 'nn hide', r: 14,
    'data-explain': 'true nearest neighbour (brute force over all points)' }, gTop);
  var nnLab = S('text', { 'class': 'lab q hide', 'font-size': 14 }, gTop); nnLab.textContent = 'true nearest';
  var cLab = S('text', { 'class': 'lab h hide' }, gTop); cLab.textContent = 'c';
  var eLab = S('text', { 'class': 'lab h hide' }, gTop); eLab.textContent = 'n';
  var fLab = S('text', { 'class': 'lab r hide' }, gTop); fLab.textContent = 'worst';
  var qG = S('g', { 'data-explain': 'query q (click the graph to move it)' }, gTop);
  var qDot = S('circle', { r: 7.5, fill: 'var(--q)', stroke: 'var(--panel)', 'stroke-width': 2.5 }, qG);
  var qLab = S('text', { 'class': 'lab q' }, qG); qLab.textContent = 'q';

  /* --- code panel --- */
  var cp = document.createElement('div'); cp.className = 'cp'; root.appendChild(cp);
  var code = document.createElement('div'); code.className = 'code'; cp.appendChild(code);
  var lineEl = CODE.map(function(t){
    var d = document.createElement('div'); d.className = 'ln'; d.innerHTML = codeHTML(t); code.appendChild(d); return d;
  });
  /* --- control bar --- */
  var bar = document.createElement('div'); bar.className = 'bar'; root.appendChild(bar);
  bar.innerHTML =
    '<button class="ctl" data-k="step">Step</button>' +
    '<button class="ctl primary" data-k="play">Play</button>' +
    '<button class="ctl" data-k="reset">Reset</button>' +
    '<label class="ef" data-explain="search width: how many results are kept">ef <input type="range" min="1" max="16" step="1" value="1" aria-label="ef"><output>1</output></label>' +
    '<span class="sp"></span>' +
    '<span class="chip" style="display:none" data-explain="dist(·, q) evaluations: the cost that ef buys recall with">distance computations <b data-k="n">0</b></span>' +
    '<span class="chip idle" data-k="res" data-explain="is the true nearest neighbour (brute force over all points) among the results?">found <b>–</b></span>';
  var bStep = bar.querySelector('[data-k=step]'), bPlay = bar.querySelector('[data-k=play]'),
      bReset = bar.querySelector('[data-k=reset]'), rng = bar.querySelector('input'), out = bar.querySelector('output'),
      chN = bar.querySelector('[data-k=n]'), chR = bar.querySelector('[data-k=res]');

  /* --- state & drawing --- */
  function cls(e, c, on){ e.classList.toggle(c, !!on); }
  function at(e, x, y){ e.setAttribute('x', x); e.setAttribute('y', y); }
  function labelAt(t, i, dx, dy){ at(t, P[i][0]+dx, P[i][1]+dy); cls(t, 'hide', false); }

  function recompute(){ st.T = trace(G, st.q, CFG.ep, st.ef); }

  function draw(){
    var T = st.T, E = st.idx >= 0 ? T.ev[st.idx] : null;
    var inV = {}, inC = {}, inW = {}, tch = {};
    if (E){ E.V.forEach(function(i){ inV[i]=1; }); E.C.forEach(function(i){ inC[i]=1; }); E.W.forEach(function(i){ inW[i]=1; });
            E.touched.forEach(function(k){ var p=k.split('-'); tch[edgeKey(+p[0],+p[1])]=1; }); }
    nodeEl.forEach(function(n, i){ cls(n, 'v', inV[i] && !inW[i]); cls(n, 'w', inW[i]); });
    ringEl.forEach(function(r, i){ cls(r, 'on', inC[i]); });
    for (var k in edgeEl) cls(edgeEl[k], 't', tch[k]);
    lineEl.forEach(function(l, i){ cls(l, 'on', E && E.lines.indexOf(i) >= 0); });
    cls(code, 'run', !!E);

    /* q */
    qDot.setAttribute('cx', st.q[0]); qDot.setAttribute('cy', st.q[1]);
    at(qLab, st.q[0]+11, st.q[1]-9);

    /* f circle and label */
    if (E && E.f !== null){
      fc.setAttribute('cx', st.q[0]); fc.setAttribute('cy', st.q[1]);
      fc.setAttribute('r', T.dq(E.f).toFixed(1)); fc.style.r = T.dq(E.f).toFixed(1)+'px';
      cls(fc, 'hide', false); cls(fLab, 'hide', true);
    } else { cls(fc, 'hide', true); cls(fLab, 'hide', true); }

    /* c, e */
    var c = E && E.c !== undefined && !E.done ? E.c : null, e = E && E.e !== undefined ? E.e : null;
    if (c !== null){ cRing.setAttribute('cx', P[c][0]); cRing.setAttribute('cy', P[c][1]); cls(cRing, 'hide', false); labelAt(cLab, c, -22, -10); }
    else { cls(cRing, 'hide', true); cls(cLab, 'hide', true); }
    if (e !== null){
      eRing.setAttribute('cx', P[e][0]); eRing.setAttribute('cy', P[e][1]); cls(eRing, 'hide', false); labelAt(eLab, e, 11, -9);
      fl.setAttribute('x1', P[c][0]); fl.setAttribute('y1', P[c][1]); fl.setAttribute('x2', P[e][0]); fl.setAttribute('y2', P[e][1]);
      cls(fl, 'on', true);
      /* the neighbour just entered the results: a light green halo */
      addRing.setAttribute('cx', P[e][0]); addRing.setAttribute('cy', P[e][1]); cls(addRing, 'hide', !inW[e]);
    } else { cls(eRing, 'hide', true); cls(eLab, 'hide', true); cls(fl, 'on', false); cls(addRing, 'hide', true); }

    /* end state */
    var done = E && E.done;
    if (done){
      var nn = T.nn; nnRing.setAttribute('cx', P[nn][0]); nnRing.setAttribute('cy', P[nn][1]); cls(nnRing, 'hide', false);
      cls(nnLab, 'hide', T.found);
      if (!T.found){
        /* label on the side away from q */
        var vx = P[nn][0]-st.q[0], vy = P[nn][1]-st.q[1], vl = Math.hypot(vx, vy) || 1;
        vx /= vl; vy /= vl;
        var lx = P[nn][0] + vx*22, ly = P[nn][1] + vy*24 + 5;
        var anchor = Math.abs(vx) < .45 ? 'middle' : (vx > 0 ? 'start' : 'end');
        if (ly < 18) ly = P[nn][1] + 30; if (ly > GH - 10) ly = P[nn][1] - 22;
        if (anchor === 'middle' && lx < 60) anchor = 'start'; if (anchor === 'middle' && lx > GW - 60) anchor = 'end';
        if (anchor === 'end' && lx < 110) { anchor = 'start'; lx = P[nn][0] + 18; }
        if (anchor === 'start' && lx > GW - 110) { anchor = 'end'; lx = P[nn][0] - 18; }
        at(nnLab, lx, ly); nnLab.setAttribute('text-anchor', anchor);
      }
      cls(fLab, 'hide', true);
    } else { cls(nnRing, 'hide', true); cls(nnLab, 'hide', true); }

    chN.textContent = E ? E.n : 0;
    chR.className = 'chip ' + (done ? (T.found ? 'ok' : 'no') : 'idle');
    chR.querySelector('b').textContent = done ? (T.found ? '✓' : '✗') : '–';
    bStep.disabled = !!done;
    bPlay.textContent = st.timer ? 'Pause' : (done ? 'Replay' : 'Play');
    out.textContent = st.ef; rng.value = st.ef;
  }

  /* autoplay: an ef = 8 run at the default query takes about 10 s */
  var BASE = (function(){
    var t = trace(G, CFG.q, CFG.ep, 8), s = 0; t.ev.forEach(function(e){ s += e.w; });
    return Math.max(90, Math.min(450, 10000 / s));
  })();
  function stop(){ if (st.timer){ clearTimeout(st.timer); st.timer = null; } }
  function tick(){
    st.timer = null;
    if (st.idx >= st.T.ev.length - 1){ draw(); return; }
    st.idx++;
    var next = st.T.ev[Math.min(st.idx + 1, st.T.ev.length - 1)];
    st.timer = st.idx < st.T.ev.length - 1 ? setTimeout(tick, BASE * next.w) : null;
    draw();
  }
  function play(){
    stop();
    if (st.idx >= st.T.ev.length - 1) st.idx = -1;
    st.timer = setTimeout(tick, st.idx < 0 ? 250 : 0);
    draw();
  }
  function toEnd(){ stop(); st.idx = st.T.ev.length - 1; draw(); }
  function toStart(){ stop(); st.idx = -1; draw(); }

  bStep.addEventListener('click', function(){ stop(); if (st.idx < st.T.ev.length - 1) st.idx++; draw(); });
  bPlay.addEventListener('click', function(){ if (st.timer){ stop(); draw(); } else play(); });
  bReset.addEventListener('click', toStart);
  rng.addEventListener('input', function(){ st.ef = +rng.value; recompute(); toEnd(); });
  /* give arrow keys back to the deck after a mouse drag */
  rng.addEventListener('pointerup', function(){ setTimeout(function(){ rng.blur(); }, 0); });
  svg.addEventListener('click', function(ev){
    var r = svg.getBoundingClientRect();
    var x = (ev.clientX - r.left) * GW / r.width, y = (ev.clientY - r.top) * GH / r.height;
    st.q = [Math.max(10, Math.min(GW-10, x)), Math.max(10, Math.min(GH-10, y))];
    recompute(); toStart();
  });

  recompute(); draw();

  return {
    step: function(n){
      var from = st.deckStep; if (n === from) return; st.deckStep = n;
      stop(); st.q = CFG.q.slice();
      if (n <= 0){ st.ef = 1; recompute(); toStart(); return; }
      st.ef = n === 1 ? 1 : 8; recompute();
      var animate = !reduce && (from === null || from < n);
      /* no autoplay: going forward shows the start, Play runs it */
      if (animate) toStart(); else toEnd();
    },
    /* deck protocol: Next first plays the step's search, then moves on */
    pending: function(){ return st.deckStep > 0 && st.idx < 0 && !st.timer; },
    busy: function(){ return !!st.timer; },
    finish: function(){ toEnd(); },
    play: function(){ play(); },
    destroy: function(){ stop(); style.remove(); root.remove(); }
  };
}});
})();
