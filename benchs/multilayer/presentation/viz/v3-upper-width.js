/* V3: width on layer 1. Same toy two-layer HNSW twice: greedy layer 1 (left) vs a wider layer 1 (right).
   The wider search hands all its results to layer 0; layer 0 runs the same ef0 search from them.
   Seeded PRNG points; counts come from this small graph, not from the experiments.
   Layer 1 (top box) and layer 0 (bottom box) are drawn separately; the searches animate step by step. */
(function(){
'use strict';

var CFG = { seed: 42, N: 90, M: 3, cap: 6, M1: 2, cap1: 4, p1: 0.2, minD: 30, ef0: 3, w: 4, q: [460, 275], start: 21 };
var GW = 578, GH = 420, PAD = 22;

function mulberry32(a){ return function(){ a|=0; a=a+0x6D2B79F5|0; var t=Math.imul(a^a>>>15,1|a);
  t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }

/* ---------- model ---------- */
function linkAll(P, ids, M, cap){
  var adj = {}; ids.forEach(function(i){ adj[i] = []; });
  function d(a,b){ return Math.hypot(P[a][0]-P[b][0], P[a][1]-P[b][1]); }
  for (var k=1;k<ids.length;k++){
    var i = ids[k], prev = ids.slice(0,k).sort(function(a,b){ return d(i,a)-d(i,b); });
    prev.slice(0, M).forEach(function(j){
      adj[i].push(j); adj[j].push(i);
      if (adj[j].length > cap){
        adj[j].sort(function(a,b){ return d(j,a)-d(j,b); });
        var drop = adj[j].pop(); adj[drop] = adj[drop].filter(function(x){ return x!==j; });
      }
    });
  }
  return adj;
}
function build(cfg){
  var r = mulberry32(cfg.seed), P = [], tries = 0;
  while (P.length < cfg.N && tries < 40000){
    tries++;
    var x = PAD + r()*(GW-2*PAD), y = PAD + r()*(GH-2*PAD), ok = true;
    for (var i=0;i<P.length;i++){ if (Math.hypot(P[i][0]-x, P[i][1]-y) < cfg.minD){ ok=false; break; } }
    if (ok) P.push([x,y]);
  }
  var all = P.map(function(_,i){ return i; });
  var up = all.filter(function(){ return r() < cfg.p1; });
  return { P: P, adj0: linkAll(P, all, cfg.M, cfg.cap), adj1: linkAll(P, up, cfg.M1, cfg.cap1), up: up };
}
/* SEARCH-LAYER from a set of entries; counts distance computations (entries included, once each) */
function searchLayer(G, adj, q, entries, ef){
  var dq = function(i){ return Math.hypot(G.P[i][0]-q[0], G.P[i][1]-q[1]); };
  var vis = {}, n = 0, C = [], W = [], order = [];
  entries.forEach(function(e){ vis[e]=1; n++; C.push(e); W.push(e); order.push(e); });
  var by = function(a,b){ return dq(a)-dq(b); };
  W.sort(by); while (W.length > ef) W.pop();
  while (C.length){
    C.sort(by); var c = C.shift();
    if (dq(c) > dq(W[W.length-1])) break;
    adj[c].forEach(function(e){
      if (vis[e]) return; vis[e]=1; n++; order.push(e);
      if (W.length < ef || dq(e) < dq(W[W.length-1])){ C.push(e); W.push(e); W.sort(by); if (W.length > ef) W.pop(); }
    });
  }
  W.sort(by);
  return { W: W, n: n, visited: order, dq: dq };
}
function run(G, q, start, w, ef0){
  var l1 = searchLayer(G, G.adj1, q, [start], w);
  var l0 = searchLayer(G, G.adj0, q, l1.W, ef0);
  var nn = 0; G.P.forEach(function(p,i){ if (l1.dq(i) < l1.dq(nn)) nn = i; });
  return { l1: l1, l0: l0, entry: l1.W[0], handed: l1.W, nn: nn, found: l0.W.indexOf(nn) >= 0,
           entryDist: l1.dq(l1.W[0]), total: l1.n + l0.n };
}

/* ---------- view ---------- */
var SY = 0.47, BH = Math.round(GH * SY) + 6;   /* each layer is a flattened plane */
var CSS = [
'.v3s{position:absolute;inset:0;font-family:var(--font);color:var(--ink)}',
'.v3s .pn{position:absolute;top:0;width:'+GW+'px}',
'.v3s .hd{display:flex;align-items:baseline;gap:12px;height:34px;font-size:22px;font-weight:600}',
'.v3s .hd .m{font-size:17px;color:var(--mute);font-weight:500}',
'.v3s svg.pl{display:block;width:'+GW+'px;height:'+(2*BH+40)+'px;overflow:visible;cursor:crosshair}',
'.v3s .box{fill:var(--panel);stroke:var(--line);stroke-width:1.5}',
'.v3s .box1{stroke:color-mix(in srgb,var(--l1) 55%,var(--line))}',
'.v3s .box0{stroke:color-mix(in srgb,var(--l0) 55%,var(--line))}',
'.v3s .ln{font-size:14px;letter-spacing:.12em;text-transform:uppercase;font-weight:600}',
'.v3s .e0{stroke:var(--line);stroke-width:1}',
'.v3s .e1{stroke:var(--l1);stroke-width:2;opacity:.5}',
'.v3s .n0{fill:var(--panel);stroke:var(--mute);stroke-width:1.1}',
'.v3s .n0.v{fill:var(--l0);stroke:var(--hot);stroke-width:2.4;r:5.5px}',
'.v3s .n1{fill:color-mix(in srgb,var(--l1) 45%,var(--panel));stroke:var(--panel);stroke-width:1.5}',
'.v3s .n1.v{fill:var(--l1);stroke:var(--hot);stroke-width:2.6}',
'.v3s .hand{fill:none;stroke:var(--l1);stroke-width:2.2}',
'.v3s .ep{fill:none;stroke:var(--ink);stroke-width:2.4}',
'.v3s .dl{stroke:var(--ink);stroke-width:2;stroke-dasharray:5 4}',
'.v3s .drop{stroke:var(--l1);stroke-width:1.6;stroke-dasharray:3 4;opacity:.8}',
'.v3s .lab{font-family:var(--mono);font-size:15px;font-weight:500;paint-order:stroke;stroke:var(--panel);stroke-width:4px;stroke-linejoin:round;fill:var(--ink)}',
'.v3s .lab.q{fill:var(--q);font-size:17px}',
'.v3s .nn{fill:none;stroke:var(--q);stroke-width:2;stroke-dasharray:4 3}',
'.v3s .cost{margin-top:4px;height:64px}',
'.v3s .row{display:flex;align-items:center;gap:10px;height:30px;font-size:17px}',
'.v3s .row .k{width:84px;color:var(--mute)}',
'.v3s .bar{height:16px;border-radius:4px;transition:width .3s ease}',
'.v3s .b1{background:var(--l1)} .v3s .b0{background:var(--l0)}',
'.v3s .num{font-family:var(--mono);font-size:16px;font-weight:500;min-width:28px}',
'.v3s .tot{font-family:var(--mono);font-size:18px;font-weight:600;margin-left:14px}',
'.v3s .ctlbar{position:absolute;right:0;top:0;display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:16px}',
'.v3s .ctlbar input{width:110px}',
'.v3s .play{position:absolute;left:0;top:-2px}',
'.v3s .verdict{font-size:17px;font-weight:600}',
'@media (prefers-reduced-motion: reduce){.v3s .bar{transition:none}}'
].join('\n');

var NS = 'http://www.w3.org/2000/svg';
function S(tag, attrs, parent, txt){ var e = document.createElementNS(NS, tag); for (var k in attrs) e.setAttribute(k, attrs[k]); if (txt !== undefined) e.textContent = txt; if (parent) parent.appendChild(e); return e; }
var RM = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function mount(el, api){
  var G = build(CFG), P = G.P;
  var start = CFG.start !== null ? CFG.start : G.up[0];
  var st = { q: CFG.q.slice(), w: CFG.w, stage: 0 }, token = 0;
  var style = document.createElement('style'); style.textContent = CSS;
  var root = document.createElement('div'); root.className = 'v3s';
  el.appendChild(style); el.appendChild(root);
  var Y1 = 0, Y0 = BH + 40;                       /* top of the layer-1 and layer-0 planes */
  function px(i){ return P[i][0]; }
  function py(i, top){ return top + 3 + P[i][1] * SY; }
  function qy(top){ return top + 3 + st.q[1] * SY; }

  function panel(left, title, sub){
    var pn = document.createElement('div'); pn.className = 'pn'; pn.style.left = left + 'px'; root.appendChild(pn);
    pn.innerHTML = '<div class="hd">'+title+' <span class="m">'+sub+'</span></div>';
    var svg = S('svg', { 'class': 'pl', viewBox: '0 0 '+GW+' '+(2*BH+40), role: 'img', 'aria-label': title + ': layer 1 above, layer 0 below' });
    pn.appendChild(svg);
    var g = { svg: svg };
    S('rect', { 'class': 'box box1', x: 0, y: Y1, width: GW, height: BH, rx: 12 }, svg);
    S('rect', { 'class': 'box box0', x: 0, y: Y0, width: GW, height: BH, rx: 12 }, svg);
    S('text', { 'class': 'ln', x: 12, y: Y1 + 20, fill: 'var(--l1)' }, svg, 'layer 1');
    S('text', { 'class': 'ln', x: 12, y: Y0 + 20, fill: 'var(--l0)' }, svg, 'layer 0');
    g.drop = S('g', {}, svg);
    var ge0 = S('g', {}, svg), ge1 = S('g', {}, svg);
    for (var a in G.adj0) G.adj0[a].forEach(function(b){ if (+a < b) S('line', { 'class':'e0', x1:px(a), y1:py(a,Y0), x2:px(b), y2:py(b,Y0) }, ge0); });
    for (var c in G.adj1) G.adj1[c].forEach(function(b){ if (+c < b) S('line', { 'class':'e1', x1:px(c), y1:py(c,Y1), x2:px(b), y2:py(b,Y1) }, ge1); });
    var gn0 = S('g', {}, svg), gn1 = S('g', {}, svg);
    g.nodes0 = P.map(function(p, i){ return S('circle', { 'class':'n0', cx:px(i), cy:py(i,Y0), r:3.4 }, gn0); });
    g.nodes1 = {}; G.up.forEach(function(i){ g.nodes1[i] = S('circle', { 'class':'n1', cx:px(i), cy:py(i,Y1), r:6.5,
      }, gn1); });
    g.top = S('g', {}, svg);
    var cost = document.createElement('div'); cost.className = 'cost'; pn.appendChild(cost);
    cost.innerHTML =
      '<div class="row" data-explain="distance computations spent on layer 1"><span class="k">layer 1</span><span class="bar b1"></span><span class="num" data-k="c1"></span></div>' +
      '<div class="row" data-explain="distance computations spent on layer 0 (same ef0 on both sides)"><span class="k">layer 0</span><span class="bar b0"></span><span class="num" data-k="c0"></span><span class="tot" data-k="tot"></span></div>';
    g.cost = cost; g.b1 = cost.querySelector('.b1'); g.b0 = cost.querySelector('.b0');
    g.c1 = cost.querySelector('[data-k=c1]'); g.c0 = cost.querySelector('[data-k=c0]'); g.tot = cost.querySelector('[data-k=tot]');
    svg.addEventListener('click', function(ev){
      var r = svg.getBoundingClientRect(), sx = GW / r.width, X = (ev.clientX - r.left) * sx, Y = (ev.clientY - r.top) * sx;
      var top = Y >= Y0 ? Y0 : Y1, yy = (Y - top - 3) / SY;
      st.q = [Math.max(8, Math.min(GW-8, X)), Math.max(8, Math.min(GH-8, yy))];
      draw(false);
    });
    return g;
  }
  var L = panel(0, 'Greedy', 'width 1'), R = panel(GW + 28, 'Wider', 'width <span data-k="w">'+st.w+'</span>');
  var wOut = R.svg.parentNode.querySelector('[data-k=w]');
  var ctl = document.createElement('div'); ctl.className = 'ctlbar';
  ctl.innerHTML = 'width <input type="range" min="2" max="8" value="'+st.w+'" aria-label="layer-1 width"><output>'+st.w+'</output>';
  R.svg.parentNode.appendChild(ctl);
  var rng = ctl.querySelector('input'), rOut = ctl.querySelector('output');
  rng.addEventListener('input', function(){ st.w = +rng.value; draw(false); });
  rng.addEventListener('pointerup', function(){ setTimeout(function(){ rng.blur(); }, 0); });

  /* k1 / k0: how many visited nodes of layer 1 / layer 0 to show (Infinity = all) */
  function drawPanel(g, res, w, k1, k0){
    var show1 = st.stage >= 1, show0 = st.stage >= 2, done1 = k1 >= res.l1.visited.length, done0 = k0 >= res.l0.visited.length;
    g.top.innerHTML = ''; g.drop.innerHTML = '';
    var inV1 = {}; if (show1) res.l1.visited.slice(0, k1).forEach(function(i){ inV1[i]=1; });
    for (var k in g.nodes1) g.nodes1[k].classList.toggle('v', !!inV1[k]);
    var inV0 = {}; if (show0) res.l0.visited.slice(0, k0).forEach(function(i){ inV0[i]=1; });
    g.nodes0.forEach(function(n, i){ n.classList.toggle('v', !!inV0[i]); });
    S('rect', { x:px(start)-11, y:py(start,Y1)-11, width:22, height:22, rx:4, fill:'none', stroke:'var(--mute)', 'stroke-width':1.8,
      }, g.top);
    S('text', { 'class':'lab', x:px(start), y:py(start,Y1)-16, 'text-anchor':'middle' }, g.top, 'start');
    if (show1 && done1){
      if (w > 1) res.handed.forEach(function(i){ S('circle', { 'class':'hand', cx:px(i), cy:py(i,Y1), r:11 }, g.top); });
      var e = res.entry;
      S('line', { 'class':'dl', x1:px(e), y1:py(e,Y1), x2:st.q[0], y2:qy(Y1) }, g.top);
      S('circle', { 'class':'ep', cx:px(e), cy:py(e,Y1), r:14 }, g.top);
      if (g.verdict) S('text', { 'class':'verdict', x:GW-14, y:Y1+22, 'text-anchor':'end', fill:g.verdict[1] }, g.top, g.verdict[0]);
      S('text', { 'class':'lab', x:(px(e)+st.q[0])/2+8, y:(py(e,Y1)+qy(Y1))/2-6 }, g.top, Math.round(res.entryDist));
    }
    if (show0){
      res.handed.forEach(function(i){ S('line', { 'class':'drop', x1:px(i), y1:py(i,Y1)+8, x2:px(i), y2:py(i,Y0) }, g.drop); });
      S('circle', { 'class':'ep', cx:px(res.entry), cy:py(res.entry,Y0), r:9 }, g.top);
      if (done0) S('circle', { 'class':'nn', cx:px(res.nn), cy:py(res.nn,Y0), r:9 }, g.top);
    }
    [Y1, Y0].forEach(function(top){
      S('circle', { cx:st.q[0], cy:qy(top), r:7, fill:'var(--q)', stroke:'var(--panel)', 'stroke-width':2.5 }, g.top);
      S('text', { 'class':'lab q', x:st.q[0]+10, y:qy(top)-8 }, g.top, 'q');
    });
    var unit = 4.2, n1 = Math.min(k1, res.l1.n), n0 = show0 ? Math.min(k0, res.l0.n) : 0;
    g.cost.style.visibility = show1 ? 'visible' : 'hidden';
    g.b1.style.width = (n1*unit)+'px'; g.c1.textContent = n1;
    g.b0.style.width = (n0*unit)+'px'; g.c0.textContent = show0 ? n0 : '';
    g.tot.textContent = show0 && done0 ? '= ' + res.total : '';
  }
  /* Play: runs the search of the current step (layer 1 at step 1, layer 0 at step 2); the deck's Next also triggers it */
  var bPlay = document.createElement('button'); bPlay.className = 'ctl primary play'; bPlay.textContent = 'Play';
  bPlay.style.position = 'absolute'; bPlay.style.left = (GW - 70) + 'px'; bPlay.style.top = '-6px';
  root.appendChild(bPlay);
  bPlay.addEventListener('click', function(){ bPlay.blur(); if (st.stage > 0){ st.pend = false; draw(true); } });
  function verdicts(a, b){
    if (b.entryDist < a.entryDist - 1e-9){ L.verdict = ['worse entry point', 'var(--q)']; R.verdict = ['better entry point', 'var(--res)']; }
    else { L.verdict = R.verdict = ['same entry point', 'var(--mute)']; }
  }
  function draw(animate, pending){
    var a = run(G, st.q, start, 1, CFG.ef0), b = run(G, st.q, start, st.w, CFG.ef0);
    verdicts(a, b);
    wOut.textContent = st.w; rOut.textContent = st.w; rng.value = st.w;
    bPlay.style.visibility = st.stage > 0 ? 'visible' : 'hidden';
    var my = ++token; st.fast = false;
    /* pending: the step's search has not been played yet */
    if (pending){
      if (st.stage === 1){ drawPanel(L, a, 1, 0, 0); drawPanel(R, b, st.w, 0, 0); }
      else { drawPanel(L, a, 1, Infinity, 0); drawPanel(R, b, st.w, Infinity, 0); }
      return;
    }
    if (!animate || RM || st.stage === 0){ drawPanel(L, a, 1, Infinity, Infinity); drawPanel(R, b, st.w, Infinity, Infinity); return; }
    var layer = st.stage === 1 ? 'l1' : 'l0', n = Math.max(a[layer].visited.length, b[layer].visited.length), k = 0;
    var dt = layer === 'l1' ? 138 : 69;   /* 20% slower than 110 / 55 */
    (function tick(){
      if (my !== token) return;
      if (layer === 'l1'){ drawPanel(L, a, 1, k, 0); drawPanel(R, b, st.w, k, 0); }
      else { drawPanel(L, a, 1, Infinity, k); drawPanel(R, b, st.w, Infinity, k); }
      if (k++ <= n){ st.anim = true; setTimeout(tick, st.fast ? 0 : dt); } else { st.anim = false; st.fast = false; }
    })();
  }
  draw(false);
  return {
    step: function(n){ var fwd = n > st.stage; st.stage = n; if (n === 0){ st.q = CFG.q.slice(); st.w = CFG.w; } st.pend = fwd && n > 0; draw(false, st.pend); },
    /* deck protocol: Next first plays the step's search, then moves on */
    pending: function(){ return !!st.pend; },
    busy: function(){ return !!st.anim; },
    finish: function(){ st.fast = true; },
    play: function(){ st.pend = false; draw(true); },
    destroy: function(){ token++; style.remove(); root.remove(); }
  };
}

VIZ.register('v3', { mount: mount, _model: { build: build, run: run, searchLayer: searchLayer, linkAll: linkAll, CFG: CFG } });
})();
