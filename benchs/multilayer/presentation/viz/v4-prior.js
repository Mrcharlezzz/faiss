/* HM: the HM-ANN author-response experiment on the same small graph as V3 (an illustration, not their data):
   layer-1 search keeps T = 4 results; each starts its own layer-0 search (4 threads), results merged. */
(function(){
'use strict';
var NS = 'http://www.w3.org/2000/svg';
function S(tag, attrs, parent, txt){ var e = document.createElementNS(NS, tag); for (var k in attrs) e.setAttribute(k, attrs[k]); if (txt !== undefined) e.textContent = txt; if (parent) parent.appendChild(e); return e; }
VIZ.register('hm', { mount: function(el){
  var M = VIZ.defs.v3._model, CFG = M.CFG, G = M.build(CFG), P = G.P, q = [270, 290], T = 4, ef0 = 3;
  var GW = 578, GH = 420, SY = .47, BH = Math.round(GH * SY) + 6, Y1 = 0, Y0 = BH + 46;
  var COL = ['var(--cand)', 'var(--hot)', 'var(--res)', 'var(--ink)'];
  var l1 = M.searchLayer(G, G.adj1, q, [CFG.start], T), ent = l1.W.slice(0, T);
  var svg = S('svg', { width: GW, height: 2 * BH + 46, viewBox: '0 0 ' + GW + ' ' + (2 * BH + 46), role: 'img',
    'aria-label': 'Layer-1 search keeps 4 nodes; each starts its own layer-0 search' });
  svg.style.cssText = 'display:block;overflow:visible';
  el.appendChild(svg);
  function px(i){ return P[i][0]; } function py(i, t){ return t + 3 + P[i][1] * SY; }
  [[Y1, 'var(--l1)', 'layer 1'], [Y0, 'var(--l0)', 'layer 0']].forEach(function(b){
    S('rect', { x: 0, y: b[0], width: GW, height: BH, rx: 12, fill: 'var(--panel)', stroke: 'color-mix(in srgb,' + b[1] + ' 55%,var(--line))', 'stroke-width': 1.5 }, svg);
    S('text', { x: 12, y: b[0] + 20, fill: b[1], 'font-size': 14, 'font-weight': 600, 'letter-spacing': '.12em' }, svg, b[2].toUpperCase());
  });
  for (var a in G.adj0) G.adj0[a].forEach(function(b){ if (+a < b) S('line', { x1: px(a), y1: py(a, Y0), x2: px(b), y2: py(b, Y0), stroke: 'var(--line)', 'stroke-width': 1 }, svg); });
  for (var c in G.adj1) G.adj1[c].forEach(function(b){ if (+c < b) S('line', { x1: px(c), y1: py(c, Y1), x2: px(b), y2: py(b, Y1), stroke: 'var(--l1)', 'stroke-width': 2, opacity: .45 }, svg); });
  /* each layer-0 search as a tree: every visited node is linked to the node whose expansion found it */
  function tree(e){
    var dq = function(i){ return Math.hypot(P[i][0] - q[0], P[i][1] - q[1]); }, by = function(a, b){ return dq(a) - dq(b); };
    var vis = {}, par = [], C = [e], W = [e]; vis[e] = 1;
    while (C.length){
      C.sort(by); var c = C.shift();
      if (dq(c) > dq(W[W.length - 1])) break;
      G.adj0[c].forEach(function(n){
        if (vis[n]) return; vis[n] = 1; par.push([c, n]);
        if (W.length < ef0 || dq(n) < dq(W[W.length - 1])){ C.push(n); W.push(n); W.sort(by); if (W.length > ef0) W.pop(); }
      });
    }
    return { vis: Object.keys(vis).map(Number), par: par };
  }
  var owner = {}, trees = ent.map(tree);
  trees.forEach(function(t, k){ t.vis.forEach(function(i){ if (owner[i] === undefined) owner[i] = k; }); });
  trees.forEach(function(t, k){ t.par.forEach(function(pq){
    S('line', { x1: px(pq[0]), y1: py(pq[0], Y0), x2: px(pq[1]), y2: py(pq[1], Y0), stroke: COL[k], 'stroke-width': 3, 'stroke-linecap': 'round', opacity: .85 }, svg); }); });
  P.forEach(function(p, i){ var k = owner[i];
    S('circle', { cx: px(i), cy: py(i, Y0), r: k === undefined ? 3.2 : 5.4, fill: k === undefined ? 'var(--panel)' : COL[k], stroke: k === undefined ? 'var(--mute)' : COL[k], 'stroke-width': 1.1 }, svg); });
  G.up.forEach(function(i){ S('circle', { cx: px(i), cy: py(i, Y1), r: 6, fill: 'color-mix(in srgb,var(--l1) 45%,var(--panel))' }, svg); });
  ent.forEach(function(e, k){
    S('line', { x1: px(e), y1: py(e, Y1) + 9, x2: px(e), y2: py(e, Y0) - 8, stroke: COL[k], 'stroke-width': 2, 'stroke-dasharray': '4 4' }, svg);
    S('circle', { cx: px(e), cy: py(e, Y1), r: 9, fill: COL[k], stroke: 'var(--panel)', 'stroke-width': 2,
      'data-explain': 'one of the T = 4 nearest nodes found on layer 1; it starts its own layer-0 search (thread ' + (k + 1) + ')' }, svg);
    S('circle', { cx: px(e), cy: py(e, Y0), r: 8, fill: 'none', stroke: COL[k], 'stroke-width': 2.4 }, svg);
  });
  [Y1, Y0].forEach(function(t){
    S('circle', { cx: q[0], cy: t + 3 + q[1] * SY, r: 9, fill: 'var(--q)', stroke: 'var(--panel)', 'stroke-width': 3 }, svg);
    S('text', { x: q[0] - 14, y: t + 3 + q[1] * SY + 7, 'text-anchor': 'end', fill: 'var(--q)', 'font-family': 'var(--mono)', 'font-size': 22, 'font-weight': 700, stroke: 'var(--panel)', 'stroke-width': 4, 'paint-order': 'stroke' }, svg, 'q');
  });
  return {};
}});
})();

/* UB: Faiss upper_beam on a small 3-layer HNSW (same graph style as the start of the deck):
   the same beam width (3) on every layer; the 3 nodes kept on each layer seed the layer below. */
(function(){
'use strict';
var NS = 'http://www.w3.org/2000/svg';
function S(tag, attrs, parent, txt){ var e = document.createElementNS(NS, tag); for (var k in attrs) e.setAttribute(k, attrs[k]); if (txt !== undefined) e.textContent = txt; if (parent) parent.appendChild(e); return e; }
VIZ.register('ub', { mount: function(el){
  var M = VIZ.defs.v3._model, CFG = M.CFG, G = M.build(CFG), P = G.P, q = CFG.q, B = 3;
  var up2 = G.up.filter(function(_, k){ return k % 3 === 0; });
  if (up2.indexOf(CFG.start) < 0) up2.unshift(CFG.start);
  var adj2 = M.linkAll(P, up2, 2, 3);
  var r2 = M.searchLayer(G, adj2, q, [CFG.start], B);
  var r1 = M.searchLayer(G, G.adj1, q, r2.W, B);
  var r0 = M.searchLayer(G, G.adj0, q, r1.W, B);
  var W = 520, SX = W / 578, SY = .34, BH = Math.round(420 * SY) + 8, GAP = 36;
  var LAY = [ { l: 2, nodes: up2, adj: adj2, res: r2, top: 0, col: 'var(--l2)' },
              { l: 1, nodes: G.up, adj: G.adj1, res: r1, top: BH + GAP, col: 'var(--l1)' },
              { l: 0, nodes: P.map(function(_, i){ return i; }), adj: G.adj0, res: r0, top: 2 * (BH + GAP), col: 'var(--l0)' } ];
  var H = 3 * BH + 2 * GAP;
  var svg = S('svg', { width: W, height: H, viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Three HNSW layers; on each the search keeps 3 nodes and passes them down' });
  svg.style.cssText = 'display:block;overflow:visible'; el.appendChild(svg);
  function px(i){ return P[i][0] * SX; } function py(i, t){ return t + 4 + P[i][1] * SY; }
  LAY.forEach(function(L){
    S('rect', { x: 0, y: L.top, width: W, height: BH, rx: 12, fill: 'var(--panel)', stroke: 'color-mix(in srgb,' + L.col + ' 55%,var(--line))', 'stroke-width': 1.5 }, svg);
    S('text', { x: 12, y: L.top + 19, fill: L.col, 'font-size': 13, 'font-weight': 600, 'letter-spacing': '.12em' }, svg, 'LAYER ' + L.l);
    for (var a in L.adj) L.adj[a].forEach(function(b){ if (+a < b) S('line', { x1: px(a), y1: py(a, L.top), x2: px(b), y2: py(b, L.top),
      stroke: L.l ? L.col : 'var(--line)', 'stroke-width': L.l ? 1.8 : 1, opacity: L.l ? .45 : 1 }, svg); });
    L.nodes.forEach(function(i){ S('circle', { cx: px(i), cy: py(i, L.top), r: L.l ? 5 : 3, fill: L.l ? 'color-mix(in srgb,' + L.col + ' 45%,var(--panel))' : 'var(--panel)', stroke: L.l ? 'none' : 'var(--mute)', 'stroke-width': 1 }, svg); });
  });
  LAY.forEach(function(L, k){
    var kept = L.res.W.slice(0, B);
    if (k < 2) kept.forEach(function(i){ S('line', { x1: px(i), y1: py(i, L.top) + 7, x2: px(i), y2: py(i, LAY[k + 1].top) - 7, stroke: 'var(--hot)', 'stroke-width': 1.8, 'stroke-dasharray': '3 4' }, svg); });
    kept.forEach(function(i){ S('circle', { cx: px(i), cy: py(i, L.top), r: 7, fill: 'var(--hot)', stroke: 'var(--panel)', 'stroke-width': 2 }, svg); });
    S('text', { x: W - 12, y: L.top + 19, 'text-anchor': 'end', fill: 'var(--mute)', 'font-family': 'var(--mono)', 'font-size': 14 }, svg, 'beam = ' + B);
    S('circle', { cx: q[0] * SX, cy: L.top + 4 + q[1] * SY, r: 6, fill: 'var(--q)', stroke: 'var(--panel)', 'stroke-width': 2 }, svg);
  });
  return {};
}});
})();
