/* R1–R3: results charts (slides 12–14). Data: window.RESULTS from data/results.js, written by
   presentation/data/export_results.py from runs 3 and 4 (same functions as the notebooks). No hand-entered numbers. */
(function(){
'use strict';
var NS = 'http://www.w3.org/2000/svg';
function S(tag, attrs, parent, text){
  var e = document.createElementNS(NS, tag);
  for (var k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}
function H(tag, cls, parent, html){ var e = document.createElement(tag); if (cls) e.className = cls; if (html !== undefined) e.innerHTML = html; if (parent) parent.appendChild(e); return e; }
function lin(d0, d1, r0, r1){ return function(v){ return r0 + (v - d0) * (r1 - r0) / (d1 - d0); }; }
function sgn(v, n){ n = n === undefined ? 1 : n; return (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(n); }
function pct(v){ return Math.round(v * 100) + '%'; }
function pct1(v){ var p = v * 100; return (p < 10 ? p.toFixed(1) : Math.round(p)) + '%'; }

/* ordered by descending dimensionality, clustered last; clustered d=32 is left out of the deck (Carlos, 2026-11-15) */
var REAL = ['Deep1M', 'GloVe-25'], CLUS = ['Clustered d=16'];
var DSL = { 'Deep1M': ['Deep1M', '96-d'], 'GloVe-25': ['GloVe-25', '25-d'],
            'Clustered d=16': ['Clustered', '16-d'], 'Clustered d=32': ['Clustered', '32-d'] };
var DSC = { 'Deep1M': 'var(--cand)', 'GloVe-25': 'var(--l2)',
            'Clustered d=16': 'var(--q)', 'Clustered d=32': 'var(--hot)' };
var SEEDS = { 'Deep1M': 5, 'GloVe-25': 10, 'Clustered d=16': 10, 'Clustered d=32': 10 };
/* width encoding: layer-1 purple ramp, lighter = narrower (in both themes), plus a marker shape */
var WS = ['2', '4', '8'];
var WC = { '2': 'var(--w2)', '4': 'var(--w4)', '8': 'var(--w8)' };
function marker(g, w, x, y, r){
  var c = WC[w];
  if (w === '2') return S('circle', { cx: x, cy: y, r: r, fill: c }, g);
  if (w === '4') return S('rect', { x: x - r, y: y - r, width: 2 * r, height: 2 * r, fill: c }, g);
  return S('path', { d: 'M' + x + ',' + (y - r * 1.25) + 'L' + (x + r * 1.15) + ',' + (y + r * .85) + 'L' + (x - r * 1.15) + ',' + (y + r * .85) + 'Z', fill: c }, g);
}

var CSS = [
':root{--w2:#B99BE8;--w4:#7B49C2;--w8:#3A1A72}',
'@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--w2:#E6DAFC;--w4:#A57FE6;--w8:#6A40C6}}',
':root[data-theme="dark"]{--w2:#E6DAFC;--w4:#A57FE6;--w8:#6A40C6}',
'.rr{position:absolute;inset:0;font-family:var(--font);color:var(--ink)}',
'.rr svg{display:block;overflow:visible}',
'.rr .ax{stroke:var(--line);stroke-width:1.2}',
'.rr .zero{stroke:var(--ink);stroke-width:1.6}',
'.rr .tk{font-family:var(--mono);font-size:16px;fill:var(--ink)}',
'.rr .al{font-size:18px;fill:var(--ink)}',
'.rr .dn{font-size:20px;font-weight:600;fill:var(--ink)}',
'.rr .dd{font-family:var(--mono);font-size:14px;fill:var(--mute)}',
'.rr .grp{font-size:13px;letter-spacing:.12em;text-transform:uppercase;fill:var(--mute)}',
'.rr .val{font-family:var(--mono);font-size:17px;font-weight:500;fill:var(--ink)}',
'.rr .seg{display:inline-flex;border:1.5px solid var(--line);border-radius:999px;overflow:hidden;background:var(--panel)}',
'.rr .seg button{font:inherit;font-size:16px;padding:5px 14px;border:0;background:transparent;color:var(--ink);cursor:pointer}',
'.rr .seg button.on{background:var(--ink);color:var(--paper)}',
'.rr .seg button:focus-visible{outline:3px solid var(--q);outline-offset:-3px}',
'.rr .bar{position:absolute;display:flex;align-items:center;gap:14px;font-size:16px;color:var(--mute)}',
'.rr .lg{display:inline-flex;align-items:center;gap:6px;font-size:16px;color:var(--ink)}',
'.rr .row{cursor:pointer}',
'.rr .row .hl{fill:var(--faint);opacity:0}',
'.rr .row.sel .hl{opacity:1}',
'.rr [data-r]{transition:opacity .4s}',
'.rr .off{opacity:0;pointer-events:none}',
'.rr .dim{opacity:.25}',
'.rr .note{font-size:14px;fill:var(--mute);font-style:italic}',
'@media (prefers-reduced-motion: reduce){.rr [data-r]{transition:none}}'
].join('\n');

function shell(el){
  var style = document.createElement('style'); style.textContent = CSS;
  var root = H('div', 'rr', null); el.appendChild(style); el.appendChild(root);
  return { root: root, destroy: function(){ style.remove(); root.remove(); } };
}
function seg(parent, items, cur, on){
  var s = H('span', 'seg', parent);
  items.forEach(function(it){
    var b = H('button', it[0] === cur ? 'on' : '', s, it[1]);
    b.addEventListener('click', function(){ s.querySelectorAll('button').forEach(function(x){ x.classList.remove('on'); }); b.classList.add('on'); on(it[0]); b.blur(); });
  });
  return { set: function(v){ s.querySelectorAll('button').forEach(function(x, i){ x.classList.toggle('on', items[i][0] === v); }); } };
}

/* ======================= R1: change vs standard across recall, one panel per dataset ======================= */
VIZ.register('r1', { mount: function(el){
  var R = window.RESULTS, sh = shell(el), root = sh.root;
  var st = { metric: 'latency', step: 0 };
  var MET = { latency: 'latency', dist: 'distance computations' };
  var DS = ['Deep1M', 'GloVe-25', 'Clustered d=16'];
  var bar = H('div', 'bar', root); bar.style.left = '0px'; bar.style.top = '0px';
  var segM = seg(bar, [['latency', 'latency'], ['dist', 'distance computations']], st.metric, function(v){ st.metric = v; draw(); });
  var lg = H('span', '', bar); lg.style.marginLeft = '18px';
  WS.forEach(function(w){
    var s = H('span', 'lg', lg); s.style.marginRight = '12px';
    var sv = S('svg', { width: 16, height: 16 }, s); marker(sv, w, 8, 8, 6); s.appendChild(document.createTextNode('width ' + w));
  });
  var PW = 372, GAP = 34, PH = 500, L = 58, T = 56, B = PH - 70;
  var panels = DS.map(function(ds, i){
    var svg = S('svg', { width: PW, height: PH, viewBox: '0 0 ' + PW + ' ' + PH, role: 'img', 'aria-label': ds + ': change in cost vs standard HNSW across recall' });
    svg.style.cssText = 'position:absolute;left:' + (i * (PW + GAP)) + 'px;top:52px;transition:opacity .4s';
    root.appendChild(svg); return svg;
  });
  function pts(ds, w, key){
    var by = R.matched[ds][w].by_ef0;
    return Object.keys(by).map(function(ef){ var c = by[ef]; return { ef: +ef, r: Math.min(c.recall, 0.9995), v: c[key] }; })
      .filter(function(p){ return p.v && p.v.mean !== null; }).sort(function(a, b){ return a.r - b.r; });
  }
  function draw(){
    segM.set(st.metric);
    var key = st.metric, all = [];
    DS.forEach(function(ds){ WS.forEach(function(w){ pts(ds, w, key).forEach(function(p){ all.push(p.v.mean); }); }); });
    var lo = Math.min(0, Math.min.apply(null, all)), hi = Math.max(0, Math.max.apply(null, all)), span = hi - lo;
    lo -= span * .1; hi += span * .1;
    var vy = lin(lo, hi, B, T), stp = span > 30 ? 10 : 5;
    panels.forEach(function(svg, i){
      var ds = DS[i]; svg.innerHTML = '';
      svg.style.opacity = i > st.step ? 0 : i < st.step && st.step < 3 ? .3 : 1;
      var rec = []; WS.forEach(function(w){ pts(ds, w, key).forEach(function(p){ rec.push(p.r); }); });
      var t = function(r){ return -Math.log10(1 - r); }, t0 = t(Math.min.apply(null, rec)), t1 = t(Math.max.apply(null, rec)), pad = (t1 - t0) * .08;
      var vx = function(r){ return lin(t0 - pad, t1 + pad, L, PW - 10)(t(r)); };
      S('text', { 'class': 'dn', x: 0, y: 22 }, svg, DSL[ds][0]);
      for (var v = Math.ceil(lo / stp) * stp; v <= hi; v += stp){
        S('line', { 'class': v === 0 ? 'zero' : 'ax', x1: L, x2: PW - 10, y1: vy(v), y2: vy(v), 'stroke-dasharray': v === 0 ? '' : '2 5' }, svg);
        S('text', { 'class': 'tk', x: L - 8, y: vy(v) + 5, 'text-anchor': 'end' }, svg, (v > 0 ? '+' : '') + v + '%');
      }
      [0.8, 0.9, 0.95, 0.99, 0.999].forEach(function(r){
        var tx = vx(r); if (tx < L - 1 || tx > PW - 9) return;
        S('line', { 'class': 'ax', x1: tx, x2: tx, y1: B, y2: B + 6 }, svg);
        S('text', { 'class': 'tk', x: tx, y: B + 24, 'text-anchor': 'middle' }, svg, String(r).replace(/^0/, ''));
      });
      S('text', { 'class': 'al', x: (L + PW) / 2, y: B + 50, 'text-anchor': 'middle' }, svg, 'recall@10 →');
      WS.forEach(function(w){
        var p = pts(ds, w, key); if (!p.length) return;
        S('polyline', { points: p.map(function(q){ return vx(q.r) + ',' + vy(q.v.mean); }).join(' '), fill: 'none', stroke: WC[w], 'stroke-width': 2.6 }, svg);
        p.forEach(function(q){ marker(svg, w, vx(q.r), vy(q.v.mean), 5.5).setAttribute('data-explain', DSL[ds].join(' ') + ', width ' + w + ', ef0 = ' + q.ef + ': recall ' + q.r.toFixed(4) + ', ' + MET[key] + ' ' + sgn(q.v.mean) + '% vs standard (95% CI ' + sgn(q.v.lo) + ' to ' + sgn(q.v.hi) + ')'); });
      });
    });
  }
  draw();
  return { step: function(n){ st.step = n; draw(); }, destroy: sh.destroy };
}});

/* ======================= R2: entry-point gain (survival) ======================= */
VIZ.register('r2', { mount: function(el){
  var R = window.RESULTS, sh = shell(el), root = sh.root;
  var st = { w: '4', thr: null, step: 0 };
  var ORDER = REAL.concat(CLUS);
  var bar = H('div', 'bar', root); bar.style.left = '0px'; bar.style.top = '0px';
  bar.appendChild(document.createTextNode('layer-1 width'));
  var segW = seg(bar, [['2', '2'], ['4', '4'], ['8', '8']], st.w, function(v){ st.w = v; draw(); });
  st.show = { 'Deep1M': 1, 'GloVe-25': 1, 'Clustered d=16': 1 };
  var THR_STEP = 0;   /* threshold line shown from the first step (Carlos, 2026-11-16); datasets appear one per step */
  var thrBox = H('span', '', bar); thrBox.style.marginLeft = '22px'; thrBox.setAttribute('data-r', THR_STEP);
  thrBox.appendChild(document.createTextNode('at least '));
  var segT = seg(thrBox, [['5', '5%'], ['10', '10%'], ['30', '30%'], ['50', '50%']], '10', function(v){ st.thr = v; draw(); });
  thrBox.appendChild(document.createTextNode(' closer'));

  var W = 790, Hh = 530, L = 78, Rr = W - 10, T = 20, B = Hh - 70;
  var svg = S('svg', { width: W, height: Hh, viewBox: '0 0 ' + W + ' ' + Hh, role: 'img',
    'aria-label': 'Distribution of how much closer the entry point got, share of all queries per bin, per dataset' });
  svg.style.position = 'absolute'; svg.style.left = '0px'; svg.style.top = '52px'; root.appendChild(svg);
  /* Smoothed histogram. The survival curve S(x) = share of all queries with gain >= x is exported on a log grid,
     20 points per decade from 0.1% to 100%. Each grid step S(x_i) - S(x_i+1) is the share of all queries in that
     slice; times 4 gives the share per 0.2-decade bin, the same unit as a 15-bin histogram, so the curve follows
     the bar tops instead of a per-decade density (which is 5x larger). Lightly smoothed with a Gaussian kernel
     (sigma = 1.5 grid steps) in log x. Summing the curve over every 4th point gives the share that changed. */
  var PER = 4;
  function pmf(e){
    var c = e.curve, raw = [];
    for (var i = 0; i + 1 < c.length; i++){
      var a = c[i][1], b = c[i + 1][1];
      raw.push({ x: Math.sqrt(c[i][0] * c[i + 1][0]), v: (a === null || b === null) ? 0 : Math.max(0, a - b) * PER });
    }
    var SG = 1.5;
    return raw.map(function(p, i){
      var sw = 0, sv = 0;
      for (var j = Math.max(0, i - 5); j <= Math.min(raw.length - 1, i + 5); j++){ var k = Math.exp(-(j - i) * (j - i) / (2 * SG * SG)); sw += k; sv += k * raw[j].v; }
      return { x: p.x, v: sv / sw };
    });
  }
  var YM = 0;
  ORDER.forEach(function(ds){ WS.forEach(function(w){ pmf(R.entry[ds][w]).forEach(function(q){ YM = Math.max(YM, q.v); }); }); });
  YM = Math.ceil(YM * 1.08 * 20) / 20;
  var lx = function(v){ return lin(-1, 2, L, Rr)(Math.log10(v)); }, ly = lin(0, YM, B, T);
  var yst = YM > .5 ? .1 : YM > .2 ? .05 : .02;
  for (var yv = 0; yv <= YM + 1e-9; yv += yst){
    S('line', { 'class': 'ax', x1: L, x2: Rr, y1: ly(yv), y2: ly(yv), 'stroke-dasharray': yv ? '2 5' : '' }, svg);
    S('text', { 'class': 'tk', x: L - 10, y: ly(yv) + 5, 'text-anchor': 'end' }, svg, Math.round(yv * 100) + '%');
  }
  [0.1, 0.3, 1, 3, 10, 30, 100].forEach(function(v){
    S('line', { 'class': 'ax', x1: lx(v), x2: lx(v), y1: B, y2: B + 6 }, svg);
    S('text', { 'class': 'tk', x: lx(v), y: B + 24, 'text-anchor': 'middle' }, svg, v + '%');
  });
  S('text', { 'class': 'al', x: (L + Rr) / 2, y: B + 52, 'text-anchor': 'middle',
    'data-explain': 'How much closer to q the entry point handed to layer 0 is, compared with standard HNSW on the same graph and query. Log scale.' }, svg, 'entry-point improvement →');
  S('text', { 'class': 'al', x: -B + 30, y: 18, transform: 'rotate(-90)',
    'data-keep': 1, 'data-explain': 'Share of all queries per 0.2-decade bin of improvement (smoothed histogram). Queries whose entry point did not change are not drawn.' }, svg, '% of queries per bin');
  var defs = S('defs', {}, svg), clip = S('clipPath', { id: 'r2thr' }, defs), clipR = S('rect', { x: 0, y: 0, width: 0, height: Hh }, clip);
  var gC = S('g', {}, svg), gThr = S('g', { 'data-r': THR_STEP }, svg);

  var fml = H('div', '', root); fml.style.cssText = 'position:absolute;left:830px;top:60px;width:354px';
  fml.innerHTML = '<div style="display:flex;align-items:center;gap:10px;font-size:22px;white-space:nowrap"><span>improvement = 1 −</span>' +
    '<span style="display:inline-flex;flex-direction:column;align-items:center;font-size:20px;white-space:nowrap"><span style="padding:0 6px 4px">d(q, ep<sub>wide</sub>)</span><span style="border-top:2px solid var(--ink);padding:4px 6px 0">d(q, ep<sub>std</sub>)</span></span></div>' +
    '<div style="font-size:16px;color:var(--mute);margin-top:10px;line-height:1.35">ep = entry point handed to layer 0</div>';
  var side = H('div', '', root); side.style.cssText = 'position:absolute;left:830px;top:230px;width:354px';
  var sideHd = H('div', '', side); sideHd.style.cssText = 'font-size:15px;letter-spacing:.12em;text-transform:uppercase;color:var(--mute);height:52px;line-height:1.35';
  var sideRows = ORDER.map(function(ds){
    var r = H('div', '', side); r.style.cssText = 'display:flex;align-items:center;gap:12px;height:62px;border-top:1.5px solid var(--line);cursor:pointer;transition:opacity .2s';
    r.setAttribute('role', 'button'); r.setAttribute('aria-pressed', 'true');
    r.addEventListener('click', function(){ st.show[ds] = st.show[ds] ? 0 : 1; draw(); });
    var sw = H('span', '', r); sw.style.cssText = 'width:22px;height:5px;border-radius:3px;background:' + DSC[ds];
    var nm = H('span', '', r, '<b style="font-weight:600;font-size:20px">' + DSL[ds][0] + '</b> <span class="m" style="color:var(--mute);font-size:15px">' + DSL[ds][1] + '</span>');
    nm.style.flex = '1';
    var v = H('span', 'm', r); v.style.cssText = 'font-size:24px';
    return { ds: ds, v: v, el: r };
  });

  function draw(){
    segW.set(st.w);
    var thr = st.step >= THR_STEP ? (st.thr || '10') : null;
    segT.set(thr || '10');
    gC.innerHTML = ''; gThr.innerHTML = '';
    if (thr) clipR.setAttribute('x', lx(+thr)), clipR.setAttribute('width', Rr - lx(+thr));
    ORDER.forEach(function(ds, k){
      if (!st.show[ds] || k > st.step) return;
      var e = R.entry[ds][st.w], c = DSC[ds], P = pmf(e);
      var d = 'M' + lx(P[0].x) + ',' + B + P.map(function(q){ return 'L' + lx(q.x).toFixed(1) + ',' + ly(q.v).toFixed(1); }).join('') + 'L' + lx(P[P.length - 1].x) + ',' + B + 'Z';
      var dl = 'M' + P.map(function(q){ return lx(q.x).toFixed(1) + ',' + ly(q.v).toFixed(1); }).join('L');
      S('path', { d: d, fill: c, 'fill-opacity': thr ? .06 : .12, stroke: 'none' }, gC);
      if (thr) S('path', { d: d, fill: c, 'fill-opacity': .4, stroke: 'none', 'clip-path': 'url(#r2thr)' }, gC);
      var ln = S('path', { d: dl, fill: 'none', stroke: c, 'stroke-width': CLUS.indexOf(ds) >= 0 ? 3.2 : 2.6, 'stroke-linejoin': 'round' }, gC);
      ln.setAttribute('data-explain', DSL[ds].join(' ') + ', width ' + st.w + ': entry point changed for ' + pct(e.changed) + ' of ' + e.queries.toLocaleString() + ' query × seed pairs; curve height = share of all queries per 0.2-decade bin');
    });
    if (thr){
      S('line', { x1: lx(+thr), x2: lx(+thr), y1: T, y2: B, stroke: 'var(--ink)', 'stroke-width': 1.6, 'stroke-dasharray': '6 5' }, gThr);
      S('text', { 'class': 'val', x: lx(+thr) + 8, y: T + 14 }, gThr, '≥ ' + thr + '% closer');
    }
    sideHd.innerHTML = '% of queries' + '<span style="float:right;text-transform:none;letter-spacing:0;font-size:15px">click to show</span>';
    sideRows.forEach(function(r){
      var k = ORDER.indexOf(r.ds);
      r.el.style.visibility = k > st.step ? 'hidden' : 'visible';
      r.el.style.opacity = st.show[r.ds] ? 1 : .35; r.el.setAttribute('aria-pressed', st.show[r.ds] ? 'true' : 'false');
      var e = R.entry[r.ds][st.w];
      r.v.textContent = thr ? pct1(e.at_least[thr]) : pct(e.changed);
    });
    root.querySelectorAll('[data-r]').forEach(function(e){ e.classList.toggle('off', +e.getAttribute('data-r') > st.step); });
  }
  draw();
  return { step: function(n){ st.step = n; if (n === 0){ st.w = '4'; st.thr = null; } draw(); }, destroy: sh.destroy };
}});

/* ======================= R3: net layer-0 cost saved per entry-point quality bin, width 4 ======================= */
VIZ.register('r3', { mount: function(el){
  /* Width 4, matched recall 0.95. Per entry-point-quality bin: bar = layer-0 distances saved per query,
     tick = extra layer-1 distances per query (the cost the saving has to clear), right column = net. */
  var R = window.RESULTS, sh = shell(el), root = sh.root, W = '4', RT = '0.95';
  var DS3 = ['Deep1M', 'GloVe-25', 'Clustered d=16'], PANELS = [];
  var PW = 376, GAPX = 28, PH = 500, L = 120, Rr = 312, T = 40, Bt = PH - 60, XMAX = 170;
  var lg = H('div', '', root, '<span style="display:inline-block;width:26px;height:14px;background:var(--l0);border-radius:3px;vertical-align:-1px;margin-right:8px"></span>layer-0 work saved'
    + '<span style="display:inline-block;width:3px;height:22px;background:var(--ink);vertical-align:-5px;margin:0 8px 0 34px"></span>extra layer-1 work'
    + '<span style="margin-left:34px;color:var(--mute)">distance computations per query · width 4 · matched recall 0.95</span>');
  lg.style.cssText = 'position:absolute;left:0;top:0;font-size:17px;color:var(--ink)';
  function rng(n){ return n === 'unchanged' ? 'unchanged' : n.replace(/(\d+\.\d)/g, function(m){ return String(Math.round(+m)); }); }
  DS3.forEach(function(ds, i){
    var svg = S('svg', { width: PW, height: PH, viewBox: '0 0 ' + PW + ' ' + PH, role: 'img', 'aria-label': ds + ': layer-0 work saved vs extra layer-1 work, per entry-point quality bin' });
    svg.style.position = 'absolute'; svg.style.left = (i * (PW + GAPX)) + 'px'; svg.style.top = '44px'; svg.style.transition = 'opacity .35s'; root.appendChild(svg); PANELS.push(svg);
    var x = lin(0, XMAX, L, Rr), B = R.balance[ds][W], keys = ['0', '1', '2', '3'];
    var tot = keys.reduce(function(a, k){ return a + B[k].queries_per_seed; }, 0);
    S('text', { 'class': 'dn', x: 0, y: 20 }, svg, DSL[ds][0]);
    S('text', { 'class': 'al', x: PW - 4, y: T + 4, 'text-anchor': 'end' }, svg, 'net');
    [0, 50, 100, 150].forEach(function(v){
      S('line', { 'class': 'ax', x1: x(v), x2: x(v), y1: T + 14, y2: Bt, 'stroke-dasharray': v ? '2 5' : '' }, svg);
      S('text', { 'class': 'tk', x: x(v), y: Bt + 22, 'text-anchor': 'middle' }, svg, v);
    });
    var rowH = (Bt - T - 14) / 4, bh = 30;
    keys.forEach(function(k, j){
      var g = B[k], sv = g.targets[RT].layer0_saved.mean, cost = g.layer1_extra.mean, net = sv - cost;
      var yc = T + 14 + rowH * (j + .5), y = yc - bh / 2;
      S('text', { 'class': 'al', x: L - 12, y: yc + 6, 'text-anchor': 'end' }, svg, j ? rng(g.name) : 'unchanged');
      S('rect', { x: L, y: y, width: Math.max(1.5, x(Math.max(0, sv)) - L), height: bh, rx: 3, fill: 'var(--l0)' }, svg);
      S('line', { x1: x(cost), x2: x(cost), y1: y - 7, y2: y + bh + 7, stroke: 'var(--ink)', 'stroke-width': 3 }, svg);
      S('text', { 'class': 'val', x: PW - 4, y: yc + 6, 'text-anchor': 'end', 'font-size': 17, style: 'fill:' + (net > 0 ? 'var(--res)' : 'var(--ink)') + ';font-weight:' + (net > 0 ? 700 : 400) }, svg, (net > 0 ? '+' : '−') + Math.abs(Math.round(net)));
    });
  });
  function show(n){ PANELS.forEach(function(p, i){ p.style.opacity = i > n ? 0 : (i < n && n < 3 ? .3 : 1); p.style.visibility = i <= n ? 'visible' : 'hidden'; }); }
  show(0);
  return { step: show, destroy: sh.destroy };
}});

/* ======================= SENS: layer-0 sensitivity to the entry point ======================= */
VIZ.register('sens', { mount: function(el){
  var D = window.SENSITIVITY, sh = shell(el), root = sh.root;
  var ORD = ['Deep1M', 'GloVe-25', 'Clustered d=16'];
  var W = 1180, Hh = 560, L = 70, Rr = W - 20, T = 30, B = Hh - 80;
  var svg = S('svg', { width: W, height: Hh, viewBox: '0 0 ' + W + ' ' + Hh, role: 'img',
    'aria-label': 'Percent of layer-0 work saved per 10 percent closer entry point, by ef0' });
  svg.style.position = 'absolute'; svg.style.left = '0px'; svg.style.top = '10px'; root.appendChild(svg);
  var lx = function(ef){ return lin(Math.log(10), Math.log(512), L + 10, Rr - 10)(Math.log(ef)); }, YM = 13, ly = lin(0, YM, B, T);
  [0, 2, 4, 6, 8, 10, 12].forEach(function(v){
    S('line', { 'class': v ? 'ax' : 'zero', x1: L, x2: Rr, y1: ly(v), y2: ly(v), 'stroke-dasharray': v ? '2 5' : '' }, svg);
    S('text', { 'class': 'tk', x: L - 10, y: ly(v) + 5, 'text-anchor': 'end' }, svg, v + '%');
  });
  D.ef0.forEach(function(ef){ S('text', { 'class': 'tk', x: lx(ef), y: B + 24, 'text-anchor': 'middle' }, svg, ef); });
  S('text', { 'class': 'al', x: (L + Rr) / 2, y: B + 52, 'text-anchor': 'middle' }, svg, 'ef0');
  S('text', { 'class': 'al', x: L, y: T - 12 }, svg, 'less layer-0 work for an entry point 10% closer');
  ORD.forEach(function(ds){
    var pts = D.data[ds].map(function(v, i){ return v ? { ef: D.ef0[i], m: v[0], lo: v[1], hi: v[2] } : null; }).filter(Boolean), c = DSC[ds];
    var band = pts.map(function(p){ return lx(p.ef) + ',' + ly(Math.min(YM, p.hi)); }).concat(pts.slice().reverse().map(function(p){ return lx(p.ef) + ',' + ly(p.lo); }));
    S('polygon', { points: band.join(' '), fill: c, 'fill-opacity': .12 }, svg);
    S('polyline', { points: pts.map(function(p){ return lx(p.ef) + ',' + ly(p.m); }).join(' '), fill: 'none', stroke: c, 'stroke-width': 3 }, svg);
    pts.forEach(function(p){ S('circle', { cx: lx(p.ef), cy: ly(p.m), r: 4.5, fill: c,
      'data-explain': DSL[ds].join(' ') + ', ef0 = ' + p.ef + ': ' + p.m.toFixed(1) + '% less layer-0 work per 10% closer entry point (95% CI ' + p.lo.toFixed(1) + '–' + p.hi.toFixed(1) + ' across seeds)' }, svg); });
    var last = pts[pts.length - 1], first = pts[0];
    var lbl = ds.indexOf('Clustered') === 0 ? DSL[ds].join(' ') : DSL[ds][0];
    var yy = ds === 'Clustered d=32' ? ly(first.m) + 24 : ds === 'Clustered d=16' ? ly(first.m) - 12 : ly(first.m) - 10;
    var xx = lx(first.ef) + 12;
    if (ds === 'Deep1M') { xx = lx(23) + 10; yy = ly(pts[1].m) - 10; }
    if (ds === 'GloVe-25') { xx = lx(23) + 6; yy = ly(pts[1].m) + 26; }
    S('text', { x: xx, y: yy, fill: c, 'font-size': 17, 'font-weight': 600 }, svg, lbl);
  });
  return { destroy: sh.destroy };
}});

/* ======================= IMPACT: who carries the layer-0 saving (width 4) ======================= */
VIZ.register('impact', { mount: function(el){
  var R = window.RESULTS, sh = shell(el), root = sh.root, W = '4', TI = 1;
  var DS3 = ['Deep1M', 'GloVe-25', 'Clustered d=16'];
  var BX = 250, BW = 600, RW = 300;
  var hd = H('div', '', root); hd.style.cssText = 'position:absolute;left:' + BX + 'px;top:0;width:' + BW + 'px;display:flex;gap:22px;font-size:16px';
  hd.innerHTML = '<span class="lg"><span style="width:16px;height:16px;background:var(--q);border-radius:3px"></span>largest-gain queries</span>' +
                 '<span class="lg"><span style="width:16px;height:16px;background:var(--line);border-radius:3px"></span>all other queries</span>';
  var nh = H('div', '', root); nh.style.cssText = 'position:absolute;left:' + (BX + BW + 40) + 'px;top:-4px;width:' + (RW - 20) + 'px;font-size:15px;letter-spacing:.1em;text-transform:uppercase;color:var(--mute);line-height:1.35';
  nh.textContent = 'net saved per query, 95% CI';
  var rn = H('div', '', root, 'matched recall 0.95'); rn.style.cssText = 'position:absolute;left:0;top:0;font-size:16px;color:var(--mute)';
  DS3.forEach(function(ds, i){
    var B = R.balance[ds][W], g = ['0', '1', '2', '3'].map(function(k){ return B[k]; });
    var tgt = '0.95';
    var N = 0, Sv = 0, X = 0;
    g.forEach(function(x){ N += x.queries_per_seed; Sv += x.targets[tgt].layer0_saved.mean * x.queries_per_seed; X += x.layer1_extra.mean * x.queries_per_seed; });
    var qShare = g[3].queries_per_seed / N, sShare = g[3].targets[tgt].layer0_saved.mean * g[3].queries_per_seed / Sv;
    var net = (Sv - X) / N, netRest = (Sv - g[3].targets[tgt].layer0_saved.mean * g[3].queries_per_seed - (X - g[3].layer1_extra.mean * g[3].queries_per_seed)) / N;
    /* per-seed numbers and 95% t intervals across seeds (data/intervals.js), when present */
    var I = window.INTERVALS && window.INTERVALS.impact[ds], sCI = null, nCI = null;
    if (I) { qShare = I.q_share.mean; sShare = I.s_share.mean; net = I.net.mean; netRest = I.net_rest.mean; sCI = I.s_share; nCI = I.net; }
    var top = 56 + i * 170;
    var nm = H('div', '', root); nm.style.cssText = 'position:absolute;left:0;top:' + (top + 20) + 'px;width:230px';
    nm.innerHTML = '<div style="font-size:24px;font-weight:600">' + DSL[ds][0] + ' <span class="m" style="font-size:15px;color:var(--mute)">' + DSL[ds][1] + '</span></div>' ;
    [['queries', qShare, 'of queries'], ['layer-0 work saved', sShare, 'of the layer-0 saving']].forEach(function(r, k){
      var y = top + k * 62;
      var lab = H('div', '', root, r[0]); lab.style.cssText = 'position:absolute;left:' + BX + 'px;top:' + y + 'px;font-size:15px;color:var(--mute)';
      var bar = H('div', '', root); bar.style.cssText = 'position:absolute;left:' + BX + 'px;top:' + (y + 22) + 'px;width:' + BW + 'px;height:28px;border-radius:6px;background:var(--line);overflow:hidden';
      var f = H('div', '', bar); f.style.cssText = 'height:100%;width:' + (r[1] * 100) + '%;background:var(--q)';
      bar.setAttribute('data-explain', DSL[ds].join(' ') + ', width 4, recall ' + tgt + ': the largest-gain group (entry point ' + g[3].name + ') is ' + pct(qShare) + ' of queries and ' + pct(sShare) + ' of the layer-0 distance computations saved (group means × group sizes).');
      var v = H('div', 'm', root, pct(r[1])); v.style.cssText = 'position:absolute;left:' + (BX + Math.max(r[1] * BW, 0) + 10) + 'px;top:' + (y + 22) + 'px;font-size:22px;line-height:28px;font-weight:500;color:' + (k ? 'var(--q)' : 'var(--ink)');
    });
    var nb = H('div', '', root); nb.style.cssText = 'position:absolute;left:' + (BX + BW + 40) + 'px;top:' + (top + 16) + 'px;width:' + RW + 'px';
    var good = nCI ? nCI.lo > 0 : net > 0;
    nb.innerHTML = '<div class="m" style="font-size:38px;font-weight:500;color:' + (good ? 'var(--res)' : 'var(--ink)') + '">' + sgn(net, 0) + '</div>' +
      (nCI ? '<div class="m" style="font-size:18px;white-space:nowrap;color:var(--mute);margin-top:2px">[' + sgn(nCI.lo, 0) + ', ' + sgn(nCI.hi, 0) + ']</div>' : '');
    nb.setAttribute('data-keep', 1); nb.setAttribute('data-explain', 'Distance computations per query, all queries: layer-0 saved ' + (Sv / N).toFixed(1) + ' minus extra layer-1 ' + (X / N).toFixed(1) + ' = ' + sgn(net) + '. Positive = less total work than standard at the same recall.');
  });
  return { destroy: sh.destroy };
}});
})();
