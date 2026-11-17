// Level-0 sensitivity to the entry point (run 3). Copied from notebooks/run3-refactor/run3_refactored.ipynb,
// "Figure 6" table rd.elasticity(queries): pct_per_10pct = % level-0 distance computations saved per 10% closer
// entry point; seed mean with 95% t interval; widths pooled. GloVe-25: same table in run4_glove25.ipynb (cell 21).
window.SENSITIVITY = {
  // median gain of the queries whose entry point changed, width 4 (notebook, entry-gain table changed_p50, %)
  typical: { 'Deep1M': 8.12 },
  ef0: [10, 23, 52, 79, 120, 182, 275, 512],
  data: {
    'Deep1M':         [[9.080,8.703,9.457],[5.357,5.125,5.589],[2.736,2.623,2.849],[1.834,1.724,1.944],[1.189,1.099,1.279],[0.750,0.686,0.814],[0.458,0.412,0.505],[0.210,0.181,0.240]],
    'GloVe-25':       [[7.367,7.232,7.501],[4.155,4.098,4.213],[2.020,1.977,2.063],[1.317,1.288,1.347],[0.835,0.813,0.856],[0.512,0.498,0.527],[0.307,0.296,0.317],[0.133,0.127,0.139]],
    'Clustered d=16': [[2.052,1.874,2.229],[1.943,1.798,2.088],[1.754,1.661,1.848],[1.546,1.437,1.654],[1.339,1.237,1.442],null,null,null]
  }
};
