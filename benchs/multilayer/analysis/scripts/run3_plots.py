"""Run 3 plot helpers. Each takes a tidy frame from run3_data plus column names and draws on an `ax`.

No Parquet reads and no aggregation beyond the seed mean and the 95% t band
(run3_data.across_seeds). Axis labels come from run3_data.LABELS.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

import run3_data as rd
from run3_data import CLUSTERED, LABELS

# One style per width: standard black, w on one blue ramp.
STYLE = {
    1: dict(color="#000000", marker="o", name="standard"),
    2: dict(color="#86b6ef", marker="s", name="w=2"),
    4: dict(color="#3987e5", marker="^", name="w=4"),
    8: dict(color="#184f95", marker="D", name="w=8"),
}
BAND = "#3987e5"  # Figure 6 percentile bands
BAND_ALPHA = 0.12  # 95% t bands

RC = {
    "figure.dpi": 100, "savefig.dpi": 100, "font.size": 10, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.2, "grid.linewidth": 0.6,
    "axes.edgecolor": "#888888", "xtick.color": "#555555", "ytick.color": "#555555",
    "legend.frameon": False,
}


def setup():
    """Apply the notebook's matplotlib style."""
    plt.rcParams.update(RC)


def label(col, ax=None, axis="y"):
    """LABELS[col], with ' (log)' added when that axis of `ax` is logarithmic."""
    text = LABELS.get(col, col)
    if ax is not None and (ax.get_yscale() if axis == "y" else ax.get_xscale()) == "log":
        text += " (log)"
    return text


# --------------------------------------------------------------------------- axes


def recall_axis(ax, dataset):
    """Clustered sets: 1 − recall on a reversed log axis (better recall to the right). Others: recall.
    Returns the column to put on x. Call before drawing so curves() can drop points off a log axis."""
    if dataset in CLUSTERED:
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel(LABELS["one_minus_recall"] + " (log, reversed)")
        return "one_minus_recall"
    ax.set_xlabel(LABELS["recall_at_10"])
    return "recall_at_10"


def ef_axis(ax, efs):
    """Log ef0 axis with one labelled tick per grid value."""
    ax.set_xscale("log")
    ax.set_xticks(list(efs))
    ax.set_xticklabels([str(e) for e in efs], rotation=45 if len(efs) > 5 else 0)
    ax.minorticks_off()
    ax.set_xlabel(LABELS["ef0"])


# --------------------------------------------------------------------------- curves


def curves(ax, df, x, y, by="width", order="ef0", dots=True, band=True, ls="-"):
    """Per value of `by`: each seed's point as a light dot, the seed mean over `order` as a line,
    and the 95% t interval of y across seeds as a band. Non-positive values are dropped on log axes."""
    d = df[list(dict.fromkeys([by, order, "seed", x, y]))].copy()
    for col, scale in ((x, ax.get_xscale()), (y, ax.get_yscale())):
        if scale == "log":
            d.loc[d[col] <= 0, col] = np.nan
    for key, dk in d.groupby(by, observed=True):
        st = STYLE[key]
        if dots:
            ax.scatter(dk[x], dk[y], s=9, color=st["color"], alpha=0.3, lw=0, zorder=2)
        g = dk.groupby(order)
        my = g[y].mean().sort_index()
        mx = my.index.to_series() if x == order else g[x].mean().reindex(my.index)
        if band:
            ci = rd.across_seeds(dk.dropna(subset=[y]), y, order).reindex(my.index)
            ax.fill_between(mx.to_numpy(), ci.ci_lo.to_numpy(), ci.ci_hi.to_numpy(), color=st["color"], alpha=BAND_ALPHA, lw=0)
        ax.plot(mx.to_numpy(), my.to_numpy(), color=st["color"], marker=st["marker"], ms=4.5, lw=1.6, ls=ls,
                mec="white", mew=0.5, zorder=3)
    if not ax.get_xlabel():
        ax.set_xlabel(label(x, ax, "x"))
    if not ax.get_ylabel():
        ax.set_ylabel(label(y, ax, "y"))


def delta_vs_standard(ax, df, x, y_cols, by="width", styles=("-", "--")):
    """Figure 3: seed-mean % delta per ef0 against mean x, one line style per y column, with the
    95% t band across seeds; rows where any y column is NaN (outside the standard range) are left out."""
    d = df.dropna(subset=list(y_cols))
    for key, dk in d.groupby(by, observed=True):
        st = STYLE[key]
        xm = dk.groupby("ef0")[x].mean()
        for col, ls in zip(y_cols, styles):
            ci = rd.across_seeds(dk, col, "ef0").reindex(xm.index)
            ax.fill_between(xm.to_numpy(), ci.ci_lo.to_numpy(), ci.ci_hi.to_numpy(), color=st["color"], alpha=0.09, lw=0)
            ax.plot(xm.to_numpy(), ci["mean"].to_numpy(), color=st["color"], ls=ls, marker=st["marker"], ms=4, lw=1.6)
    ax.axhline(0, color="#000000", lw=0.9)
    if not ax.get_ylabel():
        ax.set_ylabel(LABELS["delta_pct"])


# --------------------------------------------------------------------------- Figure 5


def entry_violins(ax, ent, log=False, max_points=30000):
    """Figure 5: one violin per width of entry_dist (seeds pooled), cut at p0.5–p99.5, with the
    interquartile bar, the median dot, and the share of queries with ep1 unchanged on top."""
    widths = sorted(ent.width.unique())
    tops = []
    for i, w in enumerate(widths):
        g = ent[ent.width == w]
        v = g.entry_dist.to_numpy()
        lo, q1, med, q3, hi = np.percentile(v, [0.5, 25, 50, 75, 99.5])
        vc = v[(v >= lo) & (v <= hi)]
        parts = ax.violinplot(vc[:: max(1, len(vc) // max_points)], positions=[i], widths=0.8, showextrema=False, points=120)
        for p in parts["bodies"]:
            p.set_facecolor(STYLE[w]["color"])
            p.set_edgecolor(STYLE[w]["color"])
            p.set_alpha(0.55)
        ax.vlines(i, q1, q3, color="#000000" if w != 1 else "#ffffff", lw=3, zorder=3)
        ax.scatter(i, med, s=22, color="white", edgecolor="#000000", lw=0.8, zorder=4)
        tops.append((i, hi, g.entry_unchanged.mean(), w))
    top = max(t[1] for t in tops)
    for i, hi, same, w in tops:
        ax.text(i, top * 1.03, "–" if w == 1 else f"{same:.0%}", ha="center", va="bottom", fontsize=9, color="#333333")
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator([2, 3, 4, 5, 6, 8, 10, 15, 20, 30]))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(top=top * 1.12)
    ax.set_xticks(range(len(widths)), [STYLE[w]["name"] for w in widths])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel(label("entry_dist", ax))


# --------------------------------------------------------------------------- Figure 6

PCT_TICKS = [0, 10, 20, 30, 50, 70, 80, 90, 5, -5, -10, -20, -50]  # priority order


def _binned(x, y, max_bins=25, min_per_bin=30):
    """Quantile bins of x: per bin the median x and the p5, p25, p50, p75, p95 of y."""
    nb = min(max_bins, len(x) // min_per_bin)
    if nb < 3:
        return None
    edges = np.unique(np.quantile(x, np.linspace(0, 1, nb + 1)))
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    out = [(np.median(x[idx == i]), *np.percentile(y[idx == i], [5, 25, 50, 75, 95]))
           for i in range(len(edges) - 1) if (idx == i).any()]
    return np.array(out).T


def _pct_ticks(lo, hi):
    """Percent labels on the log-ratio axis, skipping any that would collide."""
    kept = []
    for p in PCT_TICKS:
        v = -np.log(1 - p / 100)
        if lo <= v <= hi and all(abs(v - k) >= 0.085 * (hi - lo) for k in kept):
            kept.append(v)
    kept.sort()
    return kept, [f"{100 * (1 - np.exp(-v)):.0f}%" for v in kept]


def sensitivity_fan(ax, q, x="entry_closer_logratio", y="layer0_saved_rel_pct"):
    """Figure 6, one dataset and ef0 (run3_data.fig6_pairs): per-query level-0 saving vs how much
    closer ep1 is. Queries with x != 0 in quantile bins: p5–p95 and p25–p75 bands, pooled median
    (dotted) and one median line per width. Queries with x = 0 (ep1 unchanged) as a box at 0."""
    xv, yv, wv = q[x].to_numpy(), q[y].to_numpy(), q.width.to_numpy()
    nz = xv != 0
    tb = _binned(xv[nz], yv[nz])
    band_top = -np.inf
    if tb is not None:
        bx, p5, p25, p50, p75, p95 = tb
        band_top = p95.max()
        ax.fill_between(bx, p5, p95, color=BAND, alpha=0.14, lw=0)
        ax.fill_between(bx, p25, p75, color=BAND, alpha=0.26, lw=0)
        ax.plot(bx, p50, color="#555555", lw=1.2, ls=":")
    for w in sorted(np.unique(wv)):
        m = nz & (wv == w)
        tw = _binned(xv[m], yv[m])
        if tw is not None:
            ax.plot(tw[0], tw[3], color=STYLE[w]["color"], lw=1.6)
    xl = np.quantile(xv, [0.005, 0.995])
    xl = (min(xl[0], -0.01), max(xl[1], 0.02))
    span = xl[1] - xl[0]
    z = np.percentile(yv[~nz], [5, 25, 50, 75, 95])
    ax.bxp([dict(med=z[2], q1=z[1], q3=z[3], whislo=z[0], whishi=z[4], fliers=[])], positions=[0], widths=0.05 * span,
           showfliers=False, patch_artist=True, manage_ticks=False, boxprops=dict(facecolor="white", edgecolor="#000000"),
           medianprops=dict(color="#000000", lw=1.6), zorder=5)
    yl = np.percentile(yv, [1, 99])
    ax.axhline(0, color="#000000", lw=0.8)
    ax.set_xticks(*_pct_ticks(*xl))
    ax.set(xlim=(xl[0] - 0.05 * span, xl[1] + 0.05 * span), ylim=(min(yl[0], -2), max(yl[1], band_top, 2)),
           title=f"ef0={q.ef0.iloc[0]}  (ep1 unchanged {(~nz).mean():.0%})",
           xlabel=LABELS["entry_closer_logratio"], ylabel=LABELS[y])


def sensitivity_legend(fig, widths=(2, 4, 8)):
    handles = [Line2D([], [], color=STYLE[w]["color"], lw=1.8, label=f"median, {STYLE[w]['name']}") for w in widths] + [
        Line2D([], [], color="#555555", ls=":", label="median, pooled"),
        plt.Rectangle((0, 0), 1, 1, color=BAND, alpha=0.26, label="p25–p75"),
        plt.Rectangle((0, 0), 1, 1, color=BAND, alpha=0.14, label="p5–p95"),
        plt.Rectangle((0, 0), 1, 1, fc="white", ec="#000000", label="box at 0%: ep1 unchanged")]
    fig.legend(handles=handles, loc="outside lower center", ncol=7, fontsize=8.5)


# --------------------------------------------------------------------------- legend


def width_legend(fig, widths=(1, 2, 4, 8), y=1.0, extra=(), band=True):
    """One legend row above the figure: one entry per width, then `extra` (label, linestyle) entries."""
    handles = [Line2D([], [], color=STYLE[w]["color"], marker=STYLE[w]["marker"], lw=1.8, ms=6, label=STYLE[w]["name"])
               for w in widths]
    handles += [Line2D([], [], color="#555555", ls=ls, label=lab) for lab, ls in extra]
    if band:
        handles.append(plt.Rectangle((0, 0), 1, 1, color="#555555", alpha=0.2, label="95% t interval across seeds"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, y))


# --------------------------------------------------------------------------- Figures 7–8: entry-gain groups

GROUP_COLOR = ["#9a9a9a", "#a6d9a0", "#4fae5b", "#1b6b33"]  # unchanged, then small → large entry gain
TARGET_COLOR = ["#c4b5e8", "#9a83d6", "#6c4fc0", "#3f2a8a"]  # recall targets, low → high


def group_recall_cost(ax, c, names, x="recall_at_10", y="dist_layer0"):
    """Figure 7, one dataset and width: per entry-gain group, the recall–cost curve of the group's
    queries under w (solid) and under standard (dashed), seed means over the ef0 grid.
    Points with recall 1 are dropped on a log 1 − recall axis."""
    for gg, d in c.groupby("gain_group"):
        m = d.groupby("ef0")[[x, x + "_std", y, y + "_std"]].mean()
        if ax.get_xscale() == "log":
            m = m.where(m > 0)
        col = GROUP_COLOR[gg]
        ax.plot(m[x], m[y], color=col, lw=1.7, marker="o", ms=3.5, label=names[gg])
        ax.plot(m[x + "_std"], m[y + "_std"], color=col, lw=1.3, ls="--", marker="o", ms=3, mfc="white")
    ax.plot([], [], color="#555555", lw=1.7, label="solid: w")
    ax.plot([], [], color="#555555", lw=1.3, ls="--", label="dashed: standard")
    ax.set_ylabel(label(y, ax))


def group_balance(ax, gm, names):
    """Figure 8, one dataset and width: x = entry-gain group. Solid lines: level-0 saved at matched
    recall, one per recall target (seed mean, 95% t interval as error bars). Black dashed: extra
    level-1 work of the same queries (no recall dependence). Net = the gap between them."""
    pos = np.arange(len(names))
    for i, (t, d) in enumerate(gm.groupby("recall_target")):
        st = rd.across_seeds(d, "layer0_saved", "gain_group").reindex(range(len(names)))
        ax.errorbar(pos + (i - 1.5) * 0.04, st["mean"], yerr=[st["mean"] - st.ci_lo, st.ci_hi - st["mean"]],
                    color=TARGET_COLOR[i], marker="o", ms=4, lw=1.6, capsize=2, label=f"level-0 saved at R = {t:g}")
    l1 = rd.across_seeds(gm.drop_duplicates(["seed", "gain_group"]), "layer1_extra", "gain_group").reindex(range(len(names)))
    ax.errorbar(pos, l1["mean"], yerr=[l1["mean"] - l1.ci_lo, l1.ci_hi - l1["mean"]], color="#000000", ls="--",
                marker="s", ms=4, lw=1.4, capsize=2, label="extra level-1")
    ax.axhline(0, color="#000000", lw=0.6)
    ax.set_xticks(pos, [n.replace(" closer", "\ncloser") for n in names], fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("distance computations per query")


def entry_gain_survival(ax, q, edges=None):
    """Share of queries whose ep1 is at least x% closer under w (seeds pooled), one line per width,
    log x axis. The curve starts at the share of queries whose ep1 changed; the rest are unchanged.
    `edges` {width: [gain % ...]} marks the entry-gain group edges of Figures 7–8 as ticks on each line."""
    base = q.ef0.min()
    q = q[q.ef0 == base]
    xs = np.logspace(-1, 2, 300)
    for w, g in q.groupby("width"):
        gain = np.sort(100 * (1 - np.exp(-g.entry_closer_logratio.dropna().to_numpy())))
        n = len(g)
        surv = 1 - np.searchsorted(gain, xs, side="left") / n
        st = STYLE[w]
        ax.plot(xs, surv, color=st["color"], lw=1.8, label=f"{st['name']} (changed {(~g.entry_unchanged).mean():.0%})")
        for e in (edges or {}).get(w, []):
            ax.plot(e, 1 - np.searchsorted(gain, e, side="left") / n, marker="|", ms=10, mew=1.6, color=st["color"])
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.3, 1, 3, 10, 30, 100], ["0.1%", "0.3%", "1%", "3%", "10%", "30%", "100%"])
    ax.minorticks_off()
    ax.set(xlabel=LABELS["entry_gain_pct"], ylabel="share of queries at least this much closer", ylim=(0, None))
