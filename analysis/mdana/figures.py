"""Stage 6: figures. All read the tables written by aggregate.run() plus the per-frame time series.

Main figures (replace the old Fig. 6 and Tables 1-2):
  Fig_MD_rmsd          ligand RMSD vs time per system (replica lines, mean ± SEM band, estradiol band for reference)
  Fig_MD_interactions  a  pose retention (replica values, mean ± 95 % CI)
                       b  ΔΔE vs estradiol at ε_in = 1, 2, 4 (GB) and 2 (PB), Welch 95 % CI; decoy (negative control) set apart
                       c  ligand–residue contact occupancy (H-bond occupancy printed where ≥ 10 %)
SI figures S1-S8: receptor and helix-12 RMSD, RMSF, SRC2/charge clamp, ligand position, convergence, per-residue
decomposition, GB/PB and ε_in sensitivity, thermodynamic QC.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import layout as L
from . import figstyle as FS
from . import stats as S
from .aggregate import TABLES, RETAIN_A, CLAMP_A, DISSOC_A

PLOT_STRIDE = 10            # 100 ps between plotted points
SMOOTH = 10                 # 1-ns running mean for the plotted traces
METHODS = [("gb", 1, "GB, ε$_{in}$ = 1", "o"), ("gb", 2, "GB, ε$_{in}$ = 2", "s"),
           ("gb", 4, "GB, ε$_{in}$ = 4", "D"), ("pb", 2, "PB, ε$_{in}$ = 2", "^")]


def tab(name):
    p = TABLES / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def smoothed(ts, col):
    """Per replica: 1-ns running mean, then every 100 ps; returns time and a (rep × time) array."""
    out, t = [], None
    for rep in L.REPLICAS:
        d = ts[ts.rep == rep].sort_values("time_ns")
        if d.empty:
            continue
        y = d[col].rolling(SMOOTH, center=True, min_periods=1).mean().to_numpy()[::PLOT_STRIDE]
        t = d.time_ns.to_numpy()[::PLOT_STRIDE]
        out.append(y)
    return t, np.array(out)


def band(ax, t, Y, color, lw_rep=0.35, alpha_band=0.25, label=None, ls="-", reps=True):
    if len(Y) == 0:
        return
    if reps:
        for y in Y:
            ax.plot(t, y, color=color, lw=lw_rep, alpha=0.55, ls=ls)
    m = Y.mean(0)
    if len(Y) > 1:
        se = Y.std(0, ddof=1) / np.sqrt(len(Y))
        ax.fill_between(t, m - se, m + se, color=color, alpha=alpha_band, lw=0)
    ax.plot(t, m, color=color, lw=0.9, ls=ls, label=label)


def shade_eq(ax):
    ax.axvspan(0, L.T_EQ_NS, color="0.92", lw=0, zorder=0)


def dots_ci(ax, x, vals, color, marker="o", size=9, bounds=None):
    """Replica values as jittered dots plus mean and t-based 95 % CI (drawn clipped to `bounds`, e.g. 0-100 % for a
    fraction: with n = 3 the t interval can run past what the quantity can take)."""
    vals = np.asarray(vals, dtype=float)
    jit = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else [0]
    ax.scatter(x + np.asarray(jit), vals, s=size, color=color, alpha=0.55, lw=0, zorder=3, marker=marker)
    s = S.summarize(vals)
    if s["n"] >= 2:
        lo, hi = (s["lo"], s["hi"]) if bounds is None else (max(s["lo"], bounds[0]), min(s["hi"], bounds[1]))
        ax.errorbar(x, s["mean"], yerr=[[s["mean"] - lo], [hi - s["mean"]]], fmt="_", color=color,
                    ms=8, mew=1.0, elinewidth=0.8, capsize=0, zorder=4)
    return s


# ------------------------------------------------------------------------------------------------ main figure
HEAT_OCC = 0.5               # residues shown in the main-figure heatmap


def main_figure(ts, show=True):
    """The former single main figure, as two: ligand RMSD traces, and retention/energies/interactions."""
    return {"rmsd": fig_main_rmsd(ts, show), "interactions": fig_main_interactions(show)}


def fig_main_rmsd(ts, show=True):
    """Ligand RMSD against time, one panel per ligand."""
    lig = L.LIGANDED
    fig = plt.figure(figsize=(FS.DOUBLE, 86 * FS.MM))
    ga = GridSpec(2, 4, figure=fig, hspace=0.42, wspace=0.12, top=0.86, bottom=0.12, left=0.07, right=0.99)
    ref_ts = ts[ts.sys == L.REFERENCE]
    tr, Yr = smoothed(ref_ts, "lig_rmsd")
    ymax = max(6.0, np.nanpercentile(ts.lig_rmsd, 99.5) * 1.05) if ts.lig_rmsd.notna().any() else 6.0
    axes = []
    for k, s in enumerate(lig):
        ax = fig.add_subplot(ga[k // 4, k % 4], sharey=axes[0] if axes else None)
        axes.append(ax)
        shade_eq(ax)
        if s.key != L.REFERENCE and len(Yr):
            m, se = Yr.mean(0), Yr.std(0, ddof=1) / np.sqrt(len(Yr))
            ax.fill_between(tr, m - se, m + se, color="0.6", alpha=0.35, lw=0)
        ax.axhline(RETAIN_A, color="0.35", lw=0.5, ls=":")
        t, Y = smoothed(ts[ts.sys == s.key], "lig_rmsd")
        if t is not None:
            band(ax, t, Y, s.color, ls=s.linestyle)
        ax.set_title(s.label, color="0.0", pad=2)
        ax.set_xlim(0, L.PROD_NS)
        ax.set_ylim(0, ymax)
        ax.set_xticks([0, 50, 100])
        if k % 4:
            plt.setp(ax.get_yticklabels(), visible=False)
        else:
            ax.set_ylabel("Ligand RMSD (Å)")
        if k // 4 == 1:
            ax.set_xlabel("Time (ns)")
    handles = [Line2D([], [], color="k", lw=0.9, label="mean ± SEM"),
               Line2D([], [], color="k", lw=0.35, alpha=0.6, label="replicas"),
               Patch(color="0.6", alpha=0.35, lw=0, label="estradiol (mean ± SEM)"),
               Patch(color="0.92", lw=0, label="equilibration (excluded)"),
               Line2D([], [], color="0.35", lw=0.5, ls=":", label=f"{RETAIN_A:.0f} Å")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.07, 0.975), ncol=5, fontsize=6.5)
    fig.text(0.99, 0.975, FS.N_NOTE, ha="right", va="top", fontsize=6.5, color="0.3")
    FS.save(fig, "Fig_MD_rmsd", show=show)
    return fig


def fig_main_interactions(show=True):
    """a pose retention, b ΔΔE vs estradiol, c contact/H-bond occupancy (full width)."""
    rep = tab("structure_per_replica")
    occ = tab("occupancy")
    dd = tab("ddE_vs_estradiol")
    lig = L.LIGANDED

    fig = plt.figure(figsize=(FS.DOUBLE, 140 * FS.MM))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.0], hspace=0.38, top=0.95, bottom=0.11, left=0.11,
                  right=0.97)
    gt = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], width_ratios=[0.8, 1.2], wspace=0.55)

    # a: pose retention, systems on y as in b and c
    ax = fig.add_subplot(gt[0])
    for i, s in enumerate(lig):
        v = rep[rep.sys == s.key].retention.to_numpy() * 100 if not rep.empty else []
        if len(v):
            dots_ci_h(ax, i, v, s.color, bounds=(0, 100))
    ax.set_yticks(range(len(lig)))
    ax.set_yticklabels([s.label for s in lig])
    ax.set_ylim(len(lig) - 0.4, -0.6)
    ax.set_xlim(-3, 103)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(f"Pose retention (% frames < {RETAIN_A:.0f} Å)")
    FS.panel_label(ax, "a", x=-0.42, y=1.02)

    # b: ΔΔE forest plot
    ax = fig.add_subplot(gt[1])
    forest(ax, dd, [s for s in lig if s.key != L.REFERENCE])
    FS.panel_label(ax, "b", x=-0.30, y=1.02)

    # c: occupancy heatmap, full width
    ax = fig.add_subplot(gs[1])
    if not occ.empty:
        keep = occ.groupby("label").contact_mean.max()
        cols = sorted(set(keep[keep >= HEAT_OCC].index) | set(L.ANCHORS), key=_res_order)
        M = np.full((len(lig), len(cols)), np.nan)
        H = np.zeros_like(M)
        for i, s in enumerate(lig):
            d = occ[occ.sys == s.key].set_index("label")
            for j, c in enumerate(cols):
                M[i, j] = d.contact_mean.get(c, 0.0)
                H[i, j] = d.hbond_mean.get(c, 0.0)
        im = ax.imshow(M * 100, aspect="auto", cmap="Blues", vmin=0, vmax=100, interpolation="nearest")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if H[i, j] >= 0.10:
                    ax.text(j, i, f"{100 * H[i, j]:.0f}", ha="center", va="center", fontsize=6,
                            color="white" if M[i, j] > 0.6 else "black")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=90)
        ax.set_yticks(range(len(lig)))
        ax.set_yticklabels([s.label for s in lig])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        for j, c in enumerate(cols):
            if c in L.ANCHORS:
                ax.get_xticklabels()[j].set_fontweight("bold")
        cb = fig.colorbar(im, ax=ax, ticks=[0, 50, 100], fraction=0.025, pad=0.015)
        cb.set_label("Contact occupancy (%)", fontsize=6.5)
        cb.ax.tick_params(labelsize=6, length=1.5, pad=1)
        cb.outline.set_linewidth(0.4)
        # the ≥ HEAT_OCC column rule goes in the figure legend
        ax.text(0.0, 1.02, "numbers: H-bond occupancy (%)", transform=ax.transAxes, fontsize=6.5, color="0.3",
                va="bottom")
    FS.panel_label(ax, "c", x=-0.075, y=1.02)
    fig.text(0.97, 0.985, FS.N_NOTE, ha="right", va="top", fontsize=6.5, color="0.3")
    FS.save(fig, "Fig_MD_interactions", show=show)
    return fig


def dots_ci_h(ax, y, vals, color, size=9, bounds=None):
    """dots_ci with the value on x: replica dots plus mean and 95 % CI, for a panel with systems on y."""
    vals = np.asarray(vals, dtype=float)
    jit = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else [0]
    ax.scatter(vals, y + np.asarray(jit), s=size, color=color, alpha=0.55, lw=0, zorder=3)
    s = S.summarize(vals)
    if s["n"] >= 2:
        lo, hi = (s["lo"], s["hi"]) if bounds is None else (max(s["lo"], bounds[0]), min(s["hi"], bounds[1]))
        ax.errorbar(s["mean"], y, xerr=[[s["mean"] - lo], [hi - s["mean"]]], fmt="|", color=color,
                    ms=8, mew=1.0, elinewidth=0.8, capsize=0, zorder=4)
    return s


def _res_order(label):
    import re
    m = re.search(r"(\d+)$", label)
    base = 1000 if label.startswith("SRC2") else 0
    return base + (int(m.group(1)) if m else 0)


def forest(ax, dd, systems):
    if dd.empty:
        ax.text(0.5, 0.5, "MM/GBSA pending", transform=ax.transAxes, ha="center")
        return
    off = np.linspace(-0.27, 0.27, len(METHODS))
    for i, s in enumerate(systems):
        for k, (meth, eps, lab, mk) in enumerate(METHODS):
            d = dd[(dd.sys == s.key) & (dd.method == meth) & (dd.eps == eps)]
            if d.empty:
                continue
            r = d.iloc[0]
            y = i + off[k]
            ax.errorbar(r["diff"], y, xerr=[[r["diff"] - r["lo"]], [r["hi"] - r["diff"]]], fmt=mk, ms=2.8,
                        color=s.color, mfc=s.color if meth == "gb" and eps == 2 else "white", mew=0.7,
                        elinewidth=0.7, capsize=0)
    ax.axvline(0, color="k", lw=0.6)
    labels = [s.label for s in systems]
    # The decoy is the negative control, not a candidate: set it apart below a rule so its (unresolved, wide-CI)
    # GB score is not read as the best ligand.
    if systems and systems[-1].key == "decoy":
        ax.axhline(len(systems) - 1.5, color="0.6", lw=0.5, ls=(0, (4, 2)))
        labels[-1] = "Decoy\n(negative control)"
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("ΔΔE vs estradiol (kcal mol$^{-1}$)\n← more favourable     less favourable →")
    handles = [Line2D([], [], marker=mk, ls="", color="0.2", mfc="0.2" if (m, e) == ("gb", 2) else "white",
                      ms=3, mew=0.7, label=lab) for m, e, lab, mk in METHODS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=1, fontsize=6)


# ------------------------------------------------------------------------------------------------ SI figures
def small_multiples(ts, col, ylabel, name, systems=None, ref_line=None, show=True, ylim=None):
    systems = systems or L.SYSTEMS
    ncol = 3
    nrow = int(np.ceil(len(systems) / ncol))
    # One shared y/x label for the whole grid: a long label repeated on each 32-mm row overlaps its neighbours.
    fig, axes = plt.subplots(nrow, ncol, figsize=(FS.DOUBLE, (32 * nrow + 12) * FS.MM), sharex=True, sharey=True,
                             layout="constrained")
    axes = np.atleast_2d(axes)
    for k, s in enumerate(systems):
        ax = axes[k // ncol, k % ncol]
        shade_eq(ax)
        if ref_line is not None:
            ax.axhline(ref_line, color="0.35", lw=0.5, ls=":")
        t, Y = smoothed(ts[ts.sys == s.key], col)
        if t is not None:
            band(ax, t, Y, s.color if s.key != "apo" else "0.45", ls=s.linestyle)
        ax.set_title(s.label, pad=2)
    for k in range(len(systems), nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
        if k >= ncol:   # the panel above an empty slot is the bottom of its column: give it x tick labels
            axes[k // ncol - 1, k % ncol].xaxis.set_tick_params(labelbottom=True)
    if ylim:
        axes[0, 0].set_ylim(*ylim)
    axes[0, 0].set_xlim(0, L.PROD_NS)
    fig.supylabel(ylabel, fontsize=7)
    fig.supxlabel("Time (ns)", fontsize=7)
    fig.suptitle(FS.N_NOTE + "; lines: replicas, band: mean ± SEM, 1-ns running mean",
                 x=0.99, ha="right", fontsize=6.5, color="0.3")
    FS.save(fig, name, show=show)
    return fig


def fig_receptor(ts, show=True):
    f1 = small_multiples(ts, "bb_rmsd", "Backbone RMSD (Å)", "FigS1a_receptor_rmsd", show=show)
    f2 = small_multiples(ts, "h12_rmsd", "Helix-12 Cα RMSD (Å)", "FigS1b_h12_rmsd", show=show)
    return f1, f2


def fig_rmsf(show=True):
    rm = tab("rmsf")
    if rm.empty:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(FS.DOUBLE, 95 * FS.MM), sharex=True,
                             gridspec_kw={"height_ratios": [1.4, 1]})
    rec = rm[rm.resid <= L.REC_RESID[1]].copy()
    rec["crystal"] = rec.resid + L.CRYSTAL_OFFSET
    apo = rec[rec.sys == "apo"].set_index("crystal").rmsf_mean
    for ax in axes:
        ax.axvspan(*L.H12_CRYSTAL, color="#FFF2CC", lw=0, zorder=0)
        ax.axvspan(*L.LOOP_CRYSTAL, color="0.93", lw=0, zorder=0)
        for a, n in L.ANCHORS.items():
            ax.axvline(n, color="0.75", lw=0.4, ls=":", zorder=0)
    for s in L.SYSTEMS:
        d = rec[rec.sys == s.key].sort_values("crystal")
        if d.empty:
            continue
        axes[0].plot(d.crystal, d.rmsf_mean, color=s.color if s.key != "apo" else "0.45",
                     lw=1.1 if s.key in ("apo", L.REFERENCE) else 0.6, ls=s.linestyle, label=s.label)
        if s.key != "apo" and len(apo):
            diff = d.set_index("crystal").rmsf_mean - apo
            axes[1].plot(diff.index, diff.values, color=s.color, lw=0.6, ls=s.linestyle)
    axes[0].set_ylabel("Cα RMSF (Å)")
    axes[1].set_ylabel("RMSF − apo (Å)")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_xlabel("ERα residue")
    axes[0].legend(ncol=5, loc="upper left", fontsize=6)
    top = axes[0].get_ylim()[1]
    axes[0].text(np.mean(L.H12_CRYSTAL), top, "H12", ha="center", va="top", fontsize=6)
    axes[0].text(np.mean(L.LOOP_CRYSTAL), top, "modelled\nloop", ha="center", va="top", fontsize=6)
    for a, n in L.ANCHORS.items():
        axes[1].text(n, axes[1].get_ylim()[1], a, rotation=90, ha="right", va="top", fontsize=5.5, color="0.4")
    axes[0].set_xlim(L.REC_RESID[0] + L.CRYSTAL_OFFSET, L.REC_RESID[1] + L.CRYSTAL_OFFSET)
    fig.text(0.99, 1.0, FS.N_NOTE + "; RMSF over 20–100 ns, averaged over replicas", ha="right", va="bottom",
             fontsize=6.5, color="0.3")
    fig.tight_layout()
    FS.save(fig, "FigS2_rmsf", show=show)
    return fig


def dot_panel(ax, rep, col, ylabel, systems, scale=1.0, hline=None):
    for i, s in enumerate(systems):
        v = rep[rep.sys == s.key][col].to_numpy() * scale
        if len(v) and not np.isnan(v).all():
            dots_ci(ax, i, v, s.color if s.key != "apo" else "0.45")
    if hline is not None:
        ax.axhline(hline, color="0.35", lw=0.5, ls=":")
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels([s.label for s in systems], rotation=60, ha="right", rotation_mode="anchor")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.6, len(systems) - 0.4)


def fig_coactivator(show=True):
    rep = tab("structure_per_replica")
    if rep.empty:
        return None
    fig, axes = plt.subplots(1, 4, figsize=(FS.DOUBLE, 62 * FS.MM))
    dot_panel(axes[0], rep, "src2_rmsd", "SRC2 Cα RMSD (Å)", L.SYSTEMS)
    dot_panel(axes[1], rep, "clamp_k362", "Lys362 NZ – SRC2 O (Å)", L.SYSTEMS, hline=CLAMP_A)
    dot_panel(axes[2], rep, "clamp_e542", "Glu542 OE – SRC2 N (Å)", L.SYSTEMS, hline=CLAMP_A)
    dot_panel(axes[3], rep, "clamp_formed", "Charge clamp formed (% frames)", L.SYSTEMS, scale=100)
    for ax, l in zip(axes, "abcd"):
        FS.panel_label(ax, l, x=-0.3, y=1.02)
    fig.text(0.99, 1.0, FS.N_NOTE + "; 20–100 ns; dots: replica means, bar: mean ± 95 % CI", ha="right",
             va="bottom", fontsize=6.5, color="0.3")
    fig.tight_layout(w_pad=1.2)
    FS.save(fig, "FigS3_coactivator", show=show)
    return fig


def fig_ligand_position(ts, show=True):
    f1 = small_multiples(ts, "lig_pocket_dist", "Ligand–pocket centroid distance (Å)", "FigS4a_ligand_centroid",
                         systems=L.LIGANDED, show=show)
    f2 = small_multiples(ts, "lig_min_dist", "Ligand–protein minimum distance (Å)", "FigS4b_ligand_mindist",
                         systems=L.LIGANDED, ref_line=DISSOC_A, show=show)
    return f1, f2


def _half_panel(ax, conv, quantity, label, systems):
    """First-half vs second-half mean per replica; points on the identity line mean no drift within 20-100 ns."""
    d = conv[conv.quantity == quantity]
    for s in systems:
        e = d[d.sys == s.key]
        ax.scatter(e.first_half, e.second_half, s=9, color=s.color, edgecolor="none", label=s.label, zorder=3)
    lo, hi = np.nanmin(d[["first_half", "second_half"]].to_numpy()), np.nanmax(d[["first_half", "second_half"]].to_numpy())
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.5", lw=0.5, ls=":", zorder=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(f"{label}, 20–60 ns")
    ax.set_ylabel(f"{label}, 60–100 ns")


def fig_convergence(frames, show=True):
    """a: running mean of ΔE_GB(ε2) minus the replica's final mean (one shared axis for all systems); b, c: first vs
    second half of 20-100 ns per replica (ΔE, ligand RMSD); d: within-replica block SEM vs the between-replica SEM,
    which is the uncertainty the tables report."""
    from matplotlib.lines import Line2D
    conv = tab("convergence")
    if frames is None or frames.empty or conv.empty:
        return None
    fr = frames[(frames.method == "gb") & (frames.eps == 2)]
    systems = [s for s in L.LIGANDED if (fr.sys == s.key).any()][:8]
    fig = plt.figure(figsize=(FS.DOUBLE, 170 * FS.MM), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.45, 1.0], hspace=0.04)
    axes = top.subplots(2, 4, sharex=True, sharey=True, gridspec_kw={"wspace": 0.12})
    lim = 0.0
    for k, s in enumerate(systems):
        ax = axes[k // 4, k % 4]
        ax.axhspan(-1, 1, color="0.92", lw=0, zorder=0)
        ax.axhline(0, color="0.5", lw=0.4, zorder=1)
        for rep in L.REPLICAS:
            d = fr[(fr.sys == s.key) & (fr.rep == rep)].sort_values("frame")
            if d.empty:
                continue
            cm = S.cumulative_mean(d.TOTAL.to_numpy())
            cm = cm - cm[-1]
            lim = max(lim, np.abs(cm[len(cm) // 10:]).max())
            t = L.T_EQ_NS + np.arange(len(d)) * (L.PROD_NS - L.T_EQ_NS) / len(d)
            ax.plot(t, cm, color=s.color, lw=0.7, ls=["-", "--", ":"][rep - 1], label=f"rep {rep}", zorder=2)
        ax.set_title(s.label, pad=2)
    for k in range(len(systems), 8):
        axes[k // 4, k % 4].axis("off")
    lim = max(1.5, np.ceil(lim * 1.1 * 2) / 2)
    axes[0, 0].set_ylim(-lim, lim)
    axes[0, 0].set_xlim(L.T_EQ_NS, L.PROD_NS)
    axes[0, 0].set_xticks([20, 60, 100])
    axes[0, 0].legend(fontsize=5.5, loc="lower right", handlelength=1.6, frameon=False)
    top.supylabel("Running mean ΔE$_{GB}$ − final\n(kcal mol$^{-1}$)", fontsize=7)
    top.supxlabel("Time (ns)", fontsize=7)
    top.text(0.0, 1.0, "a", fontsize=8, fontweight="bold", va="top", transform=top.transSubfigure)

    ax_b, ax_c, ax_d = bottom.subplots(1, 3, gridspec_kw={"width_ratios": [1, 1, 1.15]})
    _half_panel(ax_b, conv, "dE_gb_eps2", "ΔE$_{GB}$ (kcal mol$^{-1}$)", systems)
    _half_panel(ax_c, conv, "lig_rmsd", "Ligand RMSD (Å)", systems)
    d = conv[conv.quantity == "dE_gb_eps2"]
    for x, s in enumerate(systems):
        e = d[d.sys == s.key]
        ok = e.plateau.astype(bool).to_numpy()
        xs = x - 0.15 + np.linspace(-0.08, 0.08, len(e))
        ax_d.scatter(xs[ok], e.block_sem.to_numpy()[ok], s=8, color=s.color, edgecolor="none", zorder=3)
        ax_d.scatter(xs[~ok], e.block_sem.to_numpy()[~ok], s=8, facecolor="none", edgecolor=s.color, lw=0.6,
                     zorder=3)
        between = e["mean"].std(ddof=1) / np.sqrt(len(e)) if len(e) > 1 else np.nan
        ax_d.scatter([x + 0.2], [between], s=16, marker="D", color=s.color, edgecolor="none", zorder=3)
    ax_d.set_xticks(range(len(systems)))
    ax_d.set_xticklabels([s.label for s in systems], rotation=90)
    for lab, s in zip(ax_d.get_xticklabels(), systems):
        lab.set_color(s.color)
    ax_d.set_ylabel("SEM of ΔE$_{GB}$ (kcal mol$^{-1}$)")
    ax_d.set_ylim(bottom=0)
    ax_d.legend(handles=[Line2D([], [], marker="o", ls="", color="0.3", ms=3, label="within replica (block SEM)"),
                         Line2D([], [], marker="o", ls="", mfc="none", mec="0.3", ms=3, label="block SEM, no plateau"),
                         Line2D([], [], marker="D", ls="", color="0.3", ms=3.5, label="between replicas (n = 3)")],
                fontsize=5.5, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, borderaxespad=0.2)
    h, lab = ax_b.get_legend_handles_labels()
    ax_b.legend(h, lab, fontsize=5.5, loc="lower right", frameon=False, handletextpad=0.1, borderaxespad=0.2,
                ncol=2, columnspacing=0.6)
    for ax, letter in ((ax_b, "b"), (ax_c, "c"), (ax_d, "d")):
        FS.panel_label(ax, letter, x=-0.3, y=1.04)

    fig.suptitle("ε$_{in}$ = 2; ΔE from one frame per 0.5 ns over 20–100 ns; grey band ±1 kcal mol$^{-1}$; "
                 + FS.N_NOTE, x=0.99, ha="right", fontsize=6.5, color="0.3")
    FS.save(fig, "FigS5_convergence", show=show)
    return fig


def fig_decomp(show=True, threshold=1.0):
    d = tab("decomp_eps2")
    if d.empty:
        return None
    d = d.copy()
    d["label"] = [L.label_resid(int(n), r) for n, r in zip(d.resnum, d.resname)]
    keep = d.groupby("label")["mean"].apply(lambda x: x.abs().max())
    labels = sorted(keep[keep >= threshold].index, key=_res_order)
    systems = [s for s in L.LIGANDED if (d.sys == s.key).any()]
    M = np.array([[d[(d.sys == s.key) & (d.label == lab)]["mean"].sum() for lab in labels] for s in systems])
    lim = np.nanmax(np.abs(M)) if M.size else 1
    fig, ax = plt.subplots(figsize=(FS.DOUBLE, 60 * FS.MM))
    im = ax.imshow(M, aspect="auto", cmap="RdBu", vmin=-lim, vmax=lim, interpolation="nearest")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels([s.label for s in systems])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("Residue contribution (kcal mol$^{-1}$)")
    cb.outline.set_linewidth(0.4)
    ax.set_title(f"MM/GBSA per-residue decomposition, ε$_{{in}}$ = 2 (residues with |mean| ≥ {threshold:g} "
                 "kcal mol$^{-1}$ in any system; negative = favourable); " + FS.N_NOTE, fontsize=6, loc="left",
                 color="0.3")
    fig.tight_layout()
    FS.save(fig, "FigS6_decomposition", show=show)
    return fig


# label offsets (points, alignment) that keep the clustered GB-vs-PB labels apart
PB_GB_LABEL = {"estradiol": (-4, -2, "right"), "imipramine": (-4, 3, "right"), "pape1": (4, -2, "left"),
               "s_ketamine": (-4, 3, "right"), "hnk": (4, -7, "left")}


def fig_sensitivity(show=True):
    e = tab("energy_per_system")
    er = tab("energy_per_replica")
    if e.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(FS.DOUBLE, 70 * FS.MM), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    gb = e[e.method == "gb"]
    for s in L.LIGANDED:
        d = gb[gb.sys == s.key].sort_values("eps")
        if d.empty:
            continue
        ax.errorbar(d.eps, d["mean"], yerr=d["sem"], color=s.color, marker="o", ms=2.5, lw=0.8, ls=s.linestyle,
                    elinewidth=0.6, capsize=0, label=s.label)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4])
    ax.set_xticklabels(["1", "2", "4"])
    ax.set_xlabel("Solute dielectric ε$_{in}$")
    ax.set_ylabel("ΔE$_{GB}$ (kcal mol$^{-1}$, mean ± SEM)")
    ax.legend(fontsize=5.5, ncol=2, loc="upper right")
    FS.panel_label(ax, "a", x=-0.2, y=1.02)
    ax = axes[1]
    piv = e.pivot_table(index="sys", columns=["method", "eps"], values=["mean", "sem"])
    if ("mean", "pb", 2) in piv and ("mean", "gb", 2) in piv:
        for s in L.LIGANDED:
            if s.key not in piv.index:
                continue
            r = piv.loc[s.key]
            ax.errorbar(r[("mean", "gb", 2)], r[("mean", "pb", 2)], xerr=r[("sem", "gb", 2)],
                        yerr=r[("sem", "pb", 2)], fmt="o", ms=3, color=s.color, elinewidth=0.6, capsize=0)
            dx, dy, ha = PB_GB_LABEL.get(s.key, (3, 2, "left"))
            ax.annotate(s.label, (r[("mean", "gb", 2)], r[("mean", "pb", 2)]), fontsize=5.5, xytext=(dx, dy),
                        textcoords="offset points", color=s.color, ha=ha)
        d = piv[[("mean", "gb", 2), ("mean", "pb", 2)]].dropna()
        if len(d) >= 3:
            rho, p = S.spearman(d.iloc[:, 0], d.iloc[:, 1])
            ax.text(0.03, 0.97, f"Spearman ρ = {rho:.2f} (n = {len(d)} systems)", transform=ax.transAxes,
                    va="top", fontsize=6)
    ax.margins(x=0.15)
    ax.set_xlabel("ΔE$_{GB}$, ε$_{in}$ = 2 (kcal mol$^{-1}$)")
    ax.set_ylabel("ΔE$_{PB}$, ε$_{in}$ = 2 (kcal mol$^{-1}$)")
    FS.panel_label(ax, "b", x=-0.22, y=1.02)
    fig.text(0.99, 1.0, FS.N_NOTE + "; end-point energies without entropy, for ranking only", ha="right",
             va="bottom", fontsize=6.5, color="0.3")
    fig.tight_layout(w_pad=2)
    FS.save(fig, "FigS7_sensitivity", show=show)
    return fig


def fig_qc(show=True):
    qc = tab("qc")
    if qc.empty:
        return None
    terms = [("Temperature", "T (K)"), ("Pressure", "P (bar)"), ("Density", "Density (kg m$^{-3}$)"),
             ("Potential", "E$_{pot}$ drift (kJ mol$^{-1}$ / 80 ns)")]
    fig, axes = plt.subplots(1, 4, figsize=(FS.DOUBLE, 62 * FS.MM))
    for ax, (term, lab) in zip(axes, terms):
        d = qc[qc.term == term]
        col = "drift_per_80ns" if term == "Potential" else "mean"
        for i, s in enumerate(L.SYSTEMS):
            v = d[d.sys == s.key][col].to_numpy()
            if len(v):
                dots_ci(ax, i, v, s.color if s.key != "apo" else "0.45")
        ax.set_xticks(range(len(L.SYSTEMS)))
        ax.set_xticklabels([s.label for s in L.SYSTEMS], rotation=60, ha="right", rotation_mode="anchor")
        ax.set_ylabel(lab)
    axes[0].axhline(310, color="0.35", lw=0.5, ls=":")
    axes[1].axhline(1, color="0.35", lw=0.5, ls=":")
    axes[3].axhline(0, color="0.35", lw=0.5, ls=":")
    fig.text(0.99, 1.0, FS.N_NOTE + "; averages over 20–100 ns", ha="right", va="bottom", fontsize=6.5, color="0.3")
    fig.tight_layout(w_pad=1.2)
    FS.save(fig, "FigS8_qc", show=show)
    return fig


def all_figures(show=False):
    from .aggregate import load_ts
    from . import mmgbsa as MM
    FS.apply()
    ts = load_ts()
    _, frames, _ = MM.collect()
    out = {"main": main_figure(ts, show), "receptor": fig_receptor(ts, show), "rmsf": fig_rmsf(show),
           "coactivator": fig_coactivator(show), "ligand_position": fig_ligand_position(ts, show),
           "convergence": fig_convergence(frames, show), "decomp": fig_decomp(show),
           "sensitivity": fig_sensitivity(show), "qc": fig_qc(show)}
    return out
