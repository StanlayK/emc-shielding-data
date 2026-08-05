# -*- coding: utf-8 -*-
"""Obrazky pro clanek 1: jednosloupcove (sirka 3.5 in), panely nad sebou,
vektorove PDF, pismo dimenzovane na cilovou velikost. Vystup: Clanek1/figures/."""
import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/stanlayk/Library/CloudStorage/OneDrive-UniverzitaTomášeBativeZlíně/Research/Měření stínění"
FIGDIR = os.path.join(ROOT, "Clanek1", "figures")
rng = np.random.default_rng(2026)
plt.rcParams.update({
    "font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
    "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 6.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "lines.linewidth": 0.8, "pdf.fonttype": 42,
})
W = 3.5  # sirka sloupce v palcich

cache = {}
def p_lin(path):
    if path not in cache:
        df = pd.read_csv(path, engine="c", dtype=np.float64)
        f = np.array([float(c) for c in df.columns[1:]])
        cache[path] = (f, np.mean(10.0 ** (df.iloc[:, 1:].to_numpy() / 10.0), axis=0))
    return cache[path]
def smooth(x, n=31):
    return np.convolve(x, np.ones(n)/n, "same")

FGRID = np.arange(1.125e9, 4.125e9 + 1, 1e6)
bls_dir = os.path.join(ROOT, "Velké BLS")
def virc_curve(pol, absf, samp):
    d = os.path.join(bls_dir, pol, absf)
    fs = os.listdir(d)
    c = [x for x in fs if "dynamic" in x and (("efe" in x) if samp == "Ref" else (f"Sample{samp}" in x)) and x.endswith(".csv")]
    f, p = p_lin(os.path.join(d, c[0]))
    return np.interp(FGRID, f, p)

ITEMS = ["A", "B", "C", "E", "Ref"]
virc_dse = {it: {} for it in ITEMS}
virc_ref = {}
for pol in ["poloha1", "poloha2"]:
    for absf in ["bez ABS", "s ABS"]:
        pD = virc_curve(pol, absf, "D")
        for it in ITEMS:
            virc_dse[it][f"{pol}|{absf}"] = 10*np.log10(pD / virc_curve(pol, absf, it))
    pR = virc_curve(pol, "bez ABS", "Ref")
    for samp in "ABCDE":
        virc_ref[(pol, samp)] = 10*np.log10(pR / virc_curve(pol, "bez ABS", samp))

gtem_dir = os.path.join(ROOT, "VNA+GTEM_Velke_BLS")
def rot(folder):
    out = {}
    for x in os.listdir(os.path.join(gtem_dir, folder)):
        m = re.match(r"^[A-F]_VNA_(\d+)\.csv$", x)
        if m:
            f, p = p_lin(os.path.join(gtem_dir, folder, x))
            out[int(m.group(1))] = np.interp(FGRID, f, p)
    return out
graw = {s: rot(s) for s in "ABCDEF"}
gtem_dse = {it: {} for it in ITEMS}
for it in ITEMS:
    fold = "F" if it == "Ref" else it
    for a, p in graw[fold].items():
        if a in graw["D"]:
            gtem_dse[it][a] = 10*np.log10(graw["D"][a] / p)

COL = dict(zip(ITEMS, plt.cm.tab10.colors))
COLS = dict(zip("ABCDE", plt.cm.tab10.colors))

# fig_virc_dse
fig, ax = plt.subplots(2, 1, figsize=(W, 4.4), sharex=True, sharey=True)
for i, pol in enumerate(["poloha1", "poloha2"]):
    for samp in "ABCDE":
        ax[i].plot(FGRID/1e9, smooth(virc_ref[(pol, samp)]), color=COLS[samp], label=samp)
    ax[i].set_title(f"VIRC, antenna position {i+1}")
    ax[i].set_ylabel("$\\Delta$SE vs. ref. [dB]")
ax[0].legend(ncol=5, columnspacing=0.9, handlelength=1.2)
ax[1].set_xlabel("f [GHz]")
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(FIGDIR, "fig_virc_dse.pdf")); plt.close(fig)

# fig_gtem_dse
fig, ax = plt.subplots(2, 1, figsize=(W, 4.6))
for it in ["A", "B", "C", "E", "Ref"]:
    cur = smooth(np.mean(list(gtem_dse[it].values()), axis=0))
    ax[0].plot(FGRID/1e9, cur, color=COL[it], label=("F (ref.)" if it == "Ref" else it))
ax[0].set_title("GTEM, $\\Delta$SE relative to sample D")
ax[0].set_xlabel("f [GHz]"); ax[0].set_ylabel("$\\Delta$SE [dB]")
ax[0].legend(ncol=5, columnspacing=0.9, handlelength=1.2)
for s in "ABCDEF":
    angs = sorted(graw[s].keys())
    band = [10*np.log10(graw[s][a].mean()) for a in angs]
    ax[1].plot(angs, band, "o-", ms=2.5, label=s)
ax[1].set_title("Band-averaged level vs. rotation")
ax[1].set_xticks(range(0, 360, 45))
ax[1].set_xlabel("sample rotation [$^\\circ$]"); ax[1].set_ylabel("$\\langle P\\rangle$ [dB]")
ax[1].legend(ncol=6, columnspacing=0.7, handlelength=1.0)
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(FIGDIR, "fig_gtem_dse.pdf")); plt.close(fig)

# fig_cross
band_means = {}
for it in ITEMS:
    v = float(np.mean(np.mean(list(virc_dse[it].values()), axis=0)))
    g = float(np.mean(np.mean(list(gtem_dse[it].values()), axis=0)))
    band_means[it] = (v, g)
fig, ax = plt.subplots(figsize=(W, 3.1))
for it in ITEMS:
    v, g = band_means[it]
    ax.scatter(v, g, color=COL[it], s=25, zorder=3)
    ax.annotate("ref. (F)" if it == "Ref" else it, (v, g), fontsize=7,
                xytext=(4, 3), textcoords="offset points")
lim = [-3, 18]
ax.plot(lim, lim, "k--", lw=0.7, label="identity")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("$\\Delta$SE vs. D — VIRC [dB]")
ax.set_ylabel("$\\Delta$SE vs. D — GTEM [dB]")
ax.legend()
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(FIGDIR, "fig_cross.pdf")); plt.close(fig)

# fig_grr
rows = []
for it in ITEMS:
    for c, s in virc_dse[it].items():
        rows.append((it, "VIRC", float(np.mean(s))))
    for c, s in gtem_dse[it].items():
        rows.append((it, "GTEM", float(np.mean(s))))
df = pd.DataFrame(rows, columns=["item", "tech", "y"])
comps = {"part-to-part": 92.9, "technique": 0.0, "item $\\times$ tech.": 0.0,
         "repeatability": 7.1}
fig, ax = plt.subplots(2, 1, figsize=(W, 4.6))
ax[0].bar(range(4), comps.values(), color=["tab:blue", "tab:red", "tab:orange", "tab:gray"])
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(comps.keys(), fontsize=7)
ax[0].set_ylabel("% of total variance")
ax[0].set_title("Gauge R\\&R variance components")
for i, v in enumerate(comps.values()):
    ax[0].text(i, v + 1.5, f"{v:.1f} %", ha="center", fontsize=7)
xpos = {it: i for i, it in enumerate(ITEMS)}
for tech, mk, c, dx in [("VIRC", "o", "tab:blue", -0.12), ("GTEM", "^", "tab:red", 0.12)]:
    sub = df[df.tech == tech]
    ax[1].scatter([xpos[i] + dx for i in sub["item"]], sub.y, marker=mk, s=16,
                  alpha=0.75, color=c, label=tech)
ax[1].set_xticks(range(5)); ax[1].set_xticklabels(["A", "B", "C", "E", "Ref."])
ax[1].set_ylabel("$\\Delta$SE vs. D [dB]"); ax[1].set_title("Measurements by item and technique")
ax[1].legend()
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(FIGDIR, "fig_grr.pdf")); plt.close(fig)

# fig_ba_deming
SM, STEP = 51, 25
pairs = {}
for it in ITEMS:
    v = smooth(np.mean(list(virc_dse[it].values()), axis=0), SM)[::STEP][2:-2]
    g = smooth(np.mean(list(gtem_dse[it].values()), axis=0), SM)[::STEP][2:-2]
    pairs[it] = (v, g)
xall = np.concatenate([p[0] for p in pairs.values()])
yall = np.concatenate([p[1] for p in pairs.values()])
def deming(x, y):
    sxx = np.var(x, ddof=1); syy = np.var(y, ddof=1); sxy = np.cov(x, y, ddof=1)[0, 1]
    b = (syy - sxx + np.sqrt((syy - sxx)**2 + 4*sxy**2)) / (2*sxy)
    return b, np.mean(y) - b*np.mean(x)
b1, b0 = deming(xall, yall)
d = yall - xall; bias = d.mean(); sd = d.std(ddof=1)
fig, ax = plt.subplots(2, 1, figsize=(W, 4.8))
for it in ITEMS:
    v, g = pairs[it]
    lab = "Ref." if it == "Ref" else it
    ax[0].scatter((v+g)/2, g-v, s=4, alpha=0.5, color=COL[it], label=lab)
    ax[1].scatter(v, g, s=4, alpha=0.5, color=COL[it], label=lab)
ax[0].axhline(bias, color="k", lw=0.9)
for L in (bias-1.96*sd, bias+1.96*sd):
    ax[0].axhline(L, color="k", lw=0.7, ls="--")
ax[0].set_title(f"Bland--Altman: bias ${bias:.2f}$ dB, LoA $[{bias-1.96*sd:.1f}, {bias+1.96*sd:+.1f}]$ dB")
ax[0].set_xlabel("mean of techniques [dB]"); ax[0].set_ylabel("GTEM $-$ VIRC [dB]")
ax[0].legend(ncol=5, columnspacing=0.7, handlelength=0.8)
xx = np.linspace(xall.min(), xall.max(), 10)
ax[1].plot(xx, xx, "k--", lw=0.7, label="identity")
ax[1].plot(xx, b0 + b1*xx, "r-", lw=1.1, label=f"Deming: $y={b1:.3f}x{b0:+.2f}$")
ax[1].set_title("Deming regression (errors in both variables)")
ax[1].set_xlabel("$\\Delta$SE — VIRC [dB]"); ax[1].set_ylabel("$\\Delta$SE — GTEM [dB]")
ax[1].legend()
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(FIGDIR, "fig_ba_deming.pdf")); plt.close(fig)

# fig_fpca
SM2, STEP2 = 31, 5
X, meta = [], []
for it in ITEMS:
    for c, s in virc_dse[it].items():
        X.append(smooth(s, SM2)[::STEP2]); meta.append((it, "VIRC"))
    for c, s in gtem_dse[it].items():
        X.append(smooth(s, SM2)[::STEP2]); meta.append((it, "GTEM"))
X = np.array(X); meta = pd.DataFrame(meta, columns=["item", "tech"])
fgr = FGRID[::STEP2] / 1e9
Xc = X - X.mean(axis=0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
expl = S**2 / (S**2).sum()
scores = U * S
fig, ax = plt.subplots(2, 1, figsize=(W, 4.8))
for k in range(3):
    ax[0].plot(fgr, Vt[k], label=f"PC{k+1} ({expl[k]*100:.0f} %)")
ax[0].set_title("Leading eigenfunctions of $\\Delta$SE$(f)$")
ax[0].set_xlabel("f [GHz]"); ax[0].set_ylabel("eigenfunction (a.u.)")
ax[0].legend()
for it in ITEMS:
    for tech, mk in [("VIRC", "o"), ("GTEM", "^")]:
        m = ((meta["item"] == it) & (meta["tech"] == tech)).values
        ax[1].scatter(scores[m, 0], scores[m, 1], marker=mk, s=18,
                      color=COL[it], alpha=0.8)
ax[1].set_title("Score plane (color = sample, marker = technique)")
ax[1].set_xlabel(f"PC1 score ({expl[0]*100:.0f} %)")
ax[1].set_ylabel(f"PC2 score ({expl[1]*100:.0f} %)")
from matplotlib.lines import Line2D
handles = [Line2D([], [], color=COL[it], marker="s", ls="", ms=5,
                  label=("Ref." if it == "Ref" else it)) for it in ITEMS]
handles += [Line2D([], [], color="gray", marker="o", ls="", ms=4, label="VIRC"),
            Line2D([], [], color="gray", marker="^", ls="", ms=4, label="GTEM")]
ax[1].legend(handles=handles, ncol=4, columnspacing=0.7, handlelength=0.8)
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(FIGDIR, "fig_fpca.pdf")); plt.close(fig)

# fig_gp
GSTEP = 10
fg = FGRID[::GSTEP]
def gp_fuse(it):
    V = np.array(list(virc_dse[it].values()))[:, ::GSTEP]
    G = np.array(list(gtem_dse[it].values()))[:, ::GSTEP]
    mV, sV2 = V.mean(0), V.var(0, ddof=1)/len(V) + 0.25
    mG, sG2 = G.mean(0), G.var(0, ddof=1)/len(G) + 0.25
    w = 1/(sV2 + sG2)
    b = np.sum(w*(mG-mV))/np.sum(w)
    yV, yG = mV + b/2, mG - b/2
    fobs = np.concatenate([fg, fg]); y = np.concatenate([yV, yG])
    nvar = np.concatenate([sV2, sG2])
    ell, sf2 = 80e6, np.var(y)
    K = sf2*np.exp(-0.5*((fobs[:, None]-fobs[None, :])/ell)**2)
    Ks = sf2*np.exp(-0.5*((fg[:, None]-fobs[None, :])/ell)**2)
    L = np.linalg.cholesky(K + np.diag(nvar) + 1e-8*sf2*np.eye(len(y)))
    al = np.linalg.solve(L.T, np.linalg.solve(L, y - y.mean()))
    mu = y.mean() + Ks @ al
    vv = np.linalg.solve(L, Ks.T)
    sd_ = np.sqrt(np.clip(sf2 - np.sum(vv**2, axis=0), 0, None))
    return mu, sd_, b, mV, mG
fig, ax = plt.subplots(2, 1, figsize=(W, 4.8), sharex=True)
for a, it in zip(ax, ["A", "C"]):
    mu, sd_, b, mV, mG = gp_fuse(it)
    a.plot(fg/1e9, mV, lw=0.6, color="tab:blue", alpha=0.65, label="VIRC mean")
    a.plot(fg/1e9, mG, lw=0.6, color="tab:red", alpha=0.65, label="GTEM mean")
    a.plot(fg/1e9, mu, lw=1.2, color="k", label="GP posterior")
    a.fill_between(fg/1e9, mu-1.96*sd_, mu+1.96*sd_, color="k", alpha=0.15,
                   label="95 % band")
    a.set_title(f"Sample {it} (offset ${b:+.2f}$ dB)")
    a.set_ylabel("$\\Delta$SE vs. D [dB]")
ax[0].legend(ncol=2, columnspacing=0.8, handlelength=1.2)
ax[1].set_xlabel("f [GHz]")
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(FIGDIR, "fig_gp.pdf")); plt.close(fig)
print("Hotovo:", sorted(x for x in os.listdir(FIGDIR) if x.endswith(".pdf")))
