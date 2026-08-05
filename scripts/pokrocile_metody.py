# -*- coding: utf-8 -*-
"""Gauge R&R, Bland-Altman + Deming, funkcionalni PCA, GP fuze VIRC+GTEM.
Vsechny grafy v anglictine. Vztazna velicina: dSE = SE relativne ke vzorku D
(spolecny artefakt v obou technikach), pasmo 1.125-4.125 GHz."""
import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/stanlayk/Library/CloudStorage/OneDrive-UniverzitaTomášeBativeZlíně/Research/Měření stínění"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vysledky")
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
rng = np.random.default_rng(12345)

cache = {}
def p_lin(path):
    if path not in cache:
        df = pd.read_csv(path, engine="c", dtype=np.float64)
        f = np.array([float(c) for c in df.columns[1:]])
        cache[path] = (f, np.mean(10.0 ** (df.iloc[:, 1:].to_numpy() / 10.0), axis=0))
    return cache[path]

def smooth(x, n):
    return np.convolve(x, np.ones(n)/n, "same")

# ---------------- data: dSE(f) rel. vzorku D ----------------
FGRID = np.arange(1.125e9, 4.125e9 + 1, 1e6)   # spolecna osa (krok 1 MHz)

# VIRC: 4 podminky (poloha1/2 x bezABS/sABS), dynamic
bls_dir = os.path.join(ROOT, "Velké BLS")
def virc_curve(poloha, absf, samp):
    d = os.path.join(bls_dir, poloha, absf)
    fs = os.listdir(d)
    if samp == "Ref":
        cand = [x for x in fs if "dynamic" in x and "efe" in x and x.endswith(".csv")]
    else:
        cand = [x for x in fs if "dynamic" in x and f"Sample{samp}" in x and x.endswith(".csv")]
    f, p = p_lin(os.path.join(d, cand[0]))
    return np.interp(FGRID, f, p)

ITEMS = ["A", "B", "C", "E", "Ref"]
virc_dse = {it: {} for it in ITEMS}           # item -> cond -> dSE(f)
for pol in ["poloha1", "poloha2"]:
    for absf in ["bez ABS", "s ABS"]:
        cond = f"{pol}|{absf}"
        pD = virc_curve(pol, absf, "D")
        for it in ITEMS:
            virc_dse[it][cond] = 10 * np.log10(pD / virc_curve(pol, absf, it))

# GTEM: rotace parovane podle uhlu; slozka F = referencni prvek
gtem_dir = os.path.join(ROOT, "VNA+GTEM_Velke_BLS")
def gtem_rot(folder):
    out = {}
    for x in os.listdir(os.path.join(gtem_dir, folder)):
        m = re.match(r"^[A-F]_VNA_(\d+)\.csv$", x)
        if m:
            f, p = p_lin(os.path.join(gtem_dir, folder, x))
            out[int(m.group(1))] = np.interp(FGRID, f, p)
    return out

gtem_raw = {s: gtem_rot(s) for s in ["A", "B", "C", "D", "E", "F"]}
gtem_dse = {it: {} for it in ITEMS}
for it in ITEMS:
    fold = "F" if it == "Ref" else it
    for ang, p in gtem_raw[fold].items():
        if ang in gtem_raw["D"]:
            gtem_dse[it][f"rot{ang}"] = 10 * np.log10(gtem_raw["D"][ang] / p)

# ---------------- 1) GAUGE R&R (pasmove prumery) ----------------
print("=== 1) Gauge R&R — response: band-averaged dSE rel. sample D ===")
rows = []
for it in ITEMS:
    for cond, s in virc_dse[it].items():
        rows.append(dict(item=it, tech="VIRC", meas=cond, y=float(np.mean(s))))
    for cond, s in gtem_dse[it].items():
        rows.append(dict(item=it, tech="GTEM", meas=cond, y=float(np.mean(s))))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "gauge_rr_data.csv"), index=False)

p_ = df.item.nunique(); t_ = 2
cell = df.groupby(["item", "tech"]).y.agg(["mean", "var", "count"])
n_h = len(cell) / np.sum(1.0 / cell["count"])          # harmonicky prumer n
ms_within = np.average(cell["var"], weights=cell["count"] - 1)
M = cell["mean"].unstack()                              # 5x2 cell means
Mi = M.mean(axis=1); Mj = M.mean(axis=0); Mg = M.values.mean()
ss_int = ((M.sub(Mi, axis=0).sub(Mj, axis=1) + Mg) ** 2).values.sum()
ms_int = n_h * ss_int / ((p_ - 1) * (t_ - 1))
ms_part = t_ * n_h * ((Mi - Mg) ** 2).sum() / (p_ - 1)
ms_tech = p_ * n_h * ((Mj - Mg) ** 2).sum() / (t_ - 1)
v_e = ms_within
v_int = max(0.0, (ms_int - v_e) / n_h)
v_tech = max(0.0, (ms_tech - ms_int) / (p_ * n_h))
v_part = max(0.0, (ms_part - ms_int) / (t_ * n_h))
grr = v_e + v_int + v_tech
tot = grr + v_part
print(f"variance components [dB^2]: part={v_part:.3f}, technique={v_tech:.3f}, "
      f"item x technique={v_int:.3f}, repeatability(within)={v_e:.3f}")
print(f"%GRR = {100*np.sqrt(grr/tot):.1f} % of total sigma, ndc = {int(1.41*np.sqrt(v_part/grr))}")
print(f"technique share of total variance = {100*v_tech/tot:.1f} %")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
comps = {"part-to-part": v_part, "technique": v_tech, "item × technique": v_int,
         "repeatability\n(within-technique)": v_e}
ax[0].bar(range(4), [100 * v / tot for v in comps.values()],
          color=["tab:blue", "tab:red", "tab:orange", "tab:gray"])
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(comps.keys(), fontsize=8)
ax[0].set_ylabel("% of total variance")
ax[0].set_title(f"Gauge R&R — variance components (%GRR = {100*np.sqrt(grr/tot):.0f} % of σ)")
for i, (k, v) in enumerate(comps.items()):
    ax[0].text(i, 100 * v / tot + 1, f"{100*v/tot:.1f} %", ha="center", fontsize=8)
xpos = {it: i for i, it in enumerate(ITEMS)}
for tech, mk, c in [("VIRC", "o", "tab:blue"), ("GTEM", "^", "tab:red")]:
    sub = df[df.tech == tech]
    ax[1].scatter([xpos[i] + (-0.12 if tech == "VIRC" else 0.12) for i in sub.item],
                  sub.y, marker=mk, s=28, alpha=0.75, color=c, label=tech)
ax[1].set_xticks(range(len(ITEMS)))
ax[1].set_xticklabels(["A", "B", "C", "E", "Reference"])
ax[1].set_ylabel("ΔSE rel. sample D [dB]"); ax[1].set_xlabel("item")
ax[1].set_title("Measurements by item and technique"); ax[1].legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig10_gauge_rr.png")); plt.close(fig)

# ---------------- 2) BLAND-ALTMAN + DEMING ----------------
print("\n=== 2) Bland-Altman + Deming (frequency-resolved, 50 MHz smoothing) ===")
SM, STEP = 51, 25   # 51 MHz okno, vzorek kazdych 25 MHz
pairs = []
for it in ITEMS:
    v = smooth(np.mean(list(virc_dse[it].values()), axis=0), SM)[::STEP]
    g = smooth(np.mean(list(gtem_dse[it].values()), axis=0), SM)[::STEP]
    for xv, xg in zip(v[2:-2], g[2:-2]):
        pairs.append((it, xv, xg))
pr = pd.DataFrame(pairs, columns=["item", "virc", "gtem"])
mean_ = (pr.virc + pr.gtem) / 2; diff_ = pr.gtem - pr.virc
bias = diff_.mean(); sd = diff_.std(ddof=1)
loa = (bias - 1.96 * sd, bias + 1.96 * sd)
print(f"n = {len(pr)} freq-resolved pairs; bias = {bias:.2f} dB, "
      f"95% limits of agreement = [{loa[0]:.2f}, {loa[1]:.2f}] dB")

def deming(x, y):
    sxx = np.var(x, ddof=1); syy = np.var(y, ddof=1); sxy = np.cov(x, y, ddof=1)[0, 1]
    b = (syy - sxx + np.sqrt((syy - sxx) ** 2 + 4 * sxy ** 2)) / (2 * sxy)
    return b, np.mean(y) - b * np.mean(x)
b1, b0 = deming(pr.virc.values, pr.gtem.values)
boot = np.array([deming(*map(np.array, zip(*[ (pr.virc[i], pr.gtem[i])
        for i in rng.integers(0, len(pr), len(pr)) ]))) for _ in range(1000)])
print(f"Deming: slope = {b1:.3f} [95% CI {np.percentile(boot[:,0],2.5):.3f}, "
      f"{np.percentile(boot[:,0],97.5):.3f}], intercept = {b0:.2f} dB "
      f"[{np.percentile(boot[:,1],2.5):.2f}, {np.percentile(boot[:,1],97.5):.2f}]")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
colors = dict(zip(ITEMS, plt.cm.tab10.colors))
for it in ITEMS:
    m = pr.item == it
    lab = "Reference" if it == "Ref" else f"Sample {it}"
    ax[0].scatter(mean_[m], diff_[m], s=7, alpha=0.5, color=colors[it], label=lab)
    ax[1].scatter(pr.virc[m], pr.gtem[m], s=7, alpha=0.5, color=colors[it], label=lab)
ax[0].axhline(bias, color="k", lw=1, label=f"bias = {bias:.2f} dB")
for L in loa:
    ax[0].axhline(L, color="k", lw=0.8, ls="--")
ax[0].text(0.98, 0.03, f"95% LoA = [{loa[0]:.2f}, {loa[1]:.2f}] dB",
           transform=ax[0].transAxes, ha="right", fontsize=8)
ax[0].set_xlabel("mean of techniques (GTEM+VIRC)/2 [dB]")
ax[0].set_ylabel("difference GTEM − VIRC [dB]")
ax[0].set_title("Bland–Altman plot (frequency-resolved ΔSE)")
ax[0].legend(fontsize=7, ncol=2)
xx = np.linspace(pr.virc.min(), pr.virc.max(), 10)
ax[1].plot(xx, xx, "k--", lw=0.8, label="identity")
ax[1].plot(xx, b0 + b1 * xx, "r-", lw=1.2,
           label=f"Deming: y = {b1:.3f}·x {'+' if b0>=0 else '−'} {abs(b0):.2f}")
ax[1].set_xlabel("ΔSE — VIRC [dB]"); ax[1].set_ylabel("ΔSE — GTEM [dB]")
ax[1].set_title("Deming regression (errors in both variables)")
ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig11_bland_altman_deming.png")); plt.close(fig)

# ---------------- 3) FUNKCIONALNI PCA ----------------
print("\n=== 3) Functional PCA of dSE(f) curves ===")
SM2, STEP2 = 31, 5
X, meta = [], []
for it in ITEMS:
    for cond, s in virc_dse[it].items():
        X.append(smooth(s, SM2)[::STEP2]); meta.append((it, "VIRC"))
    for cond, s in gtem_dse[it].items():
        X.append(smooth(s, SM2)[::STEP2]); meta.append((it, "GTEM"))
X = np.array(X); meta = pd.DataFrame(meta, columns=["item", "tech"])
fgr = FGRID[::STEP2] / 1e9
Xc = X - X.mean(axis=0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
expl = S**2 / (S**2).sum()
scores = U * S
print(f"curves: {X.shape[0]} (VIRC {sum(meta.tech=='VIRC')}, GTEM {sum(meta.tech=='GTEM')}), "
      f"grid {X.shape[1]} pts")
print("explained variance PC1-4:", np.round(expl[:4] * 100, 1), "%")
for k in range(3):
    sc = scores[:, k]
    eta_item = sum(len(sc[meta.item == i]) * (sc[meta.item == i].mean() - sc.mean())**2
                   for i in ITEMS) / ((sc - sc.mean())**2).sum()
    eta_tech = sum(len(sc[meta.tech == t]) * (sc[meta.tech == t].mean() - sc.mean())**2
                   for t in ["VIRC", "GTEM"]) / ((sc - sc.mean())**2).sum()
    print(f"PC{k+1}: eta^2(item) = {eta_item:.3f}, eta^2(technique) = {eta_tech:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for k in range(3):
    ax[0].plot(fgr, Vt[k], lw=1, label=f"PC{k+1} ({expl[k]*100:.0f} %)")
ax[0].set_xlabel("f [GHz]"); ax[0].set_ylabel("eigenfunction (a.u.)")
ax[0].set_title("Functional PCA — leading eigenfunctions of ΔSE(f)")
ax[0].legend(fontsize=8)
for it in ITEMS:
    for tech, mk in [("VIRC", "o"), ("GTEM", "^")]:
        m = (meta.item == it) & (meta.tech == tech)
        ax[1].scatter(scores[m.values, 0], scores[m.values, 1], marker=mk, s=34,
                      color=colors[it], alpha=0.8,
                      label=f"{'Ref' if it=='Ref' else it} · {tech}")
ax[1].set_xlabel(f"PC1 score ({expl[0]*100:.0f} %)")
ax[1].set_ylabel(f"PC2 score ({expl[1]*100:.0f} %)")
ax[1].set_title("Score space: curves cluster by sample, not by technique")
ax[1].legend(fontsize=6, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig12_fpca.png")); plt.close(fig)

# ---------------- 4) GP FUZE (heteroskedasticky sum, technicky offset) ----------------
print("\n=== 4) GP fusion of VIRC+GTEM (heteroscedastic noise, technique offset) ===")
GSTEP = 10  # 10 MHz
fg = FGRID[::GSTEP]
def gp_fuse(it):
    V = np.array(list(virc_dse[it].values()))[:, ::GSTEP]
    G = np.array(list(gtem_dse[it].values()))[:, ::GSTEP]
    mV, sV2 = V.mean(0), V.var(0, ddof=1) / len(V) + 0.25
    mG, sG2 = G.mean(0), G.var(0, ddof=1) / len(G) + 0.25
    w = 1 / (sV2 + sG2)
    b = np.sum(w * (mG - mV)) / np.sum(w)          # technicky offset (GTEM-VIRC)
    yV, yG = mV + b / 2, mG - b / 2                 # symetricke odstraneni offsetu
    fobs = np.concatenate([fg, fg]); y = np.concatenate([yV, yG])
    nvar = np.concatenate([sV2, sG2])
    ell, sf2 = 80e6, np.var(y)
    K = sf2 * np.exp(-0.5 * ((fobs[:, None] - fobs[None, :]) / ell) ** 2)
    Ks = sf2 * np.exp(-0.5 * ((fg[:, None] - fobs[None, :]) / ell) ** 2)
    L = np.linalg.cholesky(K + np.diag(nvar) + 1e-8 * sf2 * np.eye(len(y)))
    al = np.linalg.solve(L.T, np.linalg.solve(L, y - y.mean()))
    mu = y.mean() + Ks @ al
    v = np.linalg.solve(L, Ks.T)
    var = sf2 - np.sum(v ** 2, axis=0)
    return mu, np.sqrt(np.clip(var, 0, None)), b, mV, mG

fig, ax = plt.subplots(1, 2, figsize=(11, 4.0), sharey=False)
for a, it in zip(ax, ["A", "C"]):
    mu, sd_, b, mV, mG = gp_fuse(it)
    a.plot(fg/1e9, mV, lw=0.7, color="tab:blue", alpha=0.6, label="VIRC (mean of 4 configs)")
    a.plot(fg/1e9, mG, lw=0.7, color="tab:red", alpha=0.6, label="GTEM (mean of rotations)")
    a.plot(fg/1e9, mu, lw=1.4, color="k", label="GP posterior mean")
    a.fill_between(fg/1e9, mu - 1.96*sd_, mu + 1.96*sd_, color="k", alpha=0.15, label="95% credible band")
    a.set_xlabel("f [GHz]"); a.set_ylabel("ΔSE rel. sample D [dB]")
    a.set_title(f"Sample {it} — technique-independent estimate (offset {b:+.2f} dB)")
    a.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig13_gp_fusion.png")); plt.close(fig)

offs = {}
fused = {}
for it in ITEMS:
    mu, sd_, b, mV, mG = gp_fuse(it)
    offs[it] = b
    fused[it] = np.vstack([fg, mu, sd_])
    print(f"item {it}: technique offset (GTEM-VIRC) = {b:+.2f} dB, "
          f"median posterior sd = {np.median(sd_):.2f} dB")
np.savez_compressed(os.path.join(OUT, "gp_fused_curves.npz"), **{k: v for k, v in fused.items()})
print("\nHotovo: fig10-13 + gauge_rr_data.csv + gp_fused_curves.npz")
