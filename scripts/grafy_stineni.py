# -*- coding: utf-8 -*-
"""Opravene GTEM vypocty + grafy."""
import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/stanlayk/Library/CloudStorage/OneDrive-UniverzitaTomášeBativeZlíně/Research/Měření stínění"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vysledky")
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

cache = {}
def load(path):
    if path not in cache:
        df = pd.read_csv(path, engine="c", dtype=np.float64)
        cache[path] = (np.array([float(c) for c in df.columns[1:]]), df.iloc[:, 1:].to_numpy())
    return cache[path]

def p_lin(path):
    f, d = load(path)
    return f, np.mean(10.0 ** (d / 10.0), axis=0)

def smooth(x, n=31):
    k = np.ones(n) / n
    return np.convolve(x, k, mode="same")

# ---- GTEM (opravene: jen rotacni soubory bez S-param sufixu) ----
def gtem_avg(folder):
    pat = re.compile(r"^[A-F]_VNA_(\d+)\.csv$")
    items = sorted((int(pat.match(x).group(1)), x) for x in os.listdir(folder) if pat.match(x))
    angs = [a for a, _ in items]
    ps = [p_lin(os.path.join(folder, x)) for _, x in items]
    f0 = ps[0][0]
    arr = np.array([p for _, p in ps])
    return f0, arr.mean(axis=0), arr, angs

print("=== GTEM Velke BLS: absolutni prijate urovne (prumer pres rotace a pasmo) ===")
gtem_dir = os.path.join(ROOT, "VNA+GTEM_Velke_BLS")
gtem = {}
for s in sorted(os.listdir(gtem_dir)):
    d = os.path.join(gtem_dir, s)
    if os.path.isdir(d):
        gtem[s] = gtem_avg(d)
        f, pavg, arr, angs = gtem[s]
        print(f"vzorek {s}: {len(angs)} rotaci, pasmovy prumer {10*np.log10(pavg.mean()):7.2f} dB, "
              f"std pres rotace {np.std(10*np.log10(arr.mean(axis=1))):.2f} dB")

# referencni kandidat = vzorek s nejvyssi prijatou urovni
levels = {s: 10*np.log10(v[1].mean()) for s, v in gtem.items()}
ref = max(levels, key=levels.get)
print(f"\nNejvyssi uroven (kandidat reference): {ref}")
fR, pR, _, _ = gtem[ref]
gtem_se = {}
for s, (f, p, arr, angs) in gtem.items():
    if s != ref:
        gtem_se[s] = 10 * np.log10(pR / p)
    print(f"SE({s}) rel. {ref}: {10*np.log10(pR.mean()/p.mean()):6.2f} dB (pasmove)")

pcb_dir = os.path.join(ROOT, "Little PCB_VNA_GTEM")
pcb = {}
for s in sorted(os.listdir(pcb_dir)):
    d = os.path.join(pcb_dir, s)
    if os.path.isdir(d):
        pat = re.compile(r"^[A-F]_VNA_\d+\.csv$")
        fs = sorted(x for x in os.listdir(d) if pat.match(x))
        ps = np.array([p_lin(os.path.join(d, x))[1] for x in fs])
        fP = p_lin(os.path.join(d, fs[0]))[0]
        pcb[s] = ps.mean(axis=0)
refk = [k for k in pcb if "bez krytu" in k][0]

# ---- VIRC BLS ----
bls_dir = os.path.join(ROOT, "Velké BLS")
def bls_se(poloha, absf, stav, samp):
    d = os.path.join(bls_dir, poloha, absf)
    fs = os.listdir(d)
    ref = [x for x in fs if stav in x and "efe" in x and x.endswith(".csv")][0]
    sam = [x for x in fs if stav in x and f"Sample{samp}" in x and x.endswith(".csv")][0]
    f1, pr = p_lin(os.path.join(d, ref)); f2, ps = p_lin(os.path.join(d, sam))
    return f1, 10 * np.log10(pr / ps)

# ---- FIG 1: VIRC kryt klid vs shake ----
virc_root = os.path.join(ROOT, "VIRC data")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for i, stav in enumerate(["klid", "shake"]):
    for run in ["01", "02", "03"]:
        fr, prr = p_lin(os.path.join(virc_root, f"vna_{run}{stav}_bezkrytu_prutAntena.csv"))
        _, pss = p_lin(os.path.join(virc_root, f"vna_{run}{stav}_skrytem_prutAntena.csv"))
        ax[i].plot(fr/1e9, smooth(10*np.log10(prr/pss)), lw=0.9, label=f"run {run}")
    ax[i].set_title(f"VIRC – enclosure with rod antenna ({'static' if stav=='klid' else 'stirred field'})")
    ax[i].set_xlabel("f [GHz]"); ax[i].legend()
ax[0].set_ylabel("SE [dB] (31-point moving average)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_virc_kryt.png")); plt.close(fig)

# ---- FIG 2: Velke BLS VIRC dynamic, obe polohy ----
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for i, pol in enumerate(["poloha1", "poloha2"]):
    for samp in "ABCDE":
        f, s = bls_se(pol, "bez ABS", "dynamic", samp)
        ax[i].plot(f/1e9, smooth(s), lw=0.9, label=f"Sample {samp}")
    ax[i].set_title(f"Large BLS – VIRC dynamic, no absorber, {'position 1' if pol=='poloha1' else 'position 2'}")
    ax[i].set_xlabel("f [GHz]"); ax[i].legend(ncol=2, fontsize=7)
ax[0].set_ylabel("SE relative to reference element [dB]")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_bls_virc.png")); plt.close(fig)

# ---- FIG 3: reprodukovatelnost mezi polohami ----
fig, ax = plt.subplots(figsize=(7, 3.6))
for samp, c in zip("ABCDE", plt.cm.tab10.colors):
    f1, s1 = bls_se("poloha1", "bez ABS", "dynamic", samp)
    f2, s2 = bls_se("poloha2", "bez ABS", "dynamic", samp)
    ax.plot(f1/1e9, smooth(s1 - s2), lw=0.8, color=c, label=f"Sample {samp}")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("f [GHz]"); ax.set_ylabel("ΔSE position 1 − position 2 [dB]")
ax.set_title("VIRC: SE difference between antenna positions (dynamic, no absorber)")
ax.legend(ncol=3, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_bls_reprodukovatelnost.png")); plt.close(fig)

# ---- FIG 4: GTEM SE (rel. ref kandidat) + rotace ----
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for s, sev in gtem_se.items():
    ax[0].plot(fR/1e9, smooth(sev), lw=0.9, label=f"sample {s}")
ax[0].set_title(f"GTEM large BLS – SE relative to sample {ref} (avg. of 8 rotations)")
ax[0].set_xlabel("f [GHz]"); ax[0].set_ylabel("SE [dB]"); ax[0].legend(fontsize=7)
for s, (f, p, arr, angs) in gtem.items():
    band = 10*np.log10(arr.mean(axis=1))
    ax[1].plot(angs, band, "o-", ms=3, lw=0.8, label=f"{s}")
ax[1].set_title("GTEM: band-averaged level vs. sample rotation")
ax[1].set_xticks(range(0, 360, 45))
ax[1].set_xlabel("sample rotation [°]"); ax[1].set_ylabel("⟨P⟩ [dB]"); ax[1].legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_gtem.png")); plt.close(fig)

# ---- FIG 5: Little PCB ----
fig, ax = plt.subplots(figsize=(7, 3.6))
for s, p in pcb.items():
    if s == refk: continue
    ax.plot(fP/1e9, smooth(10*np.log10(pcb[refk]/p)), lw=0.9, label=f"shield {s}")
ax.set_title("Little PCB in GTEM – shield SE relative to bare PCB")
ax.set_xlabel("f [GHz]"); ax.set_ylabel("SE [dB]"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_pcb.png")); plt.close(fig)

# ---- FIG 6: Box otvory ----
box_dir = os.path.join(ROOT, "Box_VNA+VIRC")
shapes = {d: os.path.join(box_dir, d) for d in sorted(os.listdir(box_dir)) if os.path.isdir(os.path.join(box_dir, d))}
open_all = [v for k, v in shapes.items() if "všemi" in k][0]
def ff(folder, stav):
    for suf in (f"{stav}_abs_S21.csv", f"{stav}_S21.csv", f"{stav}_abs.csv"):
        c = [x for x in os.listdir(folder) if x.endswith(suf)]
        if c:
            return os.path.join(folder, c[0])
    return None
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for i, stav in enumerate(["static", "dynamic"]):
    fr, pr = p_lin(ff(open_all, stav))
    for name, d in shapes.items():
        if d == open_all: continue
        fpath = ff(d, stav)
        if fpath is None:
            print(f"POZOR: chybi soubor {stav} S21 v {name}")
            continue
        fs_, ps = p_lin(fpath)
        ax[i].plot(fr/1e9, smooth(10*np.log10(pr/ps)), lw=0.9, label=name.replace("Box_", "").replace("Box ", "").replace("box ", "").replace("kruh","circle").replace("trojúhelník","triangle").replace("obdelnik","rectangle").replace("čtverec","square"))
    ax[i].set_title(f"Box with apertures – IL relative to fully open box ({stav})")
    ax[i].set_xlabel("f [GHz]"); ax[i].legend(fontsize=7)
ax[0].set_ylabel("insertion loss [dB]")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig6_box.png")); plt.close(fig)

# ---- FIG 7 + cross: identicke mapovani A..E=A..E, F=referencni prvek ----
# SE v obou technikach vztazena ke STEJNEMU spolecnemu prvku: vzorku D (nejslabsi stineni).
band = (fR >= 1.125e9) & (fR <= 4.125e9)

def virc_avgpol(samp):
    f1, s1 = bls_se("poloha1", "bez ABS", "dynamic", samp)
    _, s2 = bls_se("poloha2", "bez ABS", "dynamic", samp)
    return f1, (s1 + s2) / 2

# VIRC SE rel. SampleD: SE_X - SE_D (obe rel. ref prvku, ref se vykrati)
fV, sD = virc_avgpol("D")
virc_relD = {}
for samp in "ABCE":
    _, sX = virc_avgpol(samp)
    virc_relD[samp] = sX - sD
# ref prvek rel D = -SE_D
virc_relD["Ref"] = -sD
# GTEM rel D (D je i zde nejvyssi uroven), F = kandidat na ref prvek
fD, pD, _, _ = gtem["D"]
gtem_relD = {}
for s in "ABCE":
    gtem_relD[s] = 10 * np.log10(pD / gtem[s][1])
gtem_relD["Ref"] = 10 * np.log10(pD / gtem["F"][1])

fig, ax = plt.subplots(figsize=(6.2, 4.4))
bandv = (fV >= 1.125e9) & (fV <= 4.125e9)
xs, ys, lbl = [], [], []
for k in ["A", "B", "C", "E", "Ref"]:
    xv = float(np.mean(virc_relD[k][bandv]))
    yv = float(np.mean(gtem_relD[k][band]))
    xs.append(xv); ys.append(yv); lbl.append(k)
    print(f"rel-D srovnani {k}: VIRC {xv:6.2f} dB, GTEM {yv:6.2f} dB, delta {yv-xv:6.2f} dB")
ax.scatter(xs, ys, c="tab:red")
for x, y, t in zip(xs, ys, lbl):
    ax.annotate(t if t != "Ref" else "reference element (GTEM F)", (x, y), fontsize=8, xytext=(5, 3), textcoords="offset points")
lim = [-5, 30]
ax.plot(lim, lim, "k--", lw=0.7, label="1:1")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("ΔSE relative to sample D — VIRC [dB]")
ax.set_ylabel("ΔSE relative to sample D — GTEM [dB]")
ax.set_title("Cross-technique comparison (1.125–4.125 GHz band)\nsamples A–E mapped identically, GTEM F = reference element")
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig7_cross.png")); plt.close(fig)
from scipy.stats import spearmanr, pearsonr
rs, ps_ = spearmanr(xs, ys)
rp, pp = pearsonr(xs, ys)
print(f"\nSpearman rho = {rs:.3f} (p={ps_:.3f}), Pearson r = {rp:.3f} (p={pp:.3f})")
print(f"Prumerna odchylka GTEM-VIRC: {np.mean(np.array(ys)-np.array(xs)):.2f} dB, "
      f"std {np.std(np.array(ys)-np.array(xs)):.2f} dB")

print("\nGrafy ulozeny do", OUT)
