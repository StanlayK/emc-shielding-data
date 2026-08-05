# -*- coding: utf-8 -*-
"""Diagnostika funkce VIRC ze zmerenych dat (statistika pole, uniformita, izotropie)."""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/stanlayk/Library/CloudStorage/OneDrive-UniverzitaTomášeBativeZlíně/Research/Měření stínění"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vysledky")
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

def load(path):
    df = pd.read_csv(path, engine="c", dtype=np.float64)
    return np.array([float(c) for c in df.columns[1:]]), df.iloc[:, 1:].to_numpy()

def diagnostics(path):
    """Statistika michaneho pole z jednoho shake souboru (100 sweepu x 3001 f)."""
    f, d = load(path)
    P = 10.0 ** (d / 10.0)                      # linearni vykon
    mu = P.mean(axis=0); va = P.var(axis=0, ddof=1)
    R = mu**2 / va
    K = np.where(R > 1, (R - 1) + np.sqrt(np.clip(R**2 - R, 0, None)), 0.0)  # Ricianuv K z momentu vykonu
    # stirring ratio (max-min dB pres sweepy)
    SR = d.max(axis=0) - d.min(axis=0)
    # lag-1 autokorelace pres sweepy -> efektivni pocet nezavislych vzorku
    Pc = P - mu
    num = (Pc[:-1] * Pc[1:]).mean(axis=0)
    rho1 = num / va
    rho1 = np.clip(rho1, -0.99, 0.99)
    Neff = 100 * (1 - rho1) / (1 + rho1)
    # KS test exponencialniho rozdeleni (kazda 20. frekvence)
    idx = np.arange(0, len(f), 20)
    pvals = np.array([stats.kstest(P[:, i] / mu[i], "expon").pvalue for i in idx])
    # koherencni sirka pasma z fluktuaci (median pres sweepy), krok f
    dfstep = f[1] - f[0]
    bcoh = []
    for s in range(0, 100, 10):
        x = P[s, :] / P[s, :].mean() - 1
        ac = np.correlate(x, x, "full")[len(x)-1:]
        ac /= ac[0]
        k = np.argmax(ac < 0.5)
        bcoh.append(k * dfstep)
    return dict(f=f, K=K, SR=SR, Neff=Neff, pvals=pvals, idx=idx,
                Bcoh=np.median(bcoh), mu=mu, P=P)

virc_root = os.path.join(ROOT, "VIRC data")
poz_dir = os.path.join(virc_root, "Měření různých pozic antén")
pol_dir = os.path.join(virc_root, "VNA měření polarizace antén")

# ---------- A) statistika pole: bezkrytu shake, 3 runy ----------
print("=== A) Statistika michaneho pole (VIRC root, bezkrytu, shake) ===")
diags = {}
for run in ["01", "02", "03"]:
    p = os.path.join(virc_root, f"vna_{run}shake_bezkrytu_prutAntena.csv")
    dg = diagnostics(p)
    diags[run] = dg
    print(f"run {run}: median K = {np.median(dg['K']):.3f} (90. perc. {np.percentile(dg['K'],90):.2f}), "
          f"median stirring ratio = {np.median(dg['SR']):.1f} dB, "
          f"median N_eff = {np.median(dg['Neff']):.0f}/100, "
          f"KS(exp) podil p>0.05 = {100*np.mean(dg['pvals']>0.05):.0f} %, "
          f"B_coh = {dg['Bcoh']/1e6:.1f} MHz")

# klidova opakovatelnost
print("\n=== B) Klidovy rezim: variabilita pres 100 sweepu (ocekava se ~0 dB) ===")
for run in ["01"]:
    f, d = load(os.path.join(virc_root, f"vna_{run}klid_bezkrytu_prutAntena.csv"))
    print(f"run {run} klid: median std pres sweepy = {np.median(d.std(axis=0)):.3f} dB, "
          f"95. perc. = {np.percentile(d.std(axis=0),95):.3f} dB")
    f, d2 = load(os.path.join(virc_root, f"vna_{run}shake_bezkrytu_prutAntena.csv"))
    print(f"run {run} shake: median std pres sweepy = {np.median(d2.std(axis=0)):.3f} dB (pro srovnani)")

# ---------- C) uniformita pres pozice anten ----------
print("\n=== C) Prostorova uniformita (pozice 1,3,4,5,6; shake, prumer runu 01-03) ===")
bands = [(1.125e9, 2e9), (2e9, 3e9), (3e9, 4.125e9)]
pos_band = {}
for poz in ["1", "3", "4", "5", "6"]:
    Ps = []
    for run in ["01", "02", "03"]:
        p = os.path.join(poz_dir, f"vna_pozice{poz}_shake_{run}.csv")
        f, d = load(p)
        Ps.append((10.0 ** (d / 10.0)).mean(axis=0))
    Pm = np.mean(Ps, axis=0)
    pos_band[poz] = [10*np.log10(Pm[(f >= a) & (f < b)].mean()) for a, b in bands]
tab = pd.DataFrame(pos_band, index=[f"{a/1e9:.3g}-{b/1e9:.3g} GHz" for a, b in bands]).T
print(tab.round(2).to_string())
sig = tab.std(axis=0)
print("Smerodatna odchylka pres pozice [dB]:", sig.round(2).to_dict())

# ---------- D) polarizacni izotropie + absorber ----------
print("\n=== D) Polarizace (shake, prumer runu 01-03) a vliv absorberu ===")
def pol_band(sub, suffix):
    Ps = []
    for run in ["01", "02", "03"]:
        p = os.path.join(pol_dir, sub, f"vna_{run}shake_{suffix}.csv")
        if not os.path.exists(p):
            return None
        f, d = load(p)
        Ps.append((10.0 ** (d / 10.0)).mean(axis=0))
    Pm = np.mean(Ps, axis=0)
    return f, Pm

res = {}
for sub, suf in [("Rovne", "rovne"), ("Krizem", "krizem"),
                 ("Rovne", "rovne+absorber_lezici"), ("Rovne", "rovne+absorber_stojici"),
                 ("Krizem", "krizem+absorber_lezici"), ("Krizem", "krizem+absorber_stojici")]:
    r = pol_band(sub, suf)
    if r:
        f, Pm = r
        res[suf] = Pm
        print(f"{suf:28s}: pasmovy prumer = {10*np.log10(Pm.mean()):7.2f} dB")
co, cx = res["rovne"], res["krizem"]
print(f"\nRozdil rovne-krizem (pasmovy) = {10*np.log10(co.mean()/cx.mean()):.2f} dB")
print(f"Vliv absorberu (rovne, lezici) = {10*np.log10(res['rovne'].mean()/res['rovne+absorber_lezici'].mean()):.2f} dB")
print(f"Vliv absorberu (rovne, stojici) = {10*np.log10(res['rovne'].mean()/res['rovne+absorber_stojici'].mean()):.2f} dB")

# ---------- FIG 8: K-faktor a N_eff ----------
def smooth(x, n=51):
    return np.convolve(x, np.ones(n)/n, "same")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for run, dg in diags.items():
    ax[0].plot(dg["f"]/1e9, smooth(dg["K"]), lw=0.9, label=f"run {run}")
    ax[1].plot(dg["f"]/1e9, smooth(dg["Neff"]), lw=0.9, label=f"run {run}")
ax[0].axhline(1, color="r", ls="--", lw=0.8, label="K = 1")
ax[0].set_xlabel("f [GHz]"); ax[0].set_ylabel("K̂ (Rician)"); ax[0].set_title("Rician K-factor estimate (unstirred component)")
ax[0].legend(fontsize=7)
ax[1].set_xlabel("f [GHz]"); ax[1].set_ylabel("N_eff out of 100 sweeps")
ax[1].set_title("Effective number of independent samples (lag-1 ACF)")
ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig8_virc_statistika.png")); plt.close(fig)

# ---------- FIG 9: empiricka CDF vs exponencialni ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
dg = diags["01"]
P = dg["P"]; mu = dg["mu"]; f = dg["f"]
for fx, c in zip([1.5e9, 2.5e9, 3.5e9], ["tab:blue", "tab:orange", "tab:green"]):
    i = np.argmin(np.abs(f - fx))
    x = np.sort(P[:, i] / mu[i])
    ax[0].step(x, np.arange(1, 101)/100, color=c, lw=1, label=f"{fx/1e9:.1f} GHz")
xx = np.linspace(0, 6, 200)
ax[0].plot(xx, 1-np.exp(-xx), "k--", lw=1, label="exponential (theory)")
ax[0].set_xlabel("P / ⟨P⟩"); ax[0].set_ylabel("CDF"); ax[0].set_title("Power distribution over stirring states (run 01)")
ax[0].legend(fontsize=7)
# uniformita
x = np.arange(len(tab.index))
w = 0.25
for j, b in enumerate(tab.columns):
    ax[1].bar(x + (j-1)*w, tab[b] - tab[b].mean(), w, label=b)
ax[1].set_xticks(x); ax[1].set_xticklabels([f"pos. {p}" for p in tab.index])
ax[1].set_ylabel("deviation from mean [dB]"); ax[1].set_title("Uniformity: band-averaged level vs. antenna position")
ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig9_virc_cdf_uniformita.png")); plt.close(fig)

print("\nHotovo, obrazky fig8, fig9 ulozeny.")
