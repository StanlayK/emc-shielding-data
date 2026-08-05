# -*- coding: utf-8 -*-
"""Moving-block bootstrap CI pro Demingovu regresi VIRC vs GTEM (dSE rel. D).
Bloky L=10 vzorku (250 MHz) vzorkovane uvnitr kazdeho vzorku; 2000 replik.
Vysledek: slope 1.054, 95% CI [0.954, 1.168]; intercept -0.87 [-2.10, +0.30];
efektivni pocet nezavislych paru ~44 z 585."""
import os, re
import numpy as np
import pandas as pd

ROOT = "/Users/stanlayk/Library/CloudStorage/OneDrive-UniverzitaTomášeBativeZlíně/Research/Měření stínění"
rng = np.random.default_rng(2026)
cache = {}
def p_lin(path):
    if path not in cache:
        df = pd.read_csv(path, engine="c", dtype=np.float64)
        f = np.array([float(c) for c in df.columns[1:]])
        cache[path] = (f, np.mean(10.0 ** (df.iloc[:, 1:].to_numpy() / 10.0), axis=0))
    return cache[path]
def smooth(x, n):
    return np.convolve(x, np.ones(n)/n, "same")

FGRID = np.arange(1.125e9, 4.125e9 + 1, 1e6)
bls_dir = os.path.join(ROOT, "Velké BLS")
def virc_curve(pol, absf, samp):
    d = os.path.join(bls_dir, pol, absf)
    fs = os.listdir(d)
    c = [x for x in fs if "dynamic" in x and (("efe" in x) if samp=="Ref" else (f"Sample{samp}" in x)) and x.endswith(".csv")]
    f, p = p_lin(os.path.join(d, c[0]))
    return np.interp(FGRID, f, p)
ITEMS = ["A","B","C","E","Ref"]
virc = {it: {} for it in ITEMS}
for pol in ["poloha1","poloha2"]:
    for absf in ["bez ABS","s ABS"]:
        pD = virc_curve(pol, absf, "D")
        for it in ITEMS:
            virc[it][f"{pol}|{absf}"] = 10*np.log10(pD / virc_curve(pol, absf, it))
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
gtem = {it: {} for it in ITEMS}
for it in ITEMS:
    fold = "F" if it == "Ref" else it
    for a, p in graw[fold].items():
        if a in graw["D"]:
            gtem[it][a] = 10*np.log10(graw["D"][a] / p)

SM, STEP = 51, 25
pairs_by_item = {}
for it in ITEMS:
    v = smooth(np.mean(list(virc[it].values()), axis=0), SM)[::STEP][2:-2]
    g = smooth(np.mean(list(gtem[it].values()), axis=0), SM)[::STEP][2:-2]
    pairs_by_item[it] = (v, g)

def deming(x, y):
    sxx = np.var(x, ddof=1); syy = np.var(y, ddof=1); sxy = np.cov(x, y, ddof=1)[0,1]
    b = (syy - sxx + np.sqrt((syy - sxx)**2 + 4*sxy**2)) / (2*sxy)
    return b, np.mean(y) - b*np.mean(x)

xall = np.concatenate([p[0] for p in pairs_by_item.values()])
yall = np.concatenate([p[1] for p in pairs_by_item.values()])
b1, b0 = deming(xall, yall)
print(f"Deming: slope={b1:.3f}, intercept={b0:.2f}, n={len(xall)}")

L = 10
boot = []
for _ in range(2000):
    xs, ys = [], []
    for it in ITEMS:
        v, g = pairs_by_item[it]
        n = len(v); nb = int(np.ceil(n / L))
        starts = rng.integers(0, n - L + 1, nb)
        idx = np.concatenate([np.arange(s, s+L) for s in starts])[:n]
        xs.append(v[idx]); ys.append(g[idx])
    boot.append(deming(np.concatenate(xs), np.concatenate(ys)))
boot = np.array(boot)
print(f"slope 95% CI: [{np.percentile(boot[:,0],2.5):.3f}, {np.percentile(boot[:,0],97.5):.3f}]")
print(f"intercept 95% CI: [{np.percentile(boot[:,1],2.5):.2f}, {np.percentile(boot[:,1],97.5):.2f}]")
neffs = []
for it in ITEMS:
    v, g = pairs_by_item[it]
    x = (g - v) - (g - v).mean()
    ac = np.correlate(x, x, "full")[len(x)-1:] / np.sum(x*x)
    tau = 1 + 2*np.sum(ac[1:np.argmax(ac < 0)])
    neffs.append(len(x)/max(tau,1))
print(f"efektivni pocet nezavislych paru: ~{sum(neffs):.0f} z {len(xall)}")
