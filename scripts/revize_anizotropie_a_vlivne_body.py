#!/Users/stanlayk/.venvs/emc/bin/python
# -*- coding: utf-8 -*-
"""Dve doplnkove analyzy pro revizi:
 (a) rotacni rozptyl GTEM jako funkce frekvence  -> fyzikalni vyklad odchylky
     nad 2,8 GHz (R1 bod 3, R4 bod 5, R2 limitace azimutalni rotace),
 (b) vliv jednotlivych polozek na r = 0,988 (R3 bod 7).
"""
import os, re
import numpy as np, pandas as pd
from scipy import stats

ROOT = "/Users/stanlayk/Library/CloudStorage/OneDrive-UniverzitaTomášeBativeZlíně/Research/Měření stínění"
HERE = os.path.dirname(os.path.abspath(__file__))
FGRID = np.arange(1.125e9, 4.125e9 + 1, 1e6)
ANG = [0, 45, 90, 135, 180, 225, 270, 315]
_c = {}
def p_lin(p, tries=8):
    """OneDrive obcas vrati TimeoutError, nez stihne soubor stahnout -- opakuj."""
    if p not in _c:
        import time
        for k in range(tries):
            try:
                d = pd.read_csv(p, engine="c", dtype=np.float64)
                break
            except (TimeoutError, OSError) as e:
                if k == tries - 1:
                    raise
                print(f"    [{os.path.basename(p)}] {type(e).__name__}, pokus {k+2}/{tries}", flush=True)
                time.sleep(5)
        f = np.array([float(x) for x in d.columns[1:]])
        _c[p] = (f, np.mean(10.0 ** (d.iloc[:, 1:].to_numpy() / 10.0), axis=0))
    return _c[p]
GD = os.path.join(ROOT, "VNA+GTEM_Velke_BLS")
def gtem(s, a):
    pref = "B" if s == "C" else s
    f, p = p_lin(os.path.join(GD, s, f"{pref}_VNA_{a}.csv"))
    return np.interp(FGRID, f, p)
def angles(s):
    return sorted(int(m.group(1)) for m in
                  (re.match(r"^[A-F]_VNA_(\d+)\.csv$", x) for x in os.listdir(os.path.join(GD, s))) if m)

print("=" * 76)
print("(a) ROTATION SPREAD OF THE GTEM ENSEMBLE VERSUS FREQUENCY")
print("    sd over rotations of the received level, in 250 MHz sub-bands")
print("=" * 76)
EDGES = np.arange(1.125e9, 4.126e9, 250e6)
rows = []
for s in ["A", "B", "C", "D", "E", "F"]:
    P = np.array([gtem(s, a) for a in angles(s)])          # rotace x frekvence
    sub = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (FGRID >= lo) & (FGRID < hi)
        lvl = 10 * np.log10(P[:, m].mean(axis=1))          # uroven na rotaci
        sub.append(lvl.std(ddof=1))
    rows.append(sub)
    print(f"  {s}: " + " ".join(f"{v:4.1f}" for v in sub))
M = np.array(rows)
ctr = (EDGES[:-1] + EDGES[1:]) / 2 / 1e9
print("\n  f [GHz]: " + " ".join(f"{v:4.2f}" for v in ctr))
print("  mean   : " + " ".join(f"{v:4.1f}" for v in M.mean(axis=0)))
lo = ctr < 2.8; hi = ctr >= 2.8
print(f"\n  below 2.8 GHz: mean spread {M[:, lo].mean():.2f} dB")
print(f"  above 2.8 GHz: mean spread {M[:, hi].mean():.2f} dB   -> ratio {M[:,hi].mean()/M[:,lo].mean():.2f}x")
t = stats.ttest_ind(M[:, lo].ravel(), M[:, hi].ravel(), equal_var=False)
print(f"  Welch t-test: t = {t.statistic:.2f}, p = {t.pvalue:.4f}")
r_ = stats.pearsonr(np.tile(ctr, len(rows)), M.ravel())
print(f"  correlation of spread with frequency: r = {r_[0]:.3f} (p = {r_[1]:.1e})")

print("\n" + "=" * 76)
print("(b) INFLUENCE OF SINGLE ITEMS ON THE CROSS-TECHNIQUE CORRELATION")
print("=" * 76)
V = {"A": 14.88, "B": 13.85, "C": 0.67, "E": 1.91, "Ref": 4.34}
G = {"A": 15.66, "B": 13.08, "C": 0.41, "E": 2.77, "Ref": 2.59}
IT = list(V)
x = np.array([V[i] for i in IT]); y = np.array([G[i] for i in IT])
r0 = stats.pearsonr(x, y)
print(f"  all five items: r = {r0[0]:.4f} (p = {r0[1]:.4f})")
print(f"\n  {'omitted':10}{'r':>9}{'p':>9}{'slope':>9}")
for k, it in enumerate(IT):
    m = np.ones(len(IT), bool); m[k] = False
    rr = stats.pearsonr(x[m], y[m]); sl = np.polyfit(x[m], y[m], 1)[0]
    print(f"  {it:10}{rr[0]:9.4f}{rr[1]:9.4f}{sl:9.3f}")
rs = stats.spearmanr(x, y)
print(f"\n  Spearman (rank) over all five: rho = {rs.statistic:.3f} (p = {rs.pvalue:.3f})")
print("  Interpretation: if r stays high with every single item removed, the")
print("  correlation is not carried by one point.")
