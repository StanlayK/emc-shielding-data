#!/Users/stanlayk/.venvs/emc/bin/python
# -*- coding: utf-8 -*-
"""Revize clanku 1: interpretace eta^2 u funkcionalni PCA.

Doplnuje k publikovanym eta^2 dve veci, ktere recenzenti chteji:
  - omega^2, tj. eta^2 opravene o vychylku pri malem vzorku,
  - permutacni nulove rozdeleni eta^2 pro techniku, ktere dava mez
    rozlisitelnosti navrhu (nejmensi efekt, ktery by sel odhalit).
Replikuje FPCA z pokrocile_metody.py (okno 31 MHz, decimace 5).
"""
import os, re
import numpy as np, pandas as pd

ROOT = "/Users/stanlayk/Library/CloudStorage/OneDrive-UniverzitaTomášeBativeZlíně/Research/Měření stínění"
HERE = os.path.dirname(os.path.abspath(__file__))
FGRID = np.arange(1.125e9, 4.125e9 + 1, 1e6)
ITEMS = ["A", "B", "C", "E", "Ref"]; TECHS = ["VIRC", "GTEM"]
SM, STEP = 31, 5
rng = np.random.default_rng(20261009)

_c = {}
def p_lin(path):
    if path not in _c:
        d = pd.read_csv(path, engine="c", dtype=np.float64)
        f = np.array([float(x) for x in d.columns[1:]])
        _c[path] = (f, np.mean(10.0 ** (d.iloc[:, 1:].to_numpy() / 10.0), axis=0))
    return _c[path]
def smooth(x, n): return np.convolve(x, np.ones(n) / n, "same")

def virc_p(pol, absf, s):
    d = os.path.join(ROOT, "Velké BLS", pol, absf)
    k = "efe" if s == "Ref" else f"Sample{s}"
    c = [x for x in os.listdir(d) if "dynamic" in x and k in x and x.endswith(".csv")]
    f, p = p_lin(os.path.join(d, c[0])); return np.interp(FGRID, f, p)

GD = os.path.join(ROOT, "VNA+GTEM_Velke_BLS")
def gtem_rot(s):
    fold = "F" if s == "Ref" else s
    out = {}
    for x in os.listdir(os.path.join(GD, fold)):
        m = re.match(r"^[A-F]_VNA_(\d+)\.csv$", x)
        if m:
            f, p = p_lin(os.path.join(GD, fold, x))
            out[int(m.group(1))] = np.interp(FGRID, f, p)
    return out
RAW = {s: gtem_rot(s) for s in ITEMS + ["D"]}

X, meta = [], []
for it in ITEMS:
    for pol in ["poloha1", "poloha2"]:
        for absf in ["bez ABS", "s ABS"]:
            c = 10 * np.log10(virc_p(pol, absf, "D") / virc_p(pol, absf, it))
            X.append(smooth(c, SM)[::STEP]); meta.append((it, "VIRC"))
    for a, p in sorted(RAW[it].items()):
        if a in RAW["D"]:
            c = 10 * np.log10(RAW["D"][a] / p)
            X.append(smooth(c, SM)[::STEP]); meta.append((it, "GTEM"))
X = np.array(X); meta = pd.DataFrame(meta, columns=["item", "tech"])
print(f"curves: {len(X)} (VIRC {sum(meta.tech=='VIRC')}, GTEM {sum(meta.tech=='GTEM')}), "
      f"grid {X.shape[1]} points")
U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
expl = S ** 2 / (S ** 2).sum(); scores = U * S
print("explained variance PC1-4: " + " ".join(f"{v*100:.1f}%" for v in expl[:4]))

def eff(sc, lab, groups):
    """eta^2 (deskriptivni) a omega^2 (opraveny o vychylku)."""
    n = len(sc); k = len(groups); gm = sc.mean()
    ssb = sum(len(sc[lab == g]) * (sc[lab == g].mean() - gm) ** 2 for g in groups)
    sst = ((sc - gm) ** 2).sum(); ssw = sst - ssb
    msw = ssw / (n - k)
    return ssb / sst, (ssb - (k - 1) * msw) / (sst + msw)

print("\n=== EFFECT SIZES OF THE SCORES ===")
print(f"{'':6}{'eta2 item':>11}{'omega2 item':>13}{'eta2 tech':>11}{'omega2 tech':>13}"
      f"{'null 95th':>11}{'perm. p':>10}")
NP_ = 20000
rows = []
for k in range(3):
    sc = scores[:, k]
    e_i, o_i = eff(sc, meta.item.values, ITEMS)
    e_t, o_t = eff(sc, meta.tech.values, TECHS)
    null = np.empty(NP_)
    lab = meta.tech.values.copy()
    for b in range(NP_):
        rng.shuffle(lab)
        null[b] = eff(sc, lab, TECHS)[0]
    q95 = np.percentile(null, 95); pv = (np.sum(null >= e_t) + 1) / (NP_ + 1)
    print(f"PC{k+1:<4}{e_i:11.3f}{o_i:13.3f}{e_t:11.3f}{o_t:13.3f}{q95:11.3f}{pv:10.3f}")
    rows.append(dict(component=f"PC{k+1}", explained_pct=round(expl[k]*100,1),
                     eta2_item=round(e_i,4), omega2_item=round(o_i,4),
                     eta2_tech=round(e_t,4), omega2_tech=round(o_t,4),
                     null_95th=round(q95,4), perm_p=round(pv,4)))
out = os.path.join(HERE, "revize_fpca_eta.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nWritten: {out}")
print("""
Reading of the table:
  eta^2  = share of the between-curve variance of a score attributable to a factor
           (descriptive ratio, biased upward in small samples);
  omega^2 = the same quantity corrected for that bias;
  null 95th = 95th percentile of the permutation null for the technique, i.e. the
           smallest technique effect this design could have resolved;
  perm. p = permutation p-value for the technique factor.""")
