#!/Users/stanlayk/.venvs/emc/bin/python
# -*- coding: utf-8 -*-
"""Doplnkove analyzy pro revizi clanku 1 (Results in Engineering).

Pokryva body recenzentu, ktere se tykaji nejistot a reprodukovatelnosti:
  1) reprodukce tabulky 2 vcetne explicitniho receptu,
  2) citlivost na dve nedeklarovane volby (podmnozina konfiguraci VIRC,
     nesouhlasna rotacni mnozina u vzorku F),
  3) jackknife pres rotace GTEM,
  4) gauge R&R: bodove odhady + bootstrapove intervaly + F-testy
     + jednostranne horni meze na slozku techniky a interakce,
  5) t-interval na systematicky rozdil technik z paroveho navrhu.

Spusteni:  ~/.venvs/emc/bin/python revize_nejistoty.py
Vystup:    stdout + revize_nejistoty.csv
"""
import os, re
import numpy as np
import pandas as pd
from scipy import stats

ROOT = "/Users/stanlayk/Library/CloudStorage/OneDrive-UniverzitaTomášeBativeZlíně/Research/Měření stínění"
HERE = os.path.dirname(os.path.abspath(__file__))
FGRID = np.arange(1.125e9, 4.125e9 + 1, 1e6)      # spolecne pasmo, krok 1 MHz
ITEMS = ["A", "B", "C", "E", "Ref"]
TECHS = ["VIRC", "GTEM"]
ANG8 = [0, 45, 90, 135, 180, 225, 270, 315]
ANG7 = ANG8[1:]                                    # bez 0 stupnu (u F neni)
rng = np.random.default_rng(20261009)

_cache = {}
def p_lin(path):
    """Prumer LINEARNIHO vykonu pres 100 sweepu (prevod z dB v souboru)."""
    if path not in _cache:
        d = pd.read_csv(path, engine="c", dtype=np.float64)
        f = np.array([float(c) for c in d.columns[1:]])
        _cache[path] = (f, np.mean(10.0 ** (d.iloc[:, 1:].to_numpy() / 10.0), axis=0))
    return _cache[path]

# ---------------------------------------------------------------- data
def virc_p(poloha, absorber, vzorek):
    d = os.path.join(ROOT, "Velké BLS", poloha, absorber)
    key = "efe" if vzorek == "Ref" else f"Sample{vzorek}"
    cand = [x for x in os.listdir(d) if "dynamic" in x and key in x and x.endswith(".csv")]
    if len(cand) != 1:
        raise RuntimeError(f"{d}: ocekavan 1 soubor pro {vzorek}, nalezeno {len(cand)}")
    f, p = p_lin(os.path.join(d, cand[0]))
    return np.interp(FGRID, f, p)

def gtem_p(vzorek, uhel):
    """Rotacni sada; slozka C ma soubory s prefixem B (chyba exportu, viz known_issues)."""
    fold = "F" if vzorek == "Ref" else vzorek
    pref = "B" if fold == "C" else fold
    f, p = p_lin(os.path.join(ROOT, "VNA+GTEM_Velke_BLS", fold, f"{pref}_VNA_{uhel}.csv"))
    return np.interp(FGRID, f, p)

def gtem_angles(vzorek):
    fold = "F" if vzorek == "Ref" else vzorek
    d = os.path.join(ROOT, "VNA+GTEM_Velke_BLS", fold)
    return sorted(int(m.group(1)) for m in
                  (re.match(r"^[A-F]_VNA_(\d+)\.csv$", x) for x in os.listdir(d)) if m)

CFG_NOABS = [("poloha1", "bez ABS"), ("poloha2", "bez ABS")]
CFG_ALL4  = CFG_NOABS + [("poloha1", "s ABS"), ("poloha2", "s ABS")]

def virc_band(vzorek, cfgs):
    """Pasmovy prumer dSE vuci vzorku D, prumerovano pres konfigurace."""
    return float(np.mean([np.mean(10 * np.log10(virc_p(p, a, "D") / virc_p(p, a, vzorek)))
                          for p, a in cfgs]))

def gtem_band(vzorek, ang_item, ang_pivot):
    """Pasmovy prumer dSE vuci D; vykon se prumeruje pres rotace PRED podilem."""
    ai = [a for a in ang_item if a in gtem_angles(vzorek)]
    pX = np.mean([gtem_p(vzorek, a) for a in ai], axis=0)
    pD = np.mean([gtem_p("D", a) for a in ang_pivot], axis=0)
    return float(np.mean(10 * np.log10(pD / pX)))

rows = []
def rec(blok, popis, hodnota, jednotka=""):
    rows.append(dict(block=blok, quantity=popis, value=hodnota, unit=jednotka))

# ------------------------------------------------- 1) reprodukce tab. 2
print("=" * 78)
print("1) REPRODUCTION OF TABLE 2")
print("   Recipe actually used: GTEM = linear power pooled over rotations,")
print("   pivot D always over its 8 rotations, item F over its 7;")
print("   VIRC = the two absorber-free configurations only.")
print("=" * 78)
print(f"{'Item':8}{'VIRC':>9}{'GTEM':>9}{'Diff':>9}{'published':>22}")
PUB = {"A": (14.88, 15.66), "B": (13.85, 13.08), "C": (0.67, 0.41),
       "E": (1.91, 2.77), "Ref": (4.34, 2.59)}
v0, g0 = {}, {}
for it in ITEMS:
    v0[it] = virc_band(it, CFG_NOABS)
    g0[it] = gtem_band(it, ANG8, ANG8)
    pv, pg = PUB[it]
    flag = "exact" if abs(v0[it] - pv) < 0.005 and abs(g0[it] - pg) < 0.005 else "DIFFERS"
    print(f"{it:8}{v0[it]:9.2f}{g0[it]:9.2f}{g0[it]-v0[it]:+9.2f}{pv:9.2f}{pg:7.2f}  {flag}")
d0 = np.array([g0[i] - v0[i] for i in ITEMS])
r0 = stats.pearsonr([v0[i] for i in ITEMS], [g0[i] for i in ITEMS])
print(f"\n  mean {d0.mean():+.2f} dB | sd(ddof=0) {d0.std():.2f} | sd(ddof=1) {d0.std(ddof=1):.2f}"
      f" | r = {r0[0]:.3f} (p = {r0[1]:.3f})")
print("  NOTE: the manuscript reports sd = 0.98 dB, which is the POPULATION sd;")
print("        the sample sd of the five per-item differences is 1.10 dB.")
rec("table2", "mean technique difference", round(d0.mean(), 3), "dB")
rec("table2", "sd of differences (ddof=1)", round(d0.std(ddof=1), 3), "dB")
rec("table2", "Pearson r", round(r0[0], 4), "")

# ------------------------------------------------------- 2) citlivost
print("\n" + "=" * 78)
print("2) SENSITIVITY TO TWO UNDECLARED ANALYSIS CHOICES")
print("=" * 78)
VAR = [("A  manuscript recipe (VIRC 2 cfg; pivot 8 rot., F 7 rot.)", CFG_NOABS, ANG8),
       ("B  rotation sets harmonised to the common seven",           CFG_NOABS, ANG7),
       ("C  all four VIRC configurations",                           CFG_ALL4,  ANG8),
       ("D  both changes",                                           CFG_ALL4,  ANG7)]
print(f"{'variant':58}{'mean':>8}{'sd':>7}{'95% CI of mean':>20}")
for name, cfgs, angs in VAR:
    d = np.array([gtem_band(it, angs, angs) - virc_band(it, cfgs) for it in ITEMS])
    m, s, n = d.mean(), d.std(ddof=1), len(d)
    h = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n)
    print(f"{name:58}{m:+8.2f}{s:7.2f}   [{m-h:+.2f}, {m+h:+.2f}]")
    rec("sensitivity", name.strip(), round(m, 3), "dB")

# --------------------------------------------- 3) jackknife pres rotace
print("\n" + "=" * 78)
print("3) JACKKNIFE OVER GTEM ROTATIONS (effect of omitting one orientation)")
print("=" * 78)
print(f"{'Item':8}{'n':>4}{'max |shift|':>13}{'sd of shifts':>14}")
jk = []
for it in ITEMS:
    angs = gtem_angles(it)
    full = gtem_band(it, angs, angs)
    sh = [gtem_band(it, [a for a in angs if a != dr], angs) - full for dr in angs]
    jk.append(max(abs(x) for x in sh))
    print(f"{it:8}{len(angs):4d}{max(abs(x) for x in sh):13.2f}{np.std(sh, ddof=1):14.3f}")
print(f"\n  worst case over items: {max(jk):.2f} dB")
rec("jackknife", "worst-case shift from omitting one rotation", round(max(jk), 3), "dB")

# ------------------------------------------------------- 4) gauge R&R
print("\n" + "=" * 78)
print("4) GAUGE R&R: POINT ESTIMATES, BOOTSTRAP INTERVALS, UPPER BOUNDS")
print("=" * 78)
g = pd.read_csv(os.path.join(HERE, "gauge_rr_data.csv"))
p, t = len(ITEMS), len(TECHS)
NREP = np.array([[len(g[(g.item == i) & (g.tech == j)]) for j in TECHS] for i in ITEMS])
NH = (p * t) / np.sum(1.0 / NREP)
CM = np.array([[g[(g.item == i) & (g.tech == j)].y.mean() for j in TECHS] for i in ITEMS])
cells = [g[(g.item == i) & (g.tech == j)].y.values for i in ITEMS for j in TECHS]
DFR = NREP.sum() - p * t
MSREP = sum(((c - c.mean()) ** 2).sum() for c in cells) / DFR
print(f"  cell counts VIRC/GTEM: {NREP.tolist()}   n_h = {NH:.3f}   N = {NREP.sum()}")
print(f"  MS_rep = {MSREP:.4f} dB^2 on {DFR} df")

def mom(cm, msrep):
    gm = cm.mean(); ri = cm.mean(1); cj = cm.mean(0)
    m_item = t * np.sum((ri - gm) ** 2) / (p - 1)
    m_tech = p * np.sum((cj - gm) ** 2) / (t - 1)
    m_int = np.sum((cm - ri[:, None] - cj[None, :] + gm) ** 2) / ((p - 1) * (t - 1))
    return (max(m_item - m_int, 0) / t, max(m_tech - m_int, 0) / p,
            max(m_int - msrep / NH, 0), msrep, m_item, m_tech, m_int)

def summ(c):
    s_item, s_tech, s_int, s_rep = c[:4]
    grr = s_rep + s_tech + s_int; tot = grr + s_item
    return dict(item=s_item, tech=s_tech, inter=s_int, rep=s_rep,
                sh_item=100 * s_item / tot, sh_rep=100 * s_rep / tot,
                pctGRR=100 * np.sqrt(grr / tot),
                ndc=float(np.floor(1.41 * np.sqrt(s_item / grr))))

C0 = mom(CM, MSREP); P0 = summ(C0)
print(f"\n  sigma^2_item = {P0['item']:7.2f} dB^2  ({P0['sh_item']:.1f} %)")
print(f"  sigma^2_rep  = {P0['rep']:7.2f} dB^2  ({P0['sh_rep']:.1f} %)")
print(f"  sigma^2_tech = {P0['tech']:7.2f} dB^2  (0.0 %, truncated at zero)")
print(f"  sigma^2_int  = {P0['inter']:7.2f} dB^2  (0.0 %, truncated at zero)")
print(f"  %GRR = {P0['pctGRR']:.1f} %   ndc = {P0['ndc']:.0f}")

B = 20000
MU = CM.mean()
def sim(s_item, s_tech, s_int, s_rep):
    a = rng.normal(0, np.sqrt(s_item), p); b = rng.normal(0, np.sqrt(s_tech), t)
    ab = rng.normal(0, np.sqrt(s_int), (p, t))
    cm = np.empty((p, t)); sse = 0.0
    for i in range(p):
        for j in range(t):
            y = MU + a[i] + b[j] + ab[i, j] + rng.normal(0, np.sqrt(s_rep), NREP[i, j])
            cm[i, j] = y.mean(); sse += ((y - y.mean()) ** 2).sum()
    return cm, sse / DFR

KEYS = ["item", "rep", "tech", "inter", "pctGRR", "ndc"]
boot = {k: [] for k in KEYS}
for _ in range(B):
    s = summ(mom(*sim(P0['item'], P0['tech'], P0['inter'], P0['rep'])))
    for k in KEYS: boot[k].append(s[k])
print(f"\n  Parametric bootstrap at the fitted model, B = {B}, percentile 95 % intervals:")
for k in KEYS:
    v = np.array(boot[k], float); v = v[np.isfinite(v)]
    print(f"    {k:8} {P0[k]:8.2f}   [{np.percentile(v,2.5):8.2f}, {np.percentile(v,97.5):8.2f}]")
    rec("grr_bootstrap", k, f"{P0[k]:.2f} [{np.percentile(v,2.5):.2f}, {np.percentile(v,97.5):.2f}]", "")

_, _, _, _, m_item, m_tech, m_int = C0
Ft = m_tech / m_int
print(f"\n  Test of H0: sigma^2_tech = 0   ->  F(1,4) = {Ft:.3f}, p = {1-stats.f.cdf(Ft,1,4):.3f}")
for cl, q in [(0.95, 0.05), (0.90, 0.10)]:
    b_ = max(Ft / stats.f.ppf(q, 1, 4) - 1.0, 0.0) * m_int / p
    print(f"    one-sided {cl:.0%} upper bound:  sigma_tech <= {np.sqrt(b_):.2f} dB")
    rec("grr_bounds", f"upper bound sigma_tech ({cl:.0%})", round(float(np.sqrt(b_)), 2), "dB")
Fi = m_int / (MSREP / NH)
bi = max(Fi / stats.f.ppf(0.05, 4, DFR) - 1.0, 0.0) * MSREP / NH
print(f"  Test of H0: sigma^2_int  = 0   ->  F(4,{DFR}) = {Fi:.3f}, p = {1-stats.f.cdf(Fi,4,DFR):.3f}")
print(f"    one-sided 95% upper bound:  sigma_int <= {np.sqrt(bi):.2f} dB")
rec("grr_bounds", "upper bound sigma_int (95%)", round(float(np.sqrt(bi)), 2), "dB")

# ------------------------------------- 5) parovy navrh = vazna mez
print("\n" + "=" * 78)
print("5) PAIRED DESIGN: THE BINDING CONSTRAINT ON A SYSTEMATIC TECHNIQUE OFFSET")
print("=" * 78)
for lbl, d in [("Table 2 (VIRC absorber-free)", d0),
               ("gauge R&R data (all four VIRC configurations)", CM[:, 1] - CM[:, 0])]:
    m, s, n = d.mean(), d.std(ddof=1), len(d)
    h = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n)
    lo = s * np.sqrt((n - 1) / stats.chi2.ppf(0.975, n - 1))
    hi = s * np.sqrt((n - 1) / stats.chi2.ppf(0.025, n - 1))
    print(f"  {lbl}")
    print(f"    mean {m:+.2f} dB, sd {s:.2f} dB")
    print(f"    95 % CI of the mean [{m-h:+.2f}, {m+h:+.2f}] dB;  95 % CI of sd [{lo:.2f}, {hi:.2f}] dB")
    rec("paired", f"{lbl}: 95% CI of mean", f"[{m-h:.2f}, {m+h:.2f}]", "dB")

out = os.path.join(HERE, "revize_nejistoty.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nWritten: {out}")
