#!/Users/stanlayk/.venvs/emc/bin/python
# -*- coding: utf-8 -*-
"""Obrazek 3 (fig_cross) s chybovymi useckami -- zadost recenzenta.

Body jsou hodnoty z tabulky 2 rukopisu, aby obrazek a tabulka souhlasily.
Usecky jsou standardni chyby prumeru, pricemz skala je odhadnuta jako
SDRUZENA vnitrotechnicka opakovatelnost pres polozky (vic stupnu volnosti
nez sd jedne polozky), delena odmocninou z poctu replikaci, ktere do
prumeru skutecne vstoupily: 2 konfigurace VIRC, 8 rotaci GTEM (7 u F).
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3})
ITEMS = ["A", "B", "C", "E", "Ref"]
LBL = {"A": "A", "B": "B", "C": "C", "E": "E", "Ref": "F (unshielded)"}
V = {"A": 14.88, "B": 13.85, "C": 0.67, "E": 1.91, "Ref": 4.34}
G = {"A": 15.66, "B": 13.08, "C": 0.41, "E": 2.77, "Ref": 2.59}
NV = {i: 2 for i in ITEMS}
NG = {"A": 8, "B": 8, "C": 8, "E": 8, "Ref": 7}

g = pd.read_csv(os.path.join(HERE, "gauge_rr_data.csv"))
def pooled_sd(tech):
    ss = 0.0; df = 0
    for it in ITEMS:
        y = g[(g.item == it) & (g.tech == tech)].y.values
        ss += ((y - y.mean()) ** 2).sum(); df += len(y) - 1
    return np.sqrt(ss / df), df
sV, dfV = pooled_sd("VIRC"); sG, dfG = pooled_sd("GTEM")
print(f"pooled within-technique repeatability:")
print(f"  VIRC  sd = {sV:.2f} dB on {dfV} df")
print(f"  GTEM  sd = {sG:.2f} dB on {dfG} df")
uV = {i: sV / np.sqrt(NV[i]) for i in ITEMS}
uG = {i: sG / np.sqrt(NG[i]) for i in ITEMS}
print(f"\n{'item':14}{'VIRC':>8}{'u(V)':>7}{'GTEM':>8}{'u(G)':>7}{'diff':>8}{'u(diff)':>9}{'|d|/u':>7}")
for i in ITEMS:
    ud = np.hypot(uV[i], uG[i]); d = G[i] - V[i]
    print(f"{LBL[i]:14}{V[i]:8.2f}{uV[i]:7.2f}{G[i]:8.2f}{uG[i]:7.2f}{d:+8.2f}{ud:9.2f}{abs(d)/ud:7.2f}")

fig, ax = plt.subplots(figsize=(5.0, 4.6))
lo = min(min(V.values()), min(G.values())) - 2.5
hi = max(max(V.values()), max(G.values())) + 2.5
ax.plot([lo, hi], [lo, hi], "k--", lw=0.9, zorder=1, label="identity")
cols = plt.cm.viridis(np.linspace(0.08, 0.86, len(ITEMS)))
for c, i in zip(cols, ITEMS):
    ax.errorbar(V[i], G[i], xerr=uV[i], yerr=uG[i], fmt="o", ms=6.5,
                color=c, ecolor=c, elinewidth=1.3, capsize=3.2, zorder=3,
                label=LBL[i])
    ax.annotate(LBL[i].split()[0], (V[i], G[i]), textcoords="offset points",
                xytext=(8, -11), fontsize=8.5, color=c)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
ax.set_xlabel(r"VIRC  $\overline{\Delta \mathrm{SE}}$  [dB]")
ax.set_ylabel(r"GTEM  $\overline{\Delta \mathrm{SE}}$  [dB]")
ax.set_title("Cross-technique comparison, band-averaged 1.125–4.125 GHz", fontsize=9.5)
ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
fig.tight_layout()
out = os.path.join(HERE, "fig7_cross_chybove_usecky.png")
fig.savefig(out); plt.close(fig)
print(f"\nWritten: {out}")
print("\nCaption proposal:")
print(f"""  Cross-technique comparison of the band-averaged differential SE relative
  to sample D. The dashed line is the identity. Error bars are standard
  errors of the mean, obtained from the within-technique repeatability
  pooled over items ({sV:.2f} dB for the VIRC over {dfV} df, {sG:.2f} dB for the
  GTEM cell over {dfG} df) and the number of replicates entering each mean:
  two absorber-free configurations in the VIRC, eight rotations in the GTEM
  cell and seven for variant F. Every item lies within two combined standard
  uncertainties of the identity, variant F being the most displaced at
  {abs(G["Ref"]-V["Ref"])/np.hypot(uV["Ref"],uG["Ref"]):.2f} standard uncertainties.""")
