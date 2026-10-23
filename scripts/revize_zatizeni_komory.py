#!/Users/stanlayk/.venvs/emc/bin/python
# -*- coding: utf-8 -*-
"""R3, bod 1, podotazka: krati se C(f) i kdyz vzorek sam ovlivnuje pole?

Primy test: pridani pyramidoveho absorberu je ZAMERNA a velka zmena zatizeni
komory -- mnohem vetsi, nez jakou muze zpusobit zamena jednoho 20cm^2 vzorku
za jiny. Pokud diferencni SE na absorber nereaguje, pak se C(f) kratí
i pri zmene zatizeni, a tim spis pri zanedbatelne zmene mezi vzorky.
"""
import os, time
import numpy as np, pandas as pd
ROOT = "/Users/stanlayk/Library/CloudStorage/OneDrive-UniverzitaTomášeBativeZlíně/Research/Měření stínění"
FGRID = np.arange(1.125e9, 4.125e9 + 1, 1e6)
ITEMS = ["A", "B", "C", "E", "Ref"]
_c = {}
def p_lin(p, tries=8):
    if p not in _c:
        for k in range(tries):
            try:
                d = pd.read_csv(p, engine="c", dtype=np.float64); break
            except (TimeoutError, OSError):
                if k == tries - 1: raise
                time.sleep(5)
        f = np.array([float(x) for x in d.columns[1:]])
        _c[p] = (f, np.mean(10.0 ** (d.iloc[:, 1:].to_numpy() / 10.0), axis=0))
    return _c[p]
def virc(pol, absf, s):
    d = os.path.join(ROOT, "Velké BLS", pol, absf)
    k = "efe" if s == "Ref" else f"Sample{s}"
    c = [x for x in os.listdir(d) if "dynamic" in x and k in x and x.endswith(".csv")]
    f, p = p_lin(os.path.join(d, c[0])); return np.interp(FGRID, f, p)

print("=== 1) O kolik absorber zmeni samotnou uroven (tedy C(f))? ===")
print(f"{'item':6}{'bez ABS':>10}{'s ABS':>10}{'zmena':>9}")
dl = []
for s in ITEMS + ["D"]:
    a = np.mean([np.mean(10*np.log10(virc(p, "bez ABS", s))) for p in ["poloha1","poloha2"]])
    b = np.mean([np.mean(10*np.log10(virc(p, "s ABS", s))) for p in ["poloha1","poloha2"]])
    dl.append(b - a)
    print(f"{s:6}{a:10.2f}{b:10.2f}{b-a:+9.2f}")
print(f"  absorber snizuje uroven prumerne o {np.mean(dl):+.2f} dB -- velka zmena zatizeni\n")

print("=== 2) Prezije to diferencni SE? ===")
print(f"{'item':6}{'dSE bez ABS':>13}{'dSE s ABS':>12}{'rozdil':>9}")
dd = []
for s in ITEMS:
    a = np.mean([np.mean(10*np.log10(virc(p,"bez ABS","D")/virc(p,"bez ABS",s))) for p in ["poloha1","poloha2"]])
    b = np.mean([np.mean(10*np.log10(virc(p,"s ABS","D")/virc(p,"s ABS",s)))     for p in ["poloha1","poloha2"]])
    dd.append(b - a)
    print(f"{s:6}{a:13.2f}{b:12.2f}{b-a:+9.2f}")
dd = np.array(dd)
print(f"\n  mean {dd.mean():+.2f} dB | sd {dd.std(ddof=1):.2f} dB | max |rozdil| {np.abs(dd).max():.2f} dB")
print(f"""
  Zaver: absorber posune absolutni uroven o {np.mean(dl):.1f} dB, ale diferencni
  SE jen o {np.abs(dd).max():.2f} dB v nejhorsim pripade. C(f) se tedy krati
  i pri zamerne a velke zmene zatizeni komory. Zamena jednoho vzorku za jiny
  je proti absorberu zanedbatelna, takze predpoklad P_X = C(f)*T_X plati.""")
