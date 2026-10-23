#!/Users/stanlayk/.venvs/emc/bin/python
# -*- coding: utf-8 -*-
"""Experimentalni overeni reciprocity: |S21| vs |S12| na tech samych konfiguracich.
Dulezite proto, ze VIRC budi vzorek (vzorek = vysilac) a GTEM budi celu
(vzorek = prijimac) -- tvrzeni, ze obe techniky meri tutez velicinu, na
reciprocite stoji, a rukopis ji nikde neuvadi."""
import os, numpy as np
ROOT="/Users/stanlayk/Library/CloudStorage/OneDrive-UniverzitaTomášeBativeZlíně/Research/Měření stínění"
FGRID=np.arange(1.125e9,4.125e9+1,1e6)

def load(path, fmt="dB Mag"):
    with open(path,encoding="utf8",errors="replace") as fh:
        hdr=fh.readline().rstrip("\n").split(",")
        fr=np.array([float(x) for x in hdr[3:]]); rows=[]
        for line in fh:
            p=line.rstrip("\n").split(",")
            if len(p)<4 or p[2].strip()!=fmt: continue
            rows.append(np.array([float(x) for x in p[3:3+len(fr)]]))
    return fr, np.array(rows)

def band_p(path):
    fr,Y=load(path)
    return np.interp(FGRID,fr,np.mean(10.0**(Y/10.0),axis=0))

NEW=os.path.join(ROOT,"GTEM_velkeBLS-Imag.,real")
FOLD=lambda s:"F-refereční prvek" if s=="F" else s
pairs=[(s,a) for s in ["A","D","F"] for a in [0,90,180,270]]
print("=== GTEM, velke BLS (srpnova kampan): |S21| vs |S12| ===")
print(f"{'vzorek/uhel':14}{'S21 [dB]':>11}{'S12 [dB]':>11}{'rozdil':>9}{'max|dif| po f':>15}")
diffs=[]
for s,a in pairs:
    p21=os.path.join(NEW,FOLD(s),f"GTEM_{s}_{a}_S21.csv")
    p12=os.path.join(NEW,FOLD(s),f"GTEM_{s}_{a}_S12.csv")
    if not (os.path.exists(p21) and os.path.exists(p12)):
        print(f"{s}/{a:<11} chybi soubor"); continue
    a21=band_p(p21); a12=band_p(p12)
    b21=np.mean(10*np.log10(a21)); b12=np.mean(10*np.log10(a12))
    perf=10*np.log10(a21)-10*np.log10(a12)
    diffs.append(b21-b12)
    print(f"{s+'/'+str(a)+'°':14}{b21:11.2f}{b12:11.2f}{b21-b12:+9.2f}{np.abs(perf).max():15.2f}")
if diffs:
    d=np.array(diffs)
    print(f"\n  pasmove prumery: mean rozdil {d.mean():+.3f} dB, sd {d.std(ddof=1):.3f} dB,"
          f" max |rozdil| {np.abs(d).max():.3f} dB  (n={len(d)})")
    print("  -> reciprocita je splnena v ramci teto meze, neni to predpoklad ale merena vlastnost")
