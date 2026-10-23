#!/Users/stanlayk/.venvs/emc/bin/python
# -*- coding: utf-8 -*-
"""Opakovana VIRC kampan velkych BLS (rozsireny format) vs. publikovany sloupec VIRC.
Doplnuje reprodukovatelnost na strane VIRC k te, kterou dava druha GTEM kampan.
Vzorek A nema v teto kampani dynamic S21 (chybi export), proto jen B, C, E, F."""
import os, numpy as np, pandas as pd
ROOT="/Users/stanlayk/Library/CloudStorage/OneDrive-UniverzitaTomášeBativeZlíně/Research/Měření stínění"
NEW=os.path.join(ROOT,"Velke BLS_Imag_real_VIRC")
FGRID=np.arange(1.125e9,4.125e9+1,1e6)
FOLD=lambda s:"referenční prvek" if s=="F" else s

def load_p(path,fmt="dB Mag"):
    with open(path,encoding="utf8",errors="replace") as fh:
        hdr=fh.readline().rstrip("\n").split(",")
        fr=np.array([float(x) for x in hdr[3:]]); rows=[]
        for line in fh:
            p=line.rstrip("\n").split(",")
            if len(p)<4 or p[2].strip()!=fmt: continue
            rows.append(np.array([float(x) for x in p[3:3+len(fr)]]))
    Y=np.array(rows)
    return np.interp(FGRID,fr,np.mean(10.0**(Y/10.0),axis=0))

PUB={"B":13.85,"C":0.67,"E":1.91,"F":4.34}
for tag,suf in [("bez absorbéru (srovnatelné s publikovaným)","dynamic"),
                ("s absorbérem","dynamic_abs")]:
    try:
        pD=load_p(os.path.join(NEW,"D",f"Virc_D_{suf}_S21.csv"))
    except FileNotFoundError:
        print(f"{tag}: chybi pivot D"); continue
    print("="*66); print(tag)
    print(f"{'vzorek':8}{'opakovana':>12}{'publikovano':>13}{'rozdil':>9}")
    d=[]
    for s in ["B","C","E","F"]:
        try: pX=load_p(os.path.join(NEW,FOLD(s),f"Virc_{s}_{suf}_S21.csv"))
        except FileNotFoundError: print(f"{s:8}  chybi soubor"); continue
        v=float(np.mean(10*np.log10(pD/pX))); d.append(v-PUB[s])
        print(f"{s:8}{v:12.2f}{PUB[s]:13.2f}{v-PUB[s]:+9.2f}")
    if d:
        d=np.array(d)
        print(f"\n  mean {d.mean():+.2f} dB, sd {d.std(ddof=1):.2f} dB, "
              f"max |rozdil| {np.abs(d).max():.2f} dB  (n={len(d)})")
