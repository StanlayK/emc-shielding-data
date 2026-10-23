#!/Users/stanlayk/.venvs/emc/bin/python
# -*- coding: utf-8 -*-
"""Druha (srpnova) GTEM kampan velkych BLS: 8 rotaci u vsech vzorku, rozsireny format.
Porovnani s publikovanou tabulkou 2."""
import os, re, numpy as np
ROOT="/Users/stanlayk/Library/CloudStorage/OneDrive-UniverzitaTomášeBativeZlíně/Research/Měření stínění"
NEW=os.path.join(ROOT,"GTEM_velkeBLS-Imag.,real")
OLD=os.path.join(ROOT,"VNA+GTEM_Velke_BLS")
FGRID=np.arange(1.125e9,4.125e9+1,1e6)
ANG=[0,45,90,135,180,225,270,315]
FOLD={s:("F-refereční prvek" if s=="F" else s) for s in "ABCDEF"}

def load_new(path, fmt="dB Mag"):
    """Radkove parsovani: radek Smith ma 2 hodnoty na frekvenci, proto nelze pd.read_csv."""
    freqs=None; rows=[]; ts=[]
    with open(path,encoding="utf8",errors="replace") as fh:
        hdr=fh.readline().rstrip("\n").split(",")
        freqs=np.array([float(x) for x in hdr[3:]])
        for line in fh:
            parts=line.rstrip("\n").split(",")
            if len(parts)<4 or parts[2].strip()!=fmt: continue
            v=np.array([float(x) for x in parts[3:3+len(freqs)]])
            rows.append(v); ts.append(parts[0])
    return freqs, np.array(rows), ts

def p_new(s,ang):
    f,Y,ts=load_new(os.path.join(NEW,FOLD[s],f"GTEM_{s}_{ang}_S21.csv"))
    p=np.mean(10.0**(Y/10.0),axis=0)
    return np.interp(FGRID,f,p), Y, ts

import pandas as pd
def p_old(s,ang):
    pref="B" if s=="C" else s
    d=pd.read_csv(os.path.join(OLD,s,f"{pref}_VNA_{ang}.csv"),engine="c",dtype=np.float64)
    f=np.array([float(c) for c in d.columns[1:]])
    return np.interp(FGRID,f,np.mean(10.0**(d.iloc[:,1:].to_numpy()/10.0),axis=0))

print("=== absolutni urovne (pasmovy prumer 10log10<P>, dB) ===")
print(f"{'vzorek':8}{'nova kampan':>14}{'stara kampan':>15}{'rozdil':>10}")
PN={}; PO={}
for s in "ABCDEF":
    pn=np.mean([p_new(s,a)[0] for a in ANG],axis=0)
    angs_old=[a for a in ANG if os.path.exists(os.path.join(OLD,s,f"{'B' if s=='C' else s}_VNA_{a}.csv"))]
    po=np.mean([p_old(s,a) for a in angs_old],axis=0)
    PN[s]=pn; PO[s]=po
    print(f"{s:8}{np.mean(10*np.log10(pn)):14.2f}{np.mean(10*np.log10(po)):15.2f}"
          f"{np.mean(10*np.log10(pn))-np.mean(10*np.log10(po)):10.2f}   (stara: {len(angs_old)} rotaci)")

print("\n=== dSE vuci vzorku D, pasmovy prumer (dB) ===")
PUB={"A":15.66,"B":13.08,"C":0.41,"E":2.77,"F":2.59}
print(f"{'vzorek':8}{'nova 8 rot.':>13}{'stara':>9}{'publik.':>9}{'nova-publik.':>14}")
dn={}; 
for s in ["A","B","C","E","F"]:
    v_new=float(np.mean(10*np.log10(PN["D"]/PN[s])))
    v_old=float(np.mean(10*np.log10(PO["D"]/PO[s])))
    dn[s]=v_new
    print(f"{s:8}{v_new:13.2f}{v_old:9.2f}{PUB[s]:9.2f}{v_new-PUB[s]:+14.2f}")
d=np.array([dn[s]-PUB[s] for s in ["A","B","C","E","F"]])
print(f"\n  rozdil nova vs publikovana: mean {d.mean():+.2f} dB, sd {d.std(ddof=1):.2f} dB,"
      f" rozsah [{d.min():+.2f}, {d.max():+.2f}] dB")
rr=np.corrcoef([dn[s] for s in ["A","B","C","E","F"]],[PUB[s] for s in ["A","B","C","E","F"]])[0,1]
print(f"  korelace poradi: r = {rr:.4f}")

print("\n=== casovani sweepu v nove kampani ===")
_,_,ts=p_new("F",0)
t=pd.to_datetime(pd.Series(ts))
dt=t.diff().dt.total_seconds().dropna().values
print(f"  n={len(dt)}  median {np.median(dt):.3f} s  mean {dt.mean():.3f} s  sd {dt.std(ddof=1):.3f} s"
      f"  min {dt.min():.3f}  max {dt.max():.3f}")
print(f"  celkem 100 sweepu za {(t.max()-t.min()).total_seconds():.1f} s")
