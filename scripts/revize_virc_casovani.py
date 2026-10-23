#!/Users/stanlayk/.venvs/emc/bin/python
# -*- coding: utf-8 -*-
"""Kadence sweepu ve VIRC a test periodicity agitace -- z rozsireneho exportu.
Odpovida na dve z peti chybejicich informaci o VIRC primo z dat."""
import os, numpy as np, pandas as pd
ROOT="/Users/stanlayk/Library/CloudStorage/OneDrive-UniverzitaTomášeBativeZlíně/Research/Měření stínění"
FGRID=np.arange(1.125e9,4.125e9+1,1e6)

def load(path,fmt="dB Mag"):
    with open(path,encoding="utf8",errors="replace") as fh:
        hdr=fh.readline().rstrip("\n").split(",")
        fr=np.array([float(x) for x in hdr[3:]]); rows=[]; ts=[]
        for line in fh:
            p=line.rstrip("\n").split(",")
            if len(p)<4 or p[2].strip()!=fmt: continue
            rows.append(np.array([float(x) for x in p[3:3+len(fr)]])); ts.append(p[0])
    return fr,np.array(rows),ts

V=os.path.join(ROOT,"Velke BLS_Imag_real_VIRC","referenční prvek")
for tag,fn in [("dynamic (stěny kmitají)","Virc_F_dynamic_S21.csv"),
               ("static (stěny v klidu)","Virc_F_static_S21.csv")]:
    path=os.path.join(V,fn)
    if not os.path.exists(path): print(f"chybi {fn}"); continue
    fr,Y,ts=load(path)
    t=pd.to_datetime(pd.Series(ts)); dt=t.diff().dt.total_seconds().dropna().values
    band=(fr>=1.125e9)&(fr<=4.125e9)
    p=np.mean(10.0**(Y[:,band]/10.0),axis=1); pdb=10*np.log10(p)
    print("="*70); print(f"{tag}   [{fn}]")
    print(f"  sweepu: {len(Y)}   akvizice {(t.max()-t.min()).total_seconds():.1f} s")
    print(f"  KADENCE: median {np.median(dt):.3f} s, mean {dt.mean():.3f} s, sd {dt.std(ddof=1):.3f} s,"
          f" min {dt.min():.3f}, max {dt.max():.3f}")
    print(f"  pasmovy vykon po sweepech: {pdb.mean():.2f} dB, sd {pdb.std(ddof=1):.2f} dB,"
          f" rozsah {pdb.min():.2f}..{pdb.max():.2f} dB")
    x=p-p.mean(); n=len(x); den=np.sum(x*x)
    ac=np.array([np.sum(x[:n-k]*x[k:])/den for k in range(21)])
    print("  autokorelace lag 1-10: "+" ".join(f"{v:+.3f}" for v in ac[1:11]))
    print(f"  lag-1 rho = {ac[1]:+.4f}  ->  N_eff = {n*(1-ac[1])/(1+ac[1]):.1f} ze {n}")
    tau=1+2*np.sum(ac[1:]); print(f"  integr. autokor. doba = {tau:.2f}  ->  N_eff = {n/max(tau,1):.1f}")
    X=np.abs(np.fft.rfft(x))**2; fq=np.fft.rfftfreq(n,d=float(np.median(dt)))
    k=np.argmax(X[1:])+1; g=X[1:].max()/X[1:].sum(); m=len(X)-1
    print(f"  periodogram: nejsilnejsi {fq[k]:.4f} Hz (perioda {1/fq[k]:.2f} s), "
          f"{X[k]/X[1:].mean():.2f}x stredni vykon")
    print(f"  Fisherova g = {g:.4f}, p = {min(1.0,m*(1-g)**(m-1)):.3f}"
          "   (velke p = zadna detekovatelna periodicita)")
