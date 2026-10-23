# -*- coding: utf-8 -*-
"""Rozhodujici test hypotezy o unikovem prahu v seanci 31.7.
Pokud byl v sestave pritomen pridavny unik ~-45,5 dB, musi se objevit
i ve STATICKYCH bezich tychz vzorku z teze seance."""
import os, numpy as np, pandas as pd
ROOT="/Users/stanlayk/Library/CloudStorage/OneDrive-UniverzitaTomášeBativeZlíně/Research/Měření stínění"
NEW=os.path.join(ROOT,"Velke BLS_Imag_real_VIRC")
SP="/private/tmp/claude-501/-Users-stanlayk-Library-CloudStorage-OneDrive-UniverzitaTom--eBativeZl-n--EMC-AI/7bdec777-35da-401e-ae34-d87d55324561/scratchpad"
FGRID=np.arange(1.125e9,4.125e9+1,1e6)
FOLD=lambda s:"referenční prvek" if s=="F" else s
def load(path):
    with open(path,encoding="utf8",errors="replace") as fh:
        hdr=fh.readline().rstrip("\n").split(","); fr=np.array([float(x) for x in hdr[3:]])
        rows=[]; ts=[]
        for line in fh:
            q=line.rstrip("\n").split(",")
            if len(q)<4 or q[2].strip()!="dB Mag": continue
            rows.append(np.array([float(x) for x in q[3:3+len(fr)]])); ts.append(q[0])
    return np.interp(FGRID,fr,np.mean(10.0**(np.array(rows)/10.0),axis=0)), ts[0]

# puvodni kampan (statika i dynamika) pro srovnani
def old(pol,absf,s,mode):
    d=os.path.join(ROOT,"Velké BLS",pol,absf)
    k="efe" if s=="F" else f"Sample{s}"
    c=[x for x in os.listdir(d) if mode in x and k in x and x.endswith(".csv")]
    df=pd.read_csv(os.path.join(d,c[0]),engine="c",dtype=np.float64)
    fr=np.array([float(x) for x in df.columns[1:]])
    return np.interp(FGRID,fr,np.mean(10.0**(df.iloc[:,1:].to_numpy()/10.0),axis=0))

print(f"{'vzorek':8}{'rezim':10}{'puvodni':>10}{'opakovana':>11}{'zmena':>8}  {'cas opakovane':<26}")
for s in ["A","B","C","D","E","F"]:
    for mode,suf in [("dynamic","dynamic"),("static","static")]:
        try:
            if s=="A" and mode=="dynamic":
                pn,t=load(os.path.join(SP,"letni_A","Virc_A_dynamic_S21.csv"))
            else:
                pn,t=load(os.path.join(NEW,FOLD(s),f"Virc_{s}_{suf}_S21.csv"))
            po=old("poloha1","bez ABS",s,mode)
            a=np.mean(10*np.log10(po)); b=np.mean(10*np.log10(pn))
            print(f"{s:8}{mode:10}{a:10.2f}{b:11.2f}{b-a:+8.2f}  {t:<26}")
        except Exception as e:
            print(f"{s:8}{mode:10}  {type(e).__name__}: {str(e)[:50]}")
