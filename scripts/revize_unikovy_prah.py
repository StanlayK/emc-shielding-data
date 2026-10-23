# -*- coding: utf-8 -*-
"""Hypoteza: v seanci 31.7. byl pritomen spolecny unikovy prah ~-45,5 dB,
ktery prebil vlastni unik nejlepe stinenych vzorku A a B.
Test: pokud ano, spektra A a B jsou v opakovane kampani skoro totozna,
zatimco v puvodni kampani se lisi."""
import os, numpy as np, pandas as pd
ROOT="/Users/stanlayk/Library/CloudStorage/OneDrive-UniverzitaTomášeBativeZlíně/Research/Měření stínění"
SP="/private/tmp/claude-501/-Users-stanlayk-Library-CloudStorage-OneDrive-UniverzitaTom--eBativeZl-n--EMC-AI/7bdec777-35da-401e-ae34-d87d55324561/scratchpad"
FGRID=np.arange(1.125e9,4.125e9+1,1e6)
NEW=os.path.join(ROOT,"Velke BLS_Imag_real_VIRC")
FOLD=lambda s:"referenční prvek" if s=="F" else s
def load_new(path):
    with open(path,encoding="utf8",errors="replace") as fh:
        hdr=fh.readline().rstrip("\n").split(","); fr=np.array([float(x) for x in hdr[3:]]); rows=[]
        for line in fh:
            q=line.rstrip("\n").split(",")
            if len(q)<4 or q[2].strip()!="dB Mag": continue
            rows.append(np.array([float(x) for x in q[3:3+len(fr)]]))
    return np.interp(FGRID,fr,np.mean(10.0**(np.array(rows)/10.0),axis=0))
def load_old(pol,s):
    d=os.path.join(ROOT,"Velké BLS",pol,"bez ABS")
    k="efe" if s=="F" else f"Sample{s}"
    c=[x for x in os.listdir(d) if "dynamic" in x and k in x and x.endswith(".csv")]
    df=pd.read_csv(os.path.join(d,c[0]),engine="c",dtype=np.float64)
    fr=np.array([float(x) for x in df.columns[1:]])
    return np.interp(FGRID,fr,np.mean(10.0**(df.iloc[:,1:].to_numpy()/10.0),axis=0))

newA=load_new(os.path.join(SP,"letni_A","Virc_A_dynamic_S21.csv"))
newB=load_new(os.path.join(NEW,"B","Virc_B_dynamic_S21.csv"))
newC=load_new(os.path.join(NEW,"C","Virc_C_dynamic_S21.csv"))
oldA,oldB,oldC=[load_old("poloha1",s) for s in "ABC"]

print("=== pasmove urovne (dB) ===")
print(f"{'':6}{'puvodni':>10}{'opakovana':>11}{'zmena':>8}{'cas 31.7.':>12}")
for s,o,n,t in [("A",oldA,newA,"13:33"),("B",oldB,newB,"14:00"),("C",oldC,newC,"15:16")]:
    print(f"{s:6}{np.mean(10*np.log10(o)):10.2f}{np.mean(10*np.log10(n)):11.2f}"
          f"{np.mean(10*np.log10(n))-np.mean(10*np.log10(o)):+8.2f}{t:>12}")

def cmp(x,y,lbl):
    dx=10*np.log10(x)-10*np.log10(y)
    r=np.corrcoef(10*np.log10(x),10*np.log10(y))[0,1]
    print(f"  {lbl:28} stredni rozdil {dx.mean():+7.2f} dB | sd rozdilu {dx.std():6.2f} dB | r = {r:.3f}")
print("\n=== podobnost spekter A vs B ===")
cmp(newA,newB,"opakovana kampan (31.7.)")
cmp(oldA,oldB,"puvodni kampan")
print("\n=== kontrola: A vs C ===")
cmp(newA,newC,"opakovana kampan")
cmp(oldA,oldC,"puvodni kampan")
