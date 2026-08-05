# -*- coding: utf-8 -*-
"""Katalogizace a analyza dat ze slozky 'Mereni stineni'."""
import os, re, json, sys
import numpy as np
import pandas as pd

ROOT = "/Users/stanlayk/Library/CloudStorage/OneDrive-UniverzitaTomášeBativeZlíně/Research/Měření stínění"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vysledky")
os.makedirs(OUT, exist_ok=True)

cache = {}

def load(path):
    """Vrati (freq[Hz], data[n_sweeps, n_freq] v dB)."""
    if path in cache:
        return cache[path]
    df = pd.read_csv(path, engine="c", dtype=np.float64)
    freq = np.array([float(c) for c in df.columns[1:]])
    data = df.iloc[:, 1:].to_numpy()
    cache[path] = (freq, data)
    return freq, data

def p_lin(path):
    """Prumerny linearni vykon pres sweepy (ensemble average)."""
    f, d = load(path)
    return f, np.mean(10.0 ** (d / 10.0), axis=0)

def se(ref_path, samp_path):
    f1, pr = p_lin(ref_path)
    f2, ps = p_lin(samp_path)
    if len(f1) != len(f2) or not np.allclose(f1, f2):
        # interpolace na spolecnou osu
        flo, fhi = max(f1[0], f2[0]), min(f1[-1], f2[-1])
        f = f1[(f1 >= flo) & (f1 <= fhi)]
        pr = np.interp(f, f1, pr); ps = np.interp(f, f2, ps)
        return f, 10 * np.log10(pr / ps)
    return f1, 10 * np.log10(pr / ps)

# ---------------- 1. KATALOG ----------------
catalog = []
for dirpath, _, files in os.walk(ROOT):
    for fn in sorted(files):
        if not fn.lower().endswith(".csv"):
            continue
        p = os.path.join(dirpath, fn)
        try:
            with open(p) as fh:
                header = fh.readline().strip().split(",")
            if header[0] != "Sweep_Nr":
                catalog.append(dict(path=os.path.relpath(p, ROOT), typ="tabulka/jine"))
                continue
            freqs = [float(x) for x in header[1:]]
            n_rows = sum(1 for _ in open(p)) - 1
            catalog.append(dict(
                path=os.path.relpath(p, ROOT), typ="mereni",
                n_sweeps=n_rows, n_freq=len(freqs),
                f_min_MHz=freqs[0] / 1e6, f_max_MHz=freqs[-1] / 1e6,
                df_kHz=(freqs[1] - freqs[0]) / 1e3 if len(freqs) > 1 else 0))
        except Exception as e:
            catalog.append(dict(path=os.path.relpath(p, ROOT), typ="CHYBA: %s" % e))
cat = pd.DataFrame(catalog)
cat.to_csv(os.path.join(OUT, "katalog.csv"), index=False)

m = cat[cat.typ == "mereni"].copy()
m["dataset"] = m.path.str.split("/").str[0]
grp = m.groupby(["dataset", "n_freq", "f_min_MHz", "f_max_MHz", "n_sweeps"]).size().reset_index(name="soubory")
print("=== KATALOG: konfigurace mereni ===")
print(grp.to_string(index=False))

# ---------------- 2. SE VYPOCTY ----------------
results = {}

# --- VIRC data root: kryt s prutovou antenou, runs 01-03 ---
virc_root = os.path.join(ROOT, "VIRC data")
virc_se = {}
for run in ["01", "02", "03"]:
    for stav in ["klid", "shake"]:
        ref = os.path.join(virc_root, f"vna_{run}{stav}_bezkrytu_prutAntena.csv")
        sam = os.path.join(virc_root, f"vna_{run}{stav}_skrytem_prutAntena.csv")
        if os.path.exists(ref) and os.path.exists(sam):
            f, s = se(ref, sam)
            virc_se[f"run{run}_{stav}"] = (f, s)
results["virc_kryt"] = virc_se

# --- Velke BLS (VIRC): poloha x ABS x stav, Ref vs SampleA-E ---
bls_dir = os.path.join(ROOT, "Velké BLS")
bls_se = {}
for poloha in ["poloha1", "poloha2"]:
    for absf in ["bez ABS", "s ABS"]:
        d = os.path.join(bls_dir, poloha, absf)
        if not os.path.isdir(d):
            continue
        fs = os.listdir(d)
        for stav in ["static", "dynamic"]:
            refs = [x for x in fs if stav in x and ("Refe" in x or "refe" in x) and x.endswith(".csv")]
            if not refs:
                continue
            ref = os.path.join(d, refs[0])
            for samp in "ABCDE":
                cand = [x for x in fs if stav in x and f"Sample{samp}" in x and x.endswith(".csv")]
                if cand:
                    f, s = se(ref, os.path.join(d, cand[0]))
                    bls_se[(poloha, absf, stav, samp)] = (f, s)
results["bls_virc"] = bls_se

# --- GTEM Velke BLS: rotace 0-315, ref = slozka A ---
gtem_dir = os.path.join(ROOT, "VNA+GTEM_Velke_BLS")
def gtem_plin_avg(folder):
    """Prumer linearniho vykonu pres 8 rotaci."""
    fs = sorted([x for x in os.listdir(folder) if x.endswith(".csv")])
    ps = []
    f0 = None
    for x in fs:
        f, p = p_lin(os.path.join(folder, x))
        f0 = f
        ps.append(p)
    return f0, np.mean(ps, axis=0), np.array(ps)

gtem = {}
for s in sorted(os.listdir(gtem_dir)):
    d = os.path.join(gtem_dir, s)
    if os.path.isdir(d):
        gtem[s] = gtem_plin_avg(d)
fA, pA, _ = gtem["A"]
gtem_se = {}
for s, (f, p, prots) in gtem.items():
    if s == "A":
        continue
    gtem_se[s] = (f, 10 * np.log10(pA / p), 10 * np.log10(pA[None, :] / prots))
results["gtem_bls"] = gtem_se

# --- Little PCB GTEM: ref = A-bez krytu ---
pcb_dir = os.path.join(ROOT, "Little PCB_VNA_GTEM")
pcb = {}
for s in sorted(os.listdir(pcb_dir)):
    d = os.path.join(pcb_dir, s)
    if os.path.isdir(d):
        pcb[s] = gtem_plin_avg(d)
refk = [k for k in pcb if "bez krytu" in k][0]
fA2, pA2, _ = pcb[refk]
pcb_se = {}
for s, (f, p, prots) in pcb.items():
    if s == refk:
        continue
    pcb_se[s] = (f, 10 * np.log10(pA2 / p), 10 * np.log10(pA2[None, :] / prots))
results["pcb_gtem"] = pcb_se

# --- Box s otvory (VNA+VIRC): SE tvaru vs otevreny box ---
box_dir = os.path.join(ROOT, "Box_VNA+VIRC")
shapes = {d: os.path.join(box_dir, d) for d in sorted(os.listdir(box_dir)) if os.path.isdir(os.path.join(box_dir, d))}
def find_file(folder, stav, param="abs_S21"):
    cand = [x for x in os.listdir(folder) if x.endswith(f"{stav}_{param}.csv")]
    return os.path.join(folder, cand[0]) if cand else None
open_all = [v for k, v in shapes.items() if "všemi" in k or "vsemi" in k][0]
box_se = {}
for name, d in shapes.items():
    if d == open_all:
        continue
    for stav in ["static", "dynamic"]:
        rf = find_file(open_all, stav)
        sf = find_file(d, stav)
        if rf and sf:
            f, s = se(rf, sf)
            box_se[(name, stav)] = (f, s)
results["box"] = box_se

# --- utlum kabelu ---
kab = {}
for c in ["prvni_kabel", "druhy_kabel"]:
    p = os.path.join(ROOT, "utlum_kabelu", c, "S21.csv")
    if os.path.exists(p):
        f, d = load(p)
        kab[c] = (f, d.mean(axis=0))

# ---------------- 3. SOUHRNNE STATISTIKY ----------------
def band_stats(f, s_db):
    lin = 10 ** (s_db / 10.0)
    return dict(SE_mean_dB=float(np.mean(s_db)), SE_linavg_dB=float(10*np.log10(np.mean(lin))),
                SE_min_dB=float(np.min(s_db)), SE_max_dB=float(np.max(s_db)),
                f_min_MHz=float(f[0]/1e6), f_max_MHz=float(f[-1]/1e6))

rows = []
for key, (f, s) in virc_se.items():
    rows.append(dict(dataset="VIRC kryt+prutAntena", mereni=key, **band_stats(f, s)))
for (pol, absf, stav, samp), (f, s) in bls_se.items():
    rows.append(dict(dataset="Velke BLS (VIRC)", mereni=f"{pol}|{absf}|{stav}|Sample{samp}", **band_stats(f, s)))
for s_, (f, s, _) in gtem_se.items():
    rows.append(dict(dataset="Velke BLS (GTEM)", mereni=f"vzorek {s_} (ref A, prum. 8 rotaci)", **band_stats(f, s)))
for s_, (f, s, _) in pcb_se.items():
    rows.append(dict(dataset="Little PCB (GTEM)", mereni=f"kryt {s_} (ref bez krytu)", **band_stats(f, s)))
for (name, stav), (f, s) in box_se.items():
    rows.append(dict(dataset="Box otvory (VIRC)", mereni=f"{name}|{stav}", **band_stats(f, s)))
summ = pd.DataFrame(rows)
summ.to_csv(os.path.join(OUT, "SE_souhrn.csv"), index=False)
print("\n=== SE SOUHRN ===")
print(summ.to_string(index=False, float_format=lambda x: f"{x:8.2f}"))

# rotacni variabilita GTEM
print("\n=== GTEM: variabilita pres rotace (std pasmoveho prumeru SE, dB) ===")
for s_, (f, s, per_rot) in gtem_se.items():
    bandavg = per_rot.mean(axis=1)
    print(f"vzorek {s_}: prumery po rotacich = {np.round(bandavg,2)}  std = {bandavg.std():.2f} dB")

# utlum kabelu
print("\n=== Utlum kabelu (mean S21) ===")
for c, (f, d) in kab.items():
    print(f"{c}: {f[0]/1e6:.0f}-{f[-1]/1e6:.0f} MHz, prumer {d.mean():.2f} dB, min {d.min():.2f} dB")

# ---------------- 4. CROSS-TECHNIKA: VIRC vs GTEM (Velke BLS) ----------------
# hypoteza mapovani: GTEM B..F <-> VIRC SampleA..E (GTEM A = referencni prvek)
mapping = dict(B="A", C="B", D="C", E="D", F="E")
cross = []
for g, v in mapping.items():
    if g not in gtem_se:
        continue
    fg, sg, _ = gtem_se[g]
    virc_curves = [bls_se[k] for k in bls_se if k[3] == v and k[2] == "dynamic" and k[1] == "bez ABS"]
    if not virc_curves:
        continue
    # prumer pres polohy
    fv = virc_curves[0][0]
    sv = np.mean([np.interp(fv, fk, sk) for fk, sk in virc_curves], axis=0)
    flo, fhi = max(fg[0], fv[0]), min(fg[-1], fv[-1])
    fc = fv[(fv >= flo) & (fv <= fhi)]
    if len(fc) < 10:
        cross.append(dict(gtem=g, virc=v, poznamka="zadny frekvencni prekryv",
                          gtem_band=f"{fg[0]/1e6:.0f}-{fg[-1]/1e6:.0f} MHz",
                          virc_band=f"{fv[0]/1e6:.0f}-{fv[-1]/1e6:.0f} MHz"))
        continue
    si_g = np.interp(fc, fg, sg); si_v = np.interp(fc, fv, sv)
    cross.append(dict(gtem=g, virc=v, f_lo_MHz=flo/1e6, f_hi_MHz=fhi/1e6,
                      SE_gtem_mean=float(si_g.mean()), SE_virc_mean=float(si_v.mean()),
                      delta=float(si_g.mean() - si_v.mean()),
                      pearson=float(np.corrcoef(si_g, si_v)[0, 1])))
print("\n=== CROSS: GTEM vs VIRC (Velke BLS, dynamic, bez ABS) ===")
print(pd.DataFrame(cross).to_string(index=False))

# ulozeni SE krivek pro grafy
np.savez_compressed(os.path.join(OUT, "se_curves.npz"),
    **{f"virc_{k}": np.vstack([f, s]) for k, (f, s) in virc_se.items()},
    **{f"bls_{'_'.join(k)}": np.vstack([f, s]) for k, (f, s) in bls_se.items()},
    **{f"gtem_{k}": np.vstack([f, s]) for k, (f, s, _) in gtem_se.items()},
    **{f"pcb_{k}": np.vstack([f, s]) for k, (f, s, _) in pcb_se.items()},
    **{f"box_{k[0]}_{k[1]}": np.vstack([f, s]) for k, (f, s) in box_se.items()},
    **{f"kabel_{c}": np.vstack([f, d]) for c, (f, d) in kab.items()})
print("\nHotovo. Vystupy v:", OUT)
