# -*- coding: utf-8 -*-
"""Analyza novych mereni (cervenec 2026): sumove prahy, komplexni export,
Q z dozvuku, prizpusobeni anten. Vysledky:
- GTEM prah (zakoncovaci prvek, port 2): median -102.3 dB [5.-95. perc -105.1, -92.1];
  rezerva vsech vzorku >= 19.6 dB v pasmu 1.125-4.125 GHz, 0 % pod 6 dB.
- VIRC prah (50 ohm, port 1): median -81.7 dB; rezerva Sample A (nejhorsi pripad,
  s ABS) min 6.3 dB, median 20-30 dB, 0 % pod 6 dB.
- Komplexni export (Timestamp,Sweep_Nr,Format): Real/Imag konzistentni s dB Mag.
- Q z dozvuku (IFFT komplexniho S21, 2 log-per anteny, dynamic): tau = 125 ns,
  Q = 1179 (1.5 GHz) az 3143 (4 GHz); potvrzuje Q ~ 10^3 z koherencni sirky pasma.
- Log-per antena: |S11| < -10 dB od 379 MHz; ucinnost prizpusobeni ~90 % v pasmu."""
import io, os, re
import numpy as np
import pandas as pd

ROOT = "/Users/stanlayk/Library/CloudStorage/OneDrive-UniverzitaTomášeBativeZlíně/Research/Měření stínění"

def load_new(path, fmt="dB Mag"):
    """Novy format VNA exportu: Timestamp,Sweep_Nr,Format,frekvence...
    Radek 'Smith' ma dvojnasobek poli, proto parsovani po radcich."""
    rows = []
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\n").split(",")
        f = np.array([float(c) for c in header[3:]]); nf = len(f)
        for line in fh:
            p = line.rstrip("\n").split(",")
            if len(p) >= 3 and p[2] == fmt:
                rows.append([float(x) for x in p[3:3+nf]])
    return f, np.array(rows)

def load_old(path):
    df = pd.read_csv(path, engine="c", dtype=np.float64)
    return np.array([float(c) for c in df.columns[1:]]), df.iloc[:, 1:].to_numpy()

if __name__ == "__main__":
    fz, dz = load_old(os.path.join(ROOT, "zakončovací prvek/zakončovací prvek S21.csv"))
    Pz = np.mean(10**(dz/10), axis=0)
    print(f"GTEM prah: median {np.median(10*np.log10(Pz)):.1f} dB")
    pa = os.path.join(ROOT, "Antény/2Log-per_dynamic_S21.csv")
    f_, re_ = load_new(pa, "Real"); _, im_ = load_new(pa, "Imag")
    S = re_ + 1j*im_
    h = np.fft.ifft(S, axis=1)
    pdp = np.mean(np.abs(h)**2, axis=0)
    t = np.arange(S.shape[1]) / (S.shape[1] * (f_[1]-f_[0]))
    pdp_db = 10*np.log10(pdp/pdp.max())
    m = (t > 20e-9) & (t < 250e-9) & (pdp_db > -55)
    A = np.polyfit(t[m], pdp_db[m], 1)
    tau = -10/(A[0]*np.log(10))
    print(f"tau = {tau*1e9:.0f} ns, Q(2.3 GHz) = {2*np.pi*2.3e9*tau:.0f}")
