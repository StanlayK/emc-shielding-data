# Analysis scripts

| Script | Produces |
|--------|----------|
| `analyza_stineni.py`          | data catalog, dSE curves, summary CSV |
| `grafy_stineni.py`            | dSE curve figures, cross-technique plot |
| `verifikace_virc.py`          | VIRC verification (K-factor, KS test, uniformity) |
| `pokrocile_metody.py`         | gauge R&R, Bland-Altman/Deming, FPCA, GP fusion |
| `blokovy_bootstrap_deming.py` | moving-block bootstrap CIs for Deming regression |
| `analyza_novych_dat.py`       | noise floors, reverberation time, antenna matching; parser for the extended CSV format |
| `fig_clanek.py`               | single-column vector figures of the article |

Python >= 3.9; numpy, pandas, scipy, matplotlib. Adjust the `ROOT`
constant to the repository root before running.
