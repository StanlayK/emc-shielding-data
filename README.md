# EMC Shielding Measurements — Tomas Bata University in Zlin

Living data repository of the EMC shielding research group (Faculty of
Applied Informatics, TBU in Zlin). It collects raw measurement campaigns,
analysis scripts, and derived reference results for electromagnetic
shielding characterization in the VIRC (vibrating intrinsic reverberation
chamber) and the GTEM cell.

**Data license:** CC BY 4.0 · **Code license:** MIT ·
**Cite via:** `CITATION.cff` (Zenodo DOI of the release you used)

## Repository map

| Folder | Campaign | Environment | Content |
|---|---|---|---|
| `campaigns/virc_bls/` | Large BLS samples A–E + reference | VIRC | 2 antenna positions × ±absorber × static/dynamic, 100 sweeps |
| `campaigns/gtem_bls/` | Large BLS samples A–F | GTEM | 8 rotations (45° steps), 100 sweeps |
| `campaigns/gtem_little_pcb/` | Small PCB shields (ref. = A, no C) | GTEM | 8 rotations, 100 sweeps |
| `campaigns/virc_box_apertures/` | 30 cm cube, aperture shapes | VIRC | one aperture open at a time, full S-parameters |
| `campaigns/virc_characterization/` | Chamber itself | VIRC | rod-antenna runs, antenna positions, polarization, large box |
| `campaigns/cables/` | Both measurement cables | — | full S-parameters |
| `campaigns/noise_floors/` | Terminated-port records | GTEM + VIRC | dynamic-range budget |
| `campaigns/antennas/` | Log-periodic antennas | VIRC | matching, two-antenna transfer (reverberation time) |
| `campaigns/chamber_loading/` | Loads / two antennas | VIRC | Q and ACS studies (complex traces) |
| `scripts/` | — | — | Python analysis (see `scripts/README.md`) |
| `derived/` | — | — | technique-independent SE curves, summaries, figures |
| `docs/` | — | — | measurement protocol, sample design, file formats |

## Key facts

- **Samples (BLS study):** panel of six PCBs (FR-4, Pulse W3006 chip
  antenna at 2.4 GHz, SMA feed) with a Laird BMI-S-102 two-piece shield.
  Variants: A correct, B no ground pour under antenna, C perimeter
  clearance (grounded), D perimeter clearance (not grounded), E pads not
  grounded, F no shield (reference). Design files: `docs/`.
- **Instrument:** Rohde & Schwarz ZND (two-port, 100 kHz–4.5 GHz),
  source +3 dBm, IF bandwidth 10 kHz, calibrated, remote via VXI-11.
- **Chambers:** VIRC 1×1×1 m (flexible metallized fabric, motor-driven
  wall agitation); GTEM cell (Frankonia).

## File formats

Two CSV trace formats occur (columns after the metadata are frequencies
in Hz):

1. **Legacy:** header `Sweep_Nr,<f1>,...`; one row per sweep (100 sweeps),
   |S21| in dB.
2. **Extended (from July 2026):** header `Timestamp,Sweep_Nr,Format,<f1>,...`;
   seven rows per sweep, `Format` ∈ {dB Mag, Phase, Lin Mag, Real, Imag,
   SWR, Smith}. The `Smith` row carries two values per frequency — parse
   line-wise (see `scripts/analyza_novych_dat.py`).

Known data quirks are documented in `docs/known_issues.md` (e.g., GTEM
folder `C/` contains files named `B_*` that are genuine sample-C data).

## Reproducing the article results

Dataset release v1.x accompanies the article *"Measurement-technique-
independent characterization of board-level shielding using differential
shielding effectiveness against a common reference artifact"* (Kovar,
Strajtova, Adamek, Dostalek). The script-to-figure map is in
`scripts/README.md`; scripts run with Python ≥ 3.9 (numpy, pandas, scipy,
matplotlib) and expect the layout above.

## Contributing (team)

New campaign = new folder under `campaigns/` + a row in the table above +
a short note in `docs/` (setup, instrument settings, operator, date).
Use the extended CSV export and log `*IDN?`, IF bandwidth, and source
power into the file header. Commit, push, and create a release when a
campaign is final — Zenodo archives every release automatically and
assigns a citable version DOI.
