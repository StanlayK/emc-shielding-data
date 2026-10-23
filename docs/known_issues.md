# Known data quirks

- `campaigns/gtem_bls/C/` was exported with files named `B_*` — a naming
  mishap in the VNA export, not a sample mix-up: byte-level comparison with
  `B/` confirms all eight traces are distinct, genuine sample-C data.
  **Corrected from dataset release v1.0.1 onward**, where the published copy
  carries `C_VNA_*.csv`; the raw laboratory archive retains the original
  names. Release v1.0.0 (the version first deposited) still shows `B_*`.
- `campaigns/gtem_bls/F/` holds two separate acquisition runs of the same
  specimen. The rotation run (`F_VNA_<angle>.csv`) covers seven angles —
  45°, 90°, 135°, 180°, 225°, 270°, 315° — and **lacks 0°**; the
  S-parameter run (`F_VNA_<angle>_S*.csv`) covers 0°, 45°, 90°, 135°, 180°,
  225° and lacks 270° and 315°. The two runs are NOT interchangeable: at
  the five angles present in both, their mean levels differ by about 2 dB
  and locally by up to 32 dB. Analyses therefore use the rotation run only,
  i.e. seven orientations for variant F against eight for A--E. Dropping
  0° from the other items as well shifts their band-averaged differential
  SE by +0.24 to +0.67 dB (see `scripts/revize_nejistoty.py`).
- `campaigns/gtem_little_pcb/` has no sample C (specimen no longer exists).
- VIRC "static" records are quasi-static: the flexible walls relax slowly
  after agitation stops (inter-sweep std ≈ 4 dB, spectra of adjacent
  sweeps correlated 0.88–0.99). Use ensemble means, not single sweeps.
- Legacy exports store scalar dB only; phase is unavailable before
  July 2026 (extended format).
