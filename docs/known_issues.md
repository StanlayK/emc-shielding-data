# Known data quirks

- `campaigns/gtem_bls/C/` contains files named `B_*` — an export naming
  mishap; checksums confirm the data are distinct (genuine sample C).
- `campaigns/gtem_bls/F/` lacks the 0° rotation in the plain trace set and
  mixes two export types (rotation files and full S-parameter files).
- `campaigns/gtem_little_pcb/` has no sample C (specimen no longer exists).
- VIRC "static" records are quasi-static: the flexible walls relax slowly
  after agitation stops (inter-sweep std ≈ 4 dB, spectra of adjacent
  sweeps correlated 0.88–0.99). Use ensemble means, not single sweeps.
- Legacy exports store scalar dB only; phase is unavailable before
  July 2026 (extended format).
