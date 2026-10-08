# Lepton Isolation and Signal vs. Background Classification using ATLAS Open Data (13 TeV)

My BSc Physics thesis, PMAS Arid Agriculture University, Rawalpindi (supervisor: **Dr. Asif Ilyas**).

Using the ATLAS Open Data 1-lepton dataset (√s = 13 TeV, 10 fb⁻¹), this analysis studies **W → μν** production: it examines the lepton isolation variables used to select prompt leptons, applies a full event selection, and trains a boosted decision tree to separate W+jets signal from the tt̄ and other backgrounds.

## What the analysis does

1. **Data handling** — downloads the ATLAS Open Data 1-lepton ntuples (with verified, atomic downloads and cleanup of corrupted files), reads them in chunks with `uproot`/`awkward-array`
2. **Event selection** — exactly one tight, isolated lepton (pT > 35 GeV, |η| < 2.5, track isolation pT^cone30/pT < 0.1, calorimeter isolation E_T^cone20/pT < 0.1), E_T^miss > 30 GeV, mT > 30 GeV, with full MC event weights (cross-section, luminosity, pile-up and lepton scale factors)
3. **Isolation study** — the track- and calorimeter-isolation distributions for data vs. stacked MC (W+jets, tt̄, Z+jets, single top, diboson)
4. **Kinematics** — lepton pT, E_T^miss, and the transverse-mass distribution with its Jacobian peak at the W mass
5. **Classification** — an XGBoost BDT (features: lepton pT, η, both isolation variables, E_T^miss, mT, jet multiplicity) separating W+jets signal from background, with a significance scan for the optimal BDT cut

## Headline result

- **BDT performance: AUC = 0.992** for W → μν signal vs. background (muon channel)
- Transverse-mass distribution shows the expected Jacobian peak near mW ≈ 80 GeV

## Figures (muon channel)

| Figure | Content |
|---|---|
| `figures/iso_track_mu.png` | Track isolation pT^cone30/pT — data vs. stacked MC |
| `figures/iso_calo_mu.png` | Calorimeter isolation E_T^cone20/pT — data vs. stacked MC |
| `figures/lep_pt_mu.png` | Lepton pT spectrum |
| `figures/met_mu.png` | Missing transverse energy |
| `figures/mt_mu.png` | Transverse mass mT with the W Jacobian peak |
| `figures/bdt_output_mu.png` | BDT output, signal vs. background (train/test) |
| `figures/roc_mu.png` | Background rejection vs. signal efficiency (AUC = 0.992) |

## Repository contents

- `wlnu_analysis.py` — the complete analysis script (Appendix A of the thesis). The channel is set by `CHANNEL` (`"mu"`, `"el"` or `"both"`); `QUICK_TEST` runs a reduced file set
- `thesis.pdf` — the full thesis document
- `figures/` — the result plots above

> **Note on the code:** the script is reproduced from Appendix A of the thesis. The appendix typesetting had flattened some of the original indentation; it has been restored against the printed thesis and the file verified to parse as valid Python. No logic was changed.

## Data

ATLAS Open Data is publicly available from the [ATLAS Open Data portal](https://opendata.atlas.cern) — the script downloads the 1-lepton ntuples itself from the CERN EOS and mirror URLs given in the code. The ROOT data files are not stored in this repository.

## Requirements

Python 3 with `uproot`, `awkward`, `numpy`, `matplotlib`, `xgboost`, `scikit-learn` — see `requirements.txt`.

## Author

Ubaid Ullah — [ORCID](https://orcid.org/0009-0002-9334-2204) · [LinkedIn](https://www.linkedin.com/in/ubaid-phy) · Related: [z-boson-reconstruction](https://github.com/Ubaid-Gran/z-boson-reconstruction) (NCP internship)
