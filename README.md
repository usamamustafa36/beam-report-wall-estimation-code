# Code for "Wall Estimation from Beam-Power Reports: Identifiability and Gaps to Information Bounds"

U. Mustafa and J. Iqbal, submitted to IEEE Transactions on Cognitive Communications and Networking.

Every number and figure in the paper is produced by the scripts here. Nothing is
typed in by hand. Paths such as `../paper/` refer to the paper's LaTeX folder, which sits next to this one.

## Environments

- Analysis: system `python3` (3.10) with numpy, scipy, matplotlib, torch 2.6 (CUDA optional).
- Ray tracing: `~/.venvs/sionna/bin/python` with sionna-rt 1.2.2 (Mitsuba 3.8, Dr.Jit 1.3).
- Step 7 imports the TCCN code from `../../usama/paper2026` (unchanged).

## Model

`rsrp_model.py` holds the closed-form model. It has a vertical wall n(φ)ᵀx = d, image-source
reflection, a 64-element half-wavelength ULA centred on the BS, and 16 wide beams from a
16-element sub-array (identical to `revision_common.wide_codebook` of the TCCN code). The wall
reflects as an ITU-R P.2040 slab of 0.1 m with the TE coefficient (as Sionna RT). The model also
covers:

- known paths h0;
- RSRP in dBm with Gaussian-in-dB error and optional 1 dB quantisation;
- Fisher information by forward-mode autodiff;
- the robust information (`fisher_robust`), which removes per-user position offsets and a
  wall-path phase as nuisances.

`scene_data.py` loads the traced facades.

## Order

| Step | Script | Output (`artifacts/`) |
|---|---|---|
| 1 | `step1_checks.py` | geometry, codebook, Fresnel, numerical ranks (`step1_checks.txt`) |
| 2 | `step2_checks.py` | every statement of Sec. III checked numerically (`step2_checks.txt`) |
| 3a | `step3_find_facades.py <scene> <bs>` (Sionna) | facades ranked by NLOS users |
| 3a | `step3_trace.py <GHz> <A\|B\|E>` (Sionna) | user pool (exact ray casts), h0, Sionna facade path |
| 3c | `step3_e2e_trace.py <GHz> <B\|E>` (Sionna) | same-run 64-element channels for e2e and mismatch |
| 3 | `step3_validate.py`, `step3_checks.py` | closed form vs Sionna; ordering of robust bounds |
| 3 | `step3_crb.py`, `step3_fig.py` | bounds (`step3_crb.txt`), `../paper/fig_crb.pdf` |
| 4 | `step4_mle.py` | MAP vs bound vs noise level, quantisation (`step4_mle.txt`) |
| 5 | `step5_detect.py`, `step5_fig.py` | stale-twin test, `../paper/fig_detect.pdf` |
| 6 | `step6_mismatch.py a\|b` | stale map prior; pseudo-true wall under full Sionna channels |
| 7 | `step7_predictor.py` | the TCCN attack as a choice of one complex number (`step7_predictor.json`) |
| 9 | `step9_material.py` | material classification (`step9_material.txt`) |
| 10 | `step10_learned.py` | learned DeepSets estimator vs MAP and bound (`step10_learned.json`) |

`run_all.sh` reruns steps 1–6, 9 and 10 from the traces (logs in `artifacts/logs/`). `resume_from5.sh` reruns only steps 5, 6b, 9 and 10 (use it if a run was interrupted after step 4). The traces
(3a, 3c) and step 7 are run separately:

```
for F in A B E; do for f in 3.5 28; do ~/.venvs/sionna/bin/python step3_trace.py $f $F; done; done
for F in B E; do ~/.venvs/sionna/bin/python step3_e2e_trace.py 3.5 $F; done
./run_all.sh
python3 step7_predictor.py
```

## Pitfalls found on the way

- Sionna 1.2's `paths.a` excludes the carrier phase. `paths.cfr()` adds it.
- Sionna models every wall as a 0.1 m slab. A half-space Fresnel model is off by 0.67 rad at 3.5 GHz.
- Sionna's detection of a particular path depends on ray count and receiver batch, so user pools
  come from exact ray casts.
- Robust bounds: remove only nuisance directions that are exactly uninformative. A relative
  cut-off produces false orderings. `step3_checks.py` guards this.
- MAP fits need a per-user grid start for the wall-path phase.
