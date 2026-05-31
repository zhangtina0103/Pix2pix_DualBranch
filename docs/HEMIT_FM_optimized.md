# Deprecated: use `docs/HEMIT_cond_beating_pix2pix.md`

The flagship story is:

- **Baseline:** `joint_perc` @ 80 (vanilla FM, 0.831)
- **Method:** consistent conditional FM (`hemit_cond_fm_consistent`)
- **Target:** beat pix2pix ~0.854

`train_hemit_vanilla_fm_joint_opt.sbatch` is legacy; prefer `train_hemit_cond_fm_consistent.sbatch`.
