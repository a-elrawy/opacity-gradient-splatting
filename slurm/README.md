# SLURM / Compute Canada jobs

Portable cluster templates. Machine-specific paths stay in a **local** `env.sh` (gitignored).

## Setup

```bash
cp slurm/env.sh.example slurm/env.sh
# edit OGS_ROOT, OGS_VENV, OGS_ACCOUNT
```

## Submit

```bash
sbatch --account=$OGS_ACCOUNT slurm/run_llff_default.sh
```

Each job sources `common.sh`, which loads Alliance modules when available, activates `OGS_VENV`, and `cd`s to `OGS_ROOT`.

## Adding jobs

Copy `run_llff_default.sh` or `run_mipnerf_default.sh`, keep `#SBATCH` headers generic (`REPLACE_ME` or override with `sbatch --account=...`), and call `source "$(dirname "$0")/common.sh"` before `python train.py ...`.

Do not commit personal `$HOME` / `$SCRATCH` paths or PI account strings.
