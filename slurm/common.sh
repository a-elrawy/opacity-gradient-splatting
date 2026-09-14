#!/bin/bash
# Common preamble for SLURM jobs. Source from each job script:
#   source "$(dirname "$0")/common.sh"
set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SLURM_DIR}/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${SLURM_DIR}/env.sh"
else
  echo "ERROR: ${SLURM_DIR}/env.sh not found." >&2
  echo "Copy slurm/env.sh.example to slurm/env.sh and set OGS_ROOT / OGS_VENV / OGS_ACCOUNT." >&2
  exit 1
fi

: "${OGS_ROOT:?Set OGS_ROOT in slurm/env.sh}"
: "${OGS_VENV:?Set OGS_VENV in slurm/env.sh}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# Alliance modules (harmless no-op if module is unavailable locally)
if command -v module >/dev/null 2>&1; then
  module load cuda cudnn gcc python/3.10 opencv/4.10.0 2>/dev/null || true
fi

export PATH="${OGS_VENV}/bin:${PATH}"
cd "${OGS_ROOT}"
