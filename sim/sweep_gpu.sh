#!/usr/bin/env bash
# GPU throughput sweep, one process per batch size.
#
# IMPORTANT: do NOT set XLA_PYTHON_CLIENT_PREALLOCATE=false here. That is a
# DEBUGGING flag; it disables JAX's memory pool and causes fragmentation, which
# is what made batch 1024 hard-abort earlier. With preallocation left on (and
# the fraction tuned) the same batch runs fine at 1276 designs/s.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mems
cd "/mnt/c/Users/TUSHAR/TUSHAR DUDEJA/MEMS/sim" || exit 1

export TF_CPP_MIN_LOG_LEVEL=3
export XLA_PYTHON_CLIENT_MEM_FRACTION=.92

printf '%5s %7s %9s %8s %11s %5s\n' plat batch compile run designs/s ok
for b in "$@"; do
    if ! timeout 900 python -u bench_one.py "$b" 2>/dev/null; then
        printf '%5s %7s   FAILED (OOM or timeout)\n' gpu "$b"
    fi
done
