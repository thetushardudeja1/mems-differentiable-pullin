"""
Throughput for ONE batch size, so a shell loop can sweep them in fresh
processes.

Why one per process: JAX does not promptly release device memory between
batch sizes, so sweeping them inside a single process made batch 1024 hard-
abort (an uncatchable CUDA OOM) even though 1024 runs fine on its own. Batch
16384 OOMs the 8 GB GPU outright, and 65536 crashed the WSL VM.

Usage:  python bench_one.py <batch>
Env:    JAX_PLATFORMS=cpu | cuda
"""

import sys
import time
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import vmap, jit
import beam as B

ALPHA = 0.42 * 1e-6 / 100e-6
N, BC = 40, B.CANTILEVER
M = B._unknown_count(N, BC)
XI = B.node_xi(N, BC)


def make_designs(key, batch):
    k1, k2 = jax.random.split(key)
    d_max = jax.random.uniform(k1, (batch,), minval=3.0, maxval=8.0)
    n_exp = jax.random.uniform(k2, (batch,), minval=0.8, maxval=2.0)
    return vmap(lambda d, n: jnp.maximum(1.0, d * XI ** n))(d_max, n_exp)


@jit
def warm_batch(Ds, z0s):
    def one(D, z0):
        lam, tr, res, _ = B.pullin_fold(D, alpha=ALPHA, N_t=0.0, N=N, bc=BC, z0=z0)
        return lam, res
    return vmap(one)(Ds, z0s)


if __name__ == "__main__":
    batch = int(sys.argv[1])
    dev = jax.devices()[0]
    key = jax.random.PRNGKey(0)
    Ds = make_designs(key, batch)
    z0s = vmap(lambda D: B.fold_initial_guess(D, ALPHA, 0.0, N, BC, 15))(Ds)

    t0 = time.perf_counter()
    out = warm_batch(Ds, z0s)
    jax.block_until_ready(out)
    t_compile = time.perf_counter() - t0

    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        out = warm_batch(Ds, z0s)
        jax.block_until_ready(out)
        best = min(best, time.perf_counter() - t0)

    lam, res = warm_batch(Ds, z0s)
    ok = float(jnp.mean(res < 1e-6))
    print(f"{dev.platform:>5}{batch:>8}{t_compile:>10.2f}{best:>10.4f}"
          f"{batch / best:>12.1f}{ok:>7.2f}", flush=True)
