"""
Does the GPU win at LARGE batch? (The earlier benchmark stopped at 1024.)

Earlier finding: at batch 1024 the CPU beat the GPU on the fp64 beam solver
(1430 vs 1276 designs/s), because consumer NVIDIA cards run fp64 at ~1/64 of
fp32. But 1024 designs may simply be too few to saturate the device -- the
fp64 penalty is a throughput ratio, and a GPU only reaches its throughput once
enough work is in flight. So push the batch until memory runs out and see
whether the curves cross.

Workload: warm-started fold solves, which is the realistic inner loop for
design sweeps and fabrication-variance studies (thousands of geometries, good
initial guesses available).

Run:
    JAX_PLATFORMS=cpu  python bench_large.py
    JAX_PLATFORMS=cuda python bench_large.py
"""

import time
import gc
import sys
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import vmap, jit
import beam as B

ALPHA = 0.42 * 1e-6 / 100e-6
N = 40
BC = B.CANTILEVER
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
        lam, tr, res, _ = B.pullin_fold(D, alpha=ALPHA, N_t=0.0, N=N, bc=BC,
                                        z0=z0)
        return lam, res
    return vmap(one)(Ds, z0s)


if __name__ == "__main__":
    dev = jax.devices()[0]
    print(f"device: {dev}  platform: {dev.platform}  x64: {jax.config.jax_enable_x64}")
    print(f"problem: {BC} N={N} ({M} unknowns), fold system {2*M+1}x{2*M+1}\n")

    key = jax.random.PRNGKey(0)
    print(f"{'batch':>8}{'compile s':>11}{'run s':>10}{'designs/s':>12}"
          f"{'ok':>7}{'MB/design':>11}")
    print("-" * 60)
    best = (0, 0.0)
    for batch in [256, 1024, 2048, 4096]:   # 16384+ hard-OOMs the 8GB GPU (uncatchable abort);
                                            # 65536 crashed the WSL VM itself
        key, sk = jax.random.split(key)
        try:
            Ds = make_designs(sk, batch)
            z0s = vmap(lambda D: B.fold_initial_guess(D, ALPHA, 0.0, N, BC, 15))(Ds)
            t0 = time.perf_counter()
            out = warm_batch(Ds, z0s)
            jax.block_until_ready(out)
            t_c = time.perf_counter() - t0

            best_t = float("inf")
            for _ in range(3):
                t0 = time.perf_counter()
                out = warm_batch(Ds, z0s)
                jax.block_until_ready(out)
                best_t = min(best_t, time.perf_counter() - t0)

            lam, res = warm_batch(Ds, z0s)
            ok = float(jnp.mean(res < 1e-6))
            rate = batch / best_t
            # rough working-set estimate: the (2M+1)^2 fold Jacobian in fp64
            mb = (2 * M + 1) ** 2 * 8 / 1e6
            print(f"{batch:>8}{t_c:>11.2f}{best_t:>10.4f}{rate:>12.1f}"
                  f"{ok:>7.2f}{mb:>11.3f}", flush=True)
            if rate > best[1]:
                best = (batch, rate)
            del Ds, z0s, out, lam, res
            gc.collect()
        except Exception as e:
            print(f"{batch:>8}   STOPPED: {type(e).__name__}: {str(e)[:56]}")
            break

    print(f"\npeak: {best[1]:,.0f} designs/s at batch {best[0]}")
    print(f"      = {best[1]*60:,.0f} designs/min, "
          f"{best[1]*3600:,.0f} designs/hour")
