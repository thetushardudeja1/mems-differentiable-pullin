"""
Throughput benchmark: how many MEMS designs can we evaluate per second?

Measures the real workload (batched pull-in evaluation of the distributed beam
solver via vmap), not a toy. Reports compile time separately from run time,
because JIT compilation is a one-off cost and conflating them would flatter the
numbers.

Run as:
    JAX_PLATFORMS=cpu  python bench.py
    JAX_PLATFORMS=cuda python bench.py

NOTE ON PRECISION: the beam solver uses float64 because the physical scales
span ~1e-33 (E~ w^3 d0^3) to ~1e7 (h^-4). Consumer NVIDIA cards (RTX 4060)
run fp64 at ~1/64 of fp32 rate, so a GPU fp64 result may legitimately lose to
CPU. That is a real finding, not a bug, and is reported as such.
"""

import os
import time
import jax

jax.config.update("jax_enable_x64", __import__("os").environ.get("MEMS_X64","1")=="1")

import jax.numpy as jnp
from jax import vmap, jit
import numpy as np

import beam as B

ALPHA = 0.42 * 1e-6 / 100e-6
N = 40
BC = B.CANTILEVER
M = B._unknown_count(N, BC)
XI = B.node_xi(N, BC)


def make_designs(key, batch):
    """Random but physical gap profiles: d_max in [3,8] um, exponent in [0.8,2]."""
    k1, k2 = jax.random.split(key)
    d_max = jax.random.uniform(k1, (batch,), minval=3.0, maxval=8.0)
    n_exp = jax.random.uniform(k2, (batch,), minval=0.8, maxval=2.0)
    return vmap(lambda d, n: jnp.maximum(1.0, d * XI ** n))(d_max, n_exp)


def timeit(fn, *args, repeats=3):
    """Returns (compile_seconds, best_run_seconds)."""
    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    t_compile = time.perf_counter() - t0

    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        best = min(best, time.perf_counter() - t0)
    return t_compile, best


# ---- workload 1: full pull-in evaluation, cold (includes bracketing sweep) ----
@jit
def batched_pullin_cold(Ds):
    def one(D):
        lam, tr, res, _ = B.pullin_fold(D, alpha=ALPHA, N_t=0.0, N=N, bc=BC,
                                        n_coarse=15)
        return lam, tr, res
    return vmap(one)(Ds)


# ---- workload 2: fold solve only, warm-started (the inner loop of design /
#      uncertainty sweeps, where a good initial guess is already available) ----
@jit
def batched_fold_warm(Ds, z0s):
    def one(D, z0):
        lam, tr, res, _ = B.pullin_fold(D, alpha=ALPHA, N_t=0.0, N=N, bc=BC, z0=z0)
        return lam, tr, res
    return vmap(one)(Ds, z0s)


if __name__ == "__main__":
    dev = jax.devices()[0]
    print(f"device: {dev}   platform: {dev.platform}   x64: {jax.config.jax_enable_x64}")
    print(f"problem: {BC}, N={N} ({M} unknowns), fold system {2 * M + 1}x{2 * M + 1}\n")

    key = jax.random.PRNGKey(0)

    # correctness first: a batched result must match the known reference
    D_ref = jnp.ones((1, M))
    lam, tr, res = batched_pullin_cold(D_ref)
    err = 100 * abs(float(lam[0]) - B.LAMBDA_PI_REF[BC]) / B.LAMBDA_PI_REF[BC]
    print(f"sanity: batched uniform-gap Lambda={float(lam[0]):.5f} "
          f"(ref {B.LAMBDA_PI_REF[BC]:.5f}, err {err:.2f}%), |res|={float(res[0]):.1e}")
    assert err < 2.0 and float(res[0]) < 1e-6, "batched path disagrees with reference"

    print(f"\n{'batch':>7}{'compile s':>12}{'run s':>10}{'designs/s':>12}{'ok frac':>9}")
    print("-" * 50)
    results_cold = {}
    for batch in [1, 8, 64, 256, 1024]:
        key, sk = jax.random.split(key)
        Ds = make_designs(sk, batch)
        try:
            tc, tr_ = timeit(batched_pullin_cold, Ds)
            lam, trav, res = batched_pullin_cold(Ds)
            ok = float(jnp.mean(res < 1e-6))
            results_cold[batch] = batch / tr_
            print(f"{batch:>7}{tc:>12.2f}{tr_:>10.4f}{batch / tr_:>12.1f}{ok:>9.2f}")
        except Exception as e:
            print(f"{batch:>7}   FAILED: {type(e).__name__}: {str(e)[:60]}")
            break

    # warm-started throughput
    print(f"\nwarm-started fold solve (initial guess already available):")
    print(f"{'batch':>7}{'compile s':>12}{'run s':>10}{'designs/s':>12}{'ok frac':>9}")
    print("-" * 50)
    for batch in [1, 8, 64, 256, 1024]:
        key, sk = jax.random.split(key)
        Ds = make_designs(sk, batch)
        z0s = vmap(lambda D: B.fold_initial_guess(D, ALPHA, 0.0, N, BC, 15))(Ds)
        try:
            tc, tr_ = timeit(batched_fold_warm, Ds, z0s)
            lam, trav, res = batched_fold_warm(Ds, z0s)
            ok = float(jnp.mean(res < 1e-6))
            print(f"{batch:>7}{tc:>12.2f}{tr_:>10.4f}{batch / tr_:>12.1f}{ok:>9.2f}")
        except Exception as e:
            print(f"{batch:>7}   FAILED: {type(e).__name__}: {str(e)[:60]}")
            break

    # serial reference: what a naive (non-batched) implementation would cost
    print(f"\nserial reference (loop, no vmap), 32 designs:")
    key, sk = jax.random.split(key)
    Ds = make_designs(sk, 32)
    single = jit(lambda D: B.pullin_fold(D, alpha=ALPHA, N_t=0.0, N=N, bc=BC,
                                         n_coarse=15)[0])
    jax.block_until_ready(single(Ds[0]))          # warm the cache
    t0 = time.perf_counter()
    for i in range(32):
        jax.block_until_ready(single(Ds[i]))
    t_serial = time.perf_counter() - t0
    print(f"  {t_serial:.4f} s  ->  {32 / t_serial:.1f} designs/s")
    if 32 in results_cold or 64 in results_cold:
        b = 64 if 64 in results_cold else 32
        print(f"  vmap speedup at batch {b}: {results_cold[b] / (32 / t_serial):.1f}x")
