#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------
# Author: Parham Radpay
# --------------------
"""
Standalone antisymmetric injective norm experiment.

- Generates n_samples random antisymmetric Gaussian tensors of order p and dim d.
- Approximates the injective norm via gradient ascent on (S^{d-1})^p.
- Stores both:
    * inj_raw      ≈ ||T||_inj
    * inj_over_l2  ≈ ||T||_inj / ||T||_2
- Writes results to a JSON file.
"""

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import itertools


# ---------- Antisymmetric tensor utilities ----------


def antisymmetrize_tensor(T: torch.Tensor) -> torch.Tensor:
    """
    Antisymmetrize a full tensor T over all its indices.

    T: tensor of shape (d,)*p
    Returns A where
        A_{i1...ip} = (1/p!) sum_{\pi in S_p} sign(\pi) T_{i_{\pi(1)}...i_{\pi(p)}}

    This is fine for small p (3 or 4).
    """

    p = T.ndim
    d = T.shape[0]
    assert all(s == d for s in T.shape), "T must be dx...xd (same dim on all modes)."

    A = torch.zeros_like(T)

    for perm in itertools.permutations(range(p)):
        # permutation signature
        inv = 0
        for i in range(p):
            for j in range(i + 1, p):
                if perm[i] > perm[j]:
                    inv += 1
        sign = -1 if (inv % 2) else 1

        A_perm = T.permute(*perm)
        A = A + sign * A_perm

    A = A / math.factorial(p)
    return A


def random_gaussian_tensor(
    d: int,
    p: int,
    complex_valued: bool = False,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    Generate a full Gaussian tensor of shape (d,)*p and antisymmetrize it.

    Entries are N(0, sigma^2) with sigma = sqrt(2/d).
    For complex, real and imag are N(0, sigma^2/2).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shape = (d,) * p
    std = math.sqrt(2.0 / d)

    if not complex_valued:
        T_full = torch.normal(mean=0.0, std=std, size=shape, device=device)
    else:
        real = torch.normal(
            mean=0.0, std=std / math.sqrt(2.0), size=shape, device=device
        )
        imag = torch.normal(
            mean=0.0, std=std / math.sqrt(2.0), size=shape, device=device
        )
        T_full = real + 1j * imag

    A = antisymmetrize_tensor(T_full)
    return A


# ---------- Injective norm via gradient ascent ----------


def approx_injective_norm(
    T: torch.Tensor,
    n_steps: int = 300,
    lr: float = 0.05,
    verbose: bool = False,
) -> dict:
    """
    Approximate the injective norm of an antisymmetric tensor T.

    We solve:
        max_{||x_k||=1} | <T, x_1 \otimes ... \otimes x_p> |

    using PyTorch gradient ascent on (S^{d-1})^p.

    Returns:
        {
          "inj_raw":      ≈ ||T||_inj,
          "inj_over_l2":  ≈ ||T||_inj / ||T||_2,
          "l2_norm":      ||T||_2,
        }
    """
    device = T.device
    p = T.ndim
    d = T.shape[0]

    # L2 norm of T
    l2_norm = torch.norm(T).item()

    # Parameters: x_k in R^d or C^d
    is_complex = torch.is_complex(T)
    dtype = torch.cfloat if is_complex else torch.float

    xs = [
        torch.randn(d, dtype=dtype, device=device, requires_grad=True) for _ in range(p)
    ]

    opt = torch.optim.Adam(xs, lr=lr)

    for step in range(n_steps):
        opt.zero_grad()

        # Normalize each factor to unit norm
        us = [v / v.norm() for v in xs]

        # Successive contractions to compute F = <T, x_1 \otimes... \otimes x_p>
        tmp = T
        for u in us:
            tmp = torch.tensordot(tmp, u, dims=([0], [0]))
        F = tmp  # scalar (real or complex)

        # Maximize |F|
        loss = -F.abs()
        loss.backward()
        opt.step()

        if verbose and (step % 50 == 0 or step == n_steps - 1):
            print(f"  step {step:4d}, |F| ≈ {F.abs().item():.6f}")

    # Recompute final overlap with normalized factors
    us = [v / v.norm() for v in xs]
    tmp = T
    for u in us:
        tmp = torch.tensordot(tmp, u, dims=([0], [0]))
    F_final = tmp
    inj_raw = F_final.abs().item()
    inj_over_l2 = inj_raw / l2_norm if l2_norm > 0 else float("nan")

    return {
        "inj_raw": inj_raw,
        "inj_over_l2": inj_over_l2,
        "l2_norm": l2_norm,
    }


# ---------- Sampling wrapper with progress bar + JSON ----------


def sample_antisym_injective_norms(
    p: int,
    d: int,
    n_samples: int,
    complex_valued: bool = False,
    n_steps: int = 300,
    lr: float = 0.05,
    json_path: str | Path | None = None,
    verbose: bool = False,
) -> dict:
    """
    Generate n_samples random antisymmetric Gaussian tensors of order p, dim d,
    approximate their injective norms, and save results to JSON.

    JSON structure:
      {
        "p": p,
        "d": d,
        "n_samples": n_samples,
        "complex_valued": bool,
        "n_steps": n_steps,
        "lr": lr,
        "timestamp": "...",
        "inj_raw": [...],
        "inj_over_l2": [...],
        "l2_norms": [...]
      }
    """
    if json_path is None:
        json_path = f"antisym_p{p}_d{d}_n{n_samples}_complex{complex_valued}.json"
    json_path = Path(json_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Running antisymmetric injective norm experiment:"
        f" p={p}, d={d}, n_samples={n_samples}, complex={complex_valued},"
        f" n_steps={n_steps}, lr={lr}, device={device}"
    )

    inj_raw_list = []
    inj_over_l2_list = []
    l2_list = []

    for _ in tqdm(range(n_samples), desc="Samples"):
        # 1) Antisymmetric Gaussian tensor
        T = random_gaussian_tensor(d, p, complex_valued=complex_valued, device=device)

        # 2) Approximate injective norm
        res = approx_injective_norm(T, n_steps=n_steps, lr=lr, verbose=verbose)

        inj_raw_list.append(res["inj_raw"])
        inj_over_l2_list.append(res["inj_over_l2"])
        l2_list.append(res["l2_norm"])

        del T
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results = {
        "p": int(p),
        "d": int(d),
        "n_samples": int(n_samples),
        "complex_valued": bool(complex_valued),
        "n_steps": int(n_steps),
        "lr": float(lr),
        "timestamp": datetime.now().isoformat(),
        "inj_raw": inj_raw_list,
        "inj_over_l2": inj_over_l2_list,
        "l2_norms": l2_list,
    }

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {json_path}")
    print(f"Mean inj_raw      ≈ {np.mean(inj_raw_list):.6f}")
    print(f"Mean inj_over_l2  ≈ {np.mean(inj_over_l2_list):.6f}")

    return results


# ---------- CLI entry point ----------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Standalone antisymmetric injective norm experiment."
    )
    parser.add_argument("--p", type=int, required=True, help="Order of the tensor.")
    parser.add_argument("--d", type=int, required=True, help="Dimension.")
    parser.add_argument(
        "--n-samples", type=int, required=True, help="Number of random samples."
    )
    parser.add_argument(
        "--complex",
        action="store_true",
        help="Use complex Gaussian entries (default: real).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=300,
        help="Number of gradient steps per sample (default: 300).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="Learning rate for gradient ascent (default: 0.05).",
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default=None,
        help="Output JSON path (optional).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-step info inside each optimization.",
    )

    args = parser.parse_args()

    sample_antisym_injective_norms(
        p=args.p,
        d=args.d,
        n_samples=args.n_samples,
        complex_valued=args.complex,
        n_steps=args.n_steps,
        lr=args.lr,
        json_path=args.json_path,
        verbose=args.verbose,
    )
