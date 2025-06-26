import numpy as np
from pathlib import Path

def sample_customers(rng, m, pattern):
    if pattern == "cluster":
        k = rng.integers(1, 4)
        centers = rng.uniform(-8, 8, size=(k, 2))
        scales  = rng.uniform(0.5, 2.0, size=k)
        pts = []
        for c, s in zip(centers, scales):
            pts.append(rng.normal(c, s, size=(int(np.ceil(m/k)), 2)))
        return np.vstack(pts)[:m]
    if pattern == "ring":
        angles = rng.uniform(0, 2*np.pi, m)
        radii  = rng.uniform(4, 8, m)
        return np.column_stack([radii*np.cos(angles), radii*np.sin(angles)])
    if pattern == "bimodal":
        left  = rng.normal([-6, 0], 1.0, size=(m//2, 2))
        right = rng.normal([ 6, 0], 1.0, size=(m - m//2, 2))
        return np.vstack([left, right])
    raise ValueError(f"unknown pattern {pattern}")

def sample_instance(rng, phase):
    if phase == "warmup":
        m = rng.integers(3, 7)
        pattern = "cluster"
        phi_rng = (2.0, 2.2)
    elif phase == "mid":
        m = rng.integers(6, 13)
        pattern = rng.choice(["cluster", "ring", "bimodal"], p=[.6, .3, .1])
        phi_rng = (2.0, 2.4)
    else:  # hard
        m = rng.integers(10, 31)
        pattern = rng.choice(["cluster", "ring", "bimodal"], p=[.4, .3, .3])
        phi_rng = (2.2, 2.6)

    locs = sample_customers(rng, m, pattern).astype(np.float32)
    wts  = rng.lognormal(1.0, 0.4, m).clip(1, 8).astype(np.float32)

    params = {
        "capacity" : np.float32(rng.uniform(8, 12)),
        "max_range": np.float32(rng.uniform(35, 60)),
        "alpha"    : np.float32(rng.uniform(0.6, 0.8)),
        "beta"     : np.float32(0.9),
        "phi"      : np.float32(rng.uniform(*phi_rng)),
    }
    return locs, wts, params

def build_dataset(n_instances=5000,
                  split=(0.2, 0.4, 0.4),
                  rng_seed=42,
                  out_path="drone_dataset.npz"):
    rng = np.random.default_rng(rng_seed)
    phases = np.random.choice(
        ["warmup", "mid", "hard"], size=n_instances,
        p=split
    )

    all_xy, all_wt = [], []
    cap, rng_max, alp, bet, phi, seeds = [], [], [], [], [], []

    for i, ph in enumerate(phases):
        inst_seed = rng.integers(0, 2**32 - 1, dtype=np.uint32)
        inst_rng  = np.random.default_rng(inst_seed)

        locs, wts, p = sample_instance(inst_rng, ph)
        all_xy.append(locs)
        all_wt.append(wts)
        cap.append(p["capacity"])
        rng_max.append(p["max_range"])
        alp.append(p["alpha"])
        bet.append(p["beta"])
        phi.append(p["phi"])
        seeds.append(inst_seed)

    np.savez_compressed(
        out_path,
        cust_xy=np.array(all_xy, dtype=object),
        weights=np.array(all_wt, dtype=object),
        capacity=np.array(cap, dtype=np.float32),
        max_range=np.array(rng_max, dtype=np.float32),
        alpha=np.array(alp, dtype=np.float32),
        beta=np.array(bet, dtype=np.float32),
        phi=np.array(phi, dtype=np.float32),
        seed=np.array(seeds, dtype=np.uint32)
    )
    print(f"Saved {n_instances} instances to {Path(out_path).resolve()}")

if __name__ == "__main__":
    build_dataset(n_instances=10000)