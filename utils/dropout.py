"""Clamp-aware Gaussian dropout (PLAN_V2 §5 addendum, H7). No imports: works on torch tensors and numpy arrays.

DropGaussian keeps each Gaussian with probability 1 - r and scales the survivors' opacity by 1/(1 - r), which leaves
the expected render unchanged (Proposition 1). The rasterizer clamps per-pixel alpha at ALPHA_MAX = 0.99
(forward.cu), so once opacity/(1 - r) > ALPHA_MAX the survivors are cut and the expected alpha falls to
(1 - r) * ALPHA_MAX < opacity. In DNGaussian's hard-depth pass every Gaussian has opacity 0.95, so this happens for
all of them once r > 0.0404; in the other passes it hits the most opaque Gaussians.

Clamp-aware dropout lowers each Gaussian's rate to r_i = min(r, 1 - opacity_i / ALPHA_MAX), or 0 when
opacity_i >= ALPHA_MAX. Then opacity_i / (1 - r_i) <= ALPHA_MAX, the clamp never cuts a survivor (the 2D falloff only
lowers alpha away from the centre), and the expected alpha equals the clean alpha in every pass. All passes of one
iteration threshold the same uniform draw u, so a Gaussian kept at a high rate is also kept at a lower one.
"""

ALPHA_MAX = 0.99   # per-pixel alpha cap of the rasterizer (submodules/diff-gaussian-rasterization, forward.cu)


def clamp_aware_rates(opacity, r):
    """Per-Gaussian drop rate min(r, 1 - opacity / ALPHA_MAX), floored at 0."""
    cap = 1.0 - opacity / ALPHA_MAX
    rate = cap * (cap < r) + r * (cap >= r)
    return rate * (rate > 0)


def compensated_mask(u, rate):
    """Keep where u >= rate (u uniform in [0, 1)); survivors are scaled by 1 / (1 - rate)."""
    return (u >= rate) / (1.0 - rate)
