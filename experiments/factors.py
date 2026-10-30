#!/usr/bin/env python
"""Factor table: what every variant ID means (PLAN_V2 §3.1). stats.py and plots/diag_curves.py read it from here.

    python experiments/factors.py          # prints the table for every planned ID

    ID              block   depth_scale   drop_rate (γ)   depth_render
    A, B, C, D, E   T1, V   0/1           0 or 0.2        shared (E: clean)
    D_hard, D_soft  P       1             0.2             hard_only / soft_only: the mask reaches one depth pass
    C/D/E_g010      G       as C/D/E      0.1             as C/D/E
    C/D/E_g033      G       as C/D/E      1/3             as C/D/E
    B_r1..B_r4      N       as B          0               shared: same-seed repeats of B (noise floor)
    D_r1..D_r4      N       as D          0.2             shared: same-seed repeats of D

γ is the --drop_rate flag. The rate in force at iteration t is γ·t/10000 (D1), so γ = 0.2 peaks at 0.12 at t = 6000.
Python 3.7 compatible and dependency-free.
"""
import re

LETTERS = {"A": (0.0, 0.0, "shared"), "B": (1.0, 0.0, "shared"), "C": (0.0, 0.2, "shared"),
           "D": (1.0, 0.2, "shared"), "E": (1.0, 0.2, "clean")}   # letter: (depth_scale, drop_rate, depth_render)
GAMMAS = {"g010": 0.1, "g033": 1.0 / 3}
PLANNED = (list(LETTERS) + ["D_hard", "D_soft"] + [x + g for g in ("_g010", "_g033") for x in "CDE"]
           + [x + "_r%d" % k for x in "BD" for k in range(1, 5)])
ID_RE = re.compile(r"^([A-E])(?:_(hard|soft|g010|g033|r[1-9]))?$")
TOL = 1e-4   # matrix_dose.yaml writes 1/3 as 0.333333


def factors(variant):
    """dict(variant, letter, depth_scale, drop_rate, depth_render, repeat) for a planned ID, else None."""
    m = ID_RE.match(str(variant))
    if not m:
        return None
    letter, mod = m.group(1), m.group(2)
    depth_scale, drop_rate, depth_render = LETTERS[letter]
    repeat = 0
    if mod in ("hard", "soft"):
        if letter != "D":
            return None
        depth_render = mod + "_only"
    elif mod in GAMMAS:
        if letter not in "CDE":
            return None
        drop_rate = GAMMAS[mod]
    elif mod:
        if letter not in "BD":
            return None
        repeat = int(mod[1:])
    return dict(variant=variant, letter=letter, depth_scale=depth_scale, drop_rate=drop_rate,
                depth_render=depth_render, repeat=repeat)


def mask_reaches_depth(f):
    """True if the training mask reaches at least one depth pass, so the diag.csv gap can be non-zero."""
    return f["drop_rate"] > 0 and f["depth_render"] in ("shared", "hard_only", "soft_only")


def mismatches(variant, recorded):
    """Factors where a run's recorded flags (config.json or metrics.json) contradict its ID; [] if they agree.
    Keys that are absent or None are not checked (runs made before the flags existed)."""
    f = factors(variant)
    if f is None:
        return ["%s is not a planned variant ID" % variant]
    out = []
    for k in ("depth_scale", "drop_rate"):
        v = recorded.get(k)
        if v is not None and v == v and abs(float(v) - f[k]) > TOL:   # v == v skips NaN
            out.append("%s %s, but %s means %s" % (k, v, variant, round(f[k], 6)))
    v = recorded.get("depth_render")
    if v is not None and v == v and v != f["depth_render"]:
        out.append("depth_render %s, but %s means %s" % (v, variant, f["depth_render"]))
    return out


if __name__ == "__main__":
    print("%-8s %-11s %-9s %-12s %s" % ("variant", "depth_scale", "drop_rate", "depth_render", "repeat"))
    for v in PLANNED:
        f = factors(v)
        print("%-8s %-11g %-9.4g %-12s %s" % (v, f["depth_scale"], f["drop_rate"], f["depth_render"], f["repeat"] or ""))
