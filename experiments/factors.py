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
    A_r1..A_r4      noise2  as A          0               shared: same-seed repeats of A (PLAN_V2 §5.2.7)
    C_r1..C_r4      noise2  as C          0.2             shared: same-seed repeats of C
    C_nc, D_nc      comp    as C/D        0.2             as C/D; drop_mode nc: survivors keep opacity x1 (§5.2.9)
    C_ca, D_ca      comp    as C/D        0.2             as C/D; drop_mode clamp_aware (§5.1)
    B_ds050, D_ds050  depth-dose  0.5     as B/D          shared (§5.2.6)
    B_ds200, D_ds200  depth-dose  2       as B/D          shared

drop_mode is "uniform" (DropGaussian verbatim) for every ID except the comp block. Runs made before the flag existed
(v1.0-exp) record no drop_mode; it counts as uniform.

γ is the --drop_rate flag. The rate in force at iteration t is γ·t/10000 (D1), so γ = 0.2 peaks at 0.12 at t = 6000.
Python 3.7 compatible and dependency-free.
"""
import re

LETTERS = {"A": (0.0, 0.0, "shared"), "B": (1.0, 0.0, "shared"), "C": (0.0, 0.2, "shared"),
           "D": (1.0, 0.2, "shared"), "E": (1.0, 0.2, "clean")}   # letter: (depth_scale, drop_rate, depth_render)
GAMMAS = {"g010": 0.1, "g033": 1.0 / 3}
DEPTH_DOSES = {"ds050": 0.5, "ds200": 2.0}
DROP_MODES = {"nc": "nc", "ca": "clamp_aware"}
PLANNED = (list(LETTERS) + ["D_hard", "D_soft"] + [x + g for g in ("_g010", "_g033") for x in "CDE"]
           + [x + "_r%d" % k for x in "BD" for k in range(1, 5)]
           + [x + "_r%d" % k for x in "AC" for k in range(1, 5)]
           + [x + m for m in ("_nc", "_ca") for x in "CD"]
           + [x + d for d in ("_ds050", "_ds200") for x in "BD"])
ID_RE = re.compile(r"^([A-E])(?:_(hard|soft|g010|g033|r[1-9]|nc|ca|ds050|ds200))?$")
TOL = 1e-4   # matrix_dose.yaml writes 1/3 as 0.333333


def factors(variant):
    """dict(variant, letter, depth_scale, drop_rate, depth_render, repeat, drop_mode) for a planned ID, else None."""
    m = ID_RE.match(str(variant))
    if not m:
        return None
    letter, mod = m.group(1), m.group(2)
    depth_scale, drop_rate, depth_render = LETTERS[letter]
    repeat, drop_mode = 0, "uniform"
    if mod in DROP_MODES:
        if letter not in "CD":
            return None
        drop_mode = DROP_MODES[mod]
    elif mod in DEPTH_DOSES:
        if letter not in "BD":
            return None
        depth_scale = DEPTH_DOSES[mod]
    elif mod in ("hard", "soft"):
        if letter != "D":
            return None
        depth_render = mod + "_only"
    elif mod in GAMMAS:
        if letter not in "CDE":
            return None
        drop_rate = GAMMAS[mod]
    elif mod:
        if letter not in "ABCD":
            return None
        repeat = int(mod[1:])
    return dict(variant=variant, letter=letter, depth_scale=depth_scale, drop_rate=drop_rate,
                depth_render=depth_render, repeat=repeat, drop_mode=drop_mode)


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
    for k in ("depth_render", "drop_mode"):
        v = recorded.get(k)
        if v is not None and v == v and v != f[k]:
            out.append("%s %s, but %s means %s" % (k, v, variant, f[k]))
    return out


if __name__ == "__main__":
    print("%-8s %-11s %-9s %-12s %-12s %s" % ("variant", "depth_scale", "drop_rate", "depth_render", "drop_mode",
                                             "repeat"))
    for v in PLANNED:
        f = factors(v)
        print("%-8s %-11g %-9.4g %-12s %-12s %s" % (v, f["depth_scale"], f["drop_rate"], f["depth_render"],
                                                   f["drop_mode"], f["repeat"] or ""))
