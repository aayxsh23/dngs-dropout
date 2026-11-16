#!/usr/bin/env python
"""N1 error budget (PREREG_V3 §4): split each variant's pixel error into squared bias and run-to-run variance.

    python experiments/bias_variance.py --out_root <block root> --dataset llff --n_views 3 \\
        --variants A B C D E --seeds 0 1 2 --tables results/tables/error_budget
    python experiments/bias_variance.py --out_root <dtu root> --dataset dtu --n_views 3 \\
        --mask_root data/dtu/submission_data/idrmasks --tables results/tables/error_budget
    python experiments/bias_variance.py --self_test

Reads <out_root>/<dataset>_<n>v/<variant>/<scene>/seed<k>/eval/ours_<iter>/{renders,gt}/*.png (render.py's layout).

Estimators, per pixel and channel, with r_k the render of seed k (n seeds) and y the ground truth:
    m = mean_k r_k,  s2 = var_k(r_k, ddof=1)
    mse     = mean_k (r_k - y)^2      expected error of one run
    var     = s2                      run-to-run variance
    bias2   = (m - y)^2 - s2/n        unbiased estimate of the squared bias
    mse_ens = (m - y)^2               error of the seed ensemble
Each is averaged over pixels, channels and views (pooled; on DTU only pixels whose mask is 1, as metrics_dtu.py does).
mse = bias2 + var holds exactly and is asserted. share_var = var / mse.
psnr_single = the mean over runs of the per-image PSNR averaged over views (as metrics.py does); psnr_ens is the same
for the ensemble mean.

Writes to --tables:
    error_budget_<dataset>_<n>v.csv            scene, variant, n_seeds, n_views_eval, mse, bias2, var, mse_ens,
                                               share_var, psnr_single, psnr_ens
    error_budget_runs_<dataset>_<n>v.csv       scene, variant, seed, mse, rmse, psnr: one row per run (N4 uses it)
    error_budget_contrasts_<dataset>_<n>v.csv  B-A, C-A, D-B, E-B, I_D, I_E on mse, bias2 and var, and BV3 =
                                               mse(D) - bias2(B); paired by scene, 95% t-interval; Holm across the
                                               three BV tests (BV1: bias2 of B-A, BV2: var of C-A, BV3)
    error_budget_verdict_<dataset>_<n>v.txt    BV1, BV2, BV3 and BV5 by the PREREG_V3 §4 rules
BV4 (trend over views) and BV6 (agreement across datasets) compare several of these outputs (PREREG_V3 §4, D6).
"""
import argparse, glob, math, os, re, shutil, sys, tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stats as S  # noqa: E402  (paired t-intervals and Holm, the same helpers as the §5 analysis)

QUANTS = ["mse", "bias2", "var"]
# (key, terms as {variant: coefficient}); BV3 is handled separately because it mixes two quantities.
CONTRASTS = [
    ("B-A", {"B": 1, "A": -1}),
    ("C-A", {"C": 1, "A": -1}),
    ("D-B", {"D": 1, "B": -1}),
    ("E-B", {"E": 1, "B": -1}),
    ("I_D", {"D": 1, "C": -1, "B": -1, "A": 1}),
    ("I_E", {"E": 1, "C": -1, "B": -1, "A": 1}),
]
BV_TESTS = [("BV1", "B-A", "bias2"), ("BV2", "C-A", "var"), ("BV3", "BV3", "mse(D)-bias2(B)")]
REL_TOL = 1e-6


# ---------------------------------------------------------------- image loading

def read_png(path):
    """float64 H x W x C in [0, 1]; RGBA loses its alpha (render.py writes RGB)."""
    try:
        from PIL import Image
        a = np.asarray(Image.open(path)).astype(np.float64)
        a = a / (65535.0 if a.max() > 255 else 255.0)
    except ImportError:
        import matplotlib.image as mpimg
        a = mpimg.imread(path).astype(np.float64)
    if a.ndim == 2:
        a = a[:, :, None]
    return a[:, :, :3]


def read_mask(path, size):
    """metrics_dtu.py's mask: resized to 1/4 with PIL's default filter; a pixel-channel counts where it is exactly 1."""
    from PIL import Image
    m = Image.open(path)
    m = m.resize((m.size[0] // 4, m.size[1] // 4))
    a = np.asarray(m).astype(np.float64) / 255.0
    if a.ndim == 2:
        a = a[:, :, None]
    a = a[:, :, :3]
    if a.shape[:2] != size:
        sys.exit("mask {} is {} after /4, the render is {}".format(path, a.shape[:2], size))
    return a == 1.0


def latest_iter_dir(run):
    its = sorted(glob.glob(os.path.join(run, "eval", "ours_*")), key=lambda p: int(p.rsplit("_", 1)[1]))
    return its[-1] if its else None


def mask_paths(run, scene, names, mask_root):
    """Per-view mask files: <run>/mask/<name> (eval_dtu.py's copy), else <mask_root>/<scene>[/mask]/ in sorted order,
    renamed by position as scripts/copy_mask_dtu.sh does."""
    own = [os.path.join(run, "mask", n) for n in names]
    if all(os.path.exists(p) for p in own):
        return own
    if mask_root:
        for d in (os.path.join(mask_root, scene, "mask"), os.path.join(mask_root, scene)):
            files = sorted(f for f in glob.glob(os.path.join(d, "*.png")))
            if files:
                if len(files) != len(names):
                    sys.exit("{}: {} masks for {} eval renders".format(d, len(files), len(names)))
                return files
    sys.exit("no masks for {} (looked in {}/mask and --mask_root)".format(scene, run))


# ---------------------------------------------------------------- the estimator

def budget_arrays(renders, gt, valid=None):
    """renders: n x H x W x C, gt: H x W x C, valid: H x W x C bool or None.
    Returns per-view sums and the count of pixel-channels, so views can be pooled exactly."""
    n = renders.shape[0]
    m = renders.mean(axis=0)
    s2 = renders.var(axis=0, ddof=1)
    err = renders - gt[None]
    mse_px = (err ** 2).mean(axis=0)
    bias2_px = (m - gt) ** 2 - s2 / n
    ens_px = (m - gt) ** 2
    per_run = (err ** 2)
    if valid is not None:
        sel = valid
        cnt = int(sel.sum())
        sums = dict(mse=mse_px[sel].sum(), bias2=bias2_px[sel].sum(), var=s2[sel].sum(), mse_ens=ens_px[sel].sum())
        run_mse = np.array([per_run[k][sel].mean() for k in range(n)])
        ens_mse = ens_px[sel].mean()
    else:
        cnt = mse_px.size
        sums = dict(mse=mse_px.sum(), bias2=bias2_px.sum(), var=s2.sum(), mse_ens=ens_px.sum())
        run_mse = per_run.reshape(n, -1).mean(axis=1)
        ens_mse = ens_px.mean()
    return sums, cnt, run_mse, ens_mse


def psnr(mse):
    return float("inf") if mse <= 0 else 10.0 * math.log10(1.0 / mse)


def budget_cell(seed_dirs, scene, dataset, mask_root):
    """One (scene, variant): the pooled budget over views. seed_dirs: the eval/ours_<iter> folder of each seed."""
    names = sorted(os.listdir(os.path.join(seed_dirs[0], "renders")))
    names = [x for x in names if x.endswith(".png")]
    for d in seed_dirs[1:]:
        other = sorted(x for x in os.listdir(os.path.join(d, "renders")) if x.endswith(".png"))
        if other != names:
            sys.exit("{} and {}: different eval renders ({} vs {})".format(seed_dirs[0], d, len(names), len(other)))
    if not names:
        sys.exit("{}: no eval renders".format(seed_dirs[0]))
    run_dir = os.path.dirname(os.path.dirname(seed_dirs[0]))
    masks = mask_paths(run_dir, scene, names, mask_root) if dataset == "dtu" else None
    tot = dict(mse=0.0, bias2=0.0, var=0.0, mse_ens=0.0)
    count = 0
    psnr_runs, psnr_ens = [], []
    run_sse = np.zeros(len(seed_dirs))
    for i, name in enumerate(names):
        gts = [read_png(os.path.join(d, "gt", name)) for d in seed_dirs]
        for d, g in zip(seed_dirs[1:], gts[1:]):
            if g.shape != gts[0].shape or np.abs(g - gts[0]).max() > 1e-9:
                sys.exit("ground truth differs between seeds: {} vs {}".format(
                    os.path.join(seed_dirs[0], "gt", name), os.path.join(d, "gt", name)))
        gt = gts[0]
        r = np.stack([read_png(os.path.join(d, "renders", name)) for d in seed_dirs])
        if r.shape[1:] != gt.shape:
            sys.exit("{}: render {} vs gt {}".format(name, r.shape[1:], gt.shape))
        valid = None
        if masks is not None:
            valid = np.broadcast_to(read_mask(masks[i], gt.shape[:2]), gt.shape)
            if not valid.any():
                sys.exit("mask {} has no valid pixel".format(masks[i]))
        sums, cnt, run_mse, ens_mse = budget_arrays(r, gt, valid)
        for k in tot:
            tot[k] += sums[k]
        count += cnt
        psnr_runs.append([psnr(x) for x in run_mse])
        run_sse += run_mse * cnt
        psnr_ens.append(psnr(ens_mse))
    row = {k: v / count for k, v in tot.items()}
    check_identity(row, seed_dirs[0])
    row["share_var"] = row["var"] / row["mse"] if row["mse"] > 0 else float("nan")
    row["psnr_single"] = float(np.mean(np.mean(np.array(psnr_runs), axis=0)))
    row["psnr_ens"] = float(np.mean(psnr_ens))
    row["n_seeds"] = len(seed_dirs)
    row["n_views_eval"] = len(names)
    per_run_psnr = np.mean(np.array(psnr_runs), axis=0)
    seeds = [int(re.search(r"seed(\d+)", d.replace(os.sep, "/")).group(1)) for d in seed_dirs]
    row["_runs"] = [dict(seed=k, mse=run_sse[j] / count, rmse=math.sqrt(run_sse[j] / count), psnr=per_run_psnr[j])
                    for j, k in enumerate(seeds)]
    return row


def check_identity(row, where):
    lhs, rhs = row["mse"], row["bias2"] + row["var"]
    if abs(lhs - rhs) > REL_TOL * max(abs(lhs), 1e-12):
        sys.exit("identity mse = bias2 + var fails for {}: {} vs {}".format(where, lhs, rhs))


# ---------------------------------------------------------------- collection and contrasts

def collect(out_root, dataset, n_views, variants, seeds, mask_root):
    base = os.path.join(out_root, "{}_{}v".format(dataset, n_views))
    if not os.path.isdir(base):
        sys.exit("no folder {}".format(base))
    rows, skipped, runs = [], [], []
    scenes = sorted({os.path.basename(p) for v in variants for p in glob.glob(os.path.join(base, v, "*"))
                     if os.path.isdir(p)})
    for scene in scenes:
        for v in variants:
            dirs = []
            for k in seeds:
                d = latest_iter_dir(os.path.join(base, v, scene, "seed{}".format(k)))
                if d and os.path.isdir(os.path.join(d, "renders")):
                    dirs.append(d)
            if len(dirs) < 2:
                skipped.append("{}/{}: {} seed(s) with eval renders".format(v, scene, len(dirs)))
                continue
            iters = {os.path.basename(d) for d in dirs}
            if len(iters) > 1:
                sys.exit("{}/{}: seeds evaluated at different iterations {}".format(v, scene, sorted(iters)))
            row = budget_cell(dirs, scene, dataset, mask_root)
            row.update(scene=scene, variant=v)
            runs += [dict(scene=scene, variant=v, **r) for r in row.pop("_runs")]
            rows.append(row)
            print("  {:<10} {:<6} n={} mse={:.6f} bias2={:.6f} var={:.6f}".format(
                scene, v, row["n_seeds"], row["mse"], row["bias2"], row["var"]))
    for s in skipped:
        print("SKIPPED " + s, file=sys.stderr)
    if not rows:
        sys.exit("no (scene, variant) with at least two seeds under {}".format(base))
    cols = ["scene", "variant", "n_seeds", "n_views_eval", "mse", "bias2", "var", "mse_ens", "share_var",
            "psnr_single", "psnr_ens"]
    return pd.DataFrame(rows)[cols], pd.DataFrame(runs)[["scene", "variant", "seed", "mse", "rmse", "psnr"]]


def contrast_rows(budget):
    """Paired-by-scene contrasts. Errors: lower is better, so a reduction is a negative mean."""
    rows = []
    for q in QUANTS:
        t = budget.pivot(index="scene", columns="variant", values=q)
        for key, terms in CONTRASTS:
            if not all(v in t for v in terms):
                continue
            d = sum(c * t[v] for v, c in terms.items())
            rows.append(dict(contrast=key, quantity=q, **S.paired(d, higher_is_better=False)))
    tm = budget.pivot(index="scene", columns="variant", values="mse")
    tb = budget.pivot(index="scene", columns="variant", values="bias2")
    if "D" in tm and "B" in tb:
        rows.append(dict(contrast="BV3", quantity="mse(D)-bias2(B)",
                         **S.paired(tm["D"] - tb["B"], higher_is_better=False)))
    res = pd.DataFrame(rows)
    res["bv_test"] = ""
    for name, key, q in BV_TESTS:
        res.loc[(res["contrast"] == key) & (res["quantity"] == q), "bv_test"] = name
    fam = res["bv_test"] != ""
    res["p_holm"] = float("nan")
    res.loc[fam, "p_holm"] = S.holm(res.loc[fam, "p"].tolist())
    return res


def get(res, key, q):
    r = res[(res["contrast"] == key) & (res["quantity"] == q)]
    return None if r.empty else r.iloc[0]


def verdicts(res):
    """PREREG_V3 §4 decision rules for BV1, BV2, BV3 and BV5 (BV4/BV6 need several tables)."""
    out = []

    def share(key, part, label, name):
        m, p = get(res, key, "mse"), get(res, key, part)
        if m is None or p is None:
            out.append("{}: not computed (variants missing)".format(name))
            return
        ratio = p["mean"] / m["mean"] if m["mean"] != 0 else float("nan")
        ok = m["mean"] < 0 and ratio > 0.5 and p["p_holm"] < S.ALPHA and not (p["ci_lo"] <= 0 <= p["ci_hi"])
        out.append("{}: {} share of {} = {:.2f} (Δmse {:+.3g}, Δ{} {:+.3g} [{:+.3g}, {:+.3g}], Holm p {}) -> {}".format(
            name, label, key, ratio, m["mean"], part, p["mean"], p["ci_lo"], p["ci_hi"], S.fmt_p(p["p_holm"]),
            "HOLDS" if ok else "FAILS"))

    share("B-A", "bias2", "Δbias2", "BV1")
    share("C-A", "var", "Δvar", "BV2")
    r = get(res, "BV3", "mse(D)-bias2(B)")
    if r is None or pd.isna(r["ci_lo"]):
        out.append("BV3: not computed")
    else:
        where = "below 0: dropout also reduces bias" if r["ci_hi"] < 0 else \
            "above 0: D keeps variance above B's bias floor" if r["ci_lo"] > 0 else "contains 0: D is at B's bias floor"
        out.append("BV3: mse(D) - bias2(B) = {:+.3g} [{:+.3g}, {:+.3g}] (Holm p {}) -> CI {}".format(
            r["mean"], r["ci_lo"], r["ci_hi"], S.fmt_p(r["p_holm"]), where))
    for key in ("I_D", "I_E"):
        parts = [get(res, key, q) for q in QUANTS]
        if any(x is None for x in parts):
            continue
        out.append("BV5 {}: I_mse {:+.3g} = I_bias2 {:+.3g} [{:+.3g}, {:+.3g}] + I_var {:+.3g} [{:+.3g}, {:+.3g}]".format(
            key, parts[0]["mean"], parts[1]["mean"], parts[1]["ci_lo"], parts[1]["ci_hi"],
            parts[2]["mean"], parts[2]["ci_lo"], parts[2]["ci_hi"]))
    return out


def run(a):
    print("error budget: {} {}v, variants {}, seeds {}".format(a.dataset, a.n_views, a.variants, a.seeds))
    budget, runs = collect(a.out_root, a.dataset, a.n_views, a.variants, a.seeds, a.mask_root)
    os.makedirs(a.tables, exist_ok=True)
    tag = "{}_{}v".format(a.dataset, a.n_views)
    budget.to_csv(os.path.join(a.tables, "error_budget_{}.csv".format(tag)), index=False)
    runs.to_csv(os.path.join(a.tables, "error_budget_runs_{}.csv".format(tag)), index=False)
    res = contrast_rows(budget)
    res.to_csv(os.path.join(a.tables, "error_budget_contrasts_{}.csv".format(tag)), index=False)
    lines = verdicts(res)
    with open(os.path.join(a.tables, "error_budget_verdict_{}.txt".format(tag)), "w") as f:
        f.write("Error budget {} (PREREG_V3 §4), {} scenes\n".format(tag, budget["scene"].nunique()))
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote error_budget_{0}.csv, error_budget_contrasts_{0}.csv, error_budget_verdict_{0}.txt to {1}".format(
        tag, a.tables))
    return budget, res


# ---------------------------------------------------------------- self-test

def write_png8(path, a):
    from PIL import Image
    Image.fromarray(np.clip(np.round(a * 255), 0, 255).astype(np.uint8)).save(path)


def self_test():
    rng = np.random.RandomState(15112026)
    # 1. The estimator recovers a known bias and noise (no quantisation): 3 seeds x 2 variants.
    H, W = 256, 256
    gt = 0.3 + 0.4 * rng.rand(H, W, 3)
    yy, xx = np.mgrid[0:H, 0:W]
    bias_map = 0.08 * np.sin(xx / 9.0)[:, :, None] * np.ones((1, 1, 3))
    for name, bias, sigma in (("biased", bias_map, 0.02), ("noisy", 0.5 * bias_map, 0.06)):
        r = gt[None] + bias[None] + sigma * rng.randn(3, H, W, 3)
        sums, cnt, _, _ = budget_arrays(r, gt)
        est = {k: v / cnt for k, v in sums.items()}
        check_identity(est, "self-test " + name)
        true_b2, true_var = float((bias ** 2).mean()), sigma ** 2
        for k, true in (("bias2", true_b2), ("var", true_var)):
            if abs(est[k] - true) > 0.05 * true:
                sys.exit("self-test {}: {} = {:.6g}, expected {:.6g} within 5%".format(name, k, est[k], true))
        print("self-test {}: bias2 {:.3e} (true {:.3e}), var {:.3e} (true {:.3e}), identity ok".format(
            name, est["bias2"], true_b2, est["var"], true_var))
    # A mask restricts the average to the valid pixels.
    valid = np.zeros((H, W, 3), bool)
    valid[: H // 2] = True
    r = gt[None] + bias_map[None] + 0.02 * rng.randn(3, H, W, 3)
    sums, cnt, _, _ = budget_arrays(r, gt, valid)
    if cnt != valid.sum() or abs(sums["bias2"] / cnt - (bias_map[: H // 2] ** 2).mean()) > 0.05 * (
            bias_map[: H // 2] ** 2).mean():
        sys.exit("self-test: masked average is wrong")

    # 2. End to end through files in render.py's layout: 4 scenes x A-E x 3 seeds x 2 views, plus one quirk each
    # (a variant with only one seed is skipped). Expected: depth (B, D, E) lowers bias, dropout (C, D, E) lowers
    # variance, so BV1 and BV2 hold and the CSV equals the in-memory estimator on the quantised images.
    tmp = tempfile.mkdtemp(prefix="bv_selftest_")
    try:
        h, w = 48, 64
        spec = dict(A=(1.0, 0.05), B=(0.4, 0.05), C=(1.0, 0.02), D=(0.4, 0.02), E=(0.35, 0.02))
        expect = {}
        for s, scene in enumerate(["s0", "s1", "s2", "s3"]):
            g = 0.3 + 0.4 * rng.rand(h, w, 3)
            b = (0.06 + 0.01 * s) * np.cos(np.mgrid[0:h, 0:w][1] / 7.0)[:, :, None] * np.ones((1, 1, 3))
            for v, (bscale, sig) in spec.items():
                imgs = []
                for k in range(3):
                    ev = os.path.join(tmp, "llff_3v", v, scene, "seed{}".format(k), "eval", "ours_6000")
                    os.makedirs(os.path.join(ev, "renders"))
                    os.makedirs(os.path.join(ev, "gt"))
                    views = []
                    for i in range(2):
                        img = g + bscale * b + sig * rng.randn(h, w, 3)
                        write_png8(os.path.join(ev, "renders", "{:05d}.png".format(i)), img)
                        write_png8(os.path.join(ev, "gt", "{:05d}.png".format(i)), g)
                        views.append(np.clip(np.round(img * 255), 0, 255) / 255.0)
                    imgs.append(views)
                gq = np.clip(np.round(g * 255), 0, 255) / 255.0
                tot, cnt = dict(mse=0.0, bias2=0.0, var=0.0), 0
                for i in range(2):
                    sm, c, _, _ = budget_arrays(np.stack([imgs[k][i] for k in range(3)]), gq)
                    for q in tot:
                        tot[q] += sm[q]
                    cnt += c
                expect[(scene, v)] = {q: tot[q] / cnt for q in tot}
        lone = os.path.join(tmp, "llff_3v", "A", "s_lonely", "seed0", "eval", "ours_6000", "renders")
        os.makedirs(lone)
        a = argparse.Namespace(out_root=tmp, dataset="llff", n_views=3, variants=list("ABCDE"), seeds=[0, 1, 2],
                               mask_root=None, tables=os.path.join(tmp, "tables"))
        budget, res = run(a)
        runs = pd.read_csv(os.path.join(a.tables, "error_budget_runs_llff_3v.csv"))
        if len(runs) != 60 or sorted(runs["seed"].unique()) != [0, 1, 2]:
            sys.exit("self-test: expected 60 per-run rows for seeds 0-2, got {}".format(len(runs)))
        # The mean over runs of the per-run mse is the cell's mse (same pixels, equal counts).
        mm = runs.groupby(["scene", "variant"])["mse"].mean()
        for _, row in budget.iterrows():
            if abs(mm[(row["scene"], row["variant"])] - row["mse"]) > 1e-9 * row["mse"]:
                sys.exit("self-test: per-run mse does not average to the cell mse")
        if len(budget) != 20:
            sys.exit("self-test: expected 20 (scene, variant) rows, got {}".format(len(budget)))
        for _, row in budget.iterrows():
            for q, val in expect[(row["scene"], row["variant"])].items():
                if abs(row[q] - val) > 1e-9 * max(abs(val), 1e-12):
                    sys.exit("self-test: file round trip changed {} of {}/{}".format(q, row["variant"], row["scene"]))
        verdict = open(os.path.join(a.tables, "error_budget_verdict_llff_3v.txt")).read()
        for must in ("BV1: ", "BV2: ", "BV3: ", "BV5 I_D"):
            if must not in verdict:
                sys.exit("self-test: verdict lacks " + must)
        if "BV1:" not in verdict or "-> HOLDS" not in verdict.split("BV1:")[1].split("\n")[0]:
            sys.exit("self-test: BV1 should hold on the synthetic data:\n" + verdict)
        if "-> HOLDS" not in verdict.split("BV2:")[1].split("\n")[0]:
            sys.exit("self-test: BV2 should hold on the synthetic data:\n" + verdict)
        # 3. DTU: masks from --mask_root (full resolution, resized /4 as metrics_dtu.py does); only mask == 1 counts.
        for k in range(3):
            src = os.path.join(tmp, "llff_3v", "B", "s0", "seed{}".format(k))
            shutil.copytree(src, os.path.join(tmp, "dtu_3v", "B", "scan8", "seed{}".format(k)))
        mdir = os.path.join(tmp, "idrmasks", "scan8", "mask")
        os.makedirs(mdir)
        half = np.zeros((4 * h, 4 * w, 3))
        half[:, : 2 * w] = 1.0                      # left half valid
        for i in range(2):
            write_png8(os.path.join(mdir, "{:03d}.png".format(i)), half)
        dtu, _ = collect(tmp, "dtu", 3, ["B"], [0, 1, 2], os.path.join(tmp, "idrmasks"))
        # Expected: the plain estimator on the pixels read_mask() keeps (the left half, minus the resize's blurred edge).
        valid = read_mask(os.path.join(mdir, "000.png"), (h, w))
        if not (valid[:, : w // 2 - 2].all() and not valid[:, w // 2 + 2:].any()):
            sys.exit("self-test: the resized mask does not keep the left half")
        valid = np.broadcast_to(valid, (h, w, 3))
        ev = [os.path.join(tmp, "dtu_3v", "B", "scan8", "seed{}".format(k), "eval", "ours_6000") for k in range(3)]
        gq = read_png(os.path.join(ev[0], "gt", "00000.png"))
        tot, cnt = 0.0, 0
        for i in range(2):
            r = np.stack([read_png(os.path.join(e, "renders", "{:05d}.png".format(i))) for e in ev])
            sm, c, _, _ = budget_arrays(r, gq, valid)
            tot += sm["bias2"]
            cnt += c
        if cnt >= 2 * h * w * 3 or abs(dtu.iloc[0]["bias2"] - tot / cnt) > 1e-9 * abs(tot / cnt):
            sys.exit("self-test: DTU masked budget is not the left-half budget")
        # 4. A GT that differs between seeds must stop the run.
        bad = os.path.join(tmp, "llff_3v", "B", "s0", "seed1", "eval", "ours_6000", "gt", "00000.png")
        write_png8(bad, 0.5 * np.ones((h, w, 3)))
        try:
            collect(tmp, "llff", 3, ["B"], [0, 1, 2], None)
        except SystemExit as e:
            if "ground truth differs" not in str(e):
                raise
        else:
            sys.exit("self-test: a GT that differs between seeds was not caught")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("BIAS-VARIANCE SELF-TEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_root")
    ap.add_argument("--dataset", choices=["llff", "dtu", "blender"], default="llff")
    ap.add_argument("--n_views", type=int, default=3)
    ap.add_argument("--variants", nargs="+", default=list("ABCDE"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--mask_root", help="DTU: idrmasks folder, used when a run has no mask/ copy")
    ap.add_argument("--tables", default="results/tables/error_budget")
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if not a.out_root:
        ap.error("--out_root is required (or --self_test)")
    run(a)


if __name__ == "__main__":
    main()
