# Research analysis: making the depth × dropout study publication-grade

**Wed 14 Oct 2026** · **Draft for Aayush to review and edit** · Deadline: Sun 15 Nov (team target: submit 10–11 Nov)

**What this document is.** A critical review of the research idea and of where we stand, plus a ranked list of improvements. The redrafted plan (`docs/PLAN.md` v2) will be generated from the decisions you mark in §10, and only after you have edited this document.

**What stays fixed.** These are not up for change; every recommendation sharpens or extends them:
- the research question;
- variants A–E on the DNGaussian host;
- LLFF 3-view as the main benchmark;
- the verbatim DropGaussian port with decision D1 (`r = γ · t / 10000`);
- the invariants in AGENTS.md.

Anything that would change the question is listed separately as future work (§9).

**How to use it.** Read §1. Then go through §7 and tick Accept / Modify / Reject for each item in the decision sheet (§10), adding notes anywhere. Answer the open questions at the end of §10.

---

## Contents
1. [Executive summary](#1-executive-summary)
2. [The question, stated precisely](#2-the-question-stated-precisely)
3. [Where we are on 14 Oct](#3-where-we-are-on-14-oct)
4. [Literature update and positioning](#4-literature-update-and-positioning)
5. [Critical assessment of the current design](#5-critical-assessment-of-the-current-design)
6. [Theory: what dropout does to an alpha-compositing renderer](#6-theory-what-dropout-does-to-an-alpha-compositing-renderer)
7. [Recommendations](#7-recommendations)
8. [Scope packages, effort and compute](#8-scope-packages-effort-and-compute)
9. [Future research directions](#9-future-research-directions)
10. [Decision sheet (edit this)](#10-decision-sheet-edit-this)
11. [References](#11-references)
- [Appendix A: proposed §6 v2 analysis rules](#appendix-a-proposed-6-v2-analysis-rules)
- [Appendix B: proposed diagnostics, exact definitions](#appendix-b-proposed-diagnostics-exact-definitions)

---

## 1. Executive summary

### The question is still worth answering, but the framing has to change

Nobody has published a controlled study of how a depth prior and Gaussian dropout *interact*. However, combining the two is no longer new:
- **DropoutGS** (CVPR 2025, Table 3) adds its dropout regularizer to DNGaussian with MVS initialization and reports **19.94 → 20.64 PSNR** on LLFF 3-view.
- **StereoGS** (2026) adds DropGaussian's dropout on top of stereo-depth regularization: **21.91 → 22.05**.
- **D²GS** (2025) uses depth to decide *which* Gaussians to drop.

So "are they complementary?" already has a partial, uncontrolled answer: stacking them helps a little. Our PLAN §7.3 sentence "the cross-family question is open" would not survive review.

What remains open, and what our design is well placed to answer:
1. Is the combined gain **additive, sub-additive or interfering** when measured in a factorial design?
2. Does it matter whether the dropout mask reaches the depth supervision (**D vs E**)?
3. Through **what mechanism**?
4. Under which **conditions** does the answer change: dose, number of views, initialization, dataset, depth-prior quality?

None of the papers above isolates any of these.

### Strengths to keep
- One codebase and five flag settings: a clean 2×2 factorial plus a decoupling contrast.
- E − D is the cleanest causal contrast in the design. The two variants run identical code with one switch: whether the depth passes receive the mask. They take the same number of renders and optimizer steps.
- Verdict rules written before any result, including Disha's interaction contrast.
- Diagnostics designed from day one (`loss_depth_used − loss_depth_clean`, `depth_var_masks`).
- Reproducibility that most papers lack:
  - code tags and an md5-locked depth archive;
  - CI with a dozen deterministic checks and fixture-tested analysis scripts;
  - baselines verified against each paper (this already caught that DropGaussian, CoR-GS and DropoutGS print FSGS's superseded arXiv-v1 numbers).

### Weaknesses (each has a fix in §7)

| # | Weakness | Consequence | Fix |
| --- | --- | --- | --- |
| W-1 | **Dose.** D1 makes the drop rate peak at 0.12, not DropGaussian's published 0.2. DropoutGS also found about 0.2 best | A null result could simply be under-dosing | B1 |
| W-2 | **Generality.** One host, one dataset (8 scenes), one initialization. Every prior result we will be compared with used dense (MVS) initialization, and StereoGS's own ablation shows a regularizer interaction **flipping sign** with initialization | "Holds for DNGaussian on LLFF with random init" is all we could claim | B2, B3, B4 |
| W-3 | **Inference.** With 8 scenes × 3 seeds, the smallest interaction we can reliably detect is about 0.35–0.55 dB (§5.3). The current rules call any null "redundant", and they leave a gap: a combination that beats both singles but is sub-additive gets **no verdict** (this happens on our own fixtures) | Overclaiming, or an unclassifiable result | A1–A4 |
| W-4 | **Mechanism is under-identified.** The depth-loss gap shows that the depth signal changes under masks, not that the change does harm. D and E also differ in **two** ways at once: holes in the depth render, and a bias from the rasterizer's alpha clamp (§6) | The verdict's "why" would be speculative | C1–C6 and §6 |
| W-5 | **Progress.** The experimental code (W2–W9) is not on `main`, and no Tier 1 run exists | Everything above is cheap to fix **now** and expensive after Tier 1 has run | Do the time-critical items first |

### Top recommendations, ranked by value per effort

| Rank | Recommendation | What it buys | Cost |
| --- | --- | --- | --- |
| 1 | Revise §6 before any result exists: a two-axis verdict, equivalence tests, no data-driven best(B, C); tag it as a pre-registration (A1–A4, A7) | A null result becomes a defensible claim; credibility with reviewers | ~1.5 person-days, 0 GPU |
| 2 | Log the mechanism diagnostics inside Tier 1 (C1–C3) | No re-running 120 runs later | ~0.5 day; ~1–3 % run time |
| 3 | Dose–response for C, D and E, replacing Tier 2b (B1) | Resolves 0.12 vs 0.2; doubles as a mechanism test | ~0.5 day; ~12 T4-h |
| 4 | Noise-floor block and an adaptive seed rule (B9, A6) | Our detection limit is known before we interpret anything | ~0.25 day; ~3 T4-h |
| 5 | Initialization moderator using DropGaussian's released dense clouds (B2) | Reconciles us with DropoutGS and StereoGS; tests the strongest known moderator | ~1 day; ~12 T4-h |
| 6 | Tier 2 with variant A, on T4s (B3) | Interaction per view count; removes the laptop-GPU confound | ~0.25 day; 10–20 T4-h |
| 7 | DTU 3-view as a second dataset (B4) | Generality; roughly halves the detection limit | ~2.5 person-days; 13–19 T4-h |
| 8 | A theory section (§6): compensated dropout is unbiased except at the alpha clamp, dropout acts as a variance penalty, and what that predicts for D vs E | A principled spine for the paper, with falsifiable predictions | ~1 day of writing |

### Time-critical: decide within 48 hours, before W2–W4 merge
- Items 1 and 2 above.
- Variant naming and dose support in the runner (D-c).
- Whether Tier 2 includes variant A and moves to T4s.

### Publication
CVPR 2027 paper registration is **10 Nov**, and the submission deadline is **16 Nov**, AoE. That is the same week as our course deadline. A CVPR attempt is realistic only with the "full" package (§8) and a clear effect after Tier 1.

Recommendation:
- Write in the CVPR LaTeX template from the start.
- Decide go or no-go at a gate after Tier 1 (about 26–27 Oct).
- Otherwise target TMLR (rolling submission, values rigour over state-of-the-art numbers) or a CVPR 2027 workshop, and post an arXiv preprint after the course submission.

---

## 2. The question, stated precisely

**RQ (unchanged).** In sparse-view 3DGS, are DNGaussian's monocular-depth regularization and DropGaussian's Gaussian dropout complementary, redundant or interfering? Does keeping the dropout mask away from the depth supervision (decoupling) change the answer?

**Sub-questions, proposed to make the RQ answerable and publishable:**
- **RQ1 (effect).** What are the 2×2 effects at the main dose (B − A, C − A) and the interaction contrasts I_D and I_E? Is the combination better than each single regularizer?
- **RQ2 (decoupling).** Is E − D different from 0, and in which direction?
- **RQ3 (mechanism).** Is any interference caused by:
  - corruption of the depth signal (H1);
  - the alpha-clamp bias (H2);
  - neighbour compensation (H3);
  - the variance penalty (H4)?
- **RQ4 (conditions).** How do the answers change with:
  - dropout dose;
  - number of views;
  - initialization;
  - dataset;
  - depth-prior quality?

The current plan covers RQ1, RQ2 and part of RQ3. RQ4 is the main extension the new deadline makes possible.

---

## 3. Where we are on 14 Oct

| Area | State | Evidence | What it means |
| --- | --- | --- | --- |
| DNGaussian reproduction | ✅ Done twice | Laptop 18.73 (laptop-made depth); T4 **18.93 / 0.595 / 0.294** with the canonical depth, against the paper's 19.12 / 0.591 / 0.294 | B is a faithful baseline (−0.19 dB) |
| Data | ✅ | `llff_with_depth.tar`, md5-locked | — |
| Infrastructure | ✅ | CI (11 checks on `main`, 12 once #12 merges), the merge gate and ruleset, review bundles, fixtures | A strong reproducibility story. **Stop investing here** unless something breaks |
| Analysis scripts | ✅ on fixtures (#9) | `stats.py` (paired tests, interaction contrasts, verdict), F2–F5, fail-closed sharp checks | Need generalizing to new factors: dose, initialization, dataset (Disha) |
| Reported baselines | ✅ (#10) | Every row verified; FSGS corrected to its ECCV numbers | Context table only (§5.2, T12) |
| `--seed`, `--depth_scale`, `config.json` | On branch `feat/dropout`, no PR | 1100-iteration smoke tests | Rebase onto `main` (the merge hook needs it) |
| **Dropout port, E mode, `diag.csv`, stats JSON files, runner, validation (W2–W9)** | ❌ **Not on `main`** | — | **The critical path.** About 2 days behind the 20 Oct plan |
| Experimental runs | None yet | — | Every design decision below is still free |
| Related work, writing | Not started | — | Method and theory (§6) can be written before results |
| A100 | Not pursued | — | Not needed: compute is no longer the constraint (§8) |

**Observations**
1. With 27 days to 10 Nov, **person-days, not GPU-hours, are the binding constraint**. Two Colab and two Kaggle accounts can supply well over 100 T4-hours (§8). Rank work by value per person-day.
2. W2–W9 must land first, with the cheap diagnostics (C1–C3) folded in. Every extension competes for Aayush's time.
3. The gate took three PRs in three days. It is complete now, so that time goes back to science.
4. Discipline that should continue:
   - verified baselines;
   - fixtures for the analysis scripts;
   - fail-closed checks;
   - md5-locked data.

   These are rarer in published work than they should be, and they belong in the paper's reproducibility statement.

---

## 4. Literature update and positioning

These works appeared or matter since the playbook was written (12 Oct). Numbers are LLFF 3-view PSNR unless stated, as reported by the cited paper.

| Work | Venue | What it does | Bearing on our question | What it does *not* answer |
| --- | --- | --- | --- | --- |
| DNGaussian | CVPR 2024 | Hard and soft depth passes, patch-normalized depth loss | Our host | — |
| DropGaussian | CVPR 2025 | Random dropout with 1/(1−r) compensation, linear schedule, dense COLMAP initialization. Prior-free | Our dropout | No combination with priors |
| **DropoutGS** | CVPR 2025 | Random dropout regularization: a dropped sub-model is supervised by the full model, with **no compensation**, plus edge-guided splitting. **Table 3 plug-ins** (★ = MVS initialization): DNGaussian★ 19.94 → **20.64**; FSGS★ 20.43 → 20.70; CoR-GS★ 20.45 → 20.59; 3DGS★ 19.86 → 20.53; 3DGS, random initialization, 16.46 → 18.05. Their loss keeps DNGaussian's depth term. A dropout-rate sweep finds gains up to p ≈ 0.2 | **Prior evidence that a dropout-family regularizer adds to depth regularization** | It uses a different dropout family (no compensation, consistency loss) and dense initialization. There is no factorial (no "dropout without depth" in the same host), single numbers, no mechanism, and no D vs E |
| UGOD | arXiv 2508.04968 (Aug 2025) | Uncertainty-guided opacity plus soft dropout | Shows the dropout design space is wide | No depth interaction |
| **D²GS** | arXiv 2510.08566 (Oct 2025) | **Depth-and-density-guided dropout** | A *designed* combination: depth steers the mask | Doesn't measure the interaction |
| DP-GS | APSIPA ASC 2025 | Reportedly pairs monocular depth (Depth Anything V2) with point dropout plus diffusion guidance (from the abstract; we could not read the PDF text) | Another stacked combination | Apparently no factorial |
| **DropAnSH-GS** | CVPR 2026 (arXiv 2602.20933) | Names the **"neighbour compensation effect"**: overlapping neighbours compensate for dropped Gaussians and weaken dropout. Drops anchor neighbourhoods and high-order SH | **Gives a mechanism for sub-additivity (H3):** if depth supervision packs Gaussians onto surfaces, dropout bites less | No depth |
| Fisher-information sparse-view 3DGS | arXiv 2606.20842 (Jun 2026) | Another 2026 dropout-style variant | The field is still active | — |
| **StereoGS** | arXiv 2606.30545 (Jul 2026; its repository lists ECCV 2026) | Stereo-depth regularization, opacity decay and dense initialization: 21.91. **"Ours★" adds DropGaussian's dropout: 22.05 (+0.14 dB).** In its single-run ablation, the depth × opacity-decay interaction is **−0.52 dB without dense initialization and +0.69 dB with it** | (a) Dropout's marginal value may be small when the depth prior is strong. (b) **Interaction signs depend on initialization** | No dropout ablation; single runs |
| Reliability-aware monocular depth supervision | arXiv 2607.02554 (Jul 2026) | Monocular depth is unreliable at reflections, sky and occlusion boundaries; masks it by photometric reliability | Depth-prior quality as a moderator (B5); region analysis (C5b) | — |
| DGDO-GS | Applied Sciences 2026, 16(8):4026 | Depth constraints whose strength varies in space (flat vs edge regions) | Region-specific effects of depth supervision (H5) | — |
| PairDropGS | arXiv 2605.12072 | **Withdrawn**: don't cite it as evidence | — | — |
| Context | — | As reported by StereoGS: Binocular3DGS 21.44, NexusGS 21.07, MVPGS 20.54 | Our host (~19 dB) is well below the state of the art: a reviewer concern (§7, B2 and B8) | — |

**Proposed positioning (for the Introduction).**
> Stacking dropout-style regularization onto depth-regularized pipelines has been reported to help (DropoutGS, Tab. 3; StereoGS), and depth has been used to steer dropout (D²GS). Yet no prior work measures the *interaction* of the two in a factorial design, separates *where* the dropout mask acts (on the depth supervision or only on the photometric loss), varies the *dose*, or tests *moderators* such as initialization and view count. We provide the first controlled study of this kind, with pre-registered analysis rules, mechanistic diagnostics and a theory of compensated dropout in alpha compositing.

**Required changes**
- PLAN §7.3: replace "the cross-family question is open".
- Cite DropoutGS Table 3 and StereoGS★ explicitly.
- Repeat the arXiv check right before submission. CVPR 2027 treats papers posted after 15 Sep 2026 as contemporaneous: they must be cited and discussed, though not compared against.

---

## 5. Critical assessment of the current design

### 5.1 What is already strong
- **The factorial core.** A–D form a 2×2; E adds a decoupling contrast that differs from D by a single switch, with the same renders and optimizer steps.
- **Faithful ports.** DNGaussian at its published configuration; DropGaussian's code verbatim (D1).
- **Analysis written before results**, including the interaction contrast and a mechanism test.
- **Engineering for honesty:**
  - fail-closed sharp checks on every `diag.csv`;
  - runs only from a tag;
  - one GPU type per seed;
  - the `shared_gpu` flag;
  - fixture-tested analysis code.

### 5.2 Threats to validity

Severity: **H** = could change the conclusion, **M** = weakens a claim, **L** = report it as a limitation.

| ID | Threat | Kind | Severity | Fix |
| --- | --- | --- | --- | --- |
| T1 | **Dose mismatch.** With D1 the rate peaks at 0.12, but DropGaussian ends training at 0.2 and DropoutGS finds about 0.2 best. C, D and E may be under-dosed, which biases the study towards "redundant" | Construct | **H** | B1 |
| T2 | **One initialization (random).** Every comparable published result (DropoutGS★, StereoGS, DropGaussian) uses dense initialization, and StereoGS shows interaction signs flipping with initialization | External | **H** | B2 |
| T3 | **One dataset of 8 forward-facing scenes** | External, power | **H** for generality | B4 |
| T4 | **Gaps in the verdict taxonomy.** (a) "Redundant: both CIs contain 0" treats lack of power as a finding. (b) A combination that beats both singles but is sub-additive gets no label. On our fixtures, E − B = +0.33 [+0.18, +0.47] with 8/8 wins, yet E receives neither "complementary" nor "interfering", because I_E's CI lies below 0 | Statistical | **H** | A1, A2 |
| T5 | **best(B, C) is chosen on the same data it is tested on.** The winner's mean is biased upwards (winner's curse), which inflates false "interfering" calls | Statistical | M | A3 |
| T6 | **Power.** Detectable interaction is about 0.35–0.55 dB on LLFF (§5.3); the planned "about 0.2 dB" caveat is optimistic | Statistical | **H** | A6, B4, B9 |
| T7 | **D vs E bundles two mechanisms:** holes in the masked depth render, and the alpha-clamp bias (§6.2). The gap diagnostic shows the depth loss *differs* under masks, not that this causes harm | Internal (mechanism) | **H** for the "why" | C1, C4, C6, B1 |
| T8 | B − A measures DNGaussian's *depth module as a whole*: its losses **and** the two extra optimizer steps per iteration. Fine, but say so | Construct | L | Text |
| T9 | **Densification is a mediator.** Dropped Gaussians get zero gradient that iteration, so C, D and E densify less and end with different Gaussian counts. That is part of how dropout works (faithful port), but it should be measured, not ignored | Internal | M | C3 |
| T10 | Hyperparameters were tuned for B. C, D and E inherit them | Construct | M | B1 partly; text |
| T11 | Tier 2 omits A, so there is no interaction per view count. It runs on the laptop GPU and PyTorch 1.12 (a GPU confound against Tier 1) with one seed | Statistical, internal | M | B3 |
| T12 | **Reported baselines are not comparable:** SSIM implementations differ, LPIPS backbones differ, and the 3DGS baseline ranges from 15.52 to 19.86 across papers, depending on initialization and protocol. The verified CSV says DNGaussian's paper used **scikit-image SSIM**, while D2's rationale ("the paper matches 3DGS SSIM") assumed otherwise | Measurement | M | E2 (D2 is still fine; fix its rationale) |
| T13 | Nondeterminism is not characterized at full length. Fern on the T4 at 6000 iterations gave 20.80 and 21.08 in two runs; at 1100 iterations, 17.75 vs 17.29 | Statistical | M | B9 |
| T14 | **One depth prior (DPT).** Conclusions may be specific to DPT's quality | External | M | B5 |

### 5.3 How much can we detect?

Assumptions:
- σ = the run-to-run SD of one run's PSNR (seed plus nondeterminism). Measured: 0.2–0.33.
- τ = the SD across scenes of the true effect.
- Each scene's estimate averages k seeds; the test is a paired t over n scenes.
- MDE = the minimum detectable effect at α = 0.05 with 80 % power.
- The TOST bound is the smallest equivalence margin we could confirm with 80 % power if the true effect were 0.

The values below come from a short scipy script; the formulas are in Appendix A.

| Contrast | σ | τ | Seeds | Scenes | MDE (dB) | TOST bound (dB) |
| --- | --- | --- | --- | --- | --- | --- |
| Interaction I (4 cells) | 0.2 | 0.2 | 3 | 8 | 0.35 | 0.36 |
| I | 0.3 | 0.2 | 3 | 8 | 0.46 | 0.47 |
| I | 0.3 | 0.2 | 5 | 8 | 0.39 | 0.39 |
| I | 0.3 | 0.3 | 3 | 8 | 0.53 | 0.54 |
| I | 0.3 | 0.2 | 3 | 23 (LLFF + DTU) | 0.25 | 0.25 |
| I | 0.3 | 0.2 | 5 | 23 | 0.21 | 0.21 |
| E − D, or D − B (2 cells) | 0.3 | 0.2 | 3 | 8 | 0.37 | 0.37 |
| E − D, or D − B | 0.3 | 0.2 | 3 | 23 | 0.19 | 0.20 |

**What this means**
1. On LLFF alone, interactions smaller than about 0.35–0.55 dB are below our detection limit. That is the same order as the gains in this literature (StereoGS★ +0.14, DropoutGS on CoR-GS +0.14).
2. **More scenes help much more than more seeds.** Seeds shrink only the σ part; scenes shrink both parts. DTU's 15 scenes roughly halve the MDE.
3. **"Redundant" can be claimed honestly only with an equivalence margin of at least about 0.35 dB on LLFF alone, or about 0.25 dB with DTU.** Otherwise the honest word is "inconclusive".
4. **E − D is our best-powered contrast**, and it is also the cleanest causal one. Make it primary (A4).

---

## 6. Theory: what dropout does to an alpha-compositing renderer

This is a candidate contribution: it costs about a day of writing, needs no compute, and turns the D/E design into tests of explicit predictions.

### 6.1 Setup

Along one pixel's ray, Gaussians $i = 1..n$ are sorted front to back, with per-pixel alpha $\alpha_i$ (opacity times the 2-D falloff), colour $c_i$ and depth $z_i$. The clean render is

$$C = \sum_i c_i\,\alpha_i\,T_i, \qquad T_i = \prod_{j<i}(1-\alpha_j),$$

and the depth render is the same with $z_i$ in place of $c_i$. DropGaussian draws i.i.d. masks $m_i \sim \mathrm{Bernoulli}(1-r)$ and compensates survivors. The 3DGS rasterizer clamps per-pixel alpha at $\alpha_{\max} = 0.99$ (`forward.cu`), so the effective alpha is

$$\tilde\alpha_i = \min\!\big(\alpha_{\max},\; m_i\,\alpha_i/(1-r)\big).$$

### 6.2 Results

**Proposition 1: compensated dropout is unbiased without the clamp.** If $\alpha_i/(1-r) \le \alpha_{\max}$ for every Gaussian on the ray, then $\mathbb{E}_m[\tilde C] = C$, and the same holds for depth.

*Proof.* $\mathbb{E}[\tilde\alpha_i] = \alpha_i$ and $\mathbb{E}[1-\tilde\alpha_j] = 1-\alpha_j$. Each term $c_i\,\tilde\alpha_i \prod_{j<i}(1-\tilde\alpha_j)$ is a product of independent factors, so its expectation is the product of their expectations. ∎

**Corollary: the clamp creates a downward opacity bias.** When $\alpha_i/(1-r) > \alpha_{\max}$, we get $\mathbb{E}[\tilde\alpha_i] = (1-r)\,\alpha_{\max} < \alpha_i$. This bites hardest in DNGaussian's **hard-depth pass**, where every Gaussian has opacity 0.95 at its centre:

| Drop rate r | 0.04 | 0.06 | 0.12 (γ = 0.2 at t = 6000) | 0.20 (γ = 1/3 at t = 6000) | 0.30 |
| --- | --- | --- | --- | --- | --- |
| Expected centre alpha (clean: 0.95) | 0.950 | 0.931 (−2 %) | 0.871 (−8 %) | 0.792 (−17 %) | 0.693 (−27 %) |

The clamp binds once r > 0.0404, which is from iteration 2020 at γ = 0.2 and from iteration 1212 at γ = 1/3. So in **D only**, the expected depth render is pulled towards farther surfaces, more so as the dose grows. E's depth passes see no mask and no bias. In the photometric pass, only Gaussians with opacity above 0.99 (1 − r) are affected.

**Proposition 2: dropout acts as the clean loss plus a variance penalty.** For a squared-error loss,

$$\mathbb{E}_m\,\lVert y-\tilde R\rVert^2 = \lVert y-\mathbb{E}_m\tilde R\rVert^2 + \textstyle\sum_{\text{pixels}} \mathrm{Var}_m(\tilde R).$$

With Proposition 1, $\mathbb{E}_m\tilde R = R$, the clean render. Training under masks therefore optimizes the clean loss plus a penalty on how much the render varies across masks; this is the classic view of dropout as adaptive regularization (Wager et al., 2013). For L1, D-SSIM and DNGaussian's patch-normalized depth loss the decomposition is not exact, but the Jensen gap plays the same role.

**Consequences for our variants**

| Variant | Photometric objective | Depth objective |
| --- | --- | --- |
| A, B | clean | B: clean |
| C | clean + **photometric-variance** penalty | — |
| D | clean + photometric-variance penalty | clean + **geometric-variance penalty** + clamp bias |
| E | clean + photometric-variance penalty | clean |

**So E − D isolates the effect of penalizing depth variance under masks, plus the clamp bias.** DNGaussian also normalizes each depth patch by that patch's own standard deviation (`utils/loss_utils.py`, `normalize`). That amplifies the penalty in flat, low-variance patches, where one dropped front Gaussian creates a large normalized error.

### 6.3 Neighbour compensation

DropAnSH-GS observes that overlapping neighbours make up for a dropped Gaussian, so the render varies less across masks and dropout's penalty shrinks. If depth supervision packs Gaussians onto surfaces (more overlap), **dropout is weaker inside a depth-regularized model**. That predicts sub-additivity (I < 0) even when depth supervision itself is unharmed, so I_E < 0 together with E − D ≈ 0.

### 6.4 Pre-registered hypotheses

Each hypothesis states a prediction and what would falsify it. Write them into the pre-registration (A7).

| ID | Hypothesis | Prediction | Falsified if | Measured by |
| --- | --- | --- | --- | --- |
| H1 | **Depth-signal corruption.** Masks corrupt the depth supervision in D | gap(D) > gap(C) ≥ 0; the depth-gradient cosine (masked vs clean) is below 1 and falls with dose; E − D > 0 and **grows with dose** | E ≈ D at every dose, or E − D does not grow with dose | `diag.csv` gap, C1, B1 |
| H2 | **Clamp bias** in the hard-depth pass | An offline Monte-Carlo test shows masked depth biased "far", tracking the table in §6.2; D-nc (mask without compensation in the depth passes) differs from D | Bias ≈ 0, or D ≈ D-nc | C4 |
| H3 | **Neighbour compensation** makes dropout weaker with depth on | B has smaller kNN distances and more high-opacity overlap than A; I_E < 0 even if E − D ≈ 0 | I_E ≈ 0 while redundancy differs between A and B | C3, C5a |
| H4 | **The variance penalty** is what dropout optimizes | `depth_var_masks`(D) < `depth_var_masks`(E) late in training; `rgb_var_masks` is lower in C, D and E than in A and B | No variance difference | `diag.csv`, C2 |
| H5 | **Region specificity** | D's excess error over E concentrates in flat-depth or low-texture regions and near depth edges | Excess error is spatially uniform | C5b |
| H6 | **Diminishing returns** | The magnitude of I and of C − A shrinks with more views. Dropout adds less on top of depth with better priors (B5) and with dense initialization (B2), consistent with StereoGS★ (+0.14 dB) | Gains grow with views or prior quality | B2, B3, B5 |

---

## 7. Recommendations

- **Priority.** **Must** = needed for a sound course paper. **Should** = needed for a publishable paper. **Could** = if time allows. **Future** = after 15 Nov.
- **Drift.** **0** sharpens the existing analysis. **1** asks the same question under a new condition. **2** is a new question (future work only).
- **Cost.** Person-days (pd) and T4 GPU-hours. One LLFF run is about 4.9 min on a T4.

### Group A: statistics and inference (do before any result exists)

| ID | Recommendation | Why | Cost | Priority | Drift |
| --- | --- | --- | --- | --- | --- |
| A1 | **A two-axis verdict** replaces the single-label rules. *Practical axis*: the combination vs each single, giving better than both / equivalent / worse / inconclusive. *Interaction axis*: super-additive / additive (shown equivalent to 0) / sub-additive / inconclusive. The named cells: "synergistic", "complementary and additive", "complementary but sub-additive" (partial redundancy), "redundant", "interfering" | Closes the T4 gap: every outcome gets exactly one label, and the fixture case above becomes "complementary but sub-additive" | 0.5 pd (Disha, stats.py) + 0.25 pd (PLAN §6) | Must | 0 |
| A2 | **Equivalence tests (TOST)** with a smallest effect size of interest fixed in advance. Proposal: δ = 0.25 dB PSNR, 0.010 SSIM, 0.010 LPIPS. "Redundant" and "additive" require equivalence, otherwise the verdict is "inconclusive" | Turns a null into a claim, or honestly into "inconclusive" | 0.25 pd | Must | 0 |
| A3 | **No data-driven best(B, C).** "Better than both" uses an intersection–union test: D − B > 0 **and** D − C > 0, each at 95 %, with no correction needed. "Worse than the single" uses B (the published host method) as the pre-specified reference, and C is reported alongside | Removes the winner's-curse bias (T5) | 0.25 pd | Must | 0 |
| A4 | **A hierarchy of hypotheses.** Primary: E − D, I_D and I_E at the main dose on LLFF 3-view, with Holm correction across the three. Secondary: dose slopes, views, initialization, DTU, prior quality. Exploratory: mechanism probes | Multiplicity control; makes E − D, the cleanest and best-powered contrast, primary | 0.1 pd | Must | 0 |
| A5 | **Robustness checks:** scene-bootstrap CIs and Wilcoxon signed-rank alongside the t-test; the interaction also on the MSE scale and on LPIPS. An interaction that flips sign across monotone transformations is "removable" (Loftus, 1978) | Guards against artefacts of the metric scale | 0.5 pd | Should | 0 |
| A6 | **Variance decomposition and an adaptive seed rule.** After seeds 0–1 and B9, estimate σ (within-seed and across seeds) and τ. If the projected MDE for I with 3 seeds exceeds 0.40 dB and 5 seeds would cut it by at least 0.05 dB, run seeds 3–4. The rule uses **variances only, never effect estimates** | Optimal use of compute without p-hacking | 0.25 pd | Should | 0 |
| A7 | **Pre-registration snapshot.** Tag `prereg-v1` with PLAN §6 v2 (Appendix A), §6.4's hypotheses and `stats.py`, **before the first Tier 1 metric exists**, and cite the tag in the paper | Credibility; reviewers reward it | 0.1 pd | Must | 0 |
| A8 | **Forest plots** as the main results figure: per-scene I_D, I_E and E − D with seed-level CIs and the pooled estimate | Shows heterogeneity honestly and reads well | 0.5 pd (Disha) | Should | 0 |

### Group B: design extensions (same question, more conditions)

| ID | Recommendation | Why | Cost | Priority | Drift |
| --- | --- | --- | --- | --- | --- |
| B1 | **Dose–response as a first-class factor.** Run C, D and E at γ ∈ {0.1, 0.2 (main), 1/3, 0.5}, i.e. peak rates {0.06, 0.12, 0.20, 0.30}, with A and B as anchors. Report the main table at γ = 0.2 (D1) **and** at γ = 1/3 (the published end rate). **Supersedes Tier 2b** and includes Disha's D11 change | Fixes T1. Doubles as a mechanism test: H1 predicts E − D grows with dose, H3 predicts I_E < 0 at every dose | 0.25 pd runner/config + 0.5 pd plots; 144 runs (2 seeds) ≈ 12 T4-h, or 6 T4-h with 1 seed | **Must** | 0 |
| B2 | **Initialization moderator.** Run A–E with DNGaussian's published MVS configuration (`--mvs_pcd`, `scripts/run_llff_mvs.sh`), using **DropGaussian's released dense point clouds**: the README links preprocessed clouds, and the path `3_views/dense/fused.ply` and the 3-view selection rule match ours, so there is no view leakage (verified 14 Oct). Archive them with an md5. Note: the MVS configuration also changes hyperparameters; frame it as "the two published DNGaussian configurations" | Fixes T2. Reconciles us with DropoutGS Tab. 3 (DNGaussian★ = 19.94) and with StereoGS's initialization-dependent interactions | 1 pd; 120 runs ≈ 12 T4-h | **Should** | 1 |
| B3 | **The views sweep includes A and runs on T4s** (Kaggle or Colab), not the laptop: A–E × {2, 6, 9} views × 8 scenes, with 3-view taken from Tier 1. Use 2 seeds if compute allows | I per view count (H6), with no GPU confound | 0.25 pd; 120 runs/seed ≈ 10 T4-h per seed | **Should** | 0 |
| B4 | **DTU 3-view** (15 scenes, object masks, the standard RegNeRF/DNGaussian split). DNGaussian ships `train_dtu.py`, `scripts/run_dtu.sh` and `metrics_dtu.py`. Generate DPT depth once with `dpt/get_depth_map_for_llff_dtu.py` and archive it with an md5. Port the flags and diagnostics to `train_dtu.py` | Generalization across scene type; scenes 8 → 23, roughly halving the MDE (§5.3) | ~2.5 pd (1 Disha data, 1 Aayush port, 0.5 analysis); 150–225 runs ≈ 13–19 T4-h | **Should** | 1 |
| B5 | **Depth-prior quality axis.** Canonical DPT vs Depth Anything V2 (stronger) vs degraded DPT (controlled blur or noise, deterministic) for B, D and E; A and C are unaffected. Each new depth set is generated once and md5-archived | Tests whether "dropout is insurance against bad priors" (H6); answers T14 | ~1.25 pd; 96 runs ≈ 8 T4-h | Could | 1 |
| B6 | **An interaction surface.** `--depth_scale` ∈ {0.5, 1, 2} × γ ∈ {0.1, 0.2, 1/3} for D and E, 1 seed, shown as a heatmap | A memorable figure; checks that conclusions aren't an artefact of one weighting (T10) | 0.35 pd; 96 runs ≈ 8 T4-h | Could | 0 |
| B7 | **A second dropout family:** DropoutGS-style random dropout regularization (no compensation, consistency loss against the full render) in our host, D and E style | Reconciles directly with DropoutGS Tab. 3; tests whether the dropout family matters | ~1 pd, but λ and p are not fully specified in the paper; 48 runs ≈ 4 T4-h | Could | 1 |
| B8 | **A second host:** vanilla 3DGS (DropGaussian's own codebase) with a depth loss, variants A–E | Answers "does this hold for a near state-of-the-art pipeline?" (the ~19 vs ~21–22 dB concern) | 3–5 pd | **Future** (or CVPR-only stretch) | 1 |
| B9 | **Noise-floor block on the T4:** B and D on fern and orchids, 4 same-seed repeats plus 4 seeds | Measures σ within and across seeds at full length (T13); feeds A6 | 0.1 pd; 32 runs ≈ 2.6 T4-h | **Must** | 0 |

### Group C: mechanism (log in Tier 1, so nothing needs re-running)

| ID | Recommendation | Why | Cost | Priority | Drift |
| --- | --- | --- | --- | --- | --- |
| C1 | **Depth-gradient fidelity**, as new `diag.csv` columns `depth_grad_cos` and `depth_grad_ratio`, every 100 iterations (Appendix B) | Direct evidence for H1: does the mask change *what the depth loss teaches*? | 0.3 pd; about 1 % of run time | **Must** | 0 |
| C2 | **`rgb_var_masks`**: the photometric analogue of `depth_var_masks`, with the same probe and generator | Tests H4 (the variance penalty) | 0.1 pd; < 1 % | **Must** | 0 |
| C3 | **End-of-training structure statistics** in `train_stats.json` (new keys only): opacity quantiles, scale quantiles, kNN-distance median (via `simple_knn.distCUDA2`), fraction of Gaussians with opacity > 0.9 | Redundancy proxies for H3; densification as a mediator (T9) | 0.2 pd | **Must** | 0 |
| C4 | **Clamp decomposition (H2).** Either variant `D_nc` (the mask reaches the depth passes **without** compensation), 16 runs, or an offline Monte-Carlo probe on seed-0 checkpoints (render K masks, compare mean depth with clean) | Separates "holes" from "clamp bias" inside D vs E (T7) | 0.25 pd; 1.3 T4-h | Should | 0 |
| C5 | **Offline probes**, in a separate script (`experiments/probes.py`) that never writes into `eval/`, so invariant 1 holds. (a) A test-time drop-robustness curve: PSNR vs fraction of Gaussians removed, with and without compensation, per variant. (b) Region-stratified error: eval error binned by local DPT-depth std (flat vs structured) and image gradient (edges) | H3 and H5; strong qualitative material | 1–1.5 pd; < 1 T4-h | Should | 0 |
| C6 | **Which depth pass carries the effect?** Variants `D_hard` and `D_soft`, where the mask reaches only that one pass | Localizes interference (pass-level causal test) | 0.2 pd; 32 runs ≈ 2.6 T4-h | Could | 0 |

### Group D: implementation decisions to settle before W2–W4 merge

| ID | Decision | Recommendation |
| --- | --- | --- |
| D-a | Diagnostics in Tier 1 | Fold C1–C3 into W4 and W5 now. Otherwise mechanism evidence later means re-running 120 runs |
| D-b | Checkpoints | Keep seed-0 checkpoints (already planned); C5's probes need them. All other seeds stay slim |
| D-c | **Naming new arms without breaking invariant 3.** The layout `outputs/llff_<n>v/<variant>/<scene>/seed<k>/` stays; new arms are **new variant IDs** (adding values is allowed) | `<base>[_g<γ×100>][_mvs][_dav2][_nc]`, e.g. `D_g033`, `B_mvs`, `E_dav2`, `D_nc`. DTU goes under a new prefix `outputs/dtu_3v/...` (an addition, not a rename). `stats.py` gets a factor table (variant → depth, dropout, γ, init, prior). `scripts/check.sh` step 9 must accept optional variant keys (a gate change) |
| D-d | Where runs happen | **One GPU type for everything inferential: T4** (Colab plus Kaggle). The laptop does development, smoke tests and short offline probes (C5) only. This retires the laptop's Tier 2 nights |
| D-e | RNG | Keep the global torch RNG for masks, as the official code does. D and E already consume it identically. Diagnostics keep their separate generator |
| D-f | Main-table dose | Report **both** γ = 0.2 (D1, the official formula) and γ = 1/3 (the published end rate). That ends the D1 debate with data instead of argument |

### Group E: evaluation and reporting

| ID | Recommendation | Priority |
| --- | --- | --- |
| E1 | Report PSNR, SSIM (3DGS; scikit-image in the appendix) and LPIPS (VGG), plus per-scene tables. Correct D2's rationale: per the verified CSV, DNGaussian's paper used scikit-image SSIM; the choice of 3DGS SSIM for the main table stands because only our own variants are compared | Must |
| E2 | Reported baselines go in a separate "context" table with columns for initialization and metric implementation, and **no inferential comparison** with our variants. Note in Related Work that several papers print superseded FSGS numbers | Must |
| E3 | Efficiency per variant from quiet T4 runs (already planned); add Gaussian count as a mediator | Must |
| E4 | Release artefacts: every `metrics.json` and `diag.csv`, configs, tags, md5s and the pre-registration tag, as a single archive linked from the paper | Should |
| E5 | Geometry: on DTU, depth error against GT depth where available. On LLFF, multi-view depth consistency (reprojection agreement between eval views) | Could |

### Group F: writing, framing and publication

| ID | Recommendation | Priority |
| --- | --- | --- |
| F1 | **Re-frame** (§4 positioning). Working title ideas: *"Shared Masks, Shared Blame? Depth Priors and Gaussian Dropout in Sparse-View 3DGS"* or *"When Do Depth Priors and Gaussian Dropout Help Each Other? A Controlled Study for Sparse-View 3D Gaussian Splatting"*. Contributions: (1) the factorial study (N runs, pre-registered); (2) mechanism and theory (§6); (3) a practical recipe, e.g. "decouple the depth passes" if E > D; (4) the open harness | Must |
| F2 | **Must-cite list:** DropoutGS (incl. Tab. 3), StereoGS, D²GS, DropAnSH-GS, UGOD, reliability-aware depth supervision, DGDO-GS, DNGaussian++, FSGS, CoR-GS, NexusGS, Binocular3DGS; dropout theory (Srivastava 2014; Wager 2013; Baldi & Sadowski 2013); and the closest analogy for regularizer interference, Li et al. (CVPR 2019) on the disharmony between dropout and batch norm | Must |
| F3 | **Venue strategy.** Course submission 10–11 Nov. CVPR 2027 (registration 10 Nov, paper 16 Nov AoE) **only if the gate after Tier 1 passes**: a clear effect, plus at least one moderator (B2 or B4), plus mechanism evidence. Otherwise TMLR (rolling) or CVPR 2027 workshops (typically Feb–Mar 2027), ICCV 2027 (usually early March) or BMVC 2027; check each date. 3DV 2027 and WACV 2027 deadlines have passed | Must (decide) |
| F4 | Write in the **CVPR LaTeX template** from day 1, if the course allows, so there is no reformatting at the end. Check the course's rules on reusing a submission elsewhere, and CVPR's dual-submission and anonymity rules (our public repo has names in it; don't link it in an anonymous submission) | Should |
| F5 | **Start writing now:** Method, Theory (§6), Experimental setup and Related work need no results | Must |

### What not to do
- Don't chase state-of-the-art PSNR, add diffusion or generative priors, or propose a new method in this submission (drift 2).
- Don't build more gate tooling unless something breaks.
- Don't run inferential blocks on the laptop GPU while T4 time is available.
- Don't change D1 again; B1 and D-f settle the dose question with data.
- Don't compare our variants with reported numbers inferentially.

---

## 8. Scope packages, effort and compute

Three coherent packages. Each includes the baseline work: W2–W9 (about 2–3 pd, Aayush) and writing (about 5–6 pd each).

| Package | Adds on top of Tier 1 (120 runs, 9.8 T4-h) | Extra runs | Extra T4-h | Extra person-days | What we can claim |
| --- | --- | --- | --- | --- | --- |
| **P0: course-solid** | A1–A4, A7; B9; B1 with 1 seed; B3 with 1 seed; C1–C3; E1–E3; F1, F2, F5 | ~225 | ~19 | ~3.5 | A sound answer for DNGaussian/LLFF at two doses, with interaction per view count and first mechanism evidence |
| **P1: publication core** *(recommended)* | P0 + B1 with 2 seeds + **B2** (initialization) + A5, A6, A8 + C4, C5 + §6 theory | ~430 (+80 if A6 triggers) | ~37 (+6.5) | ~8 | Reconciles with prior work; mechanism with theory; strong enough for TMLR or a workshop, borderline for CVPR |
| **P2: full** | P1 + **B4** (DTU) + B5 (prior quality) + C6; B6 or B7 if time | ~710–790 | ~61–67 | ~12 | Generality across datasets and priors, enough power to claim equivalence; a credible CVPR 2027 attempt |

**Compute supply.** One LLFF run is 4.9 min on a T4.
- Colab free, ×2 accounts: about 3–5 T4-hours per day each, when GPUs are granted.
- Kaggle, ×2 accounts: about 30 GPU-hours per week each. Check the current quota. Kaggle's T4 matches Colab's GPU type, so one-GPU-type-per-comparison still holds.

From about 17 Oct to 2 Nov that is roughly 150–250 T4-hours, so **every package fits on compute**. The binding constraint is person-days, especially Aayush's.

**Suggested split, to be finalized in the plan**
- **Aayush:**
  - W2–W9 with C1–C3;
  - runner support for new variant IDs;
  - initialization arm (B2);
  - DTU port of the flags (B4);
  - probes (C4, C5);
  - theory (§6) and the Method section.
- **Disha:**
  - §6 v2 statistics in `stats.py` (A1–A6): factor table, TOST, intersection–union test, bootstrap;
  - dose, forest and heatmap plots;
  - DTU data and its DPT depth (B4);
  - Kaggle operations;
  - Related Work.

**Gates to put in the plan**
| Gate | When | Test |
| --- | --- | --- |
| G1 | about 16–17 Oct | Code frozen, `v1.0-exp` tagged, pre-registration tagged |
| G2 | about 20–21 Oct | Seeds 0–1 and B9 done; apply A6 |
| G3 | about 26–27 Oct | Tier 1 complete: go or no-go for CVPR and P2 |
| G4 | about 3 Nov | Experiments freeze |
| G5 | 9–10 Nov | Submission-ready; CVPR registration if going |

---

## 9. Future research directions

These are after 15 Nov, or the "future work" paragraph of the paper.

1. **Mask-aware depth supervision.** Weight the depth loss per pixel by its variance across masks, keeping dropout's benefit while removing the harm H1/H4 predict. A natural method paper if E > D.
2. **Coupled schedules.** Anneal the depth weight as the drop rate rises, or the reverse; a curriculum view of two regularizers.
3. **Structured dropout × depth.** Anchor or neighbourhood dropout (DropAnSH-GS) and depth-guided dropout (D²GS) under the same factorial lens. Neighbour compensation predicts these interact differently with depth.
4. **Correcting the clamp.** Cap compensation so that α/(1−r) ≤ α_max, or use a transmittance-preserving compensation; then test whether D's deficit disappears. This tests Proposition 1's corollary.
5. **A second-order theory** of compensated dropout in 3DGS: a Wager-style expansion giving the implicit regularizer, and how patch-normalized depth losses reshape it.
6. **A regularizer-interaction atlas** for sparse-view 3DGS (depth, dropout, opacity decay, co-regularization, unpooling, diffusion priors) with one protocol and factorial designs. StereoGS's initialization-dependent interaction signs show the need; this would be a benchmark paper.
7. **State-of-the-art hosts and feed-forward models.** Repeat the D vs E test in StereoGS-like stereo-prior pipelines and in feed-forward sparse-view splatting.
8. **Uncertainty.** Does dropout training (D or E) give better-calibrated Monte-Carlo-dropout uncertainty for unseen views?

---

## 10. Decision sheet (edit this)

Mark each item **A** (accept), **M** (modify, with a note) or **R** (reject). The plan is generated from the accepted and modified items only.

**Statistics**
- [ ] A1 two-axis verdict
- [ ] A2 equivalence tests; δ = 0.25 dB / 0.010 / 0.010 → your values: ____
- [ ] A3 intersection–union test, B as the reference single
- [ ] A4 primary family = E − D, I_D, I_E (Holm)
- [ ] A5 robustness (bootstrap, Wilcoxon, MSE scale)
- [ ] A6 variance decomposition and adaptive seeds
- [ ] A7 pre-registration tag before the first Tier 1 metric
- [ ] A8 forest plots

**Design**
- [ ] B1 dose–response; seeds: 1 / 2
- [ ] B2 initialization moderator (dense clouds)
- [ ] B3 views with A, on T4; seeds: 1 / 2
- [ ] B4 DTU 3-view
- [ ] B5 depth-prior quality
- [ ] B6 interaction surface
- [ ] B7 DropoutGS-style dropout
- [ ] B8 second host (future or stretch)
- [ ] B9 noise floor

**Mechanism**
- [ ] C1 depth-gradient fidelity
- [ ] C2 `rgb_var_masks`
- [ ] C3 structure statistics
- [ ] C4 clamp decomposition: variant `D_nc` / offline probe
- [ ] C5 offline probes: (a) robustness curve, (b) regions
- [ ] C6 single-pass masking

**Implementation**
- [ ] D-a diagnostics in Tier 1
- [ ] D-b seed-0 checkpoints
- [ ] D-c variant-ID naming and the gate change to check step 9
- [ ] D-d T4 only for inferential runs
- [ ] D-e global RNG
- [ ] D-f main table at γ = 0.2 and γ = 1/3

**Reporting and writing**
- [ ] E1 metrics and the D2 rationale fix
- [ ] E2 context table
- [ ] E3 efficiency
- [ ] E4 artefact release
- [ ] E5 geometry metrics
- [ ] F1 re-framing
- [ ] F2 citations
- [ ] F3 venue strategy
- [ ] F4 CVPR template
- [ ] F5 start writing now

**Package:** ☐ P0 ☐ **P1** (recommended) ☐ P2 ☐ custom: ____

**Open questions**
1. **Venues.** Course only, course plus CVPR 2027 (gate G3), or course plus TMLR or a workshop afterwards?
2. **Course format.** Page limit, template, and is reuse for a conference submission allowed?
3. **Equivalence margins.** Are δ = 0.25 dB / 0.010 SSIM / 0.010 LPIPS acceptable to both of you?
4. **Kaggle.** Did Disha's Kaggle test run, and how many accounts can we use?
5. **Ownership.** Does the split in §8 work for Disha?
6. **Second host.** Is B8 future work, or worth 3–5 person-days for a CVPR attempt?
7. **Anything in §2** (what stays fixed) you want changed?

---

## 11. References

**Our methods**
- DNGaussian: Li et al., CVPR 2024. [arXiv 2403.06912](https://arxiv.org/abs/2403.06912) · [code](https://github.com/Fictionarry/DNGaussian)
- DropGaussian: Park, Ryu and Kim, CVPR 2025, pp. 21600–21609. [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Park_DropGaussian_Structural_Regularization_for_Sparse-view_Gaussian_Splatting_CVPR_2025_paper.html) · [arXiv 2504.00773](https://arxiv.org/abs/2504.00773) · [code](https://github.com/DCVL-3D/DropGaussian_release) (README links preprocessed sparse and dense clouds)

**Dropout-family and depth work, 2025–26**
- DropoutGS: Xu et al., CVPR 2025. [arXiv 2504.09491](https://arxiv.org/abs/2504.09491) (Table 3: plug-ins incl. DNGaussian★)
- UGOD: Guo et al. [arXiv 2508.04968](https://arxiv.org/abs/2508.04968)
- D²GS: Song et al. [arXiv 2510.08566](https://arxiv.org/abs/2510.08566)
- DropAnSH-GS: Fang et al., CVPR 2026. [arXiv 2602.20933](https://arxiv.org/abs/2602.20933)
- Fisher-information sparse-view 3DGS. [arXiv 2606.20842](https://arxiv.org/abs/2606.20842)
- StereoGS: Yuan, Ge and Cai. [arXiv 2606.30545](https://arxiv.org/abs/2606.30545) · [code](https://github.com/StringerYwh00/StereoGS)
- Reliability-aware monocular depth supervision. [arXiv 2607.02554](https://arxiv.org/abs/2607.02554)
- DGDO-GS. Applied Sciences 16(8):4026, 2026. [doi:10.3390/app16084026](https://doi.org/10.3390/app16084026)
- DP-GS, APSIPA ASC 2025. [PDF](https://apsipa.org/proceedings/2025/papers/APSIPA2025_P206.pdf) (text not verified)
- PairDropGS. [arXiv 2605.12072](https://arxiv.org/abs/2605.12072) (**withdrawn**)

**Context**
- FSGS (ECCV 2024), CoR-GS (ECCV 2024), NexusGS (CVPR 2025), Binocular3DGS (NeurIPS 2024), MVPGS (ECCV 2024); verified numbers in `results/reported_baselines.csv`
- DPT: Ranftl, Bochkovskiy and Koltun, ICCV 2021
- Depth Anything V2: Yang et al., NeurIPS 2024

**Dropout theory**
- Srivastava et al., "Dropout", JMLR 15, 2014
- Wager, Wang and Liang, "Dropout training as adaptive regularization", NeurIPS 2013
- Baldi and Sadowski, "Understanding dropout", NeurIPS 2013
- Li, Chen, Hu and Yang, "Understanding the disharmony between dropout and batch normalization by variance shift", CVPR 2019

**Statistics**
- Lakens, "Equivalence tests: a practical primer", *Social Psychological and Personality Science* 8(4), 2017
- Berger and Hsu, "Bioequivalence trials, intersection-union tests and equivalence confidence sets", *Statistical Science* 11(4), 1996
- Holm, *Scandinavian Journal of Statistics* 6(2), 1979
- Demšar, "Statistical comparisons of classifiers over multiple data sets", JMLR 7, 2006
- Bouthillier et al., "Accounting for variance in machine learning benchmarks", MLSys 2021
- Loftus, "On interpretation of interactions", *Memory & Cognition* 6, 1978

**Venues**
- [CVPR 2027 call for papers](https://cvpr.thecvf.com/Conferences/2027/CallForPapers): registration 10 Nov 2026, paper 16 Nov 2026, supplementary 23 Nov 2026 (AoE)
- [WACV 2027](https://wacv.thecvf.com/Conferences/2027/CallForPapers): round 2 closed 28 Aug 2026
- 3DV 2027: deadline about 28 Aug 2026 (passed)

---

## Appendix A: proposed §6 v2 analysis rules

Adopt by PR and tag as `prereg-v1` before any Tier 1 metric exists.

**Units.** For each variant and scene, average PSNR, SSIM and LPIPS over seeds. Contrasts are computed per scene and are paired by scene.

**Estimands at the main dose (γ = 0.2), LLFF 3-view.**
- Primary family, Holm-corrected:
  - E − D;
  - I_D = (D − C) − (B − A);
  - I_E = (E − C) − (B − A).
- Secondary:
  - B − A and C − A;
  - D − B, D − C, E − B and E − C;
  - the same contrasts at the other doses;
  - dose slopes;
  - per view count, initialization and dataset.

**Tests**
- 95 % paired t-intervals over scenes (df = n − 1), plus wins out of n.
- Equivalence: TOST at α = 0.05, i.e. the 90 % CI lies within ±δ. δ = 0.25 dB PSNR, 0.010 SSIM, 0.010 LPIPS.
- **Practical axis**, for X ∈ {D, E}:
  - *better than both*: the intersection–union test (X − B and X − C both have 95 % CI > 0);
  - *equivalent*: X − B equivalent within δ;
  - *worse*: X − B has 95 % CI < 0;
  - otherwise *inconclusive*.
- **Interaction axis:**
  - *super-additive*: I has 95 % CI > 0;
  - *additive*: I is equivalent to 0 within δ;
  - *sub-additive*: I has 95 % CI < 0;
  - otherwise *inconclusive*.
- **Named verdicts:**

  | Practical \ Interaction | Super-additive | Additive | Sub-additive | Inconclusive |
  | --- | --- | --- | --- | --- |
  | Better than both | synergistic | complementary (additive) | complementary but sub-additive | complementary |
  | Equivalent | — | — | redundant | redundant (interaction unclear) |
  | Worse | — | — | **interfering** | worse (interaction unclear) |
  | Inconclusive | — | — | partially redundant | inconclusive |

- SSIM and LPIPS must not contradict PSNR. If they do, report "metric-dependent".
- **Mechanism statement** (secondary): "interference runs through depth supervision" requires all of:
  - E − D > 0 (95 %);
  - gap(D) > gap(C) (95 %);
  - mean `depth_grad_cos` in D decreasing in dose (negative slope, 95 %);
  - E − D increasing in dose (positive slope, 95 %).

**Robustness:** scene-bootstrap CIs (10 000 resamples) and Wilcoxon signed-rank on the primary family; the interaction recomputed on MSE and LPIPS.

**Adaptive seeds:** as in A6, using variances only.

**Pooled analysis across datasets** (if B4 runs): a mixed model with fixed effects for depth × dropout × dataset and random intercepts for scene. The primary verdict stays per dataset.

**Power formulas** used in §5.3: with per-scene variance $v = c\,\sigma^2/k + \tau^2$ (c = 4 cells for I, 2 for a difference), $SE = \sqrt{v/n}$, $\mathrm{MDE} = (t_{.975,n-1} + t_{.80,n-1})\,SE$, and TOST bound $= (t_{.95,n-1} + t_{.90,n-1})\,SE$.

---

## Appendix B: proposed diagnostics, exact definitions

These are additions only, so the frozen formats (invariant 3) are respected.

**`diag.csv`, new columns (every 100 iterations, computed alongside W4's existing diagnostics, under the same rules: before densification, separate RNG):**
- `depth_grad_cos`: the cosine between $g_{mask} = \nabla_{xyz} L_{hard}(\text{render\_for\_depth}(\text{mask}))$ and $g_{clean} = \nabla_{xyz} L_{hard}(\text{render\_for\_depth}(\text{None}))$, flattened over all Gaussians, at the fixed diagnostic patch size.
  - In C, D and E, *mask* is this iteration's training mask. In E this is counterfactual: what D would have used.
  - In A and B it is a probe mask at rate 0.2 from the diagnostics generator.
  - Use `torch.autograd.grad` on `gaussians._xyz`, so neither the optimizer state nor the `.grad` buffers are touched.
- `depth_grad_ratio`: $\lVert g_{mask}\rVert / \lVert g_{clean}\rVert$.
- `rgb_var_masks`: like `depth_var_masks`, but on the photometric render: the per-pixel variance across K = 4 probe masks at rate 0.2, averaged over channels and pixels, at the same fixed view.

**`train_stats.json`, new keys:**
- `opacity_q10`, `opacity_q50`, `opacity_q90`: pre-dropout opacity of all Gaussians at the end (DNGaussian's opacity comes from its neural field, so evaluate it once).
- `frac_opacity_gt_0_9`.
- `scale_max_q50`, `scale_max_q90`: the largest axis of each Gaussian.
- `knn_dist_q50`: the median over Gaussians of $\sqrt{\text{distCUDA2}}$, the mean squared distance to the 3 nearest neighbours, from `simple_knn`.

**Offline probes (`experiments/probes.py`, never writes into `eval/`):**
- **Robustness curve.** For f ∈ {0, 0.05, 0.1, 0.2, 0.3}, remove a random fraction f of Gaussians, with and without compensation. Report eval PSNR, using 5 masks × 3 views per point, on seed-0 checkpoints.
- **Monte-Carlo depth bias.** The mean over 32 masks of the hard-pass depth render minus the clean hard-pass depth, at r = 0.12 and r = 0.2, reported as the median signed shift in normalized depth units.
- **Region-stratified error.** Bin eval pixels by the local std of DPT depth (11×11 window, terciles) and by image-gradient magnitude (terciles). Report per-bin MSE for B, D and E, and the D − E excess per bin.
