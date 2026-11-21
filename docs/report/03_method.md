# 3 Method

This section describes the code at tags `v1.0-exp`, `v1.1-exp` and `v1.2-exp`. The equations follow `docs/RESEARCH_ANALYSIS.md` §6.

## 3.1 Background: 3D Gaussian Splatting

3D Gaussian Splatting (3DGS) represents a scene as a set of anisotropic 3D Gaussians. Each Gaussian has a centre, a scale, a rotation, an opacity and a colour. To render a view, the Gaussians are projected to 2D, sorted by depth and alpha-composited front to back.

Take one pixel's ray, with Gaussians $i = 1..n$ sorted front to back. Each has a per-pixel alpha $\alpha_i$ (its opacity times the projected 2D falloff) and a colour $c_i$. The pixel colour is

$$C = \sum_i c_i\,\alpha_i\,T_i, \qquad T_i = \prod_{j<i}(1-\alpha_j).$$

Replacing $c_i$ with the Gaussian's view-space depth $z_i$ gives a rendered depth map. The rasterizer clamps every per-pixel alpha at $\alpha_{\max} = 0.99$.

With only a few training views, 3DGS overfits. It reproduces the training images with floating, view-specific Gaussians and wrong geometry, and novel views degrade. The two methods below attack this from different sides.

## 3.2 DNGaussian: depth regularization in two extra passes

DNGaussian (Li et al., CVPR 2024) supervises the rendered depth with a monocular depth estimate from DPT. Its loss is scale-invariant: depth and the DPT prior are compared after each random patch is normalized by its own mean and standard deviation, at both the local and the global level. On LLFF there is also a smoothness term from iteration 3000. Colour comes from a small neural field (hash-grid and spherical-harmonics encoders), and opacity is a per-Gaussian parameter.

Every training iteration runs **three separate rasterizations**, each with its own backward pass and optimizer step:

1. **Hard-depth pass.** Every Gaussian's opacity is fixed at 0.95, and gradients reach only the centres. This moves Gaussians onto the surface the prior suggests.
2. **Soft-depth pass** (after iteration 1000). The real opacities are used, while centres, scales and rotations are detached, so gradients reach only the opacities.
3. **Photometric pass.** The normal render, with L1 + λ·D-SSIM against the training image, plus DNGaussian's shape, scale and opacity penalties.

The depth losses never use the photometric render. One weight, `--depth_scale` (λ_d), multiplies both depth losses, and λ_d = 0 skips passes 1 and 2 entirely.

**DTU and Blender.** We use the same three passes, with DNGaussian's own trainers (`train_dtu.py`, and the non-spherical-harmonics path of `train_blender.py`). Two details differ from LLFF:
- DTU also has a background-alpha pass that pushes the render's alpha towards 0 where the training image is black. It runs in every variant and never sees the dropout mask.
- On Blender, the depth passes run every 10th iteration, as in the upstream code.

## 3.3 DropGaussian: Gaussian dropout

DropGaussian (Park et al., CVPR 2025) randomly removes Gaussians during training, so that the remaining ones have to explain the training views on their own. At each iteration it draws one mask $m_i \sim \mathrm{Bernoulli}(1-r)$ per Gaussian and scales the survivors' opacity by $1/(1-r)$:

```python
compensation = torch.nn.Dropout(p=r)(torch.ones(N))
opacity = opacity * compensation[:, None]
```

The drop rate grows linearly, $r_t = \gamma \cdot t / 10000$, with γ = 0.2. 10000 is DropGaussian's total iteration count.

**Our port.** We copy these lines verbatim into DNGaussian's renderer, as an optional per-Gaussian vector `drop_comp`. Its default is `None`, meaning no dropout. We keep the schedule unchanged, $r_t = \gamma\, t / 10000$, although we train for 6000 iterations, so at γ = 0.2 the rate peaks at $r = 0.12$.
- One mask is drawn per iteration.
- Dropout runs only inside the training loop: every rendering and evaluation path is called without a mask.
- With γ = 0 no mask is created, so the code reproduces DNGaussian exactly.
- The opacity penalty uses the opacity before dropout, so the regularizer itself is the same in every variant.
- `scripts/check.sh` checks the verbatim line and the schedule in all three trainers.

**Compensation modes** (`--drop_mode`, `train_llff.py` and `utils/dropout.py`; LLFF only):
- **`uniform`** (the default): DropGaussian verbatim, as above.
- **`nc`** (no compensation): the same random draw as `uniform`, but the survivors keep their opacity (×1 instead of ×1/(1−r)).
- **`clamp_aware`**: each Gaussian's rate is lowered to $r_i = \min(r,\, 1 - \alpha_i/0.99)$, which is 0 for $\alpha_i \ge 0.99$. This makes compensation unbiased in every pass (§3.5), at the price of a smaller effective dose for opaque Gaussians. `diag.csv` and `train_stats.json` record the effective rate.

## 3.4 Variants: where the mask acts

All variants are the same code with different flags:

| ID | λ_d | γ | Mask reaches | Role |
| --- | --- | --- | --- | --- |
| A | 0 | 0 | — | 3DGS baseline (no depth, no dropout) |
| B | 1 | 0 | — | DNGaussian as published (fixed reference) |
| C | 0 | 0.2 | photometric | DropGaussian alone |
| D | 1 | 0.2 | photometric, hard depth, soft depth | Naive combination: one mask for all three passes |
| E | 1 | 0.2 | photometric only | **Decoupled**: the depth passes render all Gaussians |
| D_hard | 1 | 0.2 | photometric, hard depth | Mask in the hard-depth pass only |
| D_soft | 1 | 0.2 | photometric, soft depth | Mask in the soft-depth pass only |

A–D form a 2×2 factorial (depth × dropout). The interaction terms are

$$I_D = (D - C) - (B - A), \qquad I_E = (E - C) - (B - A).$$

- $I < 0$ means the two regularizers add up to less than the sum of their separate effects.
- E − D measures what it costs, or gains, to let the mask reach the depth supervision.
- D_hard and D_soft split that difference between the two depth passes.

The further LLFF variants change one factor each:

| Variants | Change | Block |
| --- | --- | --- |
| C/D/E_g010, C/D/E_g033 | γ = 0.1 and 1/3 (peak rates 0.06 and 0.20) | G (dose) |
| C_nc, D_nc, C_ca, D_ca | `--drop_mode nc` / `clamp_aware` | COMP |
| B/D_ds050, B/D_ds200 | λ_d = 0.5 and 2 | DDOSE (depth dose) |
| B_r1–4, D_r1–4; A/B/C/D_r1–4 | same-seed repeats | N, NOISE2 (noise floor) |

DTU and Blender run A–E only.

## 3.5 Why the placement can matter: compensated dropout and the alpha clamp

**Proposition 1 (unbiasedness).** If $\alpha_i/(1-r) \le \alpha_{\max}$ for every Gaussian on a ray, the expected masked render equals the clean render, for colour and for depth.

*Proof.* With $\tilde\alpha_i = m_i\alpha_i/(1-r)$, $\mathbb{E}[\tilde\alpha_i] = \alpha_i$ and $\mathbb{E}[1-\tilde\alpha_j] = 1-\alpha_j$. Each term $c_i\tilde\alpha_i\prod_{j<i}(1-\tilde\alpha_j)$ is a product of independent factors, so its expectation is the product of the expectations. ∎

**The exception: the clamp.** The rasterizer uses $\tilde\alpha_i = \min(\alpha_{\max}, m_i\alpha_i/(1-r))$. When $\alpha_i/(1-r) > \alpha_{\max}$, the expectation becomes $\mathbb{E}[\tilde\alpha_i] = (1-r)\,\alpha_{\max} < \alpha_i$.

In DNGaussian's hard-depth pass, every Gaussian has opacity 0.95 at its centre. The clamp therefore binds as soon as $r > 0.0404$, which is from iteration 2020 at γ = 0.2:

| Drop rate r | 0.04 | 0.06 | 0.12 (γ = 0.2, t = 6000) | 0.20 (γ = 1/3, t = 6000) |
| --- | --- | --- | --- | --- |
| Expected centre alpha (clean 0.95) | 0.950 | 0.931 (−2 %) | 0.871 (−8 %) | 0.792 (−17 %) |

So in a fixed-opacity pass, compensated dropout cannot be unbiased above r ≈ 0.04. The lower expected opacity lets more light through to the Gaussians behind, which pulls the masked hard-pass depth **far**.

Only the passes that receive the mask are affected:
- the hard pass in D and D_hard;
- in the soft and photometric passes, only Gaussians with opacity above $0.99(1-r)$.

E's depth passes see no mask and no bias. Clamp-aware dropout (§3.3) removes the bias by lowering the rate wherever the clamp would bind.

**Proposition 2 (variance penalty).** For a squared-error loss,

$$\mathbb{E}_m\lVert y-\tilde R\rVert^2 = \lVert y-\mathbb{E}_m\tilde R\rVert^2 + \sum_{\text{pixels}}\mathrm{Var}_m(\tilde R).$$

Where Proposition 1 holds, the first term is the clean loss. Training under masks therefore adds a penalty on how much the render varies across masks (dropout as adaptive regularization; Wager et al., 2013). For L1, D-SSIM and the patch-normalized depth loss the decomposition is not exact, but the Jensen gap plays the same role.

| Variant | Photometric objective | Depth objective |
| --- | --- | --- |
| A, B | clean | B: clean |
| C | clean + photometric-variance penalty | — |
| D | clean + photometric-variance penalty | clean + **depth-variance penalty + clamp bias** |
| E | clean + photometric-variance penalty | clean |

So E − D isolates the depth-variance penalty together with the clamp bias.

## 3.6 The error budget (N1)

To explain *why* the two regularizers combine as they do, we split each variant's test error into a systematic part and a run-to-run part (`experiments/bias_variance.py`). For each scene and variant we take the eval renders $r_k$ of the $n = 3$ seeds and the ground truth $y$. Per pixel and channel:

$$\bar m = \tfrac1n\textstyle\sum_k r_k,\quad s^2 = \tfrac{1}{n-1}\sum_k (r_k-\bar m)^2,\quad \widehat{b^2} = (\bar m - y)^2 - s^2/n,\quad \widehat{\mathrm{var}} = s^2.$$

The mean single-run MSE is then exactly $\mathrm{MSE} = \widehat{b^2} + \widehat{\mathrm{var}}$; the script checks this identity. All terms are averaged over pixels, channels and views. On DTU only the pixels inside the evaluation mask count. Contrasts (B − A, C − A, D − B, the interactions) are computed per scene for MSE, bias² and variance.

"Variance" here is the spread between training runs on fixed views, which comes from the seed and the rasterizer's nondeterminism. It is not variance over different sets of training views. "Bias" includes error that no regularizer can remove, such as regions no training view sees.

## 3.7 Hypotheses

These were fixed in tags before the data they concern existed. §4.6 gives the decision rules.

**`prereg-v1`** (LLFF):
- **H1, depth-signal corruption.** The mask perturbs the depth loss in D (its masked-minus-clean gap exceeds C's). If this harms training, E > D, and E − D grows with γ.
- **H2, clamp bias.** The masked hard-pass depth is biased far, more at r = 0.2 than at r = 0.12. If D < E, the deficit sits in D_hard: D_hard − E ≈ D − E and D_soft − E ≈ 0.
- **H3, neighbour compensation.** Depth supervision packs Gaussians closer, so the model is more redundant and dropout has less to do: $I_E < 0$ even if E − D ≈ 0.
- **H4, variance penalty.** The variance of the photometric render across masks is lower in C, D and E than in A and B.
- **H6, diminishing returns.** The effects and the interaction shrink from 2 to 9 training views.

**`prereg-v2`** (H7): does DropGaussian's benefit need its biased full-dose dropout of opaque Gaussians? Tested with C_ca and D_ca, read against the effective dose.

**`prereg-v3`** (DTU, Blender, error budget):
- **X4:** the LLFF contrasts replicate (same sign, overlapping CIs).
- **X5:** D − B is larger on DTU than on LLFF.
- **H-PD:** D − B ranks LLFF ≤ DTU ≤ Blender.
- **BV1:** most of depth's gain (B − A) is a bias reduction.
- **BV2:** most of dropout's gain (C − A) is a variance reduction.
- **BV3:** whether D's MSE falls below B's bias². A value below 0 would mean dropout also removes bias.
- **BV4:** depth's bias reduction fades as views grow.
- **BV5:** the interaction splits into bias and variance parts.
- **BV6:** BV1–BV2 hold on every dataset.
- Plus the scale robustness of the interaction (N4), the depth dose (N5), the noise floor and power (G6), and the compensation contrasts.
