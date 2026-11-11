# 3 Method (draft, AY-16)

Draft of report section 3 (PLAN_V2 §9). Numbers and equations follow RESEARCH_ANALYSIS.md §6 and the code at tag `v1.0-exp`. Figure F1 is referenced but not drawn yet (AY-24).

## 3.1 Background: 3D Gaussian Splatting

3D Gaussian Splatting (3DGS) represents a scene as a set of anisotropic 3D Gaussians. Each Gaussian has a centre, a scale, a rotation, an opacity and a colour. To render a view, the Gaussians are projected to 2D, sorted by depth and alpha-composited front to back. Along one pixel's ray, with Gaussians $i = 1..n$ sorted front to back, per-pixel alpha $\alpha_i$ (opacity times the projected 2D falloff) and colour $c_i$, the pixel colour is

$$C = \sum_i c_i\,\alpha_i\,T_i, \qquad T_i = \prod_{j<i}(1-\alpha_j).$$

Replacing $c_i$ by the Gaussian's view-space depth $z_i$ gives a rendered depth map. The rasterizer clamps every per-pixel alpha at $\alpha_{\max} = 0.99$.

With only a few training views, 3DGS overfits: it reproduces the training images with floating, view-specific Gaussians and wrong geometry, and novel views degrade. The two methods below attack this from different sides.

## 3.2 DNGaussian: depth regularization in two extra passes

DNGaussian (Li et al., CVPR 2024) supervises the rendered depth with a monocular depth estimate from DPT. Its loss is scale-invariant: depth and the DPT prior are compared after normalizing each random patch (5–17 px) by its own mean and standard deviation, at the local and at the global level, with a smoothness term from iteration 3000. Colour comes from a small neural field (hash-grid and spherical-harmonics encoders); opacity is a per-Gaussian parameter.

Every training iteration runs **three separate rasterizations**, each with its own backward pass and optimizer step:

1. **Hard-depth pass.** Every Gaussian's opacity is fixed at 0.95, and gradients reach only the centres. This moves Gaussians onto the surface the prior suggests.
2. **Soft-depth pass** (from iteration 1000). The real opacities are used, while centres, scales and rotations are detached, so gradients reach only the opacities. This removes Gaussians that are in front of the surface.
3. **Photometric pass.** The normal render, with L1 + 0.2 · D-SSIM against the training image, plus DNGaussian's shape, scale and opacity penalties.

The depth losses never use the photometric render. One weight, `--depth_scale` (λ_d), scales both depth losses; λ_d = 0 skips passes 1 and 2.

## 3.3 DropGaussian: Gaussian dropout

DropGaussian (Park et al., CVPR 2025) randomly removes Gaussians during training, so that the remaining ones must explain the training views on their own. At each iteration it draws one mask $m_i \sim \mathrm{Bernoulli}(1-r)$ per Gaussian and scales the survivors' opacity by $1/(1-r)$:

```python
compensation = torch.nn.Dropout(p=r)(torch.ones(N))
opacity = opacity * compensation[:, None]
```

The drop rate grows linearly, $r_t = \gamma \cdot t / 10000$, with γ = 0.2. 10000 is DropGaussian's total iteration count.

**Our port.** We copy these lines verbatim into DNGaussian's renderer as an optional per-Gaussian vector `drop_comp` (default `None`, meaning no dropout) and keep the schedule unchanged, $r_t = \gamma\, t / 10000$, although we train for 6000 iterations (decision D1). At γ = 0.2 the rate therefore peaks at $r = 0.12$. One mask is drawn per iteration. Dropout runs only inside the training loop: every rendering and evaluation path is called without a mask. With γ = 0 no mask is created, so the code reproduces DNGaussian exactly. The opacity penalty uses the opacity before dropout (D7), so the regularizer itself is identical across variants.

## 3.4 Variants: where the mask acts

All variants are the same code with different flags (Figure F1):

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

$I < 0$ means the two regularizers add up to less than the sum of their separate effects. E − D measures what it costs, or gains, to let the mask reach the depth supervision. D_hard and D_soft split that difference between the two depth passes. Block G repeats C, D and E at γ = 0.1 and γ = 1/3 (peak rates 0.06 and 0.20).

## 3.5 Why the placement can matter: compensated dropout and the alpha clamp

**Proposition 1 (unbiasedness).** If $\alpha_i/(1-r) \le \alpha_{\max}$ for every Gaussian on a ray, the expected masked render equals the clean render, for colour and for depth.

*Proof.* With $\tilde\alpha_i = m_i\alpha_i/(1-r)$, $\mathbb{E}[\tilde\alpha_i] = \alpha_i$ and $\mathbb{E}[1-\tilde\alpha_j] = 1-\alpha_j$. Each term $c_i\tilde\alpha_i\prod_{j<i}(1-\tilde\alpha_j)$ is a product of independent factors, so its expectation is the product of the expectations. ∎

**The exception: the clamp.** The rasterizer uses $\tilde\alpha_i = \min(\alpha_{\max}, m_i\alpha_i/(1-r))$. When $\alpha_i/(1-r) > \alpha_{\max}$, $\mathbb{E}[\tilde\alpha_i] = (1-r)\,\alpha_{\max} < \alpha_i$. In DNGaussian's hard-depth pass every Gaussian has opacity 0.95 at its centre, so the clamp binds as soon as $r > 0.0404$ (from iteration 2020 at γ = 0.2):

| Drop rate r | 0.04 | 0.06 | 0.12 (γ = 0.2, t = 6000) | 0.20 (γ = 1/3, t = 6000) |
| --- | --- | --- | --- | --- |
| Expected centre alpha (clean 0.95) | 0.950 | 0.931 (−2 %) | 0.871 (−8 %) | 0.792 (−17 %) |

In a fixed-opacity pass, compensated dropout therefore cannot be unbiased above r ≈ 0.04; as far as we know this has not been pointed out before. Lower expected opacity lets more light through to Gaussians behind, so the masked hard-pass depth is pulled **far**. Only the passes that receive the mask are affected: the hard pass in D and D_hard, and, in the soft pass and the photometric pass, only Gaussians with opacity above $0.99(1-r)$. E's depth passes see no mask and no bias.

**Proposition 2 (variance penalty).** For a squared-error loss, $\mathbb{E}_m\lVert y-\tilde R\rVert^2 = \lVert y-\mathbb{E}_m\tilde R\rVert^2 + \sum_{\text{pixels}}\mathrm{Var}_m(\tilde R)$. Where Proposition 1 holds, the first term is the clean loss, so training under masks adds a penalty on how much the render varies across masks (dropout as adaptive regularization; Wager et al., 2013). For L1, D-SSIM and DNGaussian's patch-normalized depth loss the decomposition is not exact, but the Jensen gap plays the same role.

| Variant | Photometric objective | Depth objective |
| --- | --- | --- |
| A, B | clean | B: clean |
| C | clean + photometric-variance penalty | — |
| D | clean + photometric-variance penalty | clean + **depth-variance penalty + clamp bias** |
| E | clean + photometric-variance penalty | clean |

So E − D isolates the depth-variance penalty together with the clamp bias. The patch normalization amplifies the penalty in flat, low-variance patches, where one dropped front Gaussian produces a large normalized error.

## 3.6 Hypotheses

These were fixed before any result, in tag `prereg-v1` (PLAN_V2 §2.3 and §5).

- **H1, depth-signal corruption.** The mask perturbs the depth loss in D (its masked-minus-clean gap exceeds C's). If this harms training, E > D, and E − D grows with γ.
- **H2, clamp bias.** The Monte-Carlo probe shows the masked hard-pass depth biased far, more at r = 0.2 than at r = 0.12. If D < E, the deficit sits in D_hard: D_hard − E ≈ D − E and D_soft − E ≈ 0.
- **H3, neighbour compensation.** Depth supervision packs Gaussians closer (smaller kNN distances in B than in A), so the model is more redundant and dropout has less to do: $I_E < 0$ even if E − D ≈ 0.
- **H4, variance penalty.** The variance of the photometric render across masks is lower in C, D and E than in A and B, and the depth variance is lower in D than in E.
- **H6, diminishing returns.** The effects and the interaction shrink from 2 to 9 training views.
