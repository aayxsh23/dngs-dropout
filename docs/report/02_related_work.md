# 2 Related work

**[Disha: the numbers quoted from other papers below come from our DS-07 notes; check each against the paper before submission.]**

## 2.1 Sparse-view novel view synthesis

NeRF [1] needs dense views. Few-shot NeRF variants regularize it in different ways:
- **RegNeRF** [10] regularizes the geometry and appearance of patches rendered from unobserved views.
- **FreeNeRF** [8] shows that limiting the frequency of the positional encoding early in training is enough to suppress few-shot artifacts.
- **SparseNeRF** [9] distils the *ordering* of depths from a monocular estimate, rather than their values.

3DGS [2] is fast, but it overfits sparse inputs even more readily, because each Gaussian is free to fit one view. **FSGS** [7] grows Gaussians by proximity-guided unpooling and supervises depth with a monocular prior (20.31 dB on LLFF 3-view in its ECCV version). **CoR-GS** [4] trains two Gaussian fields and penalizes their disagreement, which suppresses the errors on which independently trained models differ.

## 2.2 Depth priors for sparse-view 3DGS

**DNGaussian** [3] is our depth baseline. It renders depth in two extra passes, a hard pass that moves Gaussian centres and a soft pass that tunes opacities, and compares them with a DPT [12] estimate through a patch-normalized, scale-invariant loss (§3.2). **DNGaussian++** [11] extends it. Many later methods use monocular or stereo depth in a similar role; they differ in how they align the prior's scale and how far they trust it.

All of these treat the depth prior as helpful by default. Our Blender results (§5.3) show a regime where it is not shown to help: at 8 surrounding views of synthetic objects, B − A is −0.80 dB [−1.78, +0.18].

## 2.3 Dropout for Gaussians

**DropGaussian** [5] drops each Gaussian with probability r during training and rescales the survivors' opacity by 1/(1 − r), the standard inverted-dropout compensation. It needs no extra network or prior and reports gains on LLFF, DTU and Blender. **DropoutGS** [6] proposes a related random-dropout regularizer with an edge-guided splitting strategy.

Several papers already combine dropout with a depth-regularized baseline:
- DropoutGS reports adding its dropout to DNGaussian (with MVS initialization) on LLFF 3-view: 19.94 → 20.64 dB.
- A study of co-adaptation in sparse-view 3DGS [13] adds dropout to DNGaussian: LLFF 18.93 → 19.43 dB, DTU 18.91 → 19.86 dB.
- DropAnSH-GS [14] reports DNGaussian 19.12 → 19.71 dB.

So the fact that dropout helps a depth-regularized model is not new. Each of these papers reports one combined variant against one baseline. None runs the full factorial, so none can estimate the interaction. None varies where the mask acts or how strong the depth prior is, and none asks which error component each method removes.

## 2.4 Why dropout works

Dropout was introduced for neural networks to prevent co-adaptation of units [15]. It has two complementary readings:
- **As a regularizer:** Wager et al. [16] show that, for generalized linear models, dropout is approximately an adaptive L2 penalty.
- **As an ensemble:** Baldi and Sadowski [17] show that the deterministic network with rescaled weights approximates the geometric mean of the exponentially many sub-networks.

The ensemble reading predicts our finding that dropout mainly reduces run-to-run variance (§5.5). Dropout can also interact badly with other components. Li et al. [18] show that dropout followed by batch normalization causes a *variance shift* between training and test. That is the same kind of train/test mismatch that the opacity clamp could cause under compensated dropout (H7, §3.5), which we test and do not find.

## 2.5 Positioning

| | Combined variant | Full factorial (interaction) | Where the mask acts | Depth dose | Error budget | Datasets |
| --- | --- | --- | --- | --- | --- | --- |
| DropoutGS [6] | ✓ | | | | | LLFF, DTU, Blender |
| Co-adaptation study [13] | ✓ | | | | | LLFF, DTU, Blender |
| DropAnSH-GS [14] | ✓ | | | | | LLFF, DTU, Blender |
| **Ours** | ✓ | ✓ | ✓ (D, E, D_hard, D_soft) | ✓ | ✓ | LLFF, DTU, Blender |

**[Disha: confirm the dataset column for [6], [13] and [14].]**

Our contribution is therefore not that the combination helps. It is a pre-registered, controlled measurement of *how* the two regularizers interact, *when* they stop being complementary, and *why*.

## References

[1] B. Mildenhall et al. NeRF: Representing scenes as neural radiance fields for view synthesis. ECCV 2020.

[2] B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis. 3D Gaussian Splatting for real-time radiance field rendering. ACM TOG (SIGGRAPH) 2023.

[3] J. Li et al. DNGaussian: Optimizing sparse-view 3D Gaussian radiance fields with global-local depth normalization. CVPR 2024.

[4] J. Zhang et al. CoR-GS: Sparse-view 3D Gaussian Splatting via co-regularization. ECCV 2024.

[5] H. Park et al. DropGaussian: Structural regularization for sparse-view Gaussian Splatting. CVPR 2025.

[6] Y. Xu et al. DropoutGS: Dropping out Gaussians for better sparse-view rendering. CVPR 2025.

[7] Z. Zhu et al. FSGS: Real-time few-shot view synthesis using Gaussian Splatting. ECCV 2024.

[8] J. Yang, M. Pavone, Y. Wang. FreeNeRF: Improving few-shot neural rendering with free frequency regularization. CVPR 2023.

[9] G. Wang et al. SparseNeRF: Distilling depth ranking for few-shot novel view synthesis. ICCV 2023.

[10] M. Niemeyer et al. RegNeRF: Regularizing neural radiance fields for view synthesis from sparse inputs. CVPR 2022.

[11] DNGaussian++ (extended version of [3]). IEEE TPAMI 2026.

[12] R. Ranftl, A. Bochkovskiy, V. Koltun. Vision transformers for dense prediction (DPT). ICCV 2021.

[13] Co-adaptation in sparse-view 3D Gaussian Splatting. arXiv:2508.12720, 2025.

[14] DropAnSH-GS. CVPR 2026, arXiv:2602.20933.

[15] N. Srivastava et al. Dropout: A simple way to prevent neural networks from overfitting. JMLR 2014.

[16] S. Wager, S. Wang, P. Liang. Dropout training as adaptive regularization. NeurIPS 2013.

[17] P. Baldi, P. Sadowski. Understanding dropout. NeurIPS 2013.

[18] X. Li, S. Chen, X. Hu, J. Yang. Understanding the disharmony between dropout and batch normalization by variance shift. CVPR 2019.

**[Disha: fill in the authors of [11], [13] and [14], and check the first authors of [4] and [6].]**
