# CSD361 report: compliance table

Paper: `report/main.tex` (IEEEtran conference, two columns). Bibliography: `report/refs.bib`. Number trace: `report/NUMBERS.md`.

| Course requirement | Where it is met |
| --- | --- |
| Title: concise, descriptive | Title block |
| Authors with roll numbers, one shared affiliation, no emails | `\author` block: both names, roll numbers, Dept. of CSE, Shiv Nadar University, Delhi NCR, India |
| Course line "CSD361 – Introduction to Machine Learning" | Second line of the affiliation block |
| Abstract ≤ 300 words, self-contained: problem, objectives, datasets, ML techniques, key numbers, conclusions | Abstract (283 words) |
| Keywords (4–6) | `IEEEkeywords`: 6 |
| I. Introduction: problem, motivation, problem statement, objectives, approach, contributions | Sec. I: problem statement in bold; objectives O1–O5; approach; 4 contribution bullets |
| II. Related Work: base papers explained, other work, research gap, numeric citations | Sec. II: DNGaussian (base paper 1), DropGaussian (base paper 2), few-shot NeRF/3DGS, dropout family, depth priors, "Gap" paragraph |
| III (left open by the guidelines) | Sec. III, "Background and Problem Formulation": the compositing equation, the factorial contrasts, the verdict rules, the bias–variance identity |
| IV. Methodology: justified choices, workflow diagram, pseudocode, equations, configuration | Sec. IV: Fig. 1 (TikZ workflow), Algorithm 1 (one iteration with mask routing), Proposition 1 and the alpha clamp, compensation modes, the error-budget estimator, Table I (variants), Table II (hyperparameters per dataset) |
| V. Setup: dataset description and source, size, attributes, target, preprocessing, exploratory observations, split, language/libraries/environment, procedure, metric definitions | Sec. V: Table III (datasets, splits, resolution), sources (official releases and RegNeRF masks), preprocessing (COLMAP, DPT, masks, md5), exploratory observation (nondeterminism), Table IV (10 blocks, 1,169 runs, tags), environment (PyTorch 1.12.1, RTX A4000), metrics (PSNR, SSIM, LPIPS, MSE/RMSE, MVC@5%), statistics |
| VI. Results and Discussion: tables, figures, comparisons, interpretation, MSE/RMSE where used, limitations, challenges | Sec. VI: Table V (main results), Table VI (paired contrasts and pooled), Fig. 2 (forest), Fig. 3 and Table VII (error budget in MSE), noise and power, a limitations and challenges paragraph |
| VII. Ablation analysis | Sec. VII: Table VIII (pass, compensation, depth weight, dropout dose), Table IX (view count), test-time robustness |
| VIII. Conclusion and future work: what was done, findings, each objective met or not, limitations, extensions | Sec. VIII: one clause per objective O1–O5, main lesson, 3 extensions |
| Acknowledgment (optional) | Acknowledgment section |
| References in IEEE format; cited ⇔ listed | `IEEEtran.bst`; 29 cited = 29 in `refs.bib` (checked by script) |
| ≤ 8 pages including figures and references; references ≤ 1 page | **To confirm after compiling on Overleaf**: about 6.5–7.5 pages estimated from 4,250 words of body text, 9 tables, 4 figures and 29 references |
| Datasets not redistributed; official sources given | Sec. V names the official releases; the repo README gives the links |
