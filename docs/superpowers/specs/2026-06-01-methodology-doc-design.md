# Design — Step-by-step methodology document (`notebooks/info.md`)

**Date:** 2026-06-01
**Status:** Approved (design), pending spec review
**Deliverable language:** French · **Spec language:** English

## 1. Goal

Replace the current `notebooks/info.md` — a garbled, truncated Claude chat transcript — with a clean, readable, pedagogical **step-by-step methodology walkthrough** of the cyclops-vs-meniscus statistical pipeline, in French, GitHub-renderable, consistent in numbers and framing with `paper/manuscript.md`.

The document must make explicit, for **each step of the pipeline**: what we want to fix/verify, **why this method and not another**, how it is computed **on our actual dataset** (theory + a worked numeric example), the result, its interpretation, what it triggers next, and **which notebook to open**.

### The *fil rouge* (the spine the whole document threads)

> ① We first check whether the two groups are comparable — using **SMD**, **Cliff's δ**, and **TOST**.
> ② The answer is **NO**: sex and age are imbalanced (though the *baseline cartilage state* is equivalent).
> ③ Therefore we **adapt the method** — a co-primary **sex+age-adjusted** analysis (Firth-penalised) — to verify the patellofemoral (PF) hypothesis **despite** the imbalance.

This three-beat arc is stated once up front (callout), annotated on the pipeline diagram, and referenced in the "what it triggers next" slot of the relevant sections (01 finds imbalance → 04 adapts → verified).

## 2. Deliverable

- **File:** `notebooks/info.md` (rewrite in place).
- **Git:** commit the `info.ipynb` → `info.md` swap — delete the stale committed `notebooks/info.ipynb` (already gone from disk), add the new `info.md`.
- **Format:** GitHub-renderable Markdown — Mermaid diagrams, `$$LaTeX$$` math, GitHub callouts (`> [!NOTE]` etc.), Markdown tables, embedded PNG figures (`../figures/…` paths, since the file lives in `notebooks/`).
- **Tone:** pedagogical French, tutoiement (« tu »), concrete analogies preserved/cleaned from the transcript (duels, rigged race, "demi-patient").
- **Audience:** the project owner learning the methodology concretely (not a journal reviewer — that's the manuscript's job).

## 3. Document structure

### 3.1 Header
1. **Titre:** *Méthodologie pas à pas — Cyclops vs Ménisque : du contrôle d'équilibre à la vérification de l'hypothèse fémoro-patellaire*
2. Short intro paragraph: what the study asks; what this document is and how to read it.
3. **Callout `> [!IMPORTANT]` — « Le fil rouge »**: the three-beat arc above.
4. **Mermaid pipeline diagram** 00→06, annotating which steps *check similarity*, which *adapts*, which *verifies*.
5. **Table of contents.**
6. **Mini-glossaire de notation** (table): S1/S2, Δ = S2−S1, PF = {trochlée, rotule}, FT = {pte, pti, cfe, cfi}, SMD, MWU/U, TOST, Cliff δ, Firth, OR, E-value, δ̄, δ_c, η, LOO/ELPD.

### 3.2 Per-notebook sections — the 7-point template

Each of the seven notebook sections (00…06) uses **exactly** this template:

1. **But — ce qu'on veut fixer / vérifier.** The question this step answers.
2. **Pourquoi cette méthode et pas une autre.** The explicit method justification.
3. **Comment on le calcule sur NOS données.** Theory as a `$$` formula **plus a worked numeric example on the real data**.
4. **Résultat.** The canonical numbers (from §5 below).
5. **Interprétation / analyse.** What it means.
6. **Ce que ça déclenche ensuite.** Link forward (the *fil rouge*).
7. **Notebook à consulter.** `notebooks/0X_*.ipynb`, the relevant figure(s), and the `results.json` keys.

### 3.3 Per-notebook content map

| § | Notebook | Rôle dans le fil rouge | Méthodes (point 2: pourquoi) | Exemple chiffré travaillé (point 3) | Figures |
|---|---|---|---|---|---|
| 00 | `00_eda.ipynb` | Regarder avant de tester | Descriptif d'abord ; pourquoi pas tester tout de suite ; rareté grade 3 (2× en tout) → collapse {0,1,≥2} + binarisation FT | distribution des scores, anomalies de dates, n par groupe | — |
| 01 | `01_baseline_balance.ipynb` | **« Les groupes sont-ils similaires ? »** | SMD **pas** p-value (puissance à n=20) ; Love plot ; TOST **pas** « t-test non-signif » ; Cliff δ par covariable/compartiment (tests 3 & 4) | SMD = (x̄_cyc−x̄_men)/SD_pooled ; TOST ⟺ IC 90 % dans la boîte ±0.29 → p=0.034 ; **constat : NON équilibré (sexe, âge) mais cartilage équivalent** | fig1, figS1 |
| 02 | `02_progression_total.ipynb` | L'effet PF existe-t-il ? | Cliff δ / rangs **pas** moyennes (ordinal, n petit, ex-aequo) ; permutation exacte ; BCa + IC d'inversion (BCa instable à n_men=20) ; M1 beta-binomial | **545/21/414 duels sur 980 → δ=(545−21)/980=+0.5347** ; prob. supériorité 0.767 ; 57 % vs 5 % | fig2 |
| 03 | `03_progression_sites.ipynb` | Disclosure + dilution | tableau 6 compartiments ; BH-FDR q=0.10 ; PF vs FT apparié (Wilcoxon) | patella/trochlée signif. cas ; **PTI/CFE inversés (contrôles)** ; somme-6 diluée δ=+0.204, p=0.156 | fig3, fig4 |
| 04 | `04_risk_factors.ipynb` | **CLIMAX — on adapte** (réponse au déséquilibre de 01) | Firth **pas** logistique classique (quasi-séparation, 1/20) ; ajusté sexe+âge **co-primaire** ; E-value ; sensibilités (IMC, anomalies, pivot/travail H3) | **2×2 [28/21 ; 1/19] ; Firth +½ → (28.5·19.5)/(21.5·1.5) ≈ OR 17.2** ; ajusté 13.5 ; E-value 7.77 | fig9 |
| 05 | `05_hierarchical_bayes.ipynb` | Confirmation **non-circulaire** | estimand primaire **δ̄ invariant par partition** (anti-sélection) ; contraste PF **dérivé** ; règle 1-côté pour δ̄ / 2-côtés post-hoc ; LOO teste la partition ; pourquoi exchangeable | η = β_c·t + γ·g_i + δ_c·t·g_i + u_i ; δ̄=(1/6)Σδ_c=+0.233 (P=0.66, **non concluant = dilution**) ; contraste PF +2.28 (P=0.997) | fig5, fig6 |
| 06 | `06_temporal.ipynb` | Le délai = fenêtre d'observation, **pas** médiateur | DAG : délai en aval du groupe → over-adjustment exclu ; Weibull AFT (M4) + LogNormal (M5) ; ρ intra-cas | 240 vs 528 j ; ρ=−0.03 (p=0.82) → effet **malgré** ~½ exposition | fig7 |

### 3.4 Closing
- **Synthèse — le fil rouge récapitulé.** The symmetric Cliff-δ table: baseline PF δ ≈ 0 (CI *contains* 0 → équilibré) vs effect PF δ = +0.53, CI [0.37, 0.68] (*excludes* 0 → effet réel). Same yardstick, neutral start → sharp finish.
- **Prochaine étape.** Exploratory status (H2 refuted), prospective replication, what remains to confirm; pointer to the conservative δ̄ as the decision anchor.
- **Renvoi** to `paper/manuscript.md` and `results/results.json`.

## 4. Visual & format conventions

- **Mermaid:** (a) the 00→06 pipeline with the fil rouge annotated; (b) optionally a small "similarité → non → adapte → vérifie" logic diagram in the header. Quote all node labels; keep edge labels ASCII-simple (lesson from the manuscript pass).
- **LaTeX `$$`:** SMD, U/MWU, Cliff δ (and δ = 2U/(n₁n₂)−1), TOST-as-90%-CI, logistic logit for the adjusted OR, Firth +½, BCa (z0, a), η linear predictor, δ̄, PF contrast.
- **Callouts:** `[!IMPORTANT]` fil rouge; `[!WARNING]` anti-HARKing / quasi-separation / "absence de preuve ≠ preuve d'absence"; `[!TIP]` reading rules (SMD thresholds, decision rule); `[!NOTE]` worked-example boxes.
- **Figures embedded:** fig1, figS1 (§01); fig2 (§02); fig3, fig4 (§03); fig9 (§04); fig5, fig6 (§05); fig7 (§06) — via `![…](../figures/….png)`.
- **Austin SMD thresholds:** |SMD|<0.10 négligeable ; ≥0.25 déséquilibre notable.

## 5. Canonical numbers (single source of truth — from `results/results.json`, 49/20 cohort)

All numbers in the document MUST match these. The transcript's stale figures are corrected: **50/19 → 49/20**, crude OR **16.3 → 17.2**, baseline PF SMD **−0.040 → −0.009**.

- Cohort: **49 cyclops + 20 meniscus = 69**; analysable PF = 69.
- Baseline: age SMD **+0.484** (median 30.9 vs 24.9); sex SMD **+0.37** (`table1.csv`/manuscript); BMI SMD ≈ −0.08; baseline S1 overall MWU p=0.662; **baseline PF**: MWU p=0.818, SMD −0.009, **TOST p=0.034** (bound 0.292), equivalent = True.
- PF effect: Cliff **δ=+0.5347** (large); MWU p=0.0001; permutation p=0.0002; **BCa [0.367, 0.684]**; inversion CI [0.357, 0.675]; prob. superiority 0.767; PF worsening **28/49 (57.1%) vs 1/20 (5.0%)**.
- M1 beta-binomial: cyclops 0.569 [0.438, 0.695]; meniscus 0.091 [0.013, 0.230].
- Within-case PF vs FT: **28 vs 1** worsened, Wilcoxon p=2×10⁻⁶.
- Six-compartment sum (dilution): δ=+0.204, two-sided p=0.156, one-sided p=0.078.
- M3 (exchangeable): **δ̄ = +0.233**, 94% HDI [−0.872, +1.453], **P(δ̄>0)=0.66**; γ=−3.14; convergence max R̂=1.003, min ESS=1607, 0 divergences. **Derived PF contrast = +2.28** [0.85, 3.81], **P>0=0.997**. Two-block (descriptive): δ_PF=+2.42, δ_FT=−0.76, contrast +3.18 [0.97, 5.20]. LOO best = **two_block** (weakly; Pareto-k̂ warnings).
- Odds ratios: **Firth crude OR 17.2** [2.9, 103.5] (min control cell = 1); **adjusted sex+age OR 13.5** [2.3, 80.1], p=0.004; **E-value 7.77** (LCB 2.78); +BMI 13.9 [2.3, 83.4]; excl. anomalies 14.2.
- Delay (H4): median **240 vs 528 days**, MWU p<0.0001, Cliff δ=−0.64; LogNormal medians 237 vs 483; Weibull AFT group coef −0.61; within-case **ρ=−0.03 (p=0.82)**.
- H3 modulation: no factor BH-significant.

## 6. Worked examples to preserve & clean (from the transcript)

These are good and must be carried over, corrected to the 49/20 numbers:
- **Cliff δ duel count:** cyclops Δ_PF {0:21,1:15,2:9,3:3,4:1}, meniscus {0:19,1:1}; 980 duels → 545 worse / 21 better / 414 ties → δ=524/980=+0.5347; ties breakdown 21·19 + 15·1.
- **BCa internals:** z0 from prop(δ_boot<δ̂)≈0.497, a≈−0.007 → BCa ≈ percentile → [0.367, 0.684].
- **Firth as +½:** (28.5·19.5)/(21.5·1.5) ≈ exact Firth OR 17.2; Newton-Raphson trace converging to OR 17.23; β₁=2.85, SE=0.91.
- **TOST mental model:** TOST at 5% ⟺ 90% CI of the difference fits inside [−Δ, +Δ].

## 7. Non-goals (YAGNI)
- No re-running the pipeline or regenerating results/figures (numbers are read from `results.json`, figures already exist).
- No changes to notebooks, `src/`, or the manuscript.
- No English version (manuscript already covers that audience).
- No new figures authored for this doc (reuse existing PNGs).

## 8. Acceptance criteria
- `notebooks/info.md` is clean, complete, renders on GitHub (Mermaid + `$$` + callouts + figures), with **zero truncated/garbled lines** (the current file's defining problem).
- All seven notebooks (00–06) covered, each with the full 7-point template.
- The *fil rouge* (similarity → non → adapte → vérifie) is explicit in the header and threaded through §01 → §04.
- Every number matches §5; no stale 50/19, 16.3, or −0.040 figures remain.
- Worked examples (§6) present and arithmetic-correct.
- Old `notebooks/info.ipynb` removed; change committed.

## 9. Execution note
Per the user's request, implementation will run via **subagent-driven-development at max effort**: the seven notebook sections are independent given this spec + the canonical numbers (§5) and the fixed template (§3.2), so they parallelise cleanly; header, synthesis, Mermaid/visual pass, and a final coherence+number-audit are separate tasks. The writing-plans skill will turn this spec into that task breakdown.
