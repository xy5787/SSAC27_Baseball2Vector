# Task 10 — representative-statistic choice sensitivity

Frozen before Task 10 execution, 2026-09-27. Follow-up motivated by scrutiny of the fixed representative5 comparator after viewing Tasks 1–9; not original-study preregistration.

Question: does B2V's advantage depend on choosing ISO, Contact%, BB%, BsR and Def as the five-statistic comparator?

- Preserve the frozen 2019–2025 raw snapshot, current sd10 B2V cohort, targets (primary wRC+, secondary WAR/600), PA>=100 eligibility, expanding forecast windows, history construction, Ridge alpha grid, player-grouped inner CV and seed 42 from Task 8.
- Run standalone and history-augmented models, both all transitions and the refitted sensitivity excluding transitions touching 2020. Keep 2020 in Marcel lookbacks, as in Task 8.
- Original set: one statistic per group: Power=ISO, Contact=Contact%, Discipline=BB%, Speed=BsR, Defense=Def. Original rationale was semantic coverage, not measured optimality; this analysis does not retroactively change the rationale.
- Leave-one-out: five four-statistic models, each dropping one original representative. These assess the chosen model's reliance on each proxy, not the necessity of a latent skill or causal contribution.
- Single replacements: replace exactly one original representative with each other member of its existing B2V input group. There are 17 alternatives, covering all 22 inputs; keep the other four original representatives. This is local choice sensitivity, not exhaustive evaluation of all 1,120 cross-group combinations.
- Selected representative5: at each outer origin and for each target/context/window, choose among the original and 17 replacement sets using their minimum mean inner-CV MAE (same 50-alpha grid). Ties use declared variant order, then alpha order. All selection uses outer-training data only. Reuse the outer prediction from that selected model. Do not select the best outer-test result or replace the original comparator post hoc.
- Reproduce original representative5 predictions against Task 8 to numerical tolerance before interpreting differences. Reuse saved Task 8 B2V/history+B2V predictions for identical held-out rows; do not refit or alter them.
- Report pooled and per-year MAEs; paired differences B2V versus each five-statistic set and the internally selected set; original5 versus each drop model; original5 versus each swap. Use 2,000 paired focal-player bootstrap draws (seed 42), conditional on fitted predictions, no multiplicity or model-selection uncertainty adjustment. Treat individual swap CIs as exploratory; the predeclared internally selected comparator is the main stronger-control assessment.
- Retain all variant results, including unfavorable outcomes. Do not interpret nonsignificance as equivalence. New results will first go in a report, with an explicit recommendation for whether the abstract's comparator claim should be qualified; do not silently substitute a new comparator into the submission table.
