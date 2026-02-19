# Results report

## Scope
Αναλύθηκαν αναλυτικά τα 2 notebooks:
- `chase/notebook/analytics_train.ipynb`
- `chase/notebook/analytics_eval.ipynb`

Ελέγχθηκαν τα αποθηκευμένα outputs των cells και έγινε οπτικός έλεγχος των plots (rolling curves, histograms, heatmaps, action drift, reward components, failure-rate, distance-mix).

---

## Executive Summary
- Το καλύτερο setup συνολικά είναι `minimaxq_vs_minimaxq`, ειδικά σε `p_fail=0.2` (train: capture 0.8969, eval: 0.8875).
- Το `minimaxq_vs_fp` είναι ξεκάθαρα αποτυχημένο setup: capture rate κοντά στο μηδέν σε train/eval, με steps σχεδόν πάντα στο όριο (≈30).
- Το `fp_vs_fp` είναι πολύ σταθερό από train σε eval (σχεδόν μηδενικό gap).
- Το `minimaxq_selfplay` έχει εμφανές generalization drop από train σε eval, κυρίως στο `p_fail=0.1`.
- Η μετάβαση από `p_fail=0.1` σε `p_fail=0.2` βελτιώνει σχεδόν όλα τα setups (εκτός `minimaxq_vs_fp`).

---

## 1) Core Metrics (Train)

| run | p_fail | capture_rate | avg_steps | avg_return |
|---|---:|---:|---:|---:|
| fp_vs_fp | 0.1 | 0.7286 | 18.3503 | 7.333552 |
| fp_vs_fp | 0.2 | 0.8496 | 15.2705 | 8.535520 |
| minimaxq_selfplay | 0.1 | 0.7985 | 16.6208 | 8.011651 |
| minimaxq_selfplay | 0.2 | 0.8322 | 15.4856 | 8.333358 |
| minimaxq_vs_fp | 0.1 | 0.0244 | 29.6479 | 0.163218 |
| minimaxq_vs_fp | 0.2 | 0.0188 | 29.7020 | 0.104362 |
| minimaxq_vs_minimaxq | 0.1 | 0.8636 | 14.8720 | 8.668010 |
| minimaxq_vs_minimaxq | 0.2 | 0.8969 | 13.5108 | 8.990948 |

Παρατήρηση:
- `minimaxq_vs_minimaxq` έχει το καλύτερο trade-off (υψηλό capture, χαμηλά steps, υψηλό return).
- `minimaxq_vs_fp` είναι σχεδόν “always timeout”.

---

## 2) Core Metrics (Evaluation)

| run | p_fail | capture_rate | avg_steps | avg_return |
|---|---:|---:|---:|---:|
| fp_vs_fp | 0.1 | 0.7275 | 18.6070 | 7.321210 |
| fp_vs_fp | 0.2 | 0.8485 | 15.5125 | 8.522380 |
| minimaxq_selfplay | 0.1 | 0.6970 | 18.7055 | 6.986045 |
| minimaxq_selfplay | 0.2 | 0.7750 | 16.5720 | 7.748230 |
| minimaxq_vs_fp | 0.1 | 0.0235 | 29.6275 | 0.165560 |
| minimaxq_vs_fp | 0.2 | 0.0160 | 29.7545 | 0.089815 |
| minimaxq_vs_minimaxq | 0.1 | 0.8180 | 15.8920 | 8.212805 |
| minimaxq_vs_minimaxq | 0.2 | 0.8875 | 13.8280 | 8.896390 |

---

## 3) Eval vs Train Gap

| run | p_fail | Δcapture_rate | Δavg_steps | Δavg_return |
|---|---:|---:|---:|---:|
| fp_vs_fp | 0.1 | -0.0011 | +0.2567 | -0.012342 |
| fp_vs_fp | 0.2 | -0.0011 | +0.2420 | -0.013140 |
| minimaxq_selfplay | 0.1 | -0.1015 | +2.0847 | -1.025606 |
| minimaxq_selfplay | 0.2 | -0.0572 | +1.0864 | -0.585128 |
| minimaxq_vs_fp | 0.1 | -0.0009 | -0.0204 | +0.002342 |
| minimaxq_vs_fp | 0.2 | -0.0028 | +0.0525 | -0.014547 |
| minimaxq_vs_minimaxq | 0.1 | -0.0456 | +1.0200 | -0.455205 |
| minimaxq_vs_minimaxq | 0.2 | -0.0094 | +0.3172 | -0.094558 |

Συμπέρασμα:
- `fp_vs_fp`: εξαιρετικά μικρό gap (πολύ σταθερό).
- `minimaxq_selfplay`: μεγαλύτερη απώλεια γενίκευσης.
- `minimaxq_vs_minimaxq`: καλή γενίκευση, ειδικά σε `p_fail=0.2`.
- `minimaxq_vs_fp`: παραμένει κακό και στα 2 phases.

---

## 4) Effect of `p_fail` (0.2 - 0.1)

### Train
- `fp_vs_fp`: capture +0.1210, steps -3.0798, return +1.201968
- `minimaxq_selfplay`: capture +0.0337, steps -1.1352, return +0.321707
- `minimaxq_vs_minimaxq`: capture +0.0333, steps -1.3612, return +0.322938
- `minimaxq_vs_fp`: capture -0.0056, steps +0.0541, return -0.058856

### Eval
- `fp_vs_fp`: capture +0.1210, steps -3.0945, return +1.201170
- `minimaxq_selfplay`: capture +0.0780, steps -2.1335, return +0.762185
- `minimaxq_vs_minimaxq`: capture +0.0695, steps -2.0640, return +0.683585
- `minimaxq_vs_fp`: capture -0.0075, steps +0.1270, return -0.075745

Συμπέρασμα:
- Η αύξηση του `p_fail` βοηθά συστηματικά τα καλά setups.
- Το `minimaxq_vs_fp` δεν βελτιώνεται, άρα το πρόβλημα είναι στρατηγικό και όχι απλό θέμα noise.

---

## 5) Plot-by-Plot Findings

## 5.1 Rolling capture/steps
- Οι καμπύλες train και eval δείχνουν σταθερά:
  - κορυφαίο: `minimaxq_vs_minimaxq`
  - πολύ καλό: `fp_vs_fp`, `minimaxq_selfplay`
  - αποτυχημένο: `minimaxq_vs_fp` (capture ≈ 0, steps ≈ 30)
- Στο eval, τα patterns επιβεβαιώνονται χωρίς αντιστροφή ranking.

## 5.2 Return quantiles
- `minimaxq_vs_fp`: και στο train και στο eval οι quantiles είναι κοντά στο 0 ή αρνητικές.
- `minimaxq_vs_minimaxq`: q50/q90 πολύ κοντά στο μέγιστο return, με χαμηλή q10 που κάνει συχνά “drop” (all-or-nothing συμπεριφορά σε μέρος των episodes).

## 5.3 Steps-to-capture histograms
- Για `fp_vs_fp`, `minimaxq_selfplay`, `minimaxq_vs_minimaxq`:
  - κύρια μάζα σε χαμηλά/μεσαία βήματα,
  - ξεκάθαρα λιγότερες πολύ αργές captures σε `p_fail=0.2`.
- Για `minimaxq_vs_fp`: σχεδόν μηδενική πυκνότητα capture σε όλο το εύρος.

## 5.4 Heatmaps (visits + capture locations)
- `minimaxq_vs_fp`: έντονη παρουσία runner στα boundaries/γωνίες και captures συγκεντρωμένες σε συγκεκριμένες θέσεις (κυρίως άκρες), με πολύ μικρό πλήθος captures.
- `minimaxq_vs_minimaxq`: captures πολύ πιο ομοιόμορφα κατανεμημένες στο grid (train και eval), ένδειξη robust pursuit.
- `fp_vs_fp`: ισχυρή τάση boundary visits από runner, αλλά με πολλά captures σε edge cells.

## 5.5 Action drift
- `minimaxq_vs_minimaxq`: τα action mixes σταθεροποιούνται νωρίς και μένουν σχετικά smooth.
- `minimaxq_vs_fp`: υπάρχει drift αλλά δεν οδηγεί σε βελτίωση outcomes, άρα το policy adaptation δεν είναι “productive”.
- Γενικά ο runner εμφανίζει πιο έντονη μεταβλητότητα από τον catcher.

## 5.6 Failure-rate + wall/move/other
- Σε `minimaxq_vs_fp`, και train και eval, οι fail rates των agents μένουν περίπου στο ~0.2 (εναρμονισμένες με το πολύ χαμηλό capture).
- Στα άλλα runs, ο runner τείνει να έχει υψηλότερο fail-rate από catcher, κυρίως σε `p_fail=0.2`.
- Τα wall-hit/move/other stacks είναι σταθερά στον χρόνο, χωρίς μεγάλες δομικές αλλαγές μετά το αρχικό phase.

## 5.7 Opening distance + distance-change mix
- Opening Manhattan distance: σε όλα τα runs κυμαίνεται περίπου στο 3.0–3.6, χωρίς ισχυρό μακροχρόνιο drift.
- Distance-change mix:
  - `minimaxq_vs_minimaxq`: υψηλότερο ποσοστό `closer`.
  - `minimaxq_vs_fp`: χαμηλότερο `closer`, υψηλότερο `same/farther`.
  - τα patterns είναι πολύ παρόμοια train vs eval.

## 5.8 Reward components
- Στα reward-component plots, το μεγαλύτερο σήμα έρχεται από `capture`.
- `step/wall/move/dist` φαίνονται σχεδόν flat (ιδίως στο train), που δείχνει είτε πολύ μικρή κλίμακα είτε logging/scaling θέμα για αυτά τα terms.

## 5.9 Eval seed variability
- Από τα per-seed plots:
  - `minimaxq_vs_fp`: συνεχώς πολύ χαμηλά per-seed capture rates.
  - `minimaxq_vs_minimaxq`: υψηλά per-seed capture rates με καλύτερη συγκέντρωση κοντά στο 1.
  - `fp_vs_fp` και `minimaxq_selfplay`: μέτρια προς υψηλή διασπορά, αλλά σαφώς πάνω από `minimaxq_vs_fp`.

---

## 6) Final Conclusions
- Το πιο αποτελεσματικό και πιο σταθερό ανταγωνιστικό setup είναι `minimaxq_vs_minimaxq`.
- Το `fp_vs_fp` είναι επίσης πολύ ισχυρό και με σχεδόν μηδενικό train-eval gap.
- Το `minimaxq_selfplay` είναι καλό, αλλά χάνει σημαντικά στο eval (ιδίως `p_fail=0.1`).
- Το `minimaxq_vs_fp` δεν είναι βιώσιμο setup με τα τωρινά dynamics.
- Η αύξηση `p_fail` από 0.1 σε 0.2 λειτουργεί ως “εύνοια” στα καλά setups, αλλά δεν σώζει το αποτυχημένο matchup.

---

## 7) Notes
- Το report βασίστηκε στα ήδη αποθηκευμένα outputs των notebooks και στον οπτικό έλεγχο των plots.
- Τα animations (cells με `animate_episode`) επιβεβαιώθηκαν ως διαθέσιμα, αλλά το βάρος της ανάλυσης δόθηκε στα στατικά analytics plots και στα αριθμητικά summaries.
