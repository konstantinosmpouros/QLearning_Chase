# Evaluation Figures - Interpretation Report

Το παρόν report περιγράφει **τι δείχνει** κάθε διάγραμμα του evaluation notebook, **πώς υπολογίζεται**, και τα **βασικά αποτελέσματα** που προκύπτουν.

## 1) `eval_win_rate.png`
Τι αναπαριστά:
- Συγκεντρωτικό `win_rate` ανά agent και role (CHASER/RUNNER).

Πώς υπολογίστηκε:
- Από terminal eval επεισόδια (`eval_terminal_df`) με:
- `build_agent_episode_table(...)`
- `summarize_agent_outcomes(...)`
- `plot_agent_summary_metric(..., metric='win_rate')`

Βασικά αποτελέσματα:
- Καλύτερο runner: `DynaQ+` με `0.707`.
- DQN runner: `0.652` (πολύ κοντά στο DynaQ runner `0.654`).
- Καλύτερος chaser: `DynaQ` με `0.594`.
- DQN chaser: `0.348` (χαμηλότερος από τους βασικούς tabular chasers).

## 2) `eval_avg_terminal_reward.png`
Τι αναπαριστά:
- Μέσο terminal reward ανά agent-role.

Πώς υπολογίστηκε:
- Ίδιο pipeline με το win-rate, metric `avg_reward`.

Βασικά αποτελέσματα:
- Υψηλότερο reward στους runners: `DynaQ+ RUNNER = 4.104`, `dqn RUNNER = 3.013`, `DynaQ RUNNER = 2.992`.
- Για CHASER, μόνο `DynaQ CHASER` είναι καθαρά θετικός (`1.814`).
- `dqn CHASER = -3.158`, άρα στην eval φάση ο DQN ως chaser τείνει να χάνει utility.

## 3) `eval_avg_terminal_steps.png`
Τι αναπαριστά:
- Μέσο πλήθος βημάτων μέχρι τερματισμό ανά agent-role.

Πώς υπολογίστηκε:
- Ίδιο pipeline, metric `avg_steps`.

Βασικά αποτελέσματα:
- Μεγαλύτερη διάρκεια: `DynaQ+ RUNNER = 27.46` steps.
- DQN (και στους 2 ρόλους): ~`15.95` steps.
- Πιο «γρήγορο» chaser μεταξύ των καλύτερων win-rates: `DynaQ CHASER = 11.34`.

## 4) `eval_action_mix_total.png`
Τι αναπαριστά:
- Σύνθεση terminal reasons ανά agent-role (stacked rates).

Πώς υπολογίστηκε:
- `agent_terminal_reason_mix(...)`
- `plot_agent_terminal_reason_mix(...)`

Βασικά αποτελέσματα:
- Για `DynaQ CHASER`, πιο συχνός τερματισμός: `chaser_treasure` (39%).
- Για `DynaQ+ RUNNER`, κυριαρχεί `timeout` (~48%).
- Για DQN (και CHASER και RUNNER), ο `timeout` είναι ο πιο συχνός τρόπος λήξης (~34.3%).

Σημείωση:
- Το filename λέει `action_mix`, αλλά το περιεχόμενο είναι terminal-reason mix.

## 5) `eval_terminal_step_dist.png`
Τι αναπαριστά:
- Κατανομή βήματος λήξης (`step`) ανά `terminal_reason`.

Πώς υπολογίστηκε:
- `plot_terminal_steps_hist(eval_terminal_df)` πάνω στα terminal rows.

Βασικά αποτελέσματα:
- `timeout`: σχεδόν πάντα στο τέλος (`mean_step=49.83`, median `50`).
- Treasure endings (`runner_treasure`, `chaser_treasure`) συμβαίνουν νωρίς (mean ~`7`).
- `capture` επίσης σχετικά νωρίς (mean ~`7.31`).

## 6) `eval_wumpus_vs_pit_total_rates.png`
Τι αναπαριστά:
- Για hazard endings, λόγος `pit` vs `wumpus`, ανά side (CHASER/RUNNER).

Πώς υπολογίστηκε:
- `hazard_breakdown(eval_terminal_df)`
- bar plot με `hazard_type` και `side`.

Βασικά αποτελέσματα (overall):
- Για CHASER hazard endings: `pit 91.7%`, `wumpus 8.3%`.
- Για RUNNER hazard endings: `pit 76.6%`, `wumpus 23.4%`.
- Άρα τα pits είναι η κύρια αιτία hazard-termination και για τις δύο πλευρές.

## 7) `eval_action_mix_by_match.png`
Τι αναπαριστά:
- Action share ανά run (matchup) και ανά role, σε stacked μορφή.

Πώς υπολογίστηκε:
- Από sample eval βημάτων (`eval_behavior_df`):
- `action_mix_by_role(...)`
- `plot_action_mix_by_role(...)`

Βασικά αποτελέσματα:
- Σε aggregate επίπεδο, και οι δύο ρόλοι έχουν πιο συχνή ενέργεια το `STAY`:
- CHASER `37.7%`, RUNNER `39.4%`.
- Μεγάλη ετερογένεια ανά matchup: υπάρχουν runs όπου ο STAY είναι πολύ dominant.

## 8) `eval_control_zone.png`
Τι αναπαριστά:
- Χωρική κατανομή θέσεων runner σε 2 καθεστώτα:
- `under pressure` (`dist_ab <= 2`) και `outside pressure` (`dist_ab > 2`).

Πώς υπολογίστηκε:
- `plot_control_zone_heatmaps(eval_behavior_df, run='dynaq_plus_selfplay', p_fail=0.1, pressure_dist=2)`

Βασικά αποτελέσματα για το συγκεκριμένο run:
- Υπό πίεση, πιο συχνά κελιά runner: `(3,1), (2,0), (1,1)`.
- Εκτός πίεσης, πιο συχνά κελιά: `(6,6), (1,3), (4,6)`.
- Δείχνει ξεκάθαρη αναδιάταξη χώρου κίνησης όταν ο runner πιέζεται από τον chaser.

## 9) `eval_train_vs_eval_gap.png`
Τι αναπαριστά:
- Διαφορά `runner_opening_rate_gap = eval - train` ανά run.

Πώς υπολογίστηκε:
- `train_eval_behavior_gap(train_behavior_df, eval_behavior_df)`
- bar plot στη στήλη `runner_opening_rate_gap`.

Βασικά αποτελέσματα:
- Όλα τα runs είναι αρνητικά (mean `-0.0553`): στο eval ο runner ανοίγει λιγότερο την απόσταση απ’ ό,τι στο train.
- Πιο αρνητικό run: `dqn_vs_dqn` (`-0.1094`).
- Λιγότερο αρνητικό: `fp_selfplay` (`-0.0075`).

## 10) `eval_distance_rate.png`
Τι αναπαριστά:
- `favorable_progress_rate` ανά agent-role (distance-based progress signal).

Πώς υπολογίστηκε:
- `summarize_agent_distance(eval_behavior_df)`
- `plot_agent_distance_summary(..., metric='favorable_progress_rate')`

Βασικά αποτελέσματα:
- Καλύτεροι CHASER στο metric: `DynaQ (0.262)`, `NashQ (0.260)`.
- Καλύτερος RUNNER: `FP (0.341)`, μετά `NashQ (0.288)`.
- DQN χαμηλά και στους δύο ρόλους (`CHASER 0.105`, `RUNNER 0.113`).

## 11) `eval_distance_goal_rate.png`
Τι αναπαριστά:
- `mean_goal_distance` ανά agent-role.

Πώς υπολογίστηκε:
- Ίδιο distance summary, metric `mean_goal_distance`.

Βασικά αποτελέσματα:
- Τα περισσότερα agents κινούνται γύρω στο `4.6 - 5.3`.
- DQN έχει σαφώς χαμηλότερο mean goal distance (`CHASER 3.86`, `RUNNER 3.37`), αλλά αυτό δεν συνοδεύεται από υψηλό favorable-progress rate ή υψηλό chaser win-rate.
- Άρα το μικρότερη απόσταση από μόνη της δεν αρκεί ως δείκτης στρατηγικής ποιότητας.

## Συνολικό συμπέρασμα
- Τα πιο ισχυρά eval αποτελέσματα εμφανίζονται σε `DynaQ` / `DynaQ+` (ανάλογα με ρόλο και metric).
- Ο DQN ενσωματώνεται κανονικά στην ανάλυση, έχει ανταγωνιστικό runner win-rate, αλλά παρουσιάζει αδυναμία ως chaser και έντονο generalization gap στο runner-opening behavior.
