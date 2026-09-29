# Q-Catch: tests και έλεγχος υλοποίησης

Το πακέτο περιλαμβάνει το αρχικό project, το ενημερωμένο notebook και ανεξάρτητα
pytest tests. Τα tests του pilot φορτώνουν τους πραγματικούς ορισμούς από τα
code cells του notebook μέσω του `tools/notebook_runtime.py`. Δεν υπάρχει δεύτερη,
απλουστευμένη υλοποίηση των αλγορίθμων μόνο για τα tests.

## Εγκατάσταση και εκτέλεση

Από τον φάκελο `QCatch` με Python 3.11 ή 3.12:

```bash
python -m venv .venv
```

Windows PowerShell: `.venv\Scripts\Activate.ps1`

Linux/macOS: `source .venv/bin/activate`

```bash
python -m pip install --upgrade pip
python -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
python -m pytest -q
```

Μόνο unit/behavioral tests:

```bash
python -m pytest -q -m "not integration"
```

Σύντομα end-to-end tests, με πραγματικές ενημερώσεις δικτύου:

```bash
python -m pytest -q -m integration
```

Δεν απαιτούνται GPU, API keys ή πρόσβαση στο internet μετά την εγκατάσταση.
Τα tests χρησιμοποιούν προσωρινούς φακέλους και δεν αντικαθιστούν τα πειραματικά
αποτελέσματα του χρήστη. Δεν επιβάλλουν ελάχιστο win rate: η σωστή εκτέλεση δεν
συνεπάγεται σύγκλιση ή ανωτερότητα μιας μεθόδου.

## Notebook

Άνοιξε το `notebooks/Chase_Population_Robustness_Tested.ipynb` από το project ή
τον υποφάκελο `notebooks`. Το νέο κελί εκτελεί το ανεξάρτητο test suite και
σταματά την εκτέλεση αν κάποιο test αποτύχει. Χρειάζεται ολόκληρο το πακέτο,
όχι μόνο το `.ipynb`. Σε Colab ανέβασε/αποσυμπίεσε το ZIP και όρισε working
directory τον φάκελο `QCatch` πριν εκτελέσεις το notebook.

Το `RUN_EXPERIMENTS=False` είναι η αρχική ρύθμιση για να μη ξεκινήσει κατά λάθος
πολύωρη εκπαίδευση. Για νέο πείραμα όρισε `True` και διάλεξε `smoke`, `pilot` ή
`full`. Το RUN_TAG/CODE_SIGNATURE έχει αλλάξει ώστε να μη φορτωθούν παλιά
checkpoints σαν να προέκυψαν από τον διορθωμένο κώδικα.

Πλήρης εκτέλεση του smoke notebook σε νέο προσωρινό output directory:

```bash
python tools/run_smoke_notebook.py --output verification/smoke_executed.ipynb
```

Σε περιβάλλον που δεν επιτρέπει sockets για Jupyter kernel:

```bash
python tools/run_smoke_notebook.py --inprocess --output verification/smoke_executed.ipynb
```

Και οι δύο διαδρομές εκτελούν όλα τα code cells. Η δεύτερη χρησιμοποιεί IPython
στην ίδια διεργασία, χωρίς να ανοίγει sockets. Τα προσωρινά training checkpoints
του smoke διαγράφονται μετά την ολοκλήρωση. Το εκτελεσμένο notebook κρατά τα
outputs για έλεγχο, όχι για επιστημονικά συμπεράσματα.

## Κάλυψη

- Κίνηση, όρια, εμπόδια, ανεξάρτητες αποτυχίες κίνησης, p_fail=0 και p_fail=1.
- Ασφαλείς/διαφορετικές αρχικές θέσεις και αναπαραγωγιμότητα resets.
- Σύλληψη στο ίδιο κελί και με ανταλλαγή θέσεων.
- Προτεραιότητα capture > treasure > hazard > timeout, συμπεριλαμβανομένων
  ταυτόχρονων γεγονότων και θανάτου και των δύο παικτών.
- Αναλυτικά αναμενόμενες ανταμοιβές και άθροισμα συνιστωσών στο legacy project.
- Zero-sum ανταμοιβές και τηλεσκοπική ιδιότητα potential shaping στο pilot.
- Κατάσταση με χρόνο, p_fail και ρόλο στο pilot.
- Γνωστά pure/mixed minimax παιχνίδια, άξονες ενεργειών Runner, TD terminal mask.
- FP beliefs για τις ενέργειες του αντιπάλου και σωστή μεταφορά στην οπτική Runner.
- Dueling decomposition, Double DQN selection/evaluation, target synchronization,
  warmup, replay wraparound και αποθήκευση των πραγματικών ενεργειών ανά ρόλο.
- Nash ισορροπία σε ελεγχόμενο παιχνίδι και έλεγχος μονομερούς απόκλισης.
- Στοχαστικά joint outcomes στο Dyna και αριθμός planning updates στο pilot.
- Frozen opponents χωρίς shared parameters με τους learners.
- Αξιολόγηση χωρίς αλλαγή weights, target, optimizer, replay ή learner RNG.
- Αποκλεισμός test/validation policies και holdout families από το training pool.
- Ίσο βάρος οικογενειών, fixed controls, adaptive exploration floor και κόστος
  πρόσθετων calibration βημάτων. Το adaptive calibration χρησιμοποιεί training
  opponents με χωριστό validation RNG stream, όχι τους τελικούς test opponents.
- Cache αποτελεσμάτων με fingerprint μοντέλου/αντιπάλων/ρυθμίσεων και checksum CSV.
- Εκπαίδευση όλων των arms και των δύο ρόλων σε μικρό budget, αποθήκευση/φόρτωση,
  ακριβής συνέχιση learner checkpoint και επανέναρξη διακοπείσας εκπαίδευσης.

## CI / GitHub

Το `.github/workflows/tests.yml` ενεργοποιείται σε push και pull request, τρέχει
pytest σε Python 3.11/3.12 και το smoke notebook σε Python 3.12. Η εκτέλεση στο
GitHub δεν έγινε από αυτό το περιβάλλον. Θα ξεκινήσει όταν ανέβουν τα αρχεία στο
repository. Τοπικά ελέγχθηκε Python 3.12 CPU.

Το `.gitignore` επιτρέπει πλέον το συγκεκριμένο νέο notebook, ενώ αποκλείει
training outputs. Από το υπάρχον repository, αντέγραψε τα περιεχόμενα του
φακέλου `QCatch`, έλεγξε `git diff` και πρόσθεσε τις αλλαγές με κανονικό commit.
Δες `VALIDATION_REPORT.md` πριν χρησιμοποιήσεις παλιά αποτελέσματα με νέο κώδικα.
