# Hijrah ke Barcelona — Technical document

**More Than a Hack 2027, Initial Challenge. Release: 18 November 2026.**

## 1. Approach

We estimate team possession as offline temporal segmentation with three internal
states: home (H), away (A), and dead ball (D). Proximity alone is insufficient
for passes, loose balls, and challenges, so the system combines tracking geometry,
relative motion, contact context, and temporal state/boundary predictions.
The submitted candidate is `calibration_fusion_v46_equal5`.

The release extracts 203 tracking features in two team-swapped orientations.
Missing observations remain explicit; the deep model combines normalized inputs
and observation masks. Each frozen deep model is a warm-start dilated TCN with
eight backbone blocks and a six-block probability refiner. It predicts three
state probabilities and six directed transition probabilities. Team-swapped
predictions are mapped back and averaged. Inference uses 8,192-frame chunks
with the model's 636-frame halo; future context is allowed. This is not a
causal or live-broadcast system.

For each of five folds, calibrated deep probabilities are mixed with frozen
histogram-gradient-boosting heads. Tree weights are 0.25 for folds 0 and 3 and
zero for folds 1, 2, and 4; specialist weights are zero throughout. The five
fold outputs are averaged equally before a fixed Viterbi decoder. Its settings
are team=2, dead=4, boundary_gain=2, emission_scale=0.2, dead_bias=-0.18, and
proposal_threshold=0.1. No target frame overrides or manual CSV edits are used.

## 2. Assets used and dropped

Release inference reads **only tactical raw ATD and its metadata** from the
licensed competition bundle. Official `games.py` constants define the frame
timeline, pitch dimensions, and playing-half blocks. The pipeline does not
read target labels, Smart Tagging, video, audio, or homography. Match assets
and target feature/probability exports are not redistributed.

Expanded tracking features, video ball recovery, and whistle/flag cues were
explored separately. Their benefit was not adequately demonstrated for this
release, so none is forced into the final prediction path. A zero specialist
weight means equal ablation outputs are not proof that extra features contain
no information.

## 3. Training data and protocol

Supervision comes exclusively from seven external IDSSE matches: J03WMX,
J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, and J03WR9. The source is Bassek,
Rein, Weber, and Memmert (2025), *An integrated dataset of spatiotemporal and
event data in elite soccer*, dataset DOI **10.6084/m9.figshare.28196177.v1**,
article DOI **10.1038/s41597-025-04505-y**, licensed CC BY 4.0.

Labels are supplied attributes `BallStatus` and `BallPossession` on the external
BALL track, not new annotations created by our team. Live status with owners
1/2 maps to H/A, dead status maps to D, and unavailable status/owner maps to X.
We do not claim authorship of the original labels or invent individual operator
identities beyond source documentation. Only metadata and observed-position
XML are required; external event XML is not needed by this parser.

In each fold, three development matches train the model, one calibrates it,
and the remaining development match is tested alongside J03WQQ/J03WR9. These
last two are **known retests**, not newly untouched holdouts. Splits are by match,
not random frames. Original, short-gap, long-gap, camera-gap, and drift scenarios
test robustness. Calibration and mixture selection use the calibration match.
Public target labels are reserved for format/alignment checking, not tuning.

Manual review notes concerning the target were provided and seen during final
preparation. They were not used for training, validation, candidate selection,
thresholds, frame-specific rules, or editing the release. The packaged candidate
was frozen beforehand. We do not represent unreviewed frames as correct or claim
that the notes were never seen. AI assistance supported coding, testing, packaging,
and documentation; final labels are algorithmic outputs from frozen models.

## 4. Possession rules

Possession is treated as temporally continuous unless state and boundary
evidence support a change. Loose balls and challenges have no extra class:
their H/A/D status is inferred jointly rather than immediately switching to the
nearest player. Restarts and dead-ball periods are inferred from tracking,
not manually supplied whistle/event timestamps. This is a learned operational
approximation, not a complete symbolic implementation of football laws.

Each official half is decoded separately. Outside those half blocks, D is a
timeline placeholder, not a model-inferred dead-ball decision. The CSV has no
X predictions; that does not mean all frames are certain.

## 5. Results and limitations

The V4.6 controlled experiment contains 375 evaluation rows across five folds,
three test matches per fold, five scenarios, and five methods, involving seven
unique matches. Rows are not independent new games. Calibration-selected
per-fold SpellF1 is **0.6483558219452347**, versus **0.6477615693349609** for
frozen deep V4.5. Control 203-feature and expanded-feature methods have the same
selected outputs because all specialist weights are zero.

These are **not scores for the submitted equal-five ensemble** or the private
competition match. The release uses a different fixed decoder from the evaluated
per-fold decoder, and float16-to-float32 feature conversion to match the frozen
candidate; external evaluation extracted float32. A fresh external holdout for
the entire ensemble has not been measured. Reused matches, synthetic corruption,
domain shift, missing ball observations, and ambiguous restarts limit inference.
No private score, rank, statistical significance, or uniform improvement is claimed.

## 6. Reproduction

Use Windows x64, Python 3.12.10, pinned `requirements-release.txt`, and the
official data bundle acquired separately by an authorized participant:

```powershell
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements-release.txt
python -m riset_v4.release --data-dir "D:\MTAH\data" --output "outputs\equal5\Hijrah ke Barcelona.csv"
python evaluation/validate_submission.py "outputs\equal5\Hijrah ke Barcelona.csv"
```

Five deep and five tree artifacts are stored as base64-encoded joblib files,
with binary sizes and SHA256 in `models/equal5/manifest.json`. The loader checks
integrity before deserialization; only trusted artifacts should be loaded.
Inference requires no training, target cache, annotations, or remote model API.
Existing output files are not overwritten. A final hash mismatch is retained
as `.pending`, not promoted as a successful result.

Raw-to-CSV reproduction has been verified **byte-identical** on the release
environment: 199,319 rows (frames 0–199318), header `frame,label`, SHA256:

```text
a38b30c361cacb08dcf3d60173eb137ff28f1517d8d463dd03ad666641599ae7
```

See `hasil/rilis/reproduction.json` and `dokumen/REPRODUKSI.md`. Cross-platform
bitwise determinism and all training-GPU configurations are not claimed.
Extended methodology, ablations, provenance, and references are indexed in
`dokumen/README.md`. Organizers' source, rules, and scorer are attributed to
Metrica Sports and Barça Innovation Hub; their MIT code license does not
license competition match assets.
