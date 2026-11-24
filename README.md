# Lightweight PRM Reward Heads on a Frozen Encoder

Trains six lightweight reward heads on top of a frozen Qwen2.5-0.5B step encoder
with the PQM comparative ranking loss, and asks which head architecture is worth
the parameters. The heads split into pointwise (`linear`, `mlp`) and contextual
(`cnn`, `gru`, `attention`), plus `attention_pe`, an order-aware variant added
after the perturbation analysis.

The anchor is *Process Reward Model with Q-Value Rankings* (PQM, Li & Li).
This repository reuses PQM's loss and its Math-Shepherd training corpus, but
deviates in two ways that matter: it freezes a 0.5B encoder where PQM fine-tunes
a 7B one, and it adds an architecture axis that PQM never ablates. A LoRA arm
tests what the first deviation costs.

All numbers below come from `results_final/` and `results_ablations/`, computed
on correctly parsed data (see [Data parsing](#data-parsing-and-what-it-invalidated)).
Held-out metrics use 2,223 validation trajectories that play no part in training,
early stopping or threshold selection; intervals are 95% trajectory-level (or,
for Best-of-N, question-level) paired bootstrap intervals over 2,000 resamples,
and "significant" means significant after Holm correction within the relevant
family of comparisons. (2,000 resamples floor a bootstrap p-value at 2/2,001,
about 0.001, which is coarse for large Holm families; the bootstrap scripts now
default to 20,000, while `run_experiments.py` keeps 2,000 unless given
`--bootstrap_samples`, so that it reproduces the numbers below.)

> **Correction pending (2026-11, answer grading).** The Best-of-N candidates
> were graded with an answer parser that cut gold answers at the thousands
> comma (`#### 1,200` read as `1`) and read decimals by their fractional part
> (`26.00` read as `00`). Checked against the official GSM8K test set, 327 of
> 21,104 candidate labels (1.55%) were wrong, in 84 of 1,319 questions (6.4%);
> 302 correct candidates had been marked wrong and 25 wrong ones marked correct.
> `prepare_data.py relabel` re-grades them. Numbers that need only the labels
> were recomputed and are given below next to the originals
> ([`results_final/bon_references_relabeled.csv`](results_final/bon_references_relabeled.csv),
> counts in [`results_final/bon_relabel_stats.json`](results_final/bon_relabel_stats.json));
> every number that needs PRM scores -- all head-based Best-of-N selectors,
> their differences to majority voting and Holm tests, and the GSM8K
> single-solution AUCs -- **still has to be re-run** and is quoted below as
> originally computed. The embedding caches and checkpoints stay valid
> (relabeling does not change the encoded steps), so this is a re-scoring, not
> a retraining.

## What the experiments found

**Contextual heads win on held-out step scoring.** Held-out step ROC-AUC, each
head at `lr=1e-4` with up to 30 epochs and patience 5:

| head | params | ROC-AUC [95% CI] |
|---|---|---|
| `attention` | 6.43M | **0.8692** [0.8548, 0.8831] |
| `attention_pe` | 6.43M | 0.8651 [0.8511, 0.8787] |
| `gru` | 788k | 0.8527 [0.8382, 0.8675] |
| `cnn` | 394k | 0.8486 [0.8336, 0.8633] |
| `mlp` | 263k | 0.8374 [0.8224, 0.8521] |
| `linear` | 897 | 0.7672 [0.7501, 0.7839] |
| *position-only baseline* | 0 | *0.6313* |
| *majority baseline* | 0 | *0.5000* |

13 of the 15 pairwise differences are significant; `{attention, attention_pe}`
and `{gru, cnn}` are the two indistinguishable groups. Every head clears the
position-only baseline by a wide margin, so the frozen representation carries
step-correctness signal and not just step position. The ranking holds across
seeds (attention > cnn > mlp for seeds 42, 43 and 44; cnn and mlp are
indistinguishable at seed 43) and at `lr=3e-5` (below).

**Most of that advantage comes from reading future steps.** The encoder is
causal, but a bidirectional head re-introduces access to later steps. Scoring
step *t* from the first *t* embeddings only -- the setting any online
early-exit policy faces -- removes most of the gap:

| head | causal ROC-AUC [95% CI] | penalty vs full trajectory |
|---|---|---|
| `attention_pe` | 0.8431 [0.8288, 0.8575] | −0.0220 |
| `mlp` | 0.8374 [0.8224, 0.8521] | 0.0000 |
| `gru` | 0.8321 [0.8171, 0.8463] | −0.0206 |
| `attention` | 0.8251 [0.8087, 0.8405] | −0.0441 |
| `cnn` | 0.7866 [0.7716, 0.8017] | −0.0620 |
| `linear` | 0.7672 [0.7501, 0.7839] | 0.0000 |

Under causal scoring the 263k-parameter `mlp` is statistically
indistinguishable from the best contextual heads, `cnn` loses the most but stays
above `linear`, and 11 of 15 comparisons are significant. Which contextual head
comes first is *not* stable: it is `attention_pe` at `lr=1e-4` and `gru` at
`lr=3e-5`. The exactly-zero penalty for the pointwise heads is an invariant,
asserted in the test suite, as is the exactly-zero effect of reordering steps on
plain `attention`.

**Frozen heads can judge whole solutions in distribution, but not on GSM8K.**
Scoring a whole solution by its minimum step score, held-out Math-Shepherd
trajectories reach ROC-AUC 0.72 (`linear`) to 0.87 (`attention`). On Qwen's own
GSM8K candidates every head drops to 0.63-0.72, with `linear` the best at both
learning rates -- the representation's step-correctness signal does not
survive the change of generator and task.

**No frozen head beats majority voting at Best-of-N.** Reranking 16
candidates per GSM8K test question sampled from Qwen2.5-0.5B-Instruct
(temperature 0.7, top-p 0.95; 1,319 questions):

| selector | BoN@16 [95% CI] | re-graded labels (2026-11) |
|---|---|---|
| oracle (any of 16 correct) | 0.7604 [0.7369, 0.7839] | 0.7847 [0.7612, 0.8074] |
| **majority voting** (no reward model) | **0.4655** [0.4374, 0.4927] | **0.4958** [0.4685, 0.5231] |
| best frozen head, PRM-weighted voting (`attention`, mean) | 0.4761, vs majority +0.011 [−0.003, +0.024], n.s. | needs re-run |
| best frozen head, highest score (`attention`, last) | 0.3699, vs majority −0.096 [−0.121, −0.070] | needs re-run |
| random selection | 0.3282 [0.3002, 0.3555] | 0.3374 [0.3093, 0.3647] |

Under the corrected grading majority voting gains 0.030 at N=16 (0.4655 →
0.4958; 0.3743 → 0.3929 at N=4, 0.4313 → 0.4544 at N=8). Weighted voting and
argmax selectors gain from the same re-grading, by amounts that cannot be
known without re-scoring, so the comparisons below against majority voting --
including the LoRA arm's small margins of +0.021 to +0.027 -- are unverified
until `eval.eval_bon` is re-run on the relabeled file.

Of 36 frozen selectors (6 heads x 3 aggregations x {argmax, weighted vote}),
none beats majority voting after Holm correction and 23 are significantly worse;
the same holds at `lr=3e-5` (0 better, 21 worse) and for the BCE-trained heads
(0 better, 9 of 18 worse). Taking the single highest-scoring candidate is far
worse than letting the reward reweight a vote.

**Unfreezing the encoder is what changes that.** LoRA (r=16 on q/k/v/o, one
pass over the 440k training trajectories) with the same linear value head:

| comparison | ROC-AUC difference [95% CI] |
|---|---|
| LoRA linear vs frozen linear | **+0.0779** [+0.0636, +0.0927] |
| LoRA linear vs frozen attention | −0.0241 [−0.0356, −0.0130] |

On held-out steps LoRA recovers most, not all, of what a 6.4M-parameter head on
frozen features achieves -- but on GSM8K it is the only configuration that beats
majority voting. Its PRM-weighted vote wins with all three aggregations, each
significant after Holm correction across its six selectors:

| LoRA selector | BoN@16 | vs majority [95% CI] |
|---|---|---|
| weighted vote, last | 0.4920 | +0.027 [+0.010, +0.043] |
| weighted vote, min | 0.4897 | +0.024 [+0.008, +0.040] |
| weighted vote, mean | 0.4867 | +0.021 [+0.008, +0.034] |
| highest score, last | 0.4594 | −0.006 [−0.029, +0.017], n.s. |

Those corrections are within the LoRA run's six selectors. Treating the
frozen and LoRA selectors at N=16 as one family of 42
(`python -m analysis.holm_bon_family`, written to
`results_ablations/bon_holm_frozen_plus_lora.csv`, from the committed `bon_results.csv`
files, i.e. the pre-correction labels and 2,000 resamples), none of the LoRA
weighted votes stays significant: Holm-adjusted p = 0.057 (mean), 0.068 (last)
and 0.105 (min). Which family is appropriate depends on the claim; the
cross-arm claim ("only LoRA beats majority voting") is a statement about the
combined family. Both families need re-running on the re-graded labels.

The gain is small and the arm is not compute-matched, but the direction is
clear: better step metrics on frozen features did not buy Best-of-N, while
adapting the encoder did. What limits a frozen-encoder PRM here is the frozen
representation, not the head placed on it.

**The PQM ranking loss does not beat pointwise BCE here.** Training the same
heads with per-step binary cross-entropy instead:

| head | PQM − BCE, held-out ROC-AUC [95% CI] |
|---|---|
| `linear` | −0.0117 [−0.0161, −0.0072] |
| `mlp` | −0.0151 [−0.0220, −0.0084] |
| `attention` | +0.0008 [−0.0086, +0.0100], n.s. |

BCE is significantly better for the pointwise heads and equivalent for
`attention`. This is the opposite of the anchor paper's central claim, in a
setting that differs from it in the two ways named above, so it bounds rather
than refutes that claim.

**Learning rate matters as much as architecture.** On corrected data `lr=3e-4`
is worse than `1e-4` for five of six heads by 0.017-0.048 AUC -- as large as
most architecture gaps -- and `3e-5` reaches a lower development loss than
`1e-4` for five of six. Rerunning the entire evaluation chain on the `3e-5`
heads (`results_final_lr3e-5/`) reproduces the held-out ranking with the same
significance structure (only the indistinguishable top pair swaps), the causal
collapse, the in-distribution control and the Best-of-N failure. Two finer
claims do not reproduce, and are therefore not made above: which contextual head
leads causally, and that larger heads transfer worse to GSM8K. The margin `zeta`
matters little: against the default 4, only `linear` at `zeta=2` (−0.004) and
`attention` at `zeta=8` (−0.013) are significantly worse.

## Figures

All from `results_final/` (held-out half of the Math-Shepherd validation set,
heads at `lr=1e-4`); none depends on the Best-of-N labels corrected above.

**Label bias.** Error rate by relative step position and where the first error
falls -- the reason for the position-only baseline.

![Label position bias and first-error position](results_final/data_bias.png)

**Scores around the first error.** Mean standardised Q-value at offsets from
each trajectory's first error.

![Reward behaviour around the first error](results_final/first_error_boundary.png)

**Future-context sensitivity.** Mean absolute difference between full-trajectory
and causal-prefix scores; zero for the pointwise heads by construction.

![Future-context sensitivity](results_final/full_vs_causal_scores.png)

**Perturbations.** Change in held-out ROC-AUC when steps are reversed, swapped,
or masked.

![Deterministic perturbation sensitivity](results_final/perturbation_sensitivity.png)

**Offline pruning.** Safe step savings and error coverage against the
false-prune rate on clean trajectories, for causal-prefix scores.

![Pruning trade-off](results_final/pruning_tradeoff.png)

Training curves are in `results_final/{head}_loss_curve.png`, stratified
behaviour in `results_final/behavior_by_group.png` and detection delay in
`results_final/pruning_detection_delay.png`.

## Causal heads (implemented, not yet run)

The causal-prefix numbers above take heads *trained* bidirectionally and only
truncate their input at evaluation, which is not the like-for-like comparison
for an online setting: a head that will only ever see steps <= t could have
been trained that way. `reward_heads.py` now has causal variants of every
contextual head, selected by name with `--head` in every script:

| head | change | parameters vs bidirectional |
|---|---|---|
| `gru_causal` | unidirectional GRU (`GRUHead(bidirectional=False)`) | fewer (one direction) |
| `cnn_causal` | convolutions left-padded by `kernel_size-1` | same |
| `attention_causal` | causal attention mask (step t attends to steps <= t) | same |
| `attention_pe_causal` | the same with sinusoidal positions | same |

The test suite checks that perturbing steps > t leaves every causal head's
output at steps <= t unchanged, that the bidirectional heads fail that check,
and that prefix scoring is a no-op for the causal heads. To train them under
the main protocol (lr 1e-4, up to 30 epochs, patience 5) and compare each with
its bidirectional counterpart under prefix scoring (paired bootstrap, Holm
within the four pairs), then run Best-of-N:

```bash
python run_experiments.py --only causal causal_eval   # -> checkpoints/causal/, results_causal/
```

These groups are opt-in; a plain `python run_experiments.py` still runs only
the reported experiments. No causal head has been trained yet, so there are no
numbers for them.

## Data parsing and what it invalidated

`dataset.py` originally split each Math-Shepherd solution on newlines and
labelled every line not ending in `+` or `-` as an incorrect step. A step can
span several lines -- most often the final `Step k: ...\n\n# Answer\n\n42` -- so
one labelled step became up to three, the extra ones marked wrong. On this
repository's data that made 13.5% of all error labels spurious and marked 17% of
fully correct trajectories as containing an error, with the fake errors placed
just before the final step. Steps are now the spans between `ки` markers in the
input and each label is the `+`/`-` that replaces its marker; a record whose
label text is not exactly its input with markers replaced is rejected (114 of
440,208 training and 2 of 4,447 validation records).

Every cache, checkpoint and result directory built before the fix is kept only
as a record, under `archive/` (`RESULTS_MAP.md`). Two conclusions changed materially: the LoRA arm
had shown no gain from unfreezing (it now shows the largest effect in the
study), and `cnn` had fallen below `linear` under causal scoring (it no longer
does). The study's earlier history -- a protocol that fixed `lr=1e-3`, which the
first learning-rate grid showed to be the worst value for every head and which
had inverted the architecture ranking -- also predates the fix, and `1e-3` was
not re-tested on corrected data.

### Best-of-N answer grading

`prepare_data.py` originally extracted answers with `-?\d+`: the gold answer
after `####` stopped at the first comma (14 of 1,319 gold answers truncated,
e.g. `1,200` → `1`), and a prediction took the last digit run, so `26.00`
became `00` and `$0.90` became `90`. The parser now reads the whole number
(digit-group commas, decimals, sign) and normalises it (`26.00` → `26`).
The committed `data/gsm8k_qwen0.5b_bon16.jsonl` is left untouched as the
record of what the reported numbers used; `prepare_data.py relabel` writes a
re-graded copy, `data/gsm8k_qwen0.5b_bon16_relabeled.jsonl`, keeping the old
values as `gold_answer_v0`, `pred_num_v0` and `final_correct_v0`. Gold answers
come from `data/gsm8k_test_gold.jsonl`, extracted from the official GSM8K test
set (`prepare_data.py gold`). Grading is strict numeric equality, so `$0.90`
for a question asking for cents (gold `90`) counts as wrong (10 candidates in
two questions). `eval.eval_bon --references_only` recomputes oracle, random
and majority voting from the labels alone; with the old file it reproduces
the reported reference rows exactly.

## Limitations

* **Validation questions also occur in training (inferred, not measured).**
  The reported split held out 1% of Math-Shepherd *rows*, and Math-Shepherd
  has many trajectories per question, so the validation trajectories' questions
  are, in all likelihood, almost all also present in training. Held-out step
  metrics therefore measure generalisation to new solutions of seen questions,
  not to new questions; the GSM8K Best-of-N evaluation is unaffected. The
  original `data/val.jsonl` is not in the repository, so the overlap could not
  be measured here; `python prepare_data.py overlap --train data/train.jsonl
  --val data/val.jsonl` measures it on the original files. `split` now groups
  by question by default (seeded hash of the normalised question text);
  `--split_by row` reproduces the old behaviour. No reported number has been
  re-run on a grouped split.
* **The learning-rate grid has no lower bound.** `3e-5` beats `1e-4` on
  development loss for five heads and the trend is monotone, so the optimum may
  lie lower. The main results keep the pre-specified `1e-4`; the full
  evaluation at `3e-5` shows which conclusions depend on that choice.
* **The LoRA arm is not compute-matched** (one pass against the frozen arms' up
  to 30 epochs) and uses a single seed. The handicap favours the frozen arms, so
  its advantage is if anything understated -- but its Best-of-N margin is small.
* **Seeds.** Three seeds were run for `attention`, `cnn` and `mlp` under the
  final protocol; the other three heads and all ablations rest on one seed.
* **`max_length=512` truncation.** 8.4% of validation steps are not encoded,
  75% of them error steps, in 507 of 4,445 trajectories. Re-encoding the
  validation set at 2048 tokens changes every head's AUC by −0.002 to +0.008,
  none significant after Holm, and changes no ranking. The in-distribution
  control reads its "fully correct" target from the full labels; the pruning
  analysis classifies trajectories from the encoded steps only, which can count
  a trajectory whose only error was truncated as clean.
* **Training budget.** Best epochs range from 6 (`linear`) to 30; the two heads
  that stopped at the 30-epoch cap (`mlp`, `attention_pe`) were rerun with a
  60-epoch cap, selected the same epoch-30 checkpoint and early-stopped at 35.
* **Single encoder, single training corpus.** Math-Shepherd's Monte-Carlo step
  labels are noisy, and nothing here separates that from the architecture
  question. Best-of-N uses one generator (Qwen2.5-0.5B-Instruct) on one benchmark.

## Repository layout

Core pipeline:

- `prepare_data.py` — builds the data files: the Math-Shepherd train/validation
  split, the GSM8K Best-of-N candidates, and their step-format evaluation file
- `smoke_test.py` — runs every stage below on bundled sample data in minutes
- `encoder.py` — frozen step encoder; one hidden state per `ки` marker
- `precompute_embeddings.py`, `precompute_eval_embeddings.py` — build caches;
  `--lora_path` exports from a LoRA-adapted encoder instead
- `dataset.py` — Math-Shepherd parser and loader; `collate_fn` aligns labels to
  the step markers that survive tokenization
- `reward_heads.py` — the six heads behind one interface, plus causal variants
  (`*_causal`) of the four contextual ones
- `pqm_loss.py` — PQM comparative ranking loss, plus the pointwise step-BCE
  baseline it is compared against
- `train_from_cache.py` — trains a head from cached embeddings (`--loss pqm|bce`)
- `cached_steps.py` — in-memory (or memory-mapped) store of a cache, batches
  of any size or exactly the original shards
- `train_lora.py` — the LoRA arm: adapters on the encoder plus a value head
- `run_experiments.py` — resumable driver for every cache-based experiment
  above, the LoRA arm, and their evaluation

Evaluation (`eval/`):

- `eval_step_metrics.py` — held-out step metrics
- `eval_bon.py` — Best-of-N (argmax and PRM-weighted voting) against majority
  voting and an oracle, with paired, Holm-corrected differences to majority voting
- `eval_single_from_cache.py` — single-solution scoring from a dedicated cache
- `eval_single_from_step_cache.py` — the in-distribution single-solution control
- `eval_coin_flip_baseline.py`, `summarize_results.py`

Analysis (`analysis/`):

- `analyze_data_bias.py` — label audit, majority and position-only baselines
- `analyze_head_behavior.py` — stratified, first-error-boundary and perturbation
  analyses
- `analyze_offline_pruning.py` — causal-prefix scoring and risk-budgeted pruning
- `bootstrap_step_metrics.py`, `bootstrap_causal_metrics.py` — paired
  trajectory-level confidence intervals
- `compare_lora_frozen.py` — paired comparison of arms that may use different
  caches or checkpoints (LoRA vs frozen, and the loss / zeta / lr / seed ablations)
- `holm_correction.py` — Holm-Bonferroni correction over those pairwise
  comparisons, within a chosen family
- `holm_bon_family.py` — Holm across the Best-of-N selectors of several runs
  (e.g. frozen + LoRA) as one family

Run evaluation and analysis as modules from the repository root, for example
`python -m eval.eval_step_metrics ...`.

## Getting started

```bash
pip install -r requirements.txt
python -m unittest discover -v
python smoke_test.py
```

The unit tests need only `torch` and `numpy` and run on CPU in seconds;
`.github/workflows/tests.yml` runs them on every push and pull request.

`smoke_test.py` checks that the whole pipeline runs: on the 150 bundled
Math-Shepherd trajectories and 4 GSM8K questions it encodes with Qwen2.5-0.5B
(downloaded on first use, about 1 GB), trains all six heads and a BCE variant,
runs every evaluation and statistical script, Best-of-N, and a small LoRA pass.
It takes about 2.5 minutes on an RTX 4060; `--no_lora` skips the stage that
needs `peft`. Its numbers are meaningless by design. Without network access or
a GPU, `python tests/make_tiny_encoder.py runs/tiny_encoder` builds a tiny
random-weight Qwen2 encoder and tokenizer and `python smoke_test.py --model_name
runs/tiny_encoder` runs all stages on it on CPU in under two minutes.

To reproduce the reported experiments:

```bash
python prepare_data.py split       # Math-Shepherd -> data/train.jsonl, data/val.jsonl (grouped by question;
                                   # --split_by row for the per-row split behind the reported results)
python prepare_data.py relabel     # re-grade the bundled candidates -> data/gsm8k_qwen0.5b_bon16_relabeled.jsonl
python prepare_data.py convert     # relabeled candidates -> data/single_eval.jsonl
python precompute_embeddings.py --train_file data/train.jsonl --cache_dir cache/train_fixed
python precompute_embeddings.py --train_file data/val.jsonl --cache_dir cache/val_fixed
python precompute_eval_embeddings.py --eval_file data/single_eval.jsonl --cache_dir cache/single_eval
python run_experiments.py          # everything else; --dry_run lists the steps
```

**Training speed.** The cache holds one shard per encoder batch (16
trajectories), and training originally read one shard file per optimiser step
(about 27.5k steps per epoch; even the 897-parameter `linear` head took ~109 s
per epoch on the RTX 4060). `train_from_cache.py` now loads the cache once into
flat in-memory tensors (`cached_steps.py`, `--loader memory`, the default) and
by default still takes one shard per step, which reproduces the old runs bit
for bit (checked on the smoke-test cache for `linear`, `cnn`, `gru` and
`attention`: identical checkpoints and loss histories). `--batch_size N`
decouples the batch from the shards; larger batches change the optimisation,
so the reported learning rates do not carry over and need retuning.
`--merged_cache FILE --mmap` saves the merged store once and memory-maps it
afterwards, `--store_on_device` keeps it on the GPU, and `--loader shards` is
the original loader. The in-memory store needs about the size of the cache in
RAM (twice that while it is first merged). On a synthetic CPU benchmark (24k
trajectories, hidden 896; order of magnitude only, not the GPU setting) one
shard per step from memory was at most ~1.3x faster than reading files from the
page cache, while 256-trajectory batches made a `linear` epoch ~7x and an
`attention` epoch ~1.7x faster.

Two inputs are committed because they cannot be regenerated identically: the
GSM8K candidates were sampled without a fixed seed (`prepare_data.py generate`
re-samples them with one), and the original train/validation split used an
unrecorded shuffle, so `split` yields the same sizes but different validation
rows. Everything downstream is deterministic.

`TRAINING_WORKFLOW.md` is the step-by-step manual and `RESULTS_MAP.md` says
which result directory answers which question. Of the 47 tests, several are
invariants rather than unit tests -- pointwise heads must score prefixes and full
trajectories identically, plain attention must be permutation equivariant, a
multi-line final step must stay one step -- and each of those exists because the
corresponding bug had already corrupted a conclusion.

`requirements-cpu-test.txt` pins the CPU-only environment in which this
repository's tests and offline smoke test were last run (2026-11); it is not
the environment of the reported results. The results were produced with two environments: caches and the LoRA arm with
torch 2.13 / transformers 5.13 / peft 0.21, heads and evaluation with torch 2.5.1.
On unaffected records the encoder output matches the pre-fix caches to cosine
similarity ≥ 0.99996 (fp16 rounding). Encoding the 440k training trajectories
takes about 2.5 hours and the LoRA pass about 9.5 hours on an RTX 4060 (8 GB).
