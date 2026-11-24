import os
import tempfile
import unittest

import torch
import torch.nn as nn

from analysis.analyze_offline_pruning import (
    aggregate_pruning_records,
    calibrate_threshold,
    causal_prefix_scores,
    classify_trajectory,
    stopping_index,
)
from analysis.holm_correction import bootstrap_p_value, correct_pairwise_rows, holm_adjust
from eval.eval_utils import (
    average_precision,
    best_threshold_accuracy,
    deterministic_example_split,
    roc_auc,
)
from cached_steps import CachedSteps
from dataset import parse_math_shepherd
from prepare_data import (
    grouped_split,
    gsm8k_gold,
    is_correct,
    last_number,
    normalize_number,
    question_of,
    question_overlap,
    relabel_questions,
    solution_steps,
)
from eval.eval_bon import (
    build_groups,
    draw_subsets,
    majority_choice,
    reference_curves,
    weighted_vote_choice,
)
from pqm_loss import bce_step_loss, pqm_loss
from reward_heads import (
    CAUSAL_HEADS,
    REPORTED_HEADS,
    AttentionPoolingHead,
    AttentionPoolingPositionHead,
    CNNHead,
    LinearHead,
    build_reward_head,
)


class MetricTests(unittest.TestCase):
    def test_deterministic_split_is_disjoint_and_reproducible(self):
        calibration_a, test_a = deterministic_example_split(20, 0.5, 42)
        calibration_b, test_b = deterministic_example_split(20, 0.5, 42)
        self.assertTrue(torch.equal(calibration_a, calibration_b))
        self.assertTrue(torch.equal(test_a, test_b))
        self.assertFalse((calibration_a & test_a).any())
        self.assertTrue((calibration_a | test_a).all())

    def test_tied_scores_have_unbiased_auc_and_ap(self):
        labels = torch.tensor([1, 0, 1, 0, 1])
        scores = torch.ones(5)
        self.assertAlmostEqual(roc_auc(scores, labels), 0.5)
        self.assertAlmostEqual(average_precision(scores, labels), 0.6, places=6)

    def test_threshold_search_can_predict_all_positive(self):
        scores = torch.ones(4)
        labels = torch.ones(4, dtype=torch.long)
        _, accuracy = best_threshold_accuracy(scores, labels)
        self.assertEqual(accuracy, 1.0)


class PrepareDataTests(unittest.TestCase):
    def test_candidate_steps_split_on_blank_lines_including_whitespace(self):
        text = "Intro:\n\n1. a\n   detail\n   \n2. b\r\n\r\n\n\nAnswer: 4\n"
        self.assertEqual(solution_steps(text),
                         ["Intro:", "1. a\n   detail", "2. b", "Answer: 4"])

    def test_answer_extraction(self):
        self.assertEqual(gsm8k_gold("so 3+4=7\n#### 7"), "7")
        self.assertEqual(last_number("It costs $1,250, i.e. 1250 in total."), "1250")
        self.assertIsNone(last_number("no digits here"))

    def test_gold_answer_keeps_thousands(self):
        """'#### 1,200' used to be read as 1 (the regex stopped at the comma)."""
        self.assertEqual(gsm8k_gold("4 * 300 = <<4*300=1200>>1200\n#### 1,200"), "1200")
        self.assertEqual(gsm8k_gold("#### 1,450,000"), "1450000")
        self.assertEqual(gsm8k_gold("#### -3"), "-3")
        self.assertEqual(gsm8k_gold("a #### 5 and #### 6"), "6")
        self.assertIsNone(gsm8k_gold("no marker 12"))

    def test_prediction_reads_whole_decimals(self):
        """'26.00' used to be read as 00, '$0.90' as 90."""
        self.assertEqual(last_number("Total: $26.00"), "26")
        self.assertEqual(last_number("She has $0.90 left."), "0.9")
        self.assertEqual(last_number("so 2.50 dollars"), "2.5")
        self.assertEqual(last_number("The answer is 18."), "18")
        self.assertEqual(last_number("$1,250.50 in total"), "1250.5")

    def test_prediction_numbers_and_separators(self):
        self.assertEqual(last_number("a population of 1,30,000 people"), "130000")
        self.assertEqual(last_number("pick 3,4,5"), "5")
        self.assertEqual(last_number("x = -3"), "-3")
        self.assertEqual(last_number("pages 5-3"), "3")
        self.assertEqual(last_number("1,200 and then 7"), "7")

    def test_normalisation_and_grading(self):
        self.assertEqual(normalize_number("1,200"), "1200")
        self.assertEqual(normalize_number("-0.0"), "0")
        self.assertIsNone(normalize_number("abc"))
        self.assertEqual(is_correct("18.0", "18"), 1)
        self.assertEqual(is_correct("0.9", "90"), 0)
        self.assertEqual(is_correct(None, "1"), 0)

    def test_relabel_keeps_old_values_and_counts_changes(self):
        questions = [{"question": "Q1", "gold_answer": "1", "candidates": [
            {"text": "so 1,200", "pred_num": "200", "final_correct": 0},
            {"text": "so 1", "pred_num": "1", "final_correct": 1}]},
            {"question": "Q2", "gold_answer": "26", "candidates": [
            {"text": "costs $26.00", "pred_num": "00", "final_correct": 0}]}]
        golds = [{"question": "Q1", "gold_answer": "1200"},
                 {"question": "Q2", "gold_answer": "26"}]
        records, stats = relabel_questions(questions, golds)
        self.assertEqual([c["final_correct"] for c in records[0]["candidates"]], [1, 0])
        self.assertEqual(records[1]["candidates"][0]["final_correct"], 1)
        self.assertEqual(records[0]["gold_answer_v0"], "1")
        self.assertEqual(records[0]["candidates"][0]["pred_num_v0"], "200")
        self.assertEqual(stats["label_changed"], 3)
        self.assertEqual(stats["questions_with_label_change"], 2)
        # Relabeling its own output is a no-op on the labels and keeps the originals.
        again, _ = relabel_questions(records, golds)
        self.assertEqual(again[0]["candidates"][0]["final_correct_v0"], 0)
        self.assertEqual(again, records)
        with self.assertRaises(RuntimeError):
            relabel_questions(questions, [{"question": "other", "gold_answer": "1"}] * 2)


class SplitTests(unittest.TestCase):
    @staticmethod
    def synthetic_rows(n_questions=300, per_question=5):
        rows = []
        for q in range(n_questions):
            for t in range(per_question):
                # Same question with different spacing/case must still group.
                question = "What is {0} plus {0}?".format(q)
                if t % 2:
                    question = "  " + question.upper().replace(" ", "\n ")
                rows.append({"input": question + " Step 1: try {0} ки".format(t),
                             "label": question + " Step 1: try {0} +".format(t)})
        return rows

    def test_grouped_split_has_no_question_overlap(self):
        rows = self.synthetic_rows()
        train, val = grouped_split(rows, 0.2, seed=42)
        self.assertEqual(sorted(train + val), list(range(len(rows))))
        self.assertTrue(val and train)
        stats = question_overlap([rows[i] for i in train], [rows[i] for i in val])
        self.assertEqual(stats["shared_questions"], 0)
        self.assertEqual(stats["val_rows_with_train_question"], 0)
        # Whole questions: every validation question keeps all five trajectories.
        self.assertEqual(len(val) % 5, 0)
        self.assertAlmostEqual(len(val) / len(rows), 0.2, delta=0.06)

    def test_grouped_split_is_deterministic_and_order_independent(self):
        rows = self.synthetic_rows(100, 3)
        _, val_a = grouped_split(rows, 0.3, seed=7)
        _, val_b = grouped_split(rows[::-1], 0.3, seed=7)
        questions_a = {question_of(rows[i]) for i in val_a}
        questions_b = {question_of(rows[::-1][i]) for i in val_b}
        self.assertEqual(questions_a, questions_b)
        _, val_c = grouped_split(rows, 0.3, seed=8)
        self.assertNotEqual(set(val_a), set(val_c))

    def test_row_split_leaks_questions(self):
        """What the old per-row split does with repeated questions."""
        rows = self.synthetic_rows(50, 4)
        generator = torch.Generator().manual_seed(0)
        order = torch.randperm(len(rows), generator=generator).tolist()
        val, train = order[:20], order[20:]
        stats = question_overlap([rows[i] for i in train], [rows[i] for i in val])
        self.assertGreater(stats["shared_questions"], 0)


class CachedStepsTests(unittest.TestCase):
    @staticmethod
    def write_shards(directory, n_shards=5, rows=4, hidden=6, seed=0):
        generator = torch.Generator().manual_seed(seed)
        paths = []
        for index in range(n_shards):
            lengths = torch.randint(0, 6, (rows,), generator=generator)
            width = max(int(lengths.max()), 1) + index % 2   # some shards wider than needed
            mask = torch.arange(width).unsqueeze(0) < lengths.unsqueeze(1)
            step_hidden = torch.randn(rows, width, hidden, generator=generator).half()
            step_hidden[~mask] = 0
            labels = torch.randint(0, 2, (rows, width), generator=generator)
            labels[~mask] = -100
            labels[0, 0] = -100 if mask[0, 0] else labels[0, 0]  # unlabeled valid step
            path = os.path.join(directory, "shard_{0:06d}.pt".format(index))
            torch.save({"step_hidden": step_hidden, "step_mask": mask, "labels": labels}, path)
            paths.append(path)
        return paths

    def test_shard_batches_rebuild_the_shards_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_shards(directory)
            store = CachedSteps.from_shards(paths)
            for index, path in enumerate(paths):
                shard = torch.load(path)
                step_hidden, step_mask, labels = store.shard_batch(index)
                self.assertTrue(torch.equal(step_hidden, shard["step_hidden"]))
                self.assertTrue(torch.equal(step_mask, shard["step_mask"]))
                self.assertTrue(torch.equal(labels, shard["labels"]))

    def test_random_batches_cover_every_trajectory_once(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CachedSteps.from_shards(self.write_shards(directory))
            generator = torch.Generator().manual_seed(0)
            batches = list(store.random_batches(7, generator))
            self.assertEqual(len(batches), store.n_batches(7))
            self.assertEqual(sum(b[0].shape[0] for b in batches), store.n_trajectories)
            n_steps = sum(int(b[1].sum()) for b in batches)
            self.assertEqual(n_steps, store.hidden.shape[0])
            for step_hidden, step_mask, labels in batches:
                self.assertTrue((step_hidden[~step_mask] == 0).all())
                self.assertTrue((labels[~step_mask] == -100).all())

    def test_saved_store_round_trips_with_mmap(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CachedSteps.from_shards(self.write_shards(directory))
            merged = os.path.join(directory, "merged", "store.pt")
            store.save(merged)
            loaded = CachedSteps.from_cache_dir(directory, merged, mmap=True)
            for a, b in zip(store.shard_batches(), loaded.shard_batches()):
                for x, y in zip(a, b):
                    self.assertTrue(torch.equal(x, y))

    def test_non_prefix_mask_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "shard_000000.pt")
            torch.save({"step_hidden": torch.zeros(1, 3, 2), "labels": torch.tensor([[1, -100, 1]]),
                        "step_mask": torch.tensor([[True, False, True]])}, path)
            with self.assertRaises(ValueError):
                CachedSteps.from_shards([path])


class ParserTests(unittest.TestCase):
    def test_multiline_final_step_is_one_step_with_one_label(self):
        """The '# Answer' block must not become extra (negative) steps."""
        question = "What is 2+2?"
        body = " Step 1: Add them. {0}\nStep 2: So it is 4.\n\n# Answer\n\n4 {1}\n"
        record = parse_math_shepherd(question + body.format("ки", "ки"),
                                     question + body.format("+", "+"))
        self.assertEqual(record["question"], question)
        self.assertEqual(record["labels"], [1, 1])
        self.assertEqual(record["steps"][1], "Step 2: So it is 4.\n\n# Answer\n\n4")

    def test_labels_follow_marker_positions(self):
        text = "Q? Step 1: a {0}\nStep 2: b {1}\nStep 3: c {2}\n"
        record = parse_math_shepherd(text.format("ки", "ки", "ки"), text.format("+", "-", "-"))
        self.assertEqual(record["labels"], [1, 0, 0])
        self.assertEqual(record["steps"], ["Step 1: a", "Step 2: b", "Step 3: c"])

    def test_mismatched_label_text_is_rejected(self):
        self.assertIsNone(parse_math_shepherd("Q? Step 1: a ки\n", "Q? Step 1: b +\n"))
        self.assertIsNone(parse_math_shepherd("Q? Step 1: a ки\n", "Q? Step 1: a x\n"))
        self.assertIsNone(parse_math_shepherd("Q? no markers\n", "Q? no markers\n"))


class MultipleComparisonTests(unittest.TestCase):
    def test_holm_matches_hand_computed_values(self):
        adjusted = holm_adjust([0.01, 0.04, 0.03, 0.005])
        for got, expected in zip(adjusted, [0.03, 0.06, 0.06, 0.02]):
            self.assertAlmostEqual(got, expected)

    def test_holm_ignores_nan_in_family_size(self):
        adjusted = holm_adjust([0.02, float("nan")])
        self.assertAlmostEqual(adjusted[0], 0.02)
        self.assertTrue(adjusted[1] != adjusted[1])

    def test_identical_arms_are_not_significant(self):
        """A difference of exactly zero in every resample must give p = 1."""
        self.assertEqual(bootstrap_p_value(0.0, 2000, prob_a_worse=0.0), 1.0)
        self.assertAlmostEqual(bootstrap_p_value(0.0, 2000, prob_a_worse=1.0), 2 / 2001)

    def test_family_restricts_the_correction(self):
        def row(a, b):
            return {"arm_a": a, "arm_b": b, "roc_auc_prob_a_better": "0.0",
                    "roc_auc_prob_a_worse": "1.0", "roc_auc_significant": "True"}
        rows = [row("gru_lr1e-4", "gru_lr3e-5"), row("gru_lr1e-4", "cnn_lr1e-4"),
                row("cnn_lr1e-4", "cnn_lr3e-5")]
        out, _, n = correct_pairwise_rows(rows, 2000, family="prefix")
        self.assertEqual(n, 2)
        self.assertEqual(out[1]["roc_auc_significant_holm"], "")
        self.assertAlmostEqual(out[0]["roc_auc_p_holm"], 2 * 2 / 2001)
        _, _, n_suffix = correct_pairwise_rows(rows, 2000, family="suffix")
        self.assertEqual(n_suffix, 1)

    def test_bootstrap_p_value_is_floored_and_two_sided(self):
        self.assertAlmostEqual(bootstrap_p_value(0.0, 2000), 2 / 2001)
        self.assertAlmostEqual(bootstrap_p_value(1.0, 2000), 2 / 2001)
        self.assertEqual(bootstrap_p_value(0.5, 2000), 1.0)


class BestOfNTests(unittest.TestCase):
    def test_uniform_weights_reduce_to_majority_vote(self):
        preds = ["3", "5", "5", "3", "7"]
        weights = torch.ones(5)
        for subset in ([0, 1, 2, 3, 4], [0, 1, 3], [4, 0, 1]):
            self.assertEqual(weighted_vote_choice(preds, weights, subset),
                             majority_choice(preds, subset))

    def test_weights_can_overturn_the_majority(self):
        preds = ["3", "5", "5"]
        weights = torch.tensor([0.9, 0.3, 0.3])
        self.assertEqual(weighted_vote_choice(preds, weights, [0, 1, 2]), 0)

    def test_unparseable_answers_do_not_vote(self):
        preds = ["", "", "4"]
        weights = torch.tensor([0.9, 0.9, 0.1])
        self.assertEqual(weighted_vote_choice(preds, weights, [0, 1, 2]), 2)
        self.assertEqual(weighted_vote_choice(["", ""], weights, [0, 1]), 0)

    def test_reference_selectors_on_full_candidate_sets(self):
        # Question 0: answers 4, 4, 5 with 5 correct; question 1: no correct candidate.
        groups = build_groups([0, 0, 0, 1, 1])
        correct = torch.tensor([0, 0, 1, 0, 0])
        preds = ["4", "4", "5", "1", "2"]
        generator = torch.Generator().manual_seed(0)
        subsets = draw_subsets(groups, [1, 3], 4, generator)
        self.assertEqual(subsets[3], [[(0, 1, 2)], [(3, 4)]])
        curves = reference_curves(subsets, correct, preds)
        self.assertEqual(curves[("oracle", "-", "reference")][3].tolist(), [1.0, 0.0])
        self.assertEqual(curves[("majority_vote", "-", "reference")][3].tolist(), [0.0, 0.0])
        self.assertEqual(curves[("random", "-", "reference")][3].tolist(), [0.0, 0.0])


class ArchitectureTests(unittest.TestCase):
    def test_plain_attention_is_permutation_equivariant(self):
        """Self-attention without positions cannot tell step order apart."""
        torch.manual_seed(0)
        head = AttentionPoolingHead(hidden_size=32, n_heads=4).eval()
        step_hidden = torch.randn(1, 5, 32)
        step_mask = torch.ones(1, 5, dtype=torch.bool)
        order = torch.tensor([4, 3, 2, 1, 0])
        with torch.no_grad():
            straight = head(step_hidden, step_mask)[0]
            permuted = head(step_hidden[:, order], step_mask[:, order])[0]
        self.assertTrue(torch.allclose(straight.flip(0), permuted, atol=1e-5))

    def test_positional_attention_is_order_sensitive(self):
        """The sinusoidal variant must break that equivariance, at equal size."""
        torch.manual_seed(0)
        head = AttentionPoolingPositionHead(hidden_size=32, n_heads=4).eval()
        step_hidden = torch.randn(1, 5, 32)
        step_mask = torch.ones(1, 5, dtype=torch.bool)
        order = torch.tensor([4, 3, 2, 1, 0])
        with torch.no_grad():
            straight = head(step_hidden, step_mask)[0]
            permuted = head(step_hidden[:, order], step_mask[:, order])[0]
        self.assertFalse(torch.allclose(straight.flip(0), permuted, atol=1e-5))

        plain = AttentionPoolingHead(hidden_size=32, n_heads=4)
        self.assertEqual(sum(p.numel() for p in head.parameters()),
                         sum(p.numel() for p in plain.parameters()))


class CausalHeadTests(unittest.TestCase):
    HIDDEN = 16

    def build(self, name, seed=0):
        torch.manual_seed(seed)
        return build_reward_head(name, self.HIDDEN).eval()

    def test_causal_variants_are_registered(self):
        self.assertEqual(CAUSAL_HEADS, ["attention_causal", "attention_pe_causal",
                                        "cnn_causal", "gru_causal"])
        self.assertNotIn("gru_causal", REPORTED_HEADS)
        self.assertFalse(self.build("gru_causal").gru.bidirectional)
        self.assertTrue(self.build("gru").gru.bidirectional)

    @torch.no_grad()
    def test_output_at_t_ignores_later_steps(self):
        """Perturbing steps > t must leave the score of every step <= t unchanged."""
        generator = torch.Generator().manual_seed(1)
        x = torch.randn(3, 7, self.HIDDEN, generator=generator)
        mask = torch.ones(3, 7, dtype=torch.bool)
        mask[1, 5:] = False
        mask[2, 3:] = False
        for name in CAUSAL_HEADS:
            head = self.build(name)
            base = head(x, mask)
            for t in range(6):
                perturbed = x.clone()
                perturbed[:, t + 1:] = torch.randn(3, 7 - t - 1, self.HIDDEN, generator=generator) * 5
                out = head(perturbed, mask)
                valid = mask[:, :t + 1]
                self.assertTrue(torch.allclose(out[:, :t + 1][valid], base[:, :t + 1][valid],
                                               atol=1e-5), (name, t))

    @torch.no_grad()
    def test_bidirectional_heads_do_see_later_steps(self):
        """Control: the reported contextual heads fail the same check."""
        generator = torch.Generator().manual_seed(2)
        x = torch.randn(2, 6, self.HIDDEN, generator=generator)
        mask = torch.ones(2, 6, dtype=torch.bool)
        for name in ("cnn", "gru", "attention", "attention_pe"):
            head = self.build(name)
            perturbed = x.clone()
            perturbed[:, 3:] += 5
            self.assertFalse(torch.allclose(head(perturbed, mask)[:, :3], head(x, mask)[:, :3],
                                            atol=1e-5), name)

    @torch.no_grad()
    def test_causal_heads_score_prefixes_like_full_trajectories(self):
        generator = torch.Generator().manual_seed(3)
        x = torch.randn(4, 6, self.HIDDEN, generator=generator)
        mask = torch.ones(4, 6, dtype=torch.bool)
        mask[0, 4:] = False
        mask[3, 1:] = False
        for name in CAUSAL_HEADS:
            head = self.build(name)
            causal = causal_prefix_scores(head, x, mask)
            self.assertTrue(torch.allclose(causal[mask], head(x, mask)[mask], atol=1e-5), name)

    def test_causal_variants_keep_parameter_counts(self):
        count = lambda name: sum(p.numel() for p in self.build(name).parameters())
        for name in ("cnn", "attention", "attention_pe"):
            self.assertEqual(count(name), count(name + "_causal"))
        self.assertLess(count("gru_causal"), count("gru"))


class StabilityTests(unittest.TestCase):
    def test_attention_head_survives_all_padded_trajectory(self):
        """A zero-step trajectory must not turn the whole batch into NaN."""
        torch.manual_seed(0)
        head = AttentionPoolingHead(hidden_size=32, n_heads=4)
        step_hidden = torch.randn(2, 4, 32)
        step_mask = torch.tensor([[True, True, False, False],
                                  [False, False, False, False]])
        q_values = head(step_hidden, step_mask)
        self.assertFalse(torch.isnan(q_values).any())

        q_values.sum().backward()
        for name, param in head.named_parameters():
            self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient in {name}")


    def test_pqm_loss_is_finite_for_extreme_rewards(self):
        rewards = torch.tensor([[1000.0, -1000.0, 0.0]], requires_grad=True)
        labels = torch.tensor([[1, 0, -100]])
        loss = pqm_loss(rewards, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(rewards.grad).all())

    def test_pqm_loss_handles_unsupervised_example(self):
        rewards = torch.randn(1, 3, requires_grad=True)
        labels = torch.full((1, 3), -100)
        loss = pqm_loss(rewards, labels)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss.detach()), 0.0)

    def test_pqm_loss_excludes_unsupervised_example_from_batch_mean(self):
        """An all-padding row must neither poison nor dilute the other rows."""
        torch.manual_seed(0)
        rewards = torch.randn(2, 3)
        labels = torch.tensor([[1, 0, -100], [-100, -100, -100]])
        batch_loss = pqm_loss(rewards, labels)
        alone_loss = pqm_loss(rewards[:1], labels[:1])
        self.assertTrue(torch.isfinite(batch_loss))
        self.assertAlmostEqual(float(batch_loss), float(alone_loss), places=6)

    def test_bce_step_loss_ignores_padding_and_matches_torch(self):
        rewards = torch.tensor([[2.0, -1.0, 99.0]])
        labels = torch.tensor([[1, 0, -100]])
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor([2.0, -1.0]), torch.tensor([1.0, 0.0])
        )
        self.assertAlmostEqual(float(bce_step_loss(rewards, labels)), float(expected), places=6)
        empty = bce_step_loss(torch.randn(1, 2, requires_grad=True), torch.full((1, 2), -100))
        self.assertEqual(float(empty.detach()), 0.0)

    def test_cnn_valid_outputs_ignore_padded_values(self):
        torch.manual_seed(0)
        head = CNNHead(hidden_size=8, channels=4)
        mask = torch.tensor([[True, True, True, False, False]])
        first = torch.randn(1, 5, 8)
        second = first.clone()
        second[:, 3:] = 1000.0
        with torch.no_grad():
            first_q = head(first, mask)
            second_q = head(second, mask)
        self.assertTrue(torch.allclose(first_q[:, :3], second_q[:, :3], atol=1e-6))


class OfflinePruningTests(unittest.TestCase):
    def test_pointwise_head_causal_scores_equal_full_scores(self):
        """A pointwise head cannot see the future, so prefix scoring is a no-op."""
        torch.manual_seed(0)
        head = LinearHead(hidden_size=16)
        step_hidden = torch.randn(3, 5, 16)
        step_mask = torch.tensor([[True, True, True, False, False],
                                  [True, True, False, False, False],
                                  [True, True, True, True, True]])
        causal = causal_prefix_scores(head, step_hidden, step_mask)
        full = head(step_hidden, step_mask)
        self.assertTrue(torch.allclose(causal[step_mask], full[step_mask], atol=1e-5))


    def test_trajectory_types_separate_recovery_cases(self):
        self.assertEqual(classify_trajectory(torch.tensor([1, 1, -100]))["kind"], "clean")
        self.assertEqual(classify_trajectory(torch.tensor([1, 0, 0]))["kind"], "monotone_error")
        self.assertEqual(classify_trajectory(torch.tensor([1, 0, 1]))["kind"], "recovery")

    def test_causal_prefix_scoring_hides_future_steps(self):
        class SequenceMeanHead(nn.Module):
            def forward(self, step_hidden, step_mask):
                masked = step_hidden.squeeze(-1) * step_mask
                mean = masked.sum(dim=1) / step_mask.sum(dim=1)
                return mean.unsqueeze(1).expand(-1, step_hidden.shape[1])

        hidden = torch.tensor([[[1.0], [3.0], [5.0]]])
        mask = torch.tensor([[True, True, True]])
        scores = causal_prefix_scores(SequenceMeanHead(), hidden, mask)
        self.assertTrue(torch.allclose(scores, torch.tensor([[1.0, 2.0, 3.0]])))

    def test_threshold_calibration_respects_clean_risk_budget(self):
        scores = torch.tensor([
            [0.1, 0.8],
            [0.2, 0.8],
            [0.3, 0.8],
            [0.4, 0.8],
        ])
        labels = torch.ones(4, 2, dtype=torch.long)
        calibration = torch.ones(4, dtype=torch.bool)
        threshold, observed, n_clean = calibrate_threshold(
            scores, labels, calibration, "single_low", 0.25
        )
        self.assertEqual(n_clean, 4)
        self.assertLessEqual(observed, 0.25)
        self.assertEqual(
            sum(stopping_index(row, threshold, "single_low") is not None for row in scores),
            1,
        )

    def test_safe_saving_excludes_false_prunes(self):
        records = [
            {"kind": "clean", "length": 4, "first_error": None,
             "stop": 1, "saved": 2, "delay": None},
            {"kind": "monotone_error", "length": 5, "first_error": 2,
             "stop": 2, "saved": 2, "delay": 0},
            {"kind": "monotone_error", "length": 5, "first_error": 3,
             "stop": 1, "saved": 3, "delay": -2},
        ]
        metrics = aggregate_pruning_records(records)
        self.assertAlmostEqual(metrics["clean_false_prune_rate"], 1.0)
        self.assertAlmostEqual(metrics["pre_error_false_prune_rate"], 0.5)
        self.assertAlmostEqual(metrics["error_coverage"], 0.5)
        self.assertAlmostEqual(metrics["safe_step_saving_rate"], 2 / 14)


if __name__ == "__main__":
    unittest.main()
