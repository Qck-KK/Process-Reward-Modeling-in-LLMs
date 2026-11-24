"""Best-of-N reranking evaluation on Qwen's own GSM8K candidates."""

import argparse
import collections
import csv
import glob
import json
import os

import torch

from eval.eval_utils import (
    DEFAULT_HEADS,
    HEAD_CHOICES,
    aggregate_trajectory_scores,
    get_device,
)
from analysis.holm_correction import bootstrap_p_value, holm_adjust
from reward_heads import build_reward_head


def pad_and_concat(tensors, value):
    max_steps = max(tensor.shape[1] for tensor in tensors)
    return torch.cat([
        torch.nn.functional.pad(tensor, (0, max_steps - tensor.shape[1]), value=value)
        for tensor in tensors
    ])


@torch.no_grad()
def score_cache(head, shard_paths, device, agg):
    head.eval()
    scores = []
    for path in shard_paths:
        shard = torch.load(path, map_location=device)
        step_hidden = shard["step_hidden"].to(device).float()
        step_mask = shard["step_mask"].to(device).bool()
        q_values = head(step_hidden, step_mask)
        scores.append(aggregate_trajectory_scores(q_values, step_mask, mode=agg).cpu())
    return torch.cat(scores)


def percentile_ci(samples, alpha=0.05):
    tensor = torch.tensor(samples, dtype=torch.float64)
    tensor = tensor[torch.isfinite(tensor)]
    if tensor.numel() == 0:
        return float("nan"), float("nan")
    return (float(torch.quantile(tensor, alpha / 2)),
            float(torch.quantile(tensor, 1 - alpha / 2)))


def majority_choice(preds, subset):
    counts = collections.Counter(preds[i] for i in subset if preds[i] not in ("", None))
    if not counts:
        return subset[0]
    top = counts.most_common(1)[0][0]
    for i in subset:
        if preds[i] == top:
            return i
    return subset[0]


def weighted_vote_choice(preds, weights, subset):
    totals = {}
    for i in subset:
        if preds[i] not in ("", None):
            totals[preds[i]] = totals.get(preds[i], 0.0) + float(weights[i])
    if not totals:
        return max(subset, key=lambda i: float(weights[i]))
    top = max(totals, key=totals.get)
    for i in subset:
        if preds[i] == top:
            return i
    return subset[0]


def build_groups(question_of):
    n_questions = max(question_of) + 1
    groups = [[] for _ in range(n_questions)]
    for index, qid in enumerate(question_of):
        groups[qid].append(index)
    return groups


def draw_subsets(groups, ks, subsets_per_question, generator):
    """Fixed candidate subsets, shared by every selector so comparisons are paired."""
    subsets = {}
    for k in ks:
        per_question = []
        for members in groups:
            if k >= len(members):
                per_question.append([tuple(members)])
                continue
            draws = []
            for _ in range(subsets_per_question):
                order = torch.randperm(len(members), generator=generator)[:k]
                draws.append(tuple(members[int(i)] for i in order))
            per_question.append(draws)
        subsets[k] = per_question
    return subsets


def evaluate_selector(pick_fn, subsets, correct):
    out = {}
    for k, per_question_subsets in subsets.items():
        per_question = torch.zeros(len(per_question_subsets))
        for qid, draws in enumerate(per_question_subsets):
            hits = [int(correct[pick_fn(subset)]) for subset in draws]
            per_question[qid] = sum(hits) / len(hits)
        out[k] = per_question
    return out


def reference_curves(subsets, correct, preds):
    curves = {
        ("random", "-", "reference"): evaluate_selector(lambda subset: subset[0], subsets, correct),
        ("oracle", "-", "reference"): evaluate_selector(
            lambda subset: max(subset, key=lambda i: int(correct[i])), subsets, correct),
    }
    if preds is not None:
        curves[("majority_vote", "-", "reference")] = evaluate_selector(
            lambda subset: majority_choice(preds, list(subset)), subsets, correct)
    return curves


def summarize_curves(curves, majority, ks, n_questions, bootstrap_samples, generator):
    rows = []
    question_draws = [
        torch.randint(n_questions, (n_questions,), generator=generator)
        for _ in range(bootstrap_samples)
    ]
    for (head_name, agg, method), curve in curves.items():
        for k in ks:
            per_question = curve[k]
            point = float(per_question.mean())
            samples = [float(per_question[pick].mean()) for pick in question_draws]
            low, high = percentile_ci(samples)
            diff = diff_low = diff_high = p_value = float("nan")
            if majority is not None:
                delta = per_question - majority[k]
                diff = float(delta.mean())
                resampled = [float(delta[pick].mean()) for pick in question_draws]
                diff_low, diff_high = percentile_ci(resampled)
                if method != "reference":
                    p_value = bootstrap_p_value(
                        sum(d > 0 for d in resampled) / len(resampled), len(resampled),
                        sum(d < 0 for d in resampled) / len(resampled))
            rows.append({
                "head": head_name,
                "agg": agg,
                "method": method,
                "k": k,
                "accuracy": point,
                "ci_low": low,
                "ci_high": high,
                "diff_vs_majority": diff,
                "diff_vs_majority_ci_low": diff_low,
                "diff_vs_majority_ci_high": diff_high,
                "p_vs_majority": p_value,
                "p_holm_vs_majority": float("nan"),
                "n_questions": n_questions,
            })

    for k in ks:
        family = [row for row in rows if row["k"] == k and row["p_vs_majority"] == row["p_vs_majority"]]
        for row, adjusted in zip(family, holm_adjust([row["p_vs_majority"] for row in family])):
            row["p_holm_vs_majority"] = adjusted
    return rows


def load_source_preds(source_file, records):
    """pred_num per candidate from the Best-of-N file, aligned with eval_file.

    Refuses a source whose final_correct disagrees with eval_file: that means
    single_eval.jsonl was converted from a differently graded file (e.g. the
    committed file with the pre-fix labels instead of the relabeled one).
    """
    source = [json.loads(line) for line in open(source_file, encoding="utf-8")]
    candidates = [c for d in source for c in d["candidates"]]
    if len(candidates) != len(records):
        raise RuntimeError("source_file does not align with eval_file")
    mismatched = sum(int(c["final_correct"]) != int(r["final_correct"])
                     for c, r in zip(candidates, records))
    if mismatched:
        raise RuntimeError(
            "{0} candidates are graded differently in source_file and eval_file; "
            "re-run `prepare_data.py convert` on the same source".format(mismatched))
    return [c.get("pred_num", "") for c in candidates]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default=None,
                        help="Candidate embedding cache; required unless --references_only.")
    parser.add_argument("--eval_file", required=True,
                        help="single_eval.jsonl carrying question_id/candidate_id.")
    parser.add_argument("--source_file", default=None,
                        help="Best-of-N file with pred_num, for majority voting.")
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Head checkpoints; required unless --references_only.")
    parser.add_argument("--checkpoint_pattern", default="{head}_head.pt")
    parser.add_argument("--heads", nargs="+", default=list(DEFAULT_HEADS))
    parser.add_argument("--aggs", nargs="+", default=["min", "mean", "last"])
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--subsets_per_question", type=int, default=20)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--out_name", default="bon_results")
    parser.add_argument("--references_only", action="store_true",
                        help="Only random / oracle / majority voting; needs no cache or "
                             "checkpoints. Draws the same subsets and bootstrap "
                             "resamples as a full run with the same seed.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if not args.references_only and not (args.cache_dir and args.checkpoint_dir):
        parser.error("--cache_dir and --checkpoint_dir are required unless --references_only")

    device = get_device(args.device)
    os.makedirs(args.results_dir, exist_ok=True)

    records = [json.loads(line) for line in open(args.eval_file, encoding="utf-8")]
    correct = torch.tensor([int(r["final_correct"]) for r in records])
    groups = build_groups([r["question_id"] for r in records])
    n_questions = len(groups)

    preds = load_source_preds(args.source_file, records) if args.source_file else None

    shard_paths, hidden_size = [], None
    if not args.references_only:
        shard_paths = sorted(glob.glob(os.path.join(args.cache_dir, "shard_*.pt")))
        assert shard_paths, "No cached shards in " + args.cache_dir
        with open(os.path.join(args.cache_dir, "hidden_size.txt")) as f:
            hidden_size = int(f.read().strip())
        n_cached = sum(torch.load(p, map_location="cpu")["final_correct"].shape[0]
                       for p in shard_paths)
        if n_cached != len(records):
            raise RuntimeError("cache holds {0} candidates, eval_file {1}".format(
                n_cached, len(records)))

    # Fixed subsets, shared by every head and baseline so the comparison is paired.
    # Scoring heads draws nothing from the generator, so a references-only run
    # sees exactly the subsets and resamples of a full run.
    generator = torch.Generator().manual_seed(args.seed)
    subsets = draw_subsets(groups, args.ks, args.subsets_per_question, generator)

    curves = {}
    heads = [] if args.references_only else args.heads
    for head_name in heads:
        checkpoint = os.path.join(
            args.checkpoint_dir, args.checkpoint_pattern.format(head=head_name)
        )
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(checkpoint)
        head = build_reward_head(head_name, hidden_size).to(device)
        head.load_state_dict(torch.load(checkpoint, map_location=device))
        for agg in args.aggs:
            scores = score_cache(head, shard_paths, device, agg)
            curves[(head_name, agg, "argmax")] = evaluate_selector(
                lambda subset: max(subset, key=lambda i: float(scores[i])), subsets, correct
            )
            if preds is not None:
                weights = torch.sigmoid(scores)
                curves[(head_name, agg, "weighted_vote")] = evaluate_selector(
                    lambda subset: weighted_vote_choice(preds, weights, list(subset)),
                    subsets, correct
                )
            print("  scored {0:<14} agg={1}".format(head_name, agg))

    curves.update(reference_curves(subsets, correct, preds))
    majority = curves.get(("majority_vote", "-", "reference"))
    rows = summarize_curves(curves, majority, args.ks, n_questions,
                            args.bootstrap_samples, generator)

    out_csv = os.path.join(args.results_dir, args.out_name + ".csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    out_json = os.path.join(args.results_dir, args.out_name + ".json")
    with open(out_json, "w") as f:
        json.dump({
            "cache_dir": args.cache_dir,
            "checkpoint_dir": args.checkpoint_dir,
            "eval_file": args.eval_file,
            "source_file": args.source_file,
            "references_only": args.references_only,
            "n_questions": n_questions,
            "n_candidates": len(records),
            "subsets_per_question": args.subsets_per_question,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "rows": rows,
        }, f, indent=2)
    print("[saved] " + out_csv + " / " + out_json)


if __name__ == "__main__":
    main()
