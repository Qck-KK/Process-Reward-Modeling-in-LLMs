"""Holm correction of Best-of-N selectors across several runs as one family.

`eval.eval_bon` corrects the selector-vs-majority p-values within one run (36
selectors for the six frozen heads, 6 for the LoRA head). When a claim is made
about the frozen and the LoRA arm together -- "only LoRA beats majority voting"
-- the family is their union. This script pools the `bon_results.csv` of
several runs and re-applies Holm per k over all their selectors. The runs must
share the majority-voting reference (same candidates, subsets and seed), which
is checked.

    python -m analysis.holm_bon_family \
        --run frozen:results_final/bon_results.csv \
        --run lora:results_ablations/bon_lora/bon_results.csv \
        --out results_ablations/bon_holm_frozen_plus_lora.csv
"""

import argparse
import csv
import math

from analysis.holm_correction import holm_adjust


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def pool_runs(runs, ks=None, alpha=0.05):
    """runs: list of (name, rows). Returns pooled selector rows with combined Holm."""
    majority = {}
    pooled = []
    for name, rows in runs:
        for row in rows:
            k = int(row["k"])
            if ks and k not in ks:
                continue
            if row["head"] == "majority_vote":
                accuracy = float(row["accuracy"])
                if k in majority and abs(majority[k][1] - accuracy) > 1e-9:
                    raise ValueError(
                        "majority voting differs between runs {0} and {1} at k={2}: "
                        "they are not paired".format(majority[k][0], name, k))
                majority.setdefault(k, (name, accuracy))
                continue
            p_value = float(row["p_vs_majority"])
            if row["method"] == "reference" or math.isnan(p_value):
                continue
            pooled.append({
                "run": name, "head": row["head"], "agg": row["agg"],
                "method": row["method"], "k": k,
                "accuracy": float(row["accuracy"]),
                "diff_vs_majority": float(row["diff_vs_majority"]),
                "diff_vs_majority_ci_low": float(row["diff_vs_majority_ci_low"]),
                "diff_vs_majority_ci_high": float(row["diff_vs_majority_ci_high"]),
                "p_vs_majority": p_value,
                "p_holm_within_run": float(row["p_holm_vs_majority"]),
            })
    for k in sorted({row["k"] for row in pooled}):
        family = [row for row in pooled if row["k"] == k]
        for row, adjusted in zip(family, holm_adjust([row["p_vs_majority"] for row in family])):
            row["family_size"] = len(family)
            row["p_holm_combined"] = adjusted
            row["better_than_majority_combined"] = bool(
                adjusted < alpha and row["diff_vs_majority"] > 0)
            row["worse_than_majority_combined"] = bool(
                adjusted < alpha and row["diff_vs_majority"] < 0)
    return pooled


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", action="append", required=True, metavar="NAME:CSV",
                        help="Repeatable: a run name and its bon_results.csv.")
    parser.add_argument("--ks", nargs="+", type=int, default=None,
                        help="Only these k (default: all).")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    runs = []
    for spec in args.run:
        name, path = spec.split(":", 1)
        runs.append((name, read_rows(path)))
    pooled = pool_runs(runs, args.ks, args.alpha)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pooled[0].keys()))
        writer.writeheader()
        writer.writerows(pooled)
    for k in sorted({row["k"] for row in pooled}):
        family = [row for row in pooled if row["k"] == k]
        better = [r for r in family if r["better_than_majority_combined"]]
        worse = sum(r["worse_than_majority_combined"] for r in family)
        print("k={0}: family of {1}; {2} better, {3} worse than majority after Holm".format(
            k, len(family), len(better), worse))
        for row in better:
            print("    better: {run} {head} {agg} {method} (p_holm={p_holm_combined:.4f})".format(**row))
    print("[saved] " + args.out)


if __name__ == "__main__":
    main()
