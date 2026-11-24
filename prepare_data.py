"""Build the data files: Math-Shepherd split, GSM8K Best-of-N candidates and their grading.

Subcommands: split, generate, gold, relabel, convert.
"""

import argparse
import decimal
import json
import os
import re

GSM8K_SYSTEM_PROMPT = ("You are a helpful mathematical reasoning assistant. Please solve "
                       "the math problem step by step and end your response with the "
                       "final answer.")


def split(args):
    from datasets import load_dataset

    dataset = load_dataset("peiyi9979/Math-Shepherd", split="train")
    parts = dataset.train_test_split(test_size=args.val_fraction, seed=args.seed)
    os.makedirs(os.path.dirname(args.train_out) or ".", exist_ok=True)
    for name, path in (("train", args.train_out), ("test", args.val_out)):
        with open(path, "w", encoding="utf-8") as f:
            for row in parts[name]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("[split] {0:,} trajectories -> {1}".format(len(parts[name]), path))


# A number as written in a solution: digits with optional digit-group commas
# ("1,200", also the Indian "1,30,000") and an optional decimal part ("26.00"),
# never starting inside another number. Commas that do not form digit groups
# separate numbers ("3,4,5" is three numbers). A leading "-" is a sign only when
# it does not join two words or numbers, so "5-3" yields 5 and 3 while
# "x = -3" yields -3.
NUMBER_PATTERN = re.compile(
    r"(?:(?<![\w.])-)?(?<![\d.])"
    r"(?:\d{1,3}(?:,\d{2})*(?:,\d{3})+(?!\d)(?!,\d)|\d+)(?:\.\d+)?")


def normalize_number(text):
    """Canonical string for a number: no commas, no trailing decimal zeros.

    "1,200" -> "1200", "26.00" -> "26", "2.50" -> "2.5", "-0" -> "0".
    Returns None for anything that is not a plain decimal number.
    """
    if text is None:
        return None
    text = str(text).strip().replace(",", "")
    try:
        value = decimal.Decimal(text)
    except decimal.InvalidOperation:
        return None
    if not value.is_finite():
        return None
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f")


def gsm8k_gold(answer):
    """Gold answer of a GSM8K solution: the number after the final '####'."""
    if "####" not in answer:
        return None
    match = NUMBER_PATTERN.search(answer.rsplit("####", 1)[1])
    return normalize_number(match.group(0)) if match else None


def last_number(text):
    """Last complete number in a generated solution, normalised."""
    matches = NUMBER_PATTERN.findall(text)
    return normalize_number(matches[-1]) if matches else None


def is_correct(pred, gold):
    return int(gold is not None and pred is not None
               and normalize_number(pred) == normalize_number(gold))


def generate(args):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if args.limit:
        dataset = dataset.select(range(args.limit))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for index, item in enumerate(dataset):
            messages = [{"role": "system", "content": GSM8K_SYSTEM_PROMPT},
                        {"role": "user", "content": item["question"]}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True)
            inputs = tokenizer([prompt] * args.n_samples, return_tensors="pt",
                               padding=True).to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                         temperature=args.temperature, top_p=args.top_p,
                                         do_sample=True, pad_token_id=tokenizer.pad_token_id)
            gold = gsm8k_gold(item["answer"])
            candidates = []
            for output in outputs:
                text = tokenizer.decode(output[inputs["input_ids"].shape[1]:],
                                        skip_special_tokens=True)
                pred = last_number(text)
                candidates.append({"text": text,
                                   "final_correct": is_correct(pred, gold),
                                   "pred_num": pred})
            f.write(json.dumps({"question": item["question"], "gold_answer": gold,
                                "candidates": candidates}, ensure_ascii=False) + "\n")
            f.flush()
            print("[generate] question {0}/{1}".format(index + 1, len(dataset)), flush=True)


def read_gold_answers(path):
    """Gold answers, one per GSM8K test question, in file order.

    Accepts the official GSM8K jsonl (question, answer with '#### n') or the
    compact file written by `gold` (question, gold_answer).
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            gold = (normalize_number(item["gold_answer"]) if "gold_answer" in item
                    else gsm8k_gold(item["answer"]))
            rows.append({"question": item["question"], "gold_answer": gold})
    return rows


def gold(args):
    if args.gsm8k_file:
        rows = read_gold_answers(args.gsm8k_file)
    else:
        from datasets import load_dataset
        rows = [{"question": item["question"], "gold_answer": gsm8k_gold(item["answer"])}
                for item in load_dataset("openai/gsm8k", "main", split="test")]
    missing = sum(row["gold_answer"] is None for row in rows)
    if missing:
        raise RuntimeError("{0} questions have no parseable gold answer".format(missing))
    with open(args.out, "w", encoding="utf-8") as f:
        for question_id, row in enumerate(rows):
            f.write(json.dumps({"question_id": question_id, "question": row["question"],
                                "gold_answer": row["gold_answer"]}, ensure_ascii=False) + "\n")
    print("[gold] {0:,} gold answers -> {1}".format(len(rows), args.out))


def relabel_questions(questions, golds):
    """Re-extract pred_num and final_correct with the current answer parser.

    The original values are kept as gold_answer_v0 / pred_num_v0 /
    final_correct_v0 so the change stays auditable. Returns the new records and
    counts of what changed.
    """
    if len(questions) > len(golds):
        raise RuntimeError("more Best-of-N questions than gold answers")
    stats = {"questions": 0, "candidates": 0, "gold_changed": 0, "pred_changed": 0,
             "label_changed": 0, "label_0_to_1": 0, "label_1_to_0": 0,
             "questions_with_label_change": 0}
    out = []
    for item, reference in zip(questions, golds):
        if item["question"].strip() != reference["question"].strip():
            raise RuntimeError("question text does not match the gold file: "
                               + item["question"][:80])
        gold_answer = reference["gold_answer"]
        old_gold = item.get("gold_answer_v0", item.get("gold_answer"))
        record = {"question": item["question"], "gold_answer": gold_answer,
                  "gold_answer_v0": old_gold, "candidates": []}
        stats["questions"] += 1
        stats["gold_changed"] += int(old_gold != gold_answer)
        changed = False
        for candidate in item["candidates"]:
            old_pred = candidate.get("pred_num_v0", candidate.get("pred_num"))
            old_label = int(candidate.get("final_correct_v0", candidate["final_correct"]))
            pred = last_number(candidate["text"])
            label = is_correct(pred, gold_answer)
            record["candidates"].append({
                "text": candidate["text"], "final_correct": label, "pred_num": pred,
                "final_correct_v0": old_label, "pred_num_v0": old_pred})
            stats["candidates"] += 1
            stats["pred_changed"] += int(pred != old_pred)
            if label != old_label:
                changed = True
                stats["label_changed"] += 1
                stats["label_0_to_1" if label else "label_1_to_0"] += 1
        stats["questions_with_label_change"] += int(changed)
        out.append(record)
    return out, stats


def relabel(args):
    with open(args.source, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    records, stats = relabel_questions(questions, read_gold_answers(args.gold_file))
    if os.path.abspath(args.out) == os.path.abspath(args.source):
        raise RuntimeError("refusing to overwrite the source file; choose another --out")
    with open(args.out, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("[relabel] " + json.dumps(stats))
    print("[relabel] -> " + args.out)
    if args.stats_out:
        with open(args.stats_out, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)


def solution_steps(text):
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def convert(args):
    with open(args.source, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        questions = questions[:args.limit]
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for question_id, item in enumerate(questions):
            for candidate_id, candidate in enumerate(item["candidates"]):
                f.write(json.dumps({"question": item["question"],
                                    "steps": solution_steps(candidate["text"]),
                                    "final_correct": candidate["final_correct"],
                                    "question_id": question_id,
                                    "candidate_id": candidate_id}, ensure_ascii=False) + "\n")
                n += 1
    print("[convert] {0:,} candidates from {1:,} questions -> {2}".format(
        n, len(questions), args.out))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("split", help="Math-Shepherd train/validation split")
    p.add_argument("--val_fraction", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_out", default="data/train.jsonl")
    p.add_argument("--val_out", default="data/val.jsonl")
    p.set_defaults(func=split)

    p = sub.add_parser("generate", help="sample Best-of-N candidates on GSM8K")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="first N questions only; 0 = all")
    p.add_argument("--out", default="data/gsm8k_qwen0.5b_bon16.jsonl")
    p.set_defaults(func=generate)

    p = sub.add_parser("gold", help="GSM8K test gold answers -> compact gold file")
    p.add_argument("--gsm8k_file", default=None,
                   help="official GSM8K test.jsonl; default: download openai/gsm8k")
    p.add_argument("--out", default="data/gsm8k_test_gold.jsonl")
    p.set_defaults(func=gold)

    p = sub.add_parser("relabel",
                       help="re-grade the Best-of-N candidates with the current answer parser")
    p.add_argument("--source", default="data/gsm8k_qwen0.5b_bon16.jsonl")
    p.add_argument("--gold_file", default="data/gsm8k_test_gold.jsonl")
    p.add_argument("--out", default="data/gsm8k_qwen0.5b_bon16_relabeled.jsonl")
    p.add_argument("--stats_out", default=None)
    p.set_defaults(func=relabel)

    p = sub.add_parser("convert", help="candidates -> step-format evaluation file")
    p.add_argument("--source", default="data/gsm8k_qwen0.5b_bon16_relabeled.jsonl",
                   help="run `relabel` first; the committed file carries the old labels")
    p.add_argument("--limit", type=int, default=0, help="first N questions only; 0 = all")
    p.add_argument("--out", default="data/single_eval.jsonl")
    p.set_defaults(func=convert)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
