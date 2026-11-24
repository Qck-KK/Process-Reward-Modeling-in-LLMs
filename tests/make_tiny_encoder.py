"""Build a tiny random-weight Qwen2 encoder and tokenizer for offline smoke tests.

    python tests/make_tiny_encoder.py runs/tiny_encoder
    python smoke_test.py --model_name runs/tiny_encoder

The tokenizer is a byte-level BPE trained on the bundled sample data, the model
has two layers of width 64 and random weights. Every pipeline stage then runs on
CPU in under two minutes without downloading Qwen2.5-0.5B. The numbers it
produces are meaningless; it only checks that the pipeline runs.
"""

import argparse
import json
import os

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2Model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sample_texts(n_bon_questions=20):
    texts = []
    with open(os.path.join(ROOT, "data", "dummy_train.jsonl"), encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            texts += [record.get("input", ""), record.get("label", "")]
    with open(os.path.join(ROOT, "data", "gsm8k_qwen0.5b_bon16.jsonl"), encoding="utf-8") as f:
        for index, line in enumerate(f):
            if index >= n_bon_questions:
                break
            record = json.loads(line)
            texts += [record["question"]] + [c["text"] for c in record["candidates"]]
    return texts


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir")
    parser.add_argument("--vocab_size", type=int, default=2000)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, special_tokens=["<|endoftext|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator(sample_texts(), trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<|endoftext|>",
                                   pad_token="<|endoftext|>")
    fast.save_pretrained(args.out_dir)

    torch.manual_seed(args.seed)
    config = Qwen2Config(vocab_size=len(fast) + 8, hidden_size=args.hidden_size,
                         intermediate_size=2 * args.hidden_size, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=4096)
    Qwen2Model(config).save_pretrained(args.out_dir)
    print("[tiny encoder] vocab {0}, hidden {1} -> {2}".format(
        len(fast), args.hidden_size, args.out_dir))


if __name__ == "__main__":
    main()
