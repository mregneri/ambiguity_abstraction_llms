r"""
measure_newword_accuracy.py
===========================

Purpose
-------
Evaluate next-token (top-1) accuracy for three GPT model variants
(FULL, HALF, BASE) on two subsets of their test sets:

  A) Sentences containing inserted "new words" (in the FULL test set)
  B) Sentences without any new word (in the FULL test set)

Each variant must have exactly three checkpoints:
    ckpt1.pt, ckpt2.pt, ckpt3.pt

Inputs (config .py with top-level dict `config`):
    kind             : "HOM" or "HYP"
    map_json         : path to homonym/hypernym map JSON
    full_model_dir   : directory containing ckpt1.pt, ckpt2.pt, ckpt3.pt for FULL
    half_model_dir   : directory containing ckpt1.pt, ckpt2.pt, ckpt3.pt for HALF
    base_model_dir   : directory containing ckpt1.pt, ckpt2.pt, ckpt3.pt for BASE
    full_test_json   : JSON list of token lists for FULL (also used for HALF)
    half_test_json   : JSON list of token lists for HALF (equal to FULL split)
    base_test_json   : JSON list of token lists for BASE
    out_dir          : output directory for CSV/JSON results

Outputs
--------
<out_dir>/newword_accuracy.csv
<out_dir>/newword_accuracy.json

Notes
-----
- Test JSONs are lists of token lists. We re-join tokens with spaces to get text.
- Sentence indices are computed on FULL and re-used for HALF and BASE (strict alignment).
- Detection of “contains new word” uses the same token-chunk logic as the insertion script
  (split into \w+ and punctuation parts, case-insensitive match on \w+ chunks).

Usage
-----
  python measure_newword_accuracy.py path/to/config.py
"""

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Set, Dict

import torch
import tiktoken

# --- Add absolute path for GPT model imports ---
sys.path.append("/home/nina/ambiguity/gpt_pipelines")
from model import GPTConfig, GPT    # Same model classes as in train.py


# ----------------------------
# Fixed defaults (no config knobs)
# ----------------------------
SPLITTER = "regex"            # 'regex' → (?<=[.!?])\s+ ; or 'newline' / custom regex if you ever change it here
BATCH_SIZE = 16
MAX_TOKENS_PER_SENT = 256     # (do not exceed the model’s block size)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NEWWORD_KEY = None            # set to a key string if your map JSON ever uses a specific field for surface forms

# Token-chunk logic (mirrors insertion script)
WORD_PART_RX = re.compile(r'\w+|[^\w\s]', re.UNICODE)
WORD_ONLY_RX = re.compile(r'\w+', re.UNICODE)  # “word” chunks only


# -----------------------------------------------------------------------------
# Config loading
# -----------------------------------------------------------------------------
def load_eval_config(config_path: str) -> dict:
    """Import a Python file and return its top-level `config` dict."""
    spec = importlib.util.spec_from_file_location("eval_config_module", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = getattr(module, "config", None)
    if not isinstance(cfg, dict):
        raise ValueError("Eval config must define a top-level dict named `config`.")
    return cfg


# -----------------------------------------------------------------------------
# Test data loading (mirrors prepare.py)
# -----------------------------------------------------------------------------
def load_token_docs_from_json(json_path: str) -> List[str]:
    """
    Read a JSON file holding a list of token lists (per document),
    then join tokens with spaces to reconstruct document text.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of token lists in {json_path}")
    docs = []
    for tokens in data:
        if isinstance(tokens, list) and tokens:
            docs.append(" ".join(tokens).strip())
    return docs


def split_documents_into_sentences(documents: List[str], splitter: str) -> List[str]:
    r"""
    Split documents into sentences using a shared strategy across all variants.
    splitter:
      - 'regex'   : uses (?<=[.!?])\s+
      - 'newline' : split on newline characters
      - otherwise : treated as a custom regex pattern
    """
    if splitter == "newline":
        sentences: List[str] = []
        for doc in documents:
            sentences.extend([s for s in doc.splitlines() if s.strip()])
        return sentences

    pattern = r'(?<=[.!?])\s+' if splitter == "regex" else splitter
    sentences: List[str] = []
    for doc in documents:
        parts = re.split(pattern, doc)
        sentences.extend([s.strip() for s in parts if s.strip()])
    return sentences


# -----------------------------------------------------------------------------
# New-word detection
# -----------------------------------------------------------------------------
def load_new_surface_forms(map_json_path: str, kind: str, newword_key: Optional[str]) -> Set[str]:
    """
    Extract surface forms of inserted words from the homonym/hypernym map.

    Supports:
      1) Simple dict: { "base1,base2": "newtoken", ... }
      2) Dicts with fields: {"new": "..."} or {"new_words": ["...", ...]}
      3) Nested dict/list structures (recurses)
    """
    with open(map_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    found: Set[str] = set()

    def collect(obj):
        if isinstance(obj, dict):
            # Common structured keys
            if newword_key and isinstance(obj.get(newword_key), str):
                found.add(obj[newword_key])
            if isinstance(obj.get("new"), str):
                found.add(obj["new"])
            if isinstance(obj.get("new_words"), list):
                for w in obj["new_words"]:
                    if isinstance(w, str):
                        found.add(w)

            # IMPORTANT: also collect string values of simple {pair: "newtoken"} maps
            for v in obj.values():
                if isinstance(v, str):
                    s = v.strip()
                    if s:
                        found.add(s)
                else:
                    collect(v)

        elif isinstance(obj, list):
            for it in obj:
                collect(it)

        elif isinstance(obj, str):
            s = obj.strip()
            if s:
                found.add(s)

    # If your file isn’t namespaced by kind (HOM/HYP), this simply recurses the whole dict.
    collect(data.get(kind, data))

    return {w for w in found if isinstance(w, str) and w}



def split_sentence_indices_by_newword_presence_tokenchunks(
    sentences: List[str],
    new_words: Set[str],
) -> Tuple[List[int], List[int]]:
    r"""
    Return (indices_with_newword, indices_without_newword) using the EXACT SAME
    chunking logic as the insertion script:
      - Split sentence into whitespace tokens
      - For each token, split into parts with r'\w+|[^\w\s]'
      - Count a hit if any part is a full WORD_ONLY_RX and matches new_words (case-insensitive)
    """
    new_lower = {w.lower() for w in new_words}
    idx_with, idx_without = [], []

    for i, sent in enumerate(sentences):
        found = False
        for tok in sent.split():  # safe because original docs were joined by spaces
            parts = WORD_PART_RX.findall(tok)
            for part in parts:
                if WORD_ONLY_RX.fullmatch(part) and part.lower() in new_lower:
                    found = True
                    break
            if found:
                break

        (idx_with if found else idx_without).append(i)

    return idx_with, idx_without


def assert_equal_sentence_counts(*sentence_lists: List[str]) -> None:
    """Ensure FULL/HALF/BASE sentences are index-aligned; raise if not."""
    lengths = [len(s) for s in sentence_lists]
    if len(set(lengths)) != 1:
        raise ValueError(
            f"Sentence counts must align across FULL/HALF/BASE, got {lengths}. "
            f"Use the same splitter and corresponding test corpora."
        )


# -----------------------------------------------------------------------------
# Tokenization & batching
# -----------------------------------------------------------------------------
def load_gpt2_tokenizer():
    """Return the GPT-2 BPE tokenizer (tiktoken) used in training."""
    return tiktoken.get_encoding("gpt2")


def encode_texts(enc, texts: List[str], max_tokens: Optional[int]) -> List[List[int]]:
    """Encode each text with optional truncation."""
    ids_per_text: List[List[int]] = []
    for t in texts:
        ids = enc.encode(t)
        if max_tokens is not None and len(ids) > max_tokens:
            ids = ids[:max_tokens]
        ids_per_text.append(ids)
    return ids_per_text


def build_next_token_tensors(token_id_lists: List[List[int]], pad_id: int = 0):
    """
    Create input (x) and next-token targets (y) tensors.
    For each sequence: y[t] = x[t+1]; last token has no target → set to -100 (ignore).
    """
    if not token_id_lists:
        return torch.empty(0, 0, dtype=torch.long), torch.empty(0, 0, dtype=torch.long)

    max_len = max(len(ids) for ids in token_id_lists)
    batch = len(token_id_lists)

    x = torch.full((batch, max_len), pad_id, dtype=torch.long)
    y = torch.full((batch, max_len), -100, dtype=torch.long)

    for i, ids in enumerate(token_id_lists):
        L = len(ids)
        if L == 0:
            continue
        x[i, :L] = torch.tensor(ids, dtype=torch.long)
        if L > 1:
            y[i, :L-1] = torch.tensor(ids[1:], dtype=torch.long)
    return x, y


# -----------------------------------------------------------------------------
# Model loading (exactly three checkpoints)
# -----------------------------------------------------------------------------
def expected_three_ckpts(model_dir: str) -> List[str]:
    """
    Return exactly [ckpt1.pt, ckpt2.pt, ckpt3.pt] from a model directory.
    Raise an error if any is missing.
    """
    p = Path(model_dir)
    if not p.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    ckpts = [p / f"ckpt{i}.pt" for i in (1, 2, 3)]
    missing = [str(c) for c in ckpts if not c.exists()]
    if missing:
        raise FileNotFoundError(
            "Expected exactly three checkpoints per model: ckpt1.pt, ckpt2.pt, ckpt3.pt.\n"
            f"Missing: {missing}\nModel dir: {model_dir}"
        )
    return [str(c) for c in ckpts]


def load_model_from_checkpoint(ckpt_path: str, device: str):
    """
    Rebuild GPT model from a training checkpoint:
      ckpt['model_args'] → GPTConfig
      ckpt['model']      → state_dict
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model" not in ckpt or "model_args" not in ckpt:
        raise RuntimeError(f"Unexpected checkpoint format: {ckpt_path}")
    model_cfg = GPTConfig(**ckpt["model_args"])
    model = GPT(model_cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval().to(device)
    return model


# -----------------------------------------------------------------------------
# Accuracy computation
# -----------------------------------------------------------------------------
@torch.no_grad()
def next_token_accuracy_for_indices(
    model,
    tokenizer,
    all_sentences: List[str],
    selected_indices: List[int],
    device: str,
    batch_size: int,
    max_tokens_per_sent: Optional[int],
) -> Tuple[float, int]:
    """
    Compute next-token accuracy on `all_sentences[selected_indices]`.

    Returns:
        (accuracy, n_targets) where n_targets is the number of next-token positions evaluated.
    """
    if not selected_indices:
        return float("nan"), 0

    selected_sentences = [all_sentences[i] for i in selected_indices]
    total_correct = 0
    total_targets = 0

    for start in range(0, len(selected_sentences), batch_size):
        batch_texts = selected_sentences[start:start + batch_size]
        token_ids = encode_texts(tokenizer, batch_texts, max_tokens=max_tokens_per_sent)
        x, y = build_next_token_tensors(token_ids, pad_id=0)
        if x.numel() == 0:
            continue
        x, y = x.to(device), y.to(device)

        out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out  # [B, T, vocab]
        preds = torch.argmax(logits, dim=-1)

        mask = (y != -100)
        correct = (preds == y) & mask
        total_correct += correct.sum().item()
        total_targets += mask.sum().item()

    if total_targets == 0:
        return float("nan"), 0
    return total_correct / total_targets, total_targets


def evaluate_three_runs_for_variant(
    checkpoint_paths: List[str],
    sentences: List[str],
    indices_with_new: List[int],
    indices_without_new: List[int],
    tokenizer,
    device: str,
    batch_size: int,
    max_tokens_per_sent: int,
) -> Dict[str, List[Tuple[float, int]]]:
    """
    Evaluate a model variant (FULL/HALF/BASE) for exactly three checkpoints.
    Returns dict with per-run results for both subsets.
    """
    results = {"contains_new": [], "no_new": []}
    for ckpt in checkpoint_paths:
        model = load_model_from_checkpoint(ckpt, device=device)
        acc_with, n_with = next_token_accuracy_for_indices(
            model, tokenizer, sentences, indices_with_new, device, batch_size, max_tokens_per_sent
        )
        acc_without, n_without = next_token_accuracy_for_indices(
            model, tokenizer, sentences, indices_without_new, device, batch_size, max_tokens_per_sent
        )
        results["contains_new"].append((acc_with, n_with))
        results["no_new"].append((acc_without, n_without))
    return results


def mean(values: List[float]) -> float:
    """Arithmetic mean with NaN-pass-through semantics."""
    if not values:
        return float("nan")
    return sum(values) / len(values)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to eval config .py containing a top-level `config` dict.")
    args = parser.parse_args()

    cfg = load_eval_config(args.config_path)

    # Input & output paths
    kind = cfg["kind"]
    map_json = cfg["map_json"]
    full_model_dir = cfg["full_model_dir"]
    half_model_dir = cfg["half_model_dir"]
    base_model_dir = cfg["base_model_dir"]
    full_test_json = cfg["full_test_json"]
    half_test_json = cfg["half_test_json"]
    base_test_json = cfg["base_test_json"]
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Derive readable model names for CSV
    full_name = os.path.basename(full_model_dir)
    half_name = os.path.basename(half_model_dir)
    base_name = os.path.basename(base_model_dir)

    print("\n=== PATHS ===")
    print(f"Full model dir: {full_model_dir}")
    print(f"Half model dir: {half_model_dir}")
    print(f"Base model dir: {base_model_dir}")
    print(f"Full test JSON: {full_test_json}")
    print(f"Half test JSON: {half_test_json}")
    print(f"Base test JSON: {base_test_json}")
    print(f"Map JSON:       {map_json}")
    print(f"Output dir:     {out_dir}\n")

    # Load and split test sets
    full_docs = load_token_docs_from_json(full_test_json)
    half_docs = load_token_docs_from_json(half_test_json)
    base_docs = load_token_docs_from_json(base_test_json)
    full_sentences = split_documents_into_sentences(full_docs, SPLITTER)
    half_sentences = split_documents_into_sentences(half_docs, SPLITTER)
    base_sentences = split_documents_into_sentences(base_docs, SPLITTER)
    assert_equal_sentence_counts(full_sentences, half_sentences, base_sentences)

    # Identify new words & sentence indices (computed on FULL set)
    new_words = load_new_surface_forms(map_json, kind, NEWWORD_KEY)
    idx_with_new, idx_without_new = split_sentence_indices_by_newword_presence_tokenchunks(
        full_sentences, new_words
    )

    # Precompute counts once (reused for console + summary)
    num_with = len(idx_with_new)
    num_without = len(idx_without_new)
    total_sent = num_with + num_without

    # -----------------------------------------------------------------------------
    # Save sentence splits for inspection
    # -----------------------------------------------------------------------------
    split_dir = os.path.join(out_dir, "sentence_splits")
    os.makedirs(split_dir, exist_ok=True)

    # Save all full/half/base sentences
    for name, data in [("full", full_sentences), ("half", half_sentences), ("base", base_sentences)]:
        path = os.path.join(split_dir, f"{name}_sentences.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # Save subsets for FULL ("contains_new" vs "no_new")
    full_contains_new = [full_sentences[i] for i in idx_with_new]
    full_no_new = [full_sentences[i] for i in idx_without_new]
    with open(os.path.join(split_dir, "full_contains_new.json"), "w", encoding="utf-8") as f:
        json.dump(full_contains_new, f, ensure_ascii=False, indent=2)
    with open(os.path.join(split_dir, "full_no_new.json"), "w", encoding="utf-8") as f:
        json.dump(full_no_new, f, ensure_ascii=False, indent=2)

    # IMPORTANT: HALF and BASE subsets reuse the *FULL* indices to ensure alignment
    half_contains_new = [half_sentences[i] for i in idx_with_new]
    half_no_new = [half_sentences[i] for i in idx_without_new]
    base_contains_new = [base_sentences[i] for i in idx_with_new]
    base_no_new = [base_sentences[i] for i in idx_without_new]
    with open(os.path.join(split_dir, "half_contains_new.json"), "w", encoding="utf-8") as f:
        json.dump(half_contains_new, f, ensure_ascii=False, indent=2)
    with open(os.path.join(split_dir, "half_no_new.json"), "w", encoding="utf-8") as f:
        json.dump(half_no_new, f, ensure_ascii=False, indent=2)
    with open(os.path.join(split_dir, "base_contains_new.json"), "w", encoding="utf-8") as f:
        json.dump(base_contains_new, f, ensure_ascii=False, indent=2)
    with open(os.path.join(split_dir, "base_no_new.json"), "w", encoding="utf-8") as f:
        json.dump(base_no_new, f, ensure_ascii=False, indent=2)

    # Save indices for reproducibility
    indices_info = {"contains_new": idx_with_new, "no_new": idx_without_new}
    with open(os.path.join(split_dir, "indices.json"), "w", encoding="utf-8") as f:
        json.dump(indices_info, f, indent=2)

    # Console summary of splits
    print(f"💾 Saved sentence splits for inspection → {split_dir}")
    print(f"   Sentences with new words: {num_with:,}")
    print(f"   Sentences without new words: {num_without:,}")
    print(f"   Total sentences: {total_sent:,}\n")

    tokenizer = load_gpt2_tokenizer()
    ckpts_full = expected_three_ckpts(full_model_dir)
    ckpts_half = expected_three_ckpts(half_model_dir)
    ckpts_base = expected_three_ckpts(base_model_dir)

    # Evaluate models
    res_full = evaluate_three_runs_for_variant(
        ckpts_full, full_sentences, idx_with_new, idx_without_new, tokenizer, DEVICE, BATCH_SIZE, MAX_TOKENS_PER_SENT
    )
    res_half = evaluate_three_runs_for_variant(
        ckpts_half, half_sentences, idx_with_new, idx_without_new, tokenizer, DEVICE, BATCH_SIZE, MAX_TOKENS_PER_SENT
    )
    res_base = evaluate_three_runs_for_variant(
        ckpts_base, base_sentences, idx_with_new, idx_without_new, tokenizer, DEVICE, BATCH_SIZE, MAX_TOKENS_PER_SENT
    )

    def pack_variant(r):
        return {
            "contains_new": {"runs": r["contains_new"], "mean": mean([a for a, _ in r["contains_new"]])},
            "no_new":       {"runs": r["no_new"],       "mean": mean([a for a, _ in r["no_new"]])},
        }

    packed = {"full": pack_variant(res_full), "half": pack_variant(res_half), "base": pack_variant(res_base)}

    # -----------------------------------------------------------------------------
    # Write a compact human-readable summary
    # -----------------------------------------------------------------------------
    summary_path = os.path.join(out_dir, "summary.txt")
    pct_with = (num_with / total_sent * 100.0) if total_sent else float("nan")
    pct_without = (num_without / total_sent * 100.0) if total_sent else float("nan")

    def subset_line(variant_name: str, key: str):
        runs = packed[variant_name][key]["runs"]
        mean_acc = packed[variant_name][key]["mean"]
        n_total = sum(n for _, n in runs)
        run_str = "  ".join(
            f"run{i+1}={('nan' if (a!=a) else f'{a:.6f}')} (n={n})"
            for i, (a, n) in enumerate(runs)
        )
        mean_str = "nan" if (mean_acc != mean_acc) else f"{mean_acc:.6f}"
        return f"{variant_name.upper():>4} {key:>12}: {run_str}  | mean={mean_str} | n_total={n_total}"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== New-Word Accuracy Summary ===\n")
        f.write(f"Kind: {kind}\n")
        f.write(f"Map JSON: {map_json}\n")
        f.write(f"Models:\n  FULL={full_name}\n  HALF={half_name}\n  BASE={base_name}\n")
        f.write(f"New words collected: {len(new_words)}\n")
        f.write("\n-- Sentence splits (based on FULL) --\n")
        f.write(f"contains_new: {num_with} ({pct_with:.2f}%)\n")
        f.write(f"no_new:       {num_without} ({pct_without:.2f}%)\n")
        f.write(f"total:        {total_sent}\n")

        f.write("\n-- Accuracy (per variant / subset) --\n")
        f.write(subset_line("full", "contains_new") + "\n")
        f.write(subset_line("full", "no_new") + "\n")
        f.write(subset_line("half", "contains_new") + "\n")
        f.write(subset_line("half", "no_new") + "\n")
        f.write(subset_line("base", "contains_new") + "\n")
        f.write(subset_line("base", "no_new") + "\n")

    print(f"📝 Summary written to: {summary_path}")

    # Console summary
    def print_variant_table(name, model_name, data):
        print(f"\n=== {model_name} ({name.upper()}) ===")
        for subset in ("contains_new", "no_new"):
            runs = data[subset]["runs"]
            mean_acc = data[subset]["mean"]
            print(f"{subset:>14}: ", end="")
            for i, (acc, n) in enumerate(runs, 1):
                acc_str = "nan" if acc != acc else f"{acc:.6f}"
                print(f"run{i}={acc_str} (n={n})  ", end="")
            mean_str = "nan" if mean_acc != mean_acc else f"{mean_acc:.6f}"
            print(f"| mean={mean_str}")

    print("\n=== New-Word Accuracy (3 runs) ===")
    print_variant_table("full", full_name, packed["full"])
    print_variant_table("half", half_name, packed["half"])
    print_variant_table("base", base_name, packed["base"])

    # Save results
    csv_path = os.path.join(out_dir, "newword_accuracy.csv")
    json_path = os.path.join(out_dir, "newword_accuracy.json")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model_name", "variant", "subset", "run", "accuracy", "n_targets"])
        for (variant, name) in [("full", full_name), ("half", half_name), ("base", base_name)]:
            pv = packed[variant]
            for subset in ("contains_new", "no_new"):
                for i, (acc, n) in enumerate(pv[subset]["runs"], 1):
                    writer.writerow([name, variant, subset, i, f"{acc:.6f}" if acc == acc else "nan", n])
                writer.writerow([name, variant, subset, "mean", f"{pv[subset]['mean']:.6f}" if pv[subset]['mean'] == pv[subset]['mean'] else "nan", ""])

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "kind": kind,
                "new_words_count": len(new_words),
                "full": packed["full"],
                "half": packed["half"],
                "base": packed["base"],
                "indices": {"with": idx_with_new, "without": idx_without_new},
                "meta": {
                    "splitter": SPLITTER,
                    "max_tokens_per_sent": MAX_TOKENS_PER_SENT,
                    "batch_size": BATCH_SIZE,
                    "device": DEVICE,
                    "ckpts": {"full": ckpts_full, "half": ckpts_half, "base": ckpts_base},
                    "model_names": {"full": full_name, "half": half_name, "base": base_name},
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nSaved results:\n  {csv_path}\n  {json_path}\n")


if __name__ == "__main__":
    main()