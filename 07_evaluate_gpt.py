#!/usr/bin/env python3
"""
07_evaluate_gpt.py
==================

Evaluates trained GPT models. Computes, per checkpoint and per batch category:

  Accuracy metrics (next-token prediction):
    overall, contains_new, no_new, contains_component, no_component,
    either (new OR component), neither

  Perplexity metrics:
    ppl_mod   — perplexity of the modified (experimental) model
    ppl_base  — perplexity of the base model on the SAME test tokens
                ⚠️  ppl_base is only meaningful for "overall" and "neither"
                    categories. In categories containing pseudowords, the base
                    model encounters token IDs it was never trained on, making
                    ppl_base uninterpretable. Those values are marked N/A.

Batch classification mirrors the insertion rules exactly:
  decode X → split with r'\w+|[^\w\s]' → match \w+ parts (case-insensitive).

Usage:
    python 07_evaluate_gpt.py --corpus recipe --n 10 --kind HOM
    python 07_evaluate_gpt.py --all          # all conditions
    python 07_evaluate_gpt.py --base         # base models only (ppl only)

Output:
    evaluation/new_comp_accuracy/<model_name>/eval_results.txt
"""

import os
import re
import sys
import json
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import tiktoken

from model import GPTConfig, GPT
from config import (
    CORPORA, COUNTS, KINDS, CHECKPOINT_NAMES, N_RUNS,
    EVAL_BLOCK_SIZE, EVAL_BATCH_SIZE, NEW_COMP_EVAL_DIR,
    get_gpt_model_dir, get_gpt_data_dir, get_pseudoword_map,
    get_inflection_map,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Categories ────────────────────────────────────────────────────────────────
CATEGORIES = ["overall", "contains_new", "no_new",
              "contains_component", "no_component",
              "either", "neither"]

# ppl_base is only meaningful for these two categories
PPL_BASE_VALID = {"overall", "neither"}

# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(ckpt_path: str):
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()
    return model

def get_test_data(data_dir: str) -> np.ndarray:
    return np.memmap(os.path.join(data_dir, "test.bin"), dtype=np.uint16, mode="r")

def iter_batches(test_data: np.ndarray):
    bs, bk = EVAL_BATCH_SIZE, EVAL_BLOCK_SIZE
    for i in range(0, len(test_data) - bk, bk * bs):
        xs, ys = [], []
        for j in range(bs):
            s, e = i + j * bk, i + j * bk + bk
            if e + 1 > len(test_data): break
            xs.append(torch.from_numpy(test_data[s:e].astype(np.int64)))
            ys.append(torch.from_numpy(test_data[s+1:e+1].astype(np.int64)))
        if xs:
            yield torch.stack(xs).to(DEVICE), torch.stack(ys).to(DEVICE)

# ── Word-set loading (mirrors insertion logic exactly) ────────────────────────
def load_word_sets(map_path: str, inflection_path: str):
    with open(map_path) as f:       repl_map = json.load(f)
    with open(inflection_path) as f: master  = json.load(f)

    # New words (pseudowords)
    new_words = []
    seen = set()
    for v in repl_map.values():
        w = v.strip()
        if w and w not in seen:
            new_words.append(w)
            seen.add(w)
    new_word_set = {w.lower() for w in new_words}

    # Component forms (inflected base words)
    allowed = set(new_words)
    comp_forms = []
    seen = set()
    for artificial, entry in master.items():
        if artificial not in allowed: continue
        for form_list in entry.get("forms", {}).values():
            for form in form_list:
                f = form.strip().lower()
                if f and f not in seen:
                    comp_forms.append(f)
                    seen.add(f)
    comp_forms_set = set(comp_forms)

    return new_word_set, comp_forms_set

# ── Batch classification (same regex as insertion script) ─────────────────────
_SPLIT_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_WORD_RE  = re.compile(r"^\w+$",       re.UNICODE)

def _contains_any(text: str, word_set: set) -> bool:
    for part in _SPLIT_RE.findall(text):
        if _WORD_RE.fullmatch(part) and part.lower() in word_set:
            return True
    return False

def classify_batch(X: torch.Tensor, enc, new_word_set: set, comp_forms_set: set):
    """Return per-row boolean masks for all categories."""
    has_new  = torch.zeros(X.size(0), dtype=torch.bool, device=X.device)
    has_comp = torch.zeros(X.size(0), dtype=torch.bool, device=X.device)
    for b in range(X.size(0)):
        text = enc.decode(X[b].tolist())
        has_new[b]  = _contains_any(text, new_word_set)
        has_comp[b] = _contains_any(text, comp_forms_set)
    has_either  = has_new | has_comp
    has_neither = ~has_either
    masks = {
        "overall":            torch.ones(X.size(0), dtype=torch.bool, device=X.device),
        "contains_new":       has_new,
        "no_new":             ~has_new,
        "contains_component": has_comp,
        "no_component":       ~has_comp,
        "either":             has_either,
        "neither":            has_neither,
    }
    return masks

# ── Per-checkpoint evaluation ─────────────────────────────────────────────────
@torch.no_grad()
def evaluate_checkpoint(mod_ckpt: str, base_ckpt: str | None,
                        test_data: np.ndarray, enc,
                        new_word_set: set, comp_forms_set: set) -> dict:
    """
    Evaluate one modified checkpoint (and optionally the paired base checkpoint).
    Returns a dict of {category: {metric: value}}.
    """
    mod_model  = load_model(mod_ckpt)
    base_model = load_model(base_ckpt) if base_ckpt else None

    # Accumulators: correct, total tokens; sum(log_loss)*T for ppl; total T
    acc = {cat: {"correct": 0, "total": 0,
                 "loss_mod": 0.0, "loss_base": 0.0,
                 "n_tokens": 0}
           for cat in CATEGORIES}

    for X, Y in iter_batches(test_data):
        B, T = X.size(0), Y.size(1)
        V    = None

        # Modified model forward
        logits_mod, _ = mod_model(X, Y)
        V = logits_mod.size(-1)
        preds     = torch.argmax(logits_mod, dim=-1)  # (B, T)
        correct   = (preds == Y)                       # (B, T) bool

        # Per-row cross-entropy loss for ppl (sum over T, not mean)
        loss_mod_per_row = F.cross_entropy(
            logits_mod.view(-1, V), Y.view(-1), reduction="none"
        ).view(B, T).sum(dim=1)   # (B,)

        # Base model forward (if available)
        if base_model is not None:
            logits_base, _ = base_model(X, Y)
            loss_base_per_row = F.cross_entropy(
                logits_base.view(-1, V), Y.view(-1), reduction="none"
            ).view(B, T).sum(dim=1)   # (B,)
        else:
            loss_base_per_row = None

        masks = classify_batch(X, enc, new_word_set, comp_forms_set)

        for cat, mask in masks.items():
            if not mask.any(): continue
            idx = mask.nonzero(as_tuple=True)[0]
            acc[cat]["correct"]   += correct[idx].sum().item()
            acc[cat]["total"]     += idx.numel() * T
            acc[cat]["loss_mod"]  += loss_mod_per_row[idx].sum().item()
            acc[cat]["n_tokens"]  += idx.numel() * T
            if loss_base_per_row is not None:
                acc[cat]["loss_base"] += loss_base_per_row[idx].sum().item()

    results = {}
    for cat in CATEGORIES:
        a = acc[cat]
        n = a["n_tokens"]
        results[cat] = {
            "accuracy":  a["correct"] / a["total"] if a["total"] > 0 else float("nan"),
            "ppl_mod":   math.exp(a["loss_mod"]  / n) if n > 0 else float("nan"),
            # ppl_base: only set for valid categories; None elsewhere
            "ppl_base":  (math.exp(a["loss_base"] / n)
                          if (n > 0 and base_model is not None and cat in PPL_BASE_VALID)
                          else None),
            "n_tokens":  n,
        }
    return results

# ── Aggregate over checkpoints ────────────────────────────────────────────────
def aggregate(per_ckpt: list[dict]) -> dict:
    """Average results over N checkpoint dicts."""
    avg = {}
    for cat in CATEGORIES:
        avg[cat] = {}
        for metric in ("accuracy", "ppl_mod", "ppl_base"):
            vals = [r[cat][metric] for r in per_ckpt
                    if r[cat][metric] is not None and not math.isnan(r[cat][metric])]
            avg[cat][metric] = sum(vals) / len(vals) if vals else float("nan")
        avg[cat]["n_tokens"] = per_ckpt[0][cat]["n_tokens"]  # same across runs
    return avg

# ── Write results ─────────────────────────────────────────────────────────────
def fmt(val, is_ppl_base=False, cat=None):
    if is_ppl_base and cat not in PPL_BASE_VALID:
        return "  N/A   (⚠ pseudoword tokens seen by base model)"
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "   nan"
    return f"{val:8.4f}"

def write_results(out_path: str, model_name: str, per_ckpt: list[dict], avg: dict,
                  map_path: str, inflection_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"Evaluation results — {model_name}\n")
        f.write(f"Map:         {map_path}\n")
        f.write(f"Inflections: {inflection_path}\n")
        f.write(f"Device: {DEVICE}  block_size={EVAL_BLOCK_SIZE}  batch_size={EVAL_BATCH_SIZE}\n")
        f.write(f"Note: ppl_base is N/A for categories where pseudoword tokens appear "
                f"(only 'overall' and 'neither' are valid).\n\n")

        col_w = 30
        header = f"  {'category':<{col_w}}  {'accuracy':>9}  {'ppl_mod':>9}  {'ppl_base':>9}  {'n_tokens':>10}"

        for i, ckpt_res in enumerate(per_ckpt):
            ckpt_name = CHECKPOINT_NAMES[i]
            f.write(f"── {ckpt_name} ──\n")
            f.write(header + "\n")
            f.write("  " + "-" * (len(header) - 2) + "\n")
            for cat in CATEGORIES:
                r = ckpt_res[cat]
                acc_s   = f"{r['accuracy']:9.6f}"
                ppl_m   = f"{r['ppl_mod']:9.3f}"
                pb_raw  = r["ppl_base"]
                if cat not in PPL_BASE_VALID or pb_raw is None:
                    ppl_b = "      N/A"
                else:
                    ppl_b = f"{pb_raw:9.3f}"
                f.write(f"  {cat:<{col_w}}  {acc_s}  {ppl_m}  {ppl_b}  {r['n_tokens']:>10,}\n")
            f.write("\n")

        f.write("── Average over all checkpoints ──\n")
        f.write(header + "\n")
        f.write("  " + "-" * (len(header) - 2) + "\n")
        for cat in CATEGORIES:
            r = avg[cat]
            acc_s  = f"{r['accuracy']:9.6f}"
            ppl_m  = f"{r['ppl_mod']:9.3f}"
            pb_raw = r["ppl_base"]
            if cat not in PPL_BASE_VALID or math.isnan(pb_raw):
                ppl_b = "      N/A"
            else:
                ppl_b = f"{pb_raw:9.3f}"
            f.write(f"  {cat:<{col_w}}  {acc_s}  {ppl_m}  {ppl_b}  {r['n_tokens']:>10,}\n")

# ── Evaluate one model ────────────────────────────────────────────────────────
def evaluate_model(corpus: str, n_or_base, kind: str, model_name: str) -> None:
    model_dir = get_gpt_model_dir(corpus, str(n_or_base), kind)
    data_dir  = get_gpt_data_dir(corpus, str(n_or_base), kind)
    is_base   = n_or_base == "base"

    if not os.path.exists(data_dir):
        print(f"  ⚠️  data dir not found: {data_dir} — skipping")
        return

    test_data = get_test_data(data_dir)
    enc       = tiktoken.get_encoding("gpt2")

    if is_base:
        # Base model: compute perplexity only (no new/comp words to split on)
        map_path, inf_path = None, None
        new_word_set, comp_forms_set = set(), set()
    else:
        map_path = get_pseudoword_map(corpus, n_or_base, kind)
        base_kind = kind.replace("_HALF", "")
        inf_path  = get_inflection_map(corpus, base_kind)
        new_word_set, comp_forms_set = load_word_sets(map_path, inf_path)

    # Locate base model checkpoints for ppl_base
    base_model_dir = get_gpt_model_dir(corpus, "base", None)

    per_ckpt = []
    for i, ckpt_name in enumerate(CHECKPOINT_NAMES):
        mod_ckpt  = os.path.join(model_dir, ckpt_name)
        base_ckpt = os.path.join(base_model_dir, ckpt_name)
        if not os.path.exists(mod_ckpt):
            print(f"  ⚠️  {ckpt_name} not found — skipping")
            continue
        base_ckpt_arg = base_ckpt if (not is_base and os.path.exists(base_ckpt)) else None
        print(f"  evaluating {ckpt_name} …")
        per_ckpt.append(evaluate_checkpoint(
            mod_ckpt, base_ckpt_arg, test_data, enc, new_word_set, comp_forms_set
        ))

    if not per_ckpt:
        print("  No checkpoints found — skipping")
        return

    avg      = aggregate(per_ckpt)
    out_dir  = os.path.join(NEW_COMP_EVAL_DIR, model_name)
    out_path = os.path.join(out_dir, "eval_results.txt")
    write_results(out_path, model_name, per_ckpt, avg,
                  map_path or "N/A", inf_path or "N/A")
    print(f"  ✅ Results → {out_path}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=CORPORA)
    parser.add_argument("--n",     type=int)
    parser.add_argument("--kind",  choices=KINDS)
    parser.add_argument("--base",  action="store_true", help="Evaluate base models only")
    parser.add_argument("--all",   action="store_true", help="Evaluate all models")
    args = parser.parse_args()

    if args.all:
        tasks = [(c, "base", None) for c in CORPORA] + [
            (c, n, k) for c in CORPORA for n in COUNTS for k in KINDS
        ]
    elif args.base:
        tasks = [(c, "base", None) for c in CORPORA]
    elif args.corpus and args.n and args.kind:
        tasks = [(args.corpus, args.n, args.kind)]
    else:
        parser.print_help()
        sys.exit(0)

    for corpus, n_or_base, kind in tasks:
        is_base = n_or_base == "base"
        name = (f"gpt_{corpus}_base" if is_base
                else f"gpt_{corpus}_{n_or_base}_{kind}")
        print(f"\n══ {name} ══════════════════════════════════════════")
        evaluate_model(corpus, n_or_base, kind, name)

    print("\n🎉 Evaluation complete.")

if __name__ == "__main__":
    main()
