#!/usr/bin/env python3
"""
10_analyze_disambiguation.py  —  Sense selection experiment (HALF models only)
===============================================================================

DESIGN
------
For each pseudoword X with components (Y1, Y2), 5 matched frames generate
3 sentence variants, read at the LAST TOKEN of the residual stream after MLP:

  X_neutral   "This is another warmquake, like the other one."
  X_dis_Y1    "This is another warmquake, like the other earthquake."
  X_dis_Y2    "This is another warmquake, like the other warmed."

plus component baselines in the same frame:
  Y1_neutral  "This is another earthquake, like the other one."
  Y2_neutral  "This is another warmed, like the other one."

HALF models only — in FULL models, components are absent from training so
there is no meaningful baseline representation for Y1/Y2.

MEASURES  (sense selection only)
---------
  sense_bias    cos(X_neu, Y1_neu) − cos(X_neu, Y2_neu)
                > 0 → X in neutral context is closer to Y1 (dominant sense)
                Captures which component the pseudoword drifts toward without
                any disambiguating context.

  differential  cos(X_dis_Y1, Y1_neu) − cos(X_dis_Y1, Y2_neu)   [avg both dirs]
                Measures context-driven sense selectivity.
                HOM: predicted large (genuine disambiguation)
                HYP: predicted near-zero (meanings already similar)

  base_dist     cos(X_neu, Y1_neu+Y2_neu)/2
                Baseline similarity between pseudoword and its components
                in neutral context.

Sign test on differential (last layer) tests whether context-driven sense
selectivity is significantly above chance.

Usage
-----
    # Single condition
    python 10_analyze_disambiguation.py --corpus recipe --n 50 --kind HOM_HALF

    # Compare HOM_HALF vs HYP_HALF for a corpus
    python 10_analyze_disambiguation.py --corpus recipe --n 50 --compare

    # Both corpora
    python 10_analyze_disambiguation.py --all_corpora --n 50 --compare

Output
------
    evaluation/disambig/<model_name>_disambig.txt   (human-readable)
    evaluation/disambig/<model_name>_disambig.json  (full data)
    evaluation/disambig/disambig_<corpus>_N<n>_all.json  (--compare)
"""

import os
import re
import sys
import json
import argparse
import numpy as np
import torch
import tiktoken
from scipy import stats as _stats

from model import GPTConfig, GPT
from config import (
    CORPORA, COUNTS, KINDS,
    get_gpt_model_dir, get_pseudoword_map, ROOT,
)

DISAMBIG_DIR = os.path.join(ROOT, "evaluation", "disambig")
CHECKPOINT   = "ckpt1.pt"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
ENC          = tiktoken.get_encoding("gpt2")
PRECISION    = 6

SENSE_METRICS = ("differential", "sense_bias", "base_dist")

# ── Sentence frames ───────────────────────────────────────────────────────────
FRAME_PAIRS = [
    ("This is another {X}, like the other one.",
     "This is another {X}, like the other {Y}.",         False),
    ("Here we have that {X}, just an ordinary one.",
     "Here we have that {X}, just an ordinary {Y}.",      False),
    ("She likes that {X}, like all other things.",
     "She likes that {X}, like all other {Y}.",           True),
    ("Look for my {X}, it looks like one of your things.",
     "Look for my {X}, it looks like one of your {Y}.",   True),
    ("He talked about the {X} after other topics.",
     "He talked about the {X} after other {Y}.",          True),
]

IRREGULAR = {
    "fish":"fish","sheep":"sheep","trout":"trout","salmon":"salmon",
    "squid":"squid","shrimp":"shrimp","deer":"deer","moose":"moose",
    "octopus":"octopuses","cactus":"cacti","leaf":"leaves","loaf":"loaves",
    "knife":"knives","potato":"potatoes","tomato":"tomatoes",
    "child":"children","person":"people","goose":"geese","tooth":"teeth",
    "foot":"feet","mouse":"mice",
}

def pluralize(w: str) -> str:
    w = w.strip().lower()
    if w in IRREGULAR: return IRREGULAR[w]
    if re.search(r"(s|x|z|ch|sh)$", w): return w + "es"
    if re.search(r"[^aeiou]y$", w):     return w[:-1] + "ies"
    if re.search(r"[^aeiou]o$", w):     return w + "es"
    return w + "s"

def rnd(x): return round(float(x), PRECISION) if x is not None else None

# ── Data loading ──────────────────────────────────────────────────────────────
def load_pairs(corpus: str, n: int, kind: str) -> list[tuple]:
    path = get_pseudoword_map(corpus, n, kind)
    with open(path) as f: m = json.load(f)
    pairs = []
    for comps_str, pseudo in m.items():
        parts = [c.strip().lower() for c in comps_str.split(",")]
        if len(parts) == 2 and isinstance(pseudo, str):
            pairs.append((pseudo.strip().lower(), parts[0], parts[1]))
    return pairs

# ── Model ─────────────────────────────────────────────────────────────────────
def load_model(corpus: str, n: int, kind: str) -> GPT:
    path  = os.path.join(get_gpt_model_dir(corpus, n, kind), CHECKPOINT)
    ckpt  = torch.load(path, map_location=DEVICE)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()
    return model

def register_hooks(model: GPT):
    n_layer = model.config.n_layer
    buf = {"resid_mlp": [None] * n_layer}
    handles = []
    for l in range(n_layer):
        def hook(m, i, o, _l=l):
            buf["resid_mlp"][_l] = o.detach().cpu().float().numpy()
        handles.append(model.transformer.h[l].mlp.register_forward_hook(hook))
    return buf, handles

def sentence_rep(model: GPT, sentence: str, buf: dict, n_layer: int):
    """Return list of per-layer last-token residual vectors."""
    seq = ENC.encode_ordinary(sentence)
    if not seq: return None
    X = torch.tensor([seq], dtype=torch.long, device=DEVICE)
    with torch.no_grad(): model(X)
    return [buf["resid_mlp"][l][0, -1].copy()
            if buf["resid_mlp"][l] is not None else None
            for l in range(n_layer)]

# ── Condition collection ──────────────────────────────────────────────────────
def collect_conditions(model: GPT, pseudo: str, y1: str, y2: str,
                       buf: dict, n_layer: int) -> dict:
    """Collect per-layer representation lists for each condition."""
    conds = {k: [[] for _ in range(n_layer)]
             for k in ("X_neu", "X_dis1", "X_dis2", "Y1_neu", "Y2_neu")}
    for neu_f, dis_f, plural in FRAME_PAIRS:
        y1_c = pluralize(y1) if plural else y1
        y2_c = pluralize(y2) if plural else y2
        sents = {
            "X_neu":  neu_f.format(X=pseudo),
            "X_dis1": dis_f.format(X=pseudo, Y=y1_c),
            "X_dis2": dis_f.format(X=pseudo, Y=y2_c),
            "Y1_neu": neu_f.format(X=y1),
            "Y2_neu": neu_f.format(X=y2),
        }
        for cond, sent in sents.items():
            rep = sentence_rep(model, sent, buf, n_layer)
            if rep is None: continue
            for l in range(n_layer):
                if rep[l] is not None:
                    conds[cond][l].append(rep[l])
    return conds

# ── Metrics ───────────────────────────────────────────────────────────────────
def centroid(vecs: list) -> np.ndarray | None:
    return np.mean(vecs, axis=0) if vecs else None

def cos(a, b) -> float:
    if a is None or b is None: return float("nan")
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

# ── Main run function ─────────────────────────────────────────────────────────
def run(corpus: str, n: int, kind: str, max_pairs: int,
        json_accum: list | None = None) -> dict | None:

    if "HALF" not in kind:
        print(f"  gpt_{corpus}_{n}_{kind}: FULL model — skipping (HALF only)")
        return None

    name = f"gpt_{corpus}_{n}_{kind}"
    print(f"\n== {name} ==", flush=True)

    pairs = load_pairs(corpus, n, kind)[:max_pairs]
    model = load_model(corpus, n, kind)
    nl    = model.config.n_layer
    buf, handles = register_hooks(model)

    # Accumulators per layer
    M = {m: [[] for _ in range(nl)] for m in SENSE_METRICS}
    per_pair   = []
    diffs_last = []

    try:
        for pseudo, y1, y2 in pairs:
            C = collect_conditions(model, pseudo, y1, y2, buf, nl)
            for l in range(nl):
                xn  = centroid(C["X_neu"][l])
                x1  = centroid(C["X_dis1"][l])
                x2  = centroid(C["X_dis2"][l])
                yn1 = centroid(C["Y1_neu"][l])
                yn2 = centroid(C["Y2_neu"][l])
                if any(v is None for v in [xn, x1, x2, yn1, yn2]):
                    continue

                # sense_bias: which component is X_neutral closer to?
                cos_xn_y1 = cos(xn, yn1)
                cos_xn_y2 = cos(xn, yn2)
                sense_bias = cos_xn_y1 - cos_xn_y2
                base_dist  = 0.5 * (cos_xn_y1 + cos_xn_y2)

                # differential: context-driven selectivity
                diff1 = cos(x1, yn1) - cos(x1, yn2)
                diff2 = cos(x2, yn2) - cos(x2, yn1)
                differential = 0.5 * (diff1 + diff2)

                M["sense_bias"][l].append(sense_bias)
                M["base_dist"][l].append(base_dist)
                M["differential"][l].append(differential)

                if l == nl - 1:
                    diffs_last.append(differential)
                    per_pair.append({
                        "pseudo":       pseudo,
                        "y1":           y1,
                        "y2":           y2,
                        "cos_x_y1":     rnd(cos_xn_y1),
                        "cos_x_y2":     rnd(cos_xn_y2),
                        "sense_bias":   rnd(sense_bias),   # + → Y1 preferred
                        "base_dist":    rnd(base_dist),
                        "differential": rnd(differential),
                    })
    finally:
        for h in handles: h.remove()

    record = {
        "model": name, "corpus": corpus, "n": n, "kind": kind,
        "n_pairs": len(pairs),
        "by_layer": {
            m: [rnd(np.mean(v)) if v else None for v in M[m]]
            for m in SENSE_METRICS
        },
    }

    # Sign test on differential at last layer
    diffs = np.array([d for d in diffs_last if not np.isnan(d)])
    if len(diffs) > 0:
        n_pos = int((diffs > 0).sum())
        if hasattr(_stats, "binomtest"):
            pb = _stats.binomtest(n_pos, len(diffs), 0.5).pvalue
        else:
            pb = _stats.binom_test(n_pos, len(diffs), 0.5)
        t, pt = _stats.ttest_1samp(diffs, 0)
        record["signtest"] = {
            "n_positive":         n_pos,
            "n_total":            len(diffs),
            "frac_positive":      round(n_pos / len(diffs), 3),
            "mean_differential":  rnd(float(diffs.mean())),
            "binom_p":            round(float(pb), 6),
            "ttest_t":            round(float(t), 3),
            "ttest_p":            round(float(pt), 6),
        }
    else:
        record["signtest"] = None

    # ── Write text report ──────────────────────────────────────────────────
    os.makedirs(DISAMBIG_DIR, exist_ok=True)
    out_txt = os.path.join(DISAMBIG_DIR, f"{name}_disambig.txt")
    with open(out_txt, "w") as f:
        f.write(f"Sense selection experiment — {name}\n")
        f.write(f"Pairs: {len(pairs)}  Frames: {len(FRAME_PAIRS)}\n")
        f.write("Representation: resid_mlp at last sentence token.\n\n")

        f.write(f"  {'L':<3}  {'sense_bias':>12}  {'differential':>13}  {'base_dist':>10}\n")
        f.write("  " + "-" * 44 + "\n")
        for l in range(nl):
            def g(m): return record["by_layer"][m][l]
            def fmt(v, w=12): return f"{v:>{w}.6f}" if v is not None else " " * w
            f.write(f"  L{l:<2}  {fmt(g('sense_bias'))}  "
                    f"{fmt(g('differential'), 13)}  {fmt(g('base_dist'), 10)}\n")

        f.write("\n")
        if record["signtest"]:
            st = record["signtest"]
            f.write(f"Sign test L{nl-1}: {st['n_positive']}/{st['n_total']}  "
                    f"frac_pos={st['frac_positive']}  "
                    f"mean_diff={st['mean_differential']}  "
                    f"binom_p={st['binom_p']}  "
                    f"t={st['ttest_t']}  t_p={st['ttest_p']}\n\n")

        f.write(f"  {'pseudo':<18}{'y1':<12}{'y2':<12}"
                f"{'cos_x_y1':>10}{'cos_x_y2':>10}"
                f"{'sense_bias':>11}{'base_dist':>10}{'diff':>10}\n")
        for r in per_pair:
            f.write(f"  {r['pseudo']:<18}{r['y1']:<12}{r['y2']:<12}"
                    f"{r['cos_x_y1']:>10.6f}{r['cos_x_y2']:>10.6f}"
                    f"{r['sense_bias']:>+11.6f}{r['base_dist']:>10.4f}"
                    f"{r['differential']:>10.6f}\n")

    record["per_pair"] = per_pair
    out_json = os.path.join(DISAMBIG_DIR, f"{name}_disambig.json")
    with open(out_json, "w") as f:
        json.dump(record, f, indent=2)

    # Console summary
    L = nl - 1
    bl = record["by_layer"]
    print(f"  L{L}: sense_bias={bl['sense_bias'][L]}  "
          f"diff={bl['differential'][L]}  base={bl['base_dist'][L]}", flush=True)
    if record["signtest"]:
        st = record["signtest"]
        print(f"  sign: {st['n_positive']}/{st['n_total']}  "
              f"mean={st['mean_differential']}  binom_p={st['binom_p']}", flush=True)
    print(f"  ✅ {out_txt}", flush=True)

    if json_accum is not None:
        json_accum.append(record)
    return record

# ── Compare HOM_HALF vs HYP_HALF ─────────────────────────────────────────────
def print_compare(accum: list, corpus: str, n: int) -> None:
    if len(accum) < 2: return
    hom = next((r for r in accum if "HOM" in r["kind"]), None)
    hyp = next((r for r in accum if "HYP" in r["kind"]), None)
    if not hom or not hyp: return
    nl = len(hom["by_layer"]["differential"])

    print("\n" + "=" * 70)
    print(f"HOM_HALF vs HYP_HALF  corpus={corpus}  N={n}  [resid_mlp, last token]")
    print("=" * 70)
    for m, desc in [
        ("differential", "sense selectivity (HOM > HYP predicted)"),
        ("sense_bias",   "dominant-sense bias in neutral context"),
        ("base_dist",    "X-to-component similarity, neutral context"),
    ]:
        print(f"\n  {m}  —  {desc}")
        print(f"  {'L':<3}  {'HOM_HALF':>12}  {'HYP_HALF':>12}  {'delta':>14}")
        for l in range(nl):
            h = hom["by_layer"][m][l]
            y = hyp["by_layer"][m][l]
            if h is None or y is None: continue
            print(f"  L{l:<2}  {h:>12.6f}  {y:>12.6f}  {h-y:>+14.6f}")

    print("\n  SIGN TESTS (differential, last layer):")
    for r in accum:
        st = r.get("signtest")
        if st:
            print(f"    {r['kind']:<12}: {st['n_positive']}/{st['n_total']}  "
                  f"mean={st['mean_differential']}  binom_p={st['binom_p']}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus",      choices=CORPORA)
    parser.add_argument("--n",           type=int, default=50)
    parser.add_argument("--kind",        choices=["HOM_HALF", "HYP_HALF"])
    parser.add_argument("--compare",     action="store_true",
                        help="Run both HOM_HALF and HYP_HALF and print comparison")
    parser.add_argument("--all_corpora", action="store_true",
                        help="Run for all corpora (use with --compare)")
    parser.add_argument("--max_pairs",   type=int, default=50)
    args = parser.parse_args()

    os.makedirs(DISAMBIG_DIR, exist_ok=True)
    corpora = CORPORA if args.all_corpora else ([args.corpus] if args.corpus else [])

    if not corpora:
        parser.error("--corpus or --all_corpora required")

    if args.compare or args.all_corpora:
        for corpus in corpora:
            accum = []
            for kind in ["HOM_HALF", "HYP_HALF"]:
                run(corpus, args.n, kind, args.max_pairs, json_accum=accum)
            print_compare(accum, corpus, args.n)
            out = os.path.join(DISAMBIG_DIR,
                               f"disambig_{corpus}_N{args.n}_all.json")
            with open(out, "w") as f:
                json.dump(accum, f, indent=2)
            print(f"\n  ✅ {out}")
    else:
        if not args.kind:
            parser.error("--kind required (HOM_HALF or HYP_HALF)")
        run(args.corpus, args.n, args.kind, args.max_pairs)

if __name__ == "__main__":
    main()
