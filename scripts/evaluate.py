"""
scripts/evaluate.py — compute accuracy, F1, and confusion matrix for any results file.

Usage:
    python scripts/evaluate.py results/mbpp_oracle_450_balanced_qwen_enhanced_results.json
    python scripts/evaluate.py results/mbpp_oracle_450_balanced_qwen_enhanced_results.json \\
                               results/mbpp_oracle_450_balanced_llama3_enhanced_results.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

LABELS = ["VALID", "GROUNDED_ERROR", "HALLUCINATION"]


def compute_metrics(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    total = 0
    correct = 0
    # confusion[true][pred] = count
    confusion = defaultdict(lambda: defaultdict(int))
    per_class = {lbl: {"tp": 0, "fp": 0, "fn": 0} for lbl in LABELS}

    for s in data:
        pred = s.get("predicted_label")
        true = s.get("oracle_label") or s.get("true_label") or s.get("label")
        if pred is None or true is None:
            continue
        total += 1
        confusion[true][pred] += 1
        if pred == true:
            correct += 1
            per_class[pred]["tp"] += 1
        else:
            per_class[true]["fn"] += 1
            per_class[pred]["fp"] += 1

    accuracy = correct / total if total else 0.0

    f1s = []
    class_metrics = {}
    for lbl in LABELS:
        tp = per_class[lbl]["tp"]
        fp = per_class[lbl]["fp"]
        fn = per_class[lbl]["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
        class_metrics[lbl] = {"precision": prec, "recall": rec, "f1": f1}

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return {
        "path": path,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": class_metrics,
        "confusion": {t: dict(preds) for t, preds in confusion.items()},
    }


def print_report(m: dict, label: str = "") -> None:
    name = label or os.path.basename(m["path"])
    print(f"\n{'='*62}")
    print(f"  {name}")
    print(f"{'='*62}")
    print(f"  Samples  : {m['total']}   Correct: {m['correct']}")
    print(f"  Accuracy : {m['accuracy']:.3f}")
    print(f"  Macro F1 : {m['macro_f1']:.3f}")
    print()
    print(f"  {'Class':<18} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print(f"  {'-'*38}")
    for lbl in LABELS:
        c = m["per_class"].get(lbl, {})
        print(
            f"  {lbl:<18} {c.get('precision',0):>6.3f} "
            f"{c.get('recall',0):>6.3f} {c.get('f1',0):>6.3f}"
        )
    print()
    print(f"  Confusion matrix (rows=true, cols=pred):")
    header = f"  {'':>18}" + "".join(f" {l:>15}" for l in LABELS)
    print(header)
    for true_lbl in LABELS:
        row = f"  {'TRUE '+true_lbl:>18}"
        for pred_lbl in LABELS:
            cnt = m["confusion"].get(true_lbl, {}).get(pred_lbl, 0)
            row += f" {cnt:>15}"
        print(row)
    print(f"{'='*62}")


def print_comparison(baseline: dict, enhanced: dict) -> None:
    print(f"\n{'='*70}")
    print("  BASELINE vs ENHANCED COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Metric':<25} {'Baseline':>10} {'Enhanced':>10} {'Delta':>10}")
    print(f"  {'-'*55}")

    def row(name, b_val, e_val, fmt=".3f"):
        delta = e_val - b_val
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<25} {b_val:>{10}.{fmt[1:-1]}f} {e_val:>{10}.{fmt[1:-1]}f} "
              f"{sign+f'{delta:.3f}':>10}")

    row("Accuracy", baseline["accuracy"], enhanced["accuracy"])
    row("Macro F1", baseline["macro_f1"], enhanced["macro_f1"])
    for lbl in LABELS:
        for metric in ("precision", "recall", "f1"):
            b = baseline["per_class"].get(lbl, {}).get(metric, 0)
            e = enhanced["per_class"].get(lbl, {}).get(metric, 0)
            row(f"  {lbl[:10]} {metric[:4]}", b, e)
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate classifier results.")
    parser.add_argument("files", nargs="+", help="Results JSON file(s) to evaluate.")
    args = parser.parse_args()

    results = []
    for path in args.files:
        if not os.path.exists(path):
            print(f"[SKIP] Not found: {path}")
            continue
        m = compute_metrics(path)
        print_report(m)
        results.append(m)

    if len(results) == 2:
        print_comparison(results[0], results[1])


if __name__ == "__main__":
    main()
