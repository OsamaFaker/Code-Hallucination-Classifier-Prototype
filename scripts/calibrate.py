"""
scripts/calibrate.py — fit Platt-scaling calibrators from classifier results.

Platt scaling trains a per-model LogisticRegression on raw confidence scores
so that a predicted confidence of 0.8 actually means ~80% of those predictions
are correct.  The fitted calibrators are stored in src/calibrators.json and
are loaded automatically by CodeClassifier when enable_calibration=True.

Usage:
    python scripts/calibrate.py \
        --results results/mbpp_oracle_450_balanced_qwen_enhanced_results.json \
                  results/mbpp_oracle_450_balanced_llama3_enhanced_results.json
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


def _load_pairs(path: str):
    """Return (confidence, correct) pairs from a classified results JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = []
    for s in data:
        pred = s.get("predicted_label")
        true = s.get("oracle_label") or s.get("true_label") or s.get("label")
        conf = s.get("confidence")
        if pred is None or true is None or conf is None:
            continue
        correct = 1 if pred == true else 0
        pairs.append((float(conf), correct))
    return pairs


def _fit_calibrator(pairs):
    """Fit and return a LogisticRegression Platt scaler from (confidence, correct) pairs."""
    X = np.array([[p[0]] for p in pairs])
    y = np.array([p[1] for p in pairs])

    if len(set(y)) < 2:
        print("  [WARN] Only one class present — calibration skipped (trivial data)")
        return None

    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    lr.fit(X, y)

    cv = cross_val_score(lr, X, y, cv=5, scoring="neg_brier_score")
    brier = -cv.mean()
    baseline_brier = float(np.mean(y) * (1 - np.mean(y)))

    coef = float(lr.coef_[0][0])
    intercept = float(lr.intercept_[0])

    print(f"    Coeff={coef:.4f}  Intercept={intercept:.4f}")
    print(f"    Brier score (CV): {brier:.4f}  (baseline naïve: {baseline_brier:.4f})")
    print(f"    Improvement:      {100*(baseline_brier - brier)/baseline_brier:.1f}%")

    return {"coef": coef, "intercept": intercept}


def apply_calibration(confidence: float, calibrator: dict) -> float:
    """Apply a stored Platt calibrator dict to a raw confidence score."""
    import math
    logit = calibrator["coef"] * confidence + calibrator["intercept"]
    return 1.0 / (1.0 + math.exp(-logit))


def main():
    parser = argparse.ArgumentParser(description="Fit Platt-scaling calibrators.")
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="One or more classified results JSON files.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("src", "calibrators.json"),
        help="Where to write the calibrators JSON (default: src/calibrators.json).",
    )
    args = parser.parse_args()

    if not _SKLEARN:
        print("[ERROR] scikit-learn and numpy are required: pip install scikit-learn numpy")
        sys.exit(1)

    calibrators = {}
    for path in args.results:
        if not os.path.exists(path):
            print(f"[SKIP] Not found: {path}")
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        # Infer model key from filename (e.g. "..._qwen_results" → "qwen")
        model_key = stem
        for candidate in ("qwen", "llama3"):
            if candidate in stem:
                model_key = candidate
                break

        print(f"\nFitting calibrator for: {path}  (key={model_key})")
        pairs = _load_pairs(path)
        print(f"  Samples: {len(pairs)}  (correct={sum(p[1] for p in pairs)}/{len(pairs)})")
        cal = _fit_calibrator(pairs)
        if cal:
            calibrators[model_key] = cal

    if not calibrators:
        print("\n[ERROR] No calibrators fitted. Check that results files exist and have confidence scores.")
        sys.exit(1)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(calibrators, f, indent=2)
    print(f"\nCalibrators saved to: {args.output}")
    print(json.dumps(calibrators, indent=2))


if __name__ == "__main__":
    main()
