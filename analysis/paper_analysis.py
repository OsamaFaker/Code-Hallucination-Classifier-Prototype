"""
analysis/paper_analysis.py - reproduce the numbers reported in the paper.

Reads only the committed results files; uses no model and executes no code.

    python analysis/paper_analysis.py            # human-readable report
    python analysis/paper_analysis.py --json     # machine-readable output

Configurations (Section 5.3 / Table 5):
  uniform guessing     expected accuracy of a uniform random 3-way guess
  non-stdlib rule      HALLUCINATION if the code imports any module outside
                       sys.stdlib_module_names, else VALID
  Phase 1 only         static-checker verdicts; everything else -> VALID
  Phases 1-2           pipeline without the oracle executor (oracle_exec -> VALID)
  full pipeline        committed predictions (Phases 1-3)

Confidence intervals: 95% percentile cluster bootstrap over prompt_id
(10,000 replicates, numpy default_rng(20261011); clusters are indexed in order
of first appearance in the results file, and each replicate draws 164 cluster
indices with rng.integers). A proportion whose point estimate equals 1 gets an
exact Clopper-Pearson interval instead.

Requires Python >= 3.10 (sys.stdlib_module_names).
"""
import argparse
import ast
import json
import os
import sys
from collections import Counter

import numpy as np
from scipy.stats import beta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = {
    "qwen": os.path.join(ROOT, "results", "mbpp_oracle_450_balanced_qwen_enhanced_results.json"),
    "llama3": os.path.join(ROOT, "results", "mbpp_oracle_450_balanced_llama3_enhanced_results.json"),
}
LABELS = ["VALID", "GROUNDED_ERROR", "HALLUCINATION"]
STATIC_SOURCES = {"fabrication", "attribute", "arity", "signature", "misuse",
                  "logic", "function_name", "type", "semantic", "control_flow"}
VERIFIER_SOURCE = "cov_verifier"
ORACLE_SOURCE = "oracle_exec"
# Names from scripts/generate_oracle_dataset.py::_FAKE_LIBS that are NOT
# registered on PyPI (see analysis/check_pypi.py, checked 11 Oct 2026).
VERIFIED_FABRICATIONS = {"calctools", "codeutils", "listops", "sortinglib"}

N_BOOT = 10_000
SEED = 20261011
ALPHA = 0.05


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load():
    data = {}
    for model, path in RESULTS.items():
        with open(path, encoding="utf-8") as f:
            data[model] = json.load(f)
    ids = [[s["id"] for s in d] for d in data.values()]
    if any(i != ids[0] for i in ids):
        sys.exit("[ERROR] results files do not list the same samples in the same order")
    for model, d in data.items():
        unknown = {s["checker_source"] for s in d} - STATIC_SOURCES - {VERIFIER_SOURCE, ORACLE_SOURCE}
        if unknown:
            sys.exit(f"[ERROR] {model}: unknown checker_source values {sorted(unknown)}")
    return data


def top_level_imports(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module.split(".")[0])
    return mods


# ---------------------------------------------------------------------------
# Predictions per configuration
# ---------------------------------------------------------------------------
def pred_nonstdlib(samples):
    std = sys.stdlib_module_names
    return ["HALLUCINATION" if top_level_imports(s["code"]) - std else "VALID" for s in samples]


def pred_phase1(samples):
    return [s["predicted_label"] if s["checker_source"] in STATIC_SOURCES else "VALID" for s in samples]


def pred_phases12(samples):
    return ["VALID" if s["checker_source"] == ORACLE_SOURCE else s["predicted_label"] for s in samples]


def pred_full(samples):
    return [s["predicted_label"] for s in samples]


# ---------------------------------------------------------------------------
# Metrics on (…, 3, 3) confusion arrays: conf[..., true, pred]
# ---------------------------------------------------------------------------
def _div(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.divide(a, b, out=np.zeros(np.broadcast(a, b).shape), where=b > 0)


def metrics_from_conf(conf):
    """Return {name: (value_array, numerator_array, denominator_array|None)}."""
    tp = np.diagonal(conf, axis1=-2, axis2=-1)
    n = conf.sum(axis=(-2, -1))
    true_tot = conf.sum(axis=-1)
    pred_tot = conf.sum(axis=-2)
    out = {"accuracy": (_div(tp.sum(-1), n), tp.sum(-1), n)}
    f1s = []
    for k, lbl in enumerate(LABELS):
        p = _div(tp[..., k], pred_tot[..., k])
        r = _div(tp[..., k], true_tot[..., k])
        f1 = _div(2 * p * r, p + r)
        f1s.append(f1)
        out[f"{lbl}.precision"] = (p, tp[..., k], pred_tot[..., k])
        out[f"{lbl}.recall"] = (r, tp[..., k], true_tot[..., k])
        out[f"{lbl}.f1"] = (f1, None, None)
    out["macro_f1"] = (np.mean(f1s, axis=0), None, None)
    return out


def clopper_pearson(x, n, alpha=ALPHA):
    lo = 0.0 if x == 0 else beta.ppf(alpha / 2, x, n - x + 1)
    hi = 1.0 if x == n else beta.ppf(1 - alpha / 2, x + 1, n - x)
    return float(lo), float(hi)


class Bootstrap:
    """Cluster bootstrap over prompt_id with one shared set of replicates."""

    def __init__(self, samples):
        # Clusters in order of first appearance; the order fixes the replicates.
        pids = [s["prompt_id"] for s in samples]
        order = {pid: i for i, pid in enumerate(dict.fromkeys(pids))}
        self.cidx = np.array([order[pid] for pid in pids])
        self.k = len(order)
        draws = np.random.default_rng(SEED).integers(0, self.k, size=(N_BOOT, self.k))
        self.weights = np.zeros((N_BOOT, self.k))
        np.add.at(self.weights, (np.arange(N_BOOT)[:, None], draws), 1)

    def cluster_conf(self, true, pred, mask=None):
        c = np.zeros((self.k, 3, 3))
        for i, (t, p) in enumerate(zip(true, pred)):
            if mask is None or mask[i]:
                c[self.cidx[i], LABELS.index(t), LABELS.index(p)] += 1
        return c

    def evaluate(self, true, pred):
        cc = self.cluster_conf(true, pred)
        point = metrics_from_conf(cc.sum(axis=0))
        boot = metrics_from_conf(np.tensordot(self.weights, cc, axes=1))
        res = {}
        for name, (val, num, den) in point.items():
            v = float(val)
            if num is not None and v == 1.0:
                lo, hi = clopper_pearson(int(num), int(den))
                method = "clopper-pearson"
            else:
                lo, hi = np.percentile(boot[name][0], [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
                method = "cluster-bootstrap"
            res[name] = {"value": v, "ci": [float(lo), float(hi)], "ci_method": method}
            if num is not None:
                res[name]["count"] = [int(num), int(den)]
        res["confusion"] = cc.sum(axis=0).astype(int).tolist()
        return res


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyse():
    data = load()
    base = data["qwen"]
    true = [s["oracle_label"] for s in base]
    bs = Bootstrap(base)

    configs = {
        "non_stdlib_rule": pred_nonstdlib(base),
        "phase1_only": pred_phase1(base),
        "phases12_qwen": pred_phases12(data["qwen"]),
        "phases12_llama3": pred_phases12(data["llama3"]),
        "full_qwen": pred_full(data["qwen"]),
        "full_llama3": pred_full(data["llama3"]),
    }
    if pred_phase1(data["llama3"]) != configs["phase1_only"]:
        sys.exit("[ERROR] Phase 1 predictions differ between models")

    out = {
        "n_samples": len(base),
        "n_problems": bs.k,
        "label_counts": dict(Counter(true)),
        "bootstrap": {"replicates": N_BOOT, "seed": SEED, "cluster": "prompt_id",
                      "interval": "percentile", "alpha": ALPHA},
        "python": sys.version.split()[0],
        "uniform_guessing_accuracy": 1 / len(LABELS),
        "configurations": {name: bs.evaluate(true, p) for name, p in configs.items()},
        "full_pipeline_models_agree": configs["full_qwen"] == configs["full_llama3"],
    }

    # Phase decomposition and verifier verdicts
    phases = {}
    for model, d in data.items():
        src = Counter(s["checker_source"] for s in d)
        phases[model] = {
            "phase1_static": sum(v for k, v in src.items() if k in STATIC_SOURCES),
            "phase2_verifier_final": src[VERIFIER_SOURCE],
            "phase3_oracle_override": src[ORACLE_SOURCE],
            "by_checker_source": dict(src),
            "verifier_verdicts": dict(Counter(
                s["taxonomy_tag"] for s in d
                if s["checker_source"] in (VERIFIER_SOURCE, ORACLE_SOURCE))),
            "verifier_non_valid": sum(
                1 for s in d if s["checker_source"] == VERIFIER_SOURCE and s["predicted_label"] != "VALID"),
        }
    out["phases"] = phases

    # Error analysis: GROUNDED_ERROR predicted HALLUCINATION (full pipeline)
    errors = {}
    for model, d in data.items():
        e = [s for s in d if s["oracle_label"] == "GROUNDED_ERROR" and s["predicted_label"] == "HALLUCINATION"]
        errors[model] = {
            "total": len(e),
            "by_checker_source": dict(Counter(s["checker_source"] for s in e).most_common()),
            "by_execution_exc_type": dict(Counter(s["execution_exc_type"] for s in e).most_common()),
        }
    out["error_analysis_ge_as_hallucination"] = errors

    # Verified fabrications: HALLUCINATION samples importing a name confirmed
    # absent from PyPI
    fab = []
    for model, d in data.items():
        imports_fab = [s for s in d if top_level_imports(s["code"]) & VERIFIED_FABRICATIONS]
        hits = [s for s in imports_fab if s["oracle_label"] == "HALLUCINATION"]
        found = sum(s["predicted_label"] == "HALLUCINATION" for s in hits)
        lo, hi = clopper_pearson(found, len(hits))
        fab.append((model, {
            "names": sorted(VERIFIED_FABRICATIONS),
            "n_samples": len(hits),
            "n_other_labels_importing_names": dict(Counter(
                s["oracle_label"] for s in imports_fab if s["oracle_label"] != "HALLUCINATION")),
            "recall": {"value": found / len(hits) if hits else 0.0, "count": [found, len(hits)],
                       "ci": [lo, hi], "ci_method": "clopper-pearson"},
        }))
    out["verified_fabrications"] = dict(fab)
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _fmt(m):
    return f"{m['value']:.3f} [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}]"


def print_report(r):
    c = r["configurations"]
    print(f"n_samples {r['n_samples']}  n_problems {r['n_problems']}  "
          f"labels {r['label_counts']}  python {r['python']}")
    print(f"95% CIs: cluster bootstrap over prompt_id, {N_BOOT} replicates, seed {SEED}; "
          f"Clopper-Pearson where a proportion equals 1")

    print("\nTable 4 - full pipeline (Qwen; Llama 3 identical: "
          f"{r['full_pipeline_models_agree']})")
    f = c["full_qwen"]
    print(f"  accuracy  {_fmt(f['accuracy'])}    macro F1  {_fmt(f['macro_f1'])}")
    print(f"  {'class':<15} {'precision':<22} {'recall':<22} {'F1':<22}")
    for lbl in LABELS:
        print(f"  {lbl:<15} {_fmt(f[lbl + '.precision']):<22} "
              f"{_fmt(f[lbl + '.recall']):<22} {_fmt(f[lbl + '.f1']):<22}")
    print(f"  confusion (rows=true, cols=pred, order {LABELS}): {f['confusion']}")

    print("\nTable 5 - ablation and reference points")
    print(f"  {'configuration':<26} {'accuracy':<22} {'macro F1':<22} {'GE F1':<8} {'HALL F1':<8}")
    print(f"  {'uniform guessing':<26} {r['uniform_guessing_accuracy']:.3f}")
    rows = [("non-stdlib rule", "non_stdlib_rule"), ("Phase 1 only", "phase1_only"),
            ("Phases 1-2 (Qwen)", "phases12_qwen"), ("Phases 1-2 (Llama 3)", "phases12_llama3"),
            ("Full pipeline (Qwen)", "full_qwen"), ("Full pipeline (Llama 3)", "full_llama3")]
    for label, key in rows:
        m = c[key]
        print(f"  {label:<26} {_fmt(m['accuracy']):<22} {_fmt(m['macro_f1']):<22} "
              f"{m['GROUNDED_ERROR.f1']['value']:<8.3f} {m['HALLUCINATION.f1']['value']:<8.3f}")

    print("\nTable 6 - GROUNDED_ERROR predicted as HALLUCINATION")
    for model, e in r["error_analysis_ge_as_hallucination"].items():
        print(f"  {model:<7} total {e['total']}")
        print(f"          by checker   {e['by_checker_source']}")
        print(f"          by exception {e['by_execution_exc_type']}")

    print("\nSection 5.3 - phase decomposition")
    for model, p in r["phases"].items():
        print(f"  {model:<7} Phase 1 {p['phase1_static']}  Phase 2 final {p['phase2_verifier_final']}  "
              f"Phase 3 override {p['phase3_oracle_override']}  "
              f"verifier non-VALID {p['verifier_non_valid']}  verdicts {p['verifier_verdicts']}")
    for model, v in r["verified_fabrications"].items():
        rec = v["recall"]
        print(f"  {model:<7} verified fabrications {v['n_samples']} of "
              f"{r['label_counts']['HALLUCINATION']} HALLUCINATION samples; recall "
              f"{rec['count'][0]}/{rec['count'][1]} = {rec['value']:.3f}, "
              f"95% CP [{rec['ci'][0]:.3f}, {rec['ci'][1]:.3f}]")


def main():
    if sys.version_info < (3, 10):
        sys.exit("[ERROR] Python >= 3.10 required (sys.stdlib_module_names)")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", action="store_true", help="print results as JSON")
    args = ap.parse_args()
    r = analyse()
    if args.json:
        print(json.dumps(r, indent=2))
    else:
        print_report(r)


if __name__ == "__main__":
    main()
