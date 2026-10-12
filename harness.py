"""
v6 evaluation harness — classify 500 samples with a local LLM verifier.

Usage:
    python scripts/classify.py [--model qwen|llama3] [--output PATH] [--workers N]

Default model is qwen (qwen2.5-coder:7b), the dissertation baseline.
See README.md § "Choosing the verifier model" for details.
"""
import argparse
import os
import sys
import json
import time
import statistics
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.dataset import CodeSample
from src.classifier import CodeClassifier
from src.models import ClassificationResult, HallucinationCategory


def _oracle_post_verify(
    result: ClassificationResult,
    code: str,
    test_assertions: str,
    timeout: int = 5,
) -> ClassificationResult:
    """
    Run the oracle executor against test assertions and override a VALID
    result if execution fails.  Called after static + LLM verification
    to catch false negatives that neither checker detected.
    """
    from src.oracle_executor import (
        execute_code,
        LABEL_VALID,
        LABEL_HALLUCINATION,
    )
    exec_label, error_msg, exc_type = execute_code(code, test_assertions, timeout=timeout)

    if exec_label == LABEL_VALID:
        return result

    result.oracle_exec_label = exec_label
    result.oracle_exec_error = error_msg

    if exec_label == LABEL_HALLUCINATION:
        result.is_valid = False
        result.is_grounded_error = False
        result.hallucination_category = HallucinationCategory.FUNCTION
        result.checker_source = "oracle_exec"
        result.explanation = (
            f"Oracle execution revealed hallucination "
            f"({exc_type}: {error_msg[:120]})"
        )
    else:
        result.is_valid = False
        result.is_grounded_error = True
        result.hallucination_category = None
        result.checker_source = "oracle_exec"
        result.explanation = (
            f"Oracle execution revealed grounded error "
            f"({exc_type or 'AssertionError'}: {error_msg[:120]})"
        )

    exp = result.explanation
    result.short_explanation = (exp[:97] + "...") if len(exp) > 100 else exp
    return result


# Supported model aliases → Ollama model identifiers
MODEL_MAP = {
    "qwen":   "qwen2.5-coder:7b",   # dissertation baseline
    "llama3": "llama3:latest",       # general-purpose alternative
}


# ---------------------------------------------------------------------------
# Task 5: model-presence check
# ---------------------------------------------------------------------------
def check_model_present(model_id: str) -> None:
    """Exit with a non-zero status if model_id is not installed in Ollama."""
    try:
        with urllib.request.urlopen(
            "http://localhost:11434/api/tags", timeout=10
        ) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"[ERROR] Could not reach Ollama at http://localhost:11434: {e}")
        print("        Make sure Ollama is running before starting the evaluation.")
        sys.exit(1)

    available = [m["name"] for m in data.get("models", [])]
    if model_id not in available:
        print(f"[ERROR] Model '{model_id}' is not installed in Ollama.")
        print(f"        Available models: {available if available else '(none)'}")
        print(f"        Run:  ollama pull {model_id}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parsing (Tasks 1, 4, 6)
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="v6 evaluation harness — classify code samples with a local LLM verifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/classify.py --model qwen\n"
            "  python scripts/classify.py --model llama3 --workers 4\n"
            "  python scripts/classify.py --model qwen --output my_results.json\n"
        ),
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_MAP.keys()),
        default="qwen",
        help=(
            "Verifier model: 'qwen' -> qwen2.5-coder:7b (default, dissertation baseline); "
            "'llama3' -> llama3:latest."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Override the output JSON path. "
            "Default: dataset_v4_500_classified_v6_<model>.json"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help=(
            "Number of parallel LLM-verifier workers (default: 3). "
            "Fewer workers reduce memory pressure; more may improve throughput "
            "when sufficient RAM is available."
        ),
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Override the input dataset JSON path. "
            "Default: dataset_v4_500_unclassified.json"
        ),
    )
    parser.add_argument(
        "--use-cov",
        action="store_true",
        default=False,
        help="Enable Chain-of-Verification 3-way prompt instead of binary YES/NO verifier.",
    )
    parser.add_argument(
        "--use-selfcheck",
        action="store_true",
        default=False,
        help="Enable SelfCheckGPT multi-temperature consistency check (requires --use-cov).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------
def _load_checkpoint(output_file: str, n: int):
    """Return a partial results list if a checkpoint exists, else [None]*n."""
    final_results = [None] * n
    if not os.path.exists(output_file):
        return final_results

    try:
        with open(output_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except Exception:
        return final_results

    from src.models import ClassificationResult, SeverityLevel, HallucinationCategory
    for i, s in enumerate(existing):
        if "predicted_label" not in s:
            continue
        cat_str = s.get("hallucination_category")
        cat_enum = None
        if cat_str:
            try:
                cat_enum = HallucinationCategory[cat_str]
            except KeyError:
                pass
        res = ClassificationResult(
            is_valid=(s["predicted_label"] == "VALID"),
            is_grounded_error=(s["predicted_label"] == "GROUNDED_ERROR"),
            hallucination_category=cat_enum,
            explanation=s.get("explanation", ""),
            confidence=s.get("confidence", 0.0),
            severity=SeverityLevel.MEDIUM,
            checker_source=s.get("checker_source", "unknown"),
        )
        res.attempts_task = s.get("attempts_task")
        res.verifier_confidence = s.get("verifier_confidence")
        res.verifier_justification = s.get("verifier_justification")
        res.line = s.get("line")
        res.bad_token = s.get("bad_token")
        res.short_explanation = s.get("short_explanation")
        final_results[i] = res

    loaded = sum(1 for r in final_results if r is not None)
    if loaded:
        print(f"[checkpoint] Resumed {loaded} previously completed samples from {output_file}")
    return final_results


# ---------------------------------------------------------------------------
# Results writer
# ---------------------------------------------------------------------------
def save_results(samples, final_results, call_latencies, output_file):
    results_list = []
    for i, s in enumerate(samples):
        res = final_results[i]
        if res:
            exp = res.explanation
            res.short_explanation = (exp[:97] + "...") if len(exp) > 100 else exp

            predicted_label = "VALID"
            if res.is_grounded_error:
                predicted_label = "GROUNDED_ERROR"
            elif not res.is_valid:
                predicted_label = "HALLUCINATION"

            s.update(res.to_dict())
            s["predicted_label"] = predicted_label
            s["short_explanation"] = res.short_explanation

            if i in call_latencies:
                s["latency_ms"] = round(call_latencies[i], 1)

        results_list.append(s)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def run() -> None:
    args = parse_args()
    model_id = MODEL_MAP[args.model]
    output_file = args.output or f"dataset_v4_500_classified_v6_{args.model}.json"
    workers = args.workers
    input_file = args.input or "dataset_v4_500_unclassified.json"

    # Task 5: fail loudly before any work if the model is absent
    check_model_present(model_id)

    # Communicate model choice to CodeClassifier via env vars.
    # LLMVerifier precedence: constructor arg > LLM_MODEL env var > hardcoded default.
    # The harness uses the env-var channel so CodeClassifier needs no modification.
    os.environ["LLM_PROVIDER"] = "ollama"
    os.environ["LLM_MODEL"] = model_id

    run_start = time.perf_counter()

    if not os.path.exists(input_file):
        print(f"[ERROR] Input file not found: {input_file}")
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} samples.")
    print(f"Model   : {model_id}  (--model {args.model})")
    print(f"Output  : {output_file}")
    print(f"Workers : {workers}")
    print("Starting v6 evaluation (Parallel Robust Mode)...")

    classifier = CodeClassifier(
        gt_version="v1",
        use_dynamic_gt=True,
        enable_cache=False,
        use_cov=args.use_cov,
        use_selfcheck=args.use_selfcheck,
    )

    if not classifier._llm_verifier:
        print("[CRITICAL ERROR] LLM Verifier initialization failed.")
        sys.exit(1)

    final_results = _load_checkpoint(output_file, len(samples))

    # Phase 1: Static Funnel
    print("\nPhase 1: Static Funnel...")
    to_verify = []
    for i, s in enumerate(samples):
        if final_results[i] is not None:
            continue
        cs = CodeSample(id=s["id"], prompt=s["prompt"], generated_code=s["code"] or "")
        try:
            res = classifier.classify_static(cs)
            if not res.is_valid:
                final_results[i] = res
            else:
                to_verify.append((i, cs, res))
        except Exception as e:
            print(f"Static Error {s['id']}: {e}")

    print(f"Static Funnel done. {len(to_verify)} samples need LLM verification.")

    # Phase 2: Parallel LLM Verifier + optional oracle post-verification
    print(f"\nPhase 2: Parallel Verifier Dispatch (Workers={workers})...")

    # Detect whether the dataset carries test assertions (oracle dataset format).
    has_tests = any(s.get("tests") for s in samples)
    if has_tests:
        print("  [oracle] Test assertions detected — will run oracle executor on VALID samples.")

    call_latencies: dict = {}
    error_count = 0

    def verify_worker(task):
        idx, cs, s_res = task
        t0 = time.perf_counter()
        try:
            result = classifier.verify_semantic(cs, s_res)
            latency_ms = (time.perf_counter() - t0) * 1000

            # Oracle post-verification: execute code against test assertions to
            # catch false negatives that the LLM verifier passed through.
            if result.is_valid and has_tests:
                test_assertions = samples[idx].get("tests", "")
                if test_assertions:
                    result = _oracle_post_verify(result, cs.generated_code, test_assertions)

            return idx, result, latency_ms, False
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            print(f"Verifier Error {cs.id}: {e}")
            return idx, s_res, latency_ms, True

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(verify_worker, t): t for t in to_verify}

        count = 0
        for future in as_completed(futures):
            idx, res, lat_ms, is_error = future.result()
            final_results[idx] = res
            call_latencies[idx] = lat_ms
            if is_error:
                error_count += 1
            count += 1
            processed = sum(1 for x in final_results if x is not None)
            print(f"  Processed {processed}/{len(samples)} (Verified {count}/{len(to_verify)})...")

            if count % 10 == 0:
                save_results(samples, final_results, call_latencies, output_file)

    save_results(samples, final_results, call_latencies, output_file)

    run_end = time.perf_counter()
    total_wall_s = run_end - run_start
    total_wall_min = total_wall_s / 60

    # Timing summary
    print(f"\nEvaluation finished.")
    print(f"\n{'='*60}")
    print(f"TIMING SUMMARY  (model: {model_id})")
    print(f"{'='*60}")
    print(f"Total wall-clock time      : {total_wall_s:.1f} s  ({total_wall_min:.2f} min)")
    print(f"Verifier invocations       : {len(call_latencies)}")
    print(f"Errors / timeouts          : {error_count}")

    if call_latencies:
        latencies = list(call_latencies.values())
        latencies_sorted = sorted(latencies)
        mean_lat = statistics.mean(latencies)
        median_lat = statistics.median(latencies)
        p95_idx = min(int(0.95 * len(latencies_sorted)), len(latencies_sorted) - 1)
        p95_lat = latencies_sorted[p95_idx]
        total_verifier_s = sum(latencies) / 1000
        speedup = total_verifier_s / total_wall_s if total_wall_s > 0 else float("nan")

        print(f"Mean latency / call        : {mean_lat:.0f} ms")
        print(f"Median latency / call      : {median_lat:.0f} ms")
        print(f"p95 latency / call         : {p95_lat:.0f} ms")
        print(f"Total verifier time (sum)  : {total_verifier_s:.1f} s  ({total_verifier_s/60:.2f} min)")
        print(f"Sequential-equivalent time : {total_verifier_s:.1f} s  ({total_verifier_s/60:.2f} min)")
        print(f"Speedup (seq / observed)   : {speedup:.2f}x")

    print(f"{'='*60}")

    # Label distribution
    valid = sum(1 for s in samples if s.get("predicted_label") == "VALID")
    grounded = sum(1 for s in samples if s.get("predicted_label") == "GROUNDED_ERROR")
    hallucinations = sum(1 for s in samples if s.get("predicted_label") == "HALLUCINATION")
    errors = sum(1 for s in samples if s.get("predicted_label") == "ERROR")
    print(f"\nLabel distribution:")
    print(f"  VALID          : {valid}")
    print(f"  GROUNDED_ERROR : {grounded}")
    print(f"  HALLUCINATION  : {hallucinations}")
    print(f"  ERROR          : {errors}")
    print(f"\nOutput written to: {output_file}")


if __name__ == "__main__":
    run()
