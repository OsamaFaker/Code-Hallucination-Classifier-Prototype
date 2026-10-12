"""
analysis/check_pypi.py - check which fabrication-list names are registered on PyPI.

The names are scripts/generate_oracle_dataset.py::_FAKE_LIBS, the modules the
dataset generator asks the model to import for HALLUCINATION samples. A name
that is registered on PyPI is not a verified fabrication. Result as of
11 Oct 2026: 11 of 15 registered; calctools, codeutils, listops and sortinglib
are not.

    python analysis/check_pypi.py            # table
    python analysis/check_pypi.py --json

Queries https://pypi.org/pypi/<name>/json (HTTP 200 = registered, 404 = not).
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

FAKE_LIBS = [
    "sortinglib", "mathtools", "pyhelpers", "datautils", "pymath",
    "listops", "numutils", "stringtools", "arraylib", "calctools",
    "pyalgo", "seqtools", "mathlib", "codeutils", "pystructs",
]


def check(name: str, timeout: float = 15.0) -> dict:
    url = f"https://pypi.org/pypi/{name}/json"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            info = json.load(resp).get("info", {})
            return {"name": name, "registered": True, "status": resp.status,
                    "pypi_name": info.get("name"), "version": info.get("version")}
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"name": name, "registered": False, "status": 404}
        return {"name": name, "registered": None, "status": e.code, "error": str(e)}
    except urllib.error.URLError as e:
        return {"name": name, "registered": None, "status": None, "error": str(e.reason)}


def main():
    ap = argparse.ArgumentParser(description="Check fabrication-list names against PyPI.")
    ap.add_argument("--json", action="store_true", help="print results as JSON")
    args = ap.parse_args()

    results = [check(n) for n in FAKE_LIBS]
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            state = {True: "registered", False: "NOT registered", None: "error"}[r["registered"]]
            extra = f"  {r.get('pypi_name')} {r.get('version')}" if r["registered"] else r.get("error", "")
            print(f"  {r['name']:<12} {state:<15}{extra}")
        n_reg = sum(r["registered"] is True for r in results)
        print(f"\n{n_reg} of {len(results)} registered on PyPI; not registered: "
              f"{', '.join(r['name'] for r in results if r['registered'] is False)}")
    if any(r["registered"] is None for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
