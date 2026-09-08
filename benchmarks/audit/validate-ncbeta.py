#!/usr/bin/env python3
"""Resolve ordinary-grid noncentral-beta CDF disagreements without using R as oracle.

Requires mpmath (the installed version is recorded in summary.json). Example:
  python benchmarks/audit/validate-ncbeta.py build/r-audit/candidate/differences.tsv \
      --deps build/core-audit/python-deps --output build/ncbeta-reference

Inputs are the round-trip binary64 arguments/results produced by the audit.
The input is deliberately frozen to the 584 flagged rows (235 unique arguments)
from the 1,024-input grid with a=2.5, b=7 and noncentrality=11; a different
workload must be reviewed before changing the expected counts below.
The reference uses the defining Poisson mixture of regularized incomplete beta
integrals, evaluated independently for every component at two precisions. It
bounds all omitted probability mass by a geometric bound on the Poisson tail.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("differences", type=Path)
    parser.add_argument("output_dir", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--deps", type=Path, help="Optional directory containing mpmath")
    parser.add_argument("--dps", type=int, default=80)
    parser.add_argument("--verify-dps", type=int, default=120)
    args = parser.parse_args()
    if args.output is not None and args.output_dir is not None:
        parser.error("Specify the output directory only once")
    args.output = args.output or args.output_dir or Path("build/ncbeta-reference")
    return args


def main():
    args = arguments()
    if args.deps:
        sys.path.insert(0, str(args.deps.resolve()))
    import mpmath as mp
    if args.dps < 40 or args.verify_dps < args.dps + 20:
        raise ValueError("Use at least 40 digits and 20 additional verification digits")
    with args.differences.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if reader.fieldnames != ["key", "x", "R", "Java", "scaled_error"]:
            raise ValueError("Unexpected input columns")
        rows = list(reader)
    expected = {"ncbeta:p:0:0": 156, "ncbeta:p:0:1": 166,
                "ncbeta:p:1:0": 79, "ncbeta:p:1:1": 183}
    counts = dict.fromkeys(expected, 0)
    seen = set()
    for row in rows:
        if row["key"] not in expected or None in row or None in row.values():
            raise ValueError("Unexpected workload or malformed row")
        if not all(math.isfinite(float(row[name])) for name in ("x", "R", "Java", "scaled_error")):
            raise ValueError("Nonfinite input/result")
        identity = (row["key"], float(row["x"]))
        if identity in seen:
            raise ValueError("Duplicate input row")
        seen.add(identity)
        counts[row["key"]] += 1
    if counts != expected:
        raise ValueError("Missing or extra rows for the frozen 584-row audit: " + repr(counts))
    values = sorted({float(row["x"]) for row in rows})
    if len(values) != 235:
        raise ValueError("Expected exactly 235 unique arguments")
    if any(not 0 < value < 1 for value in values):
        raise ValueError("This ordinary-grid validator requires 0 < x < 1")

    def mixture(value, digits):
        with mp.workdps(digits):
            x = mp.mpf(value)  # exact input binary64, not its decimal approximation
            a, b, mean = mp.mpf("2.5"), mp.mpf(7), mp.mpf("5.5")
            weight = mp.exp(-mean)
            lower = mp.mpf(0)
            upper = mp.mpf(0)
            target = mp.power(10, -(digits - 10))
            for k in range(10000):
                # Evaluate the incomplete beta integral independently at each k.
                # Use the nearer endpoint, then complement at high precision.
                if x <= mp.mpf("0.5"):
                    lo = mp.betainc(a + k, b, 0, x, regularized=True)
                    hi = 1 - lo
                else:
                    hi = mp.betainc(b, a + k, 0, 1 - x, regularized=True)
                    lo = 1 - hi
                lower += weight * lo
                upper += weight * hi
                next_weight = weight * mean / (k + 1)
                ratio_bound = mean / (k + 2)
                if ratio_bound < 1:
                    remainder = next_weight / (1 - ratio_bound)
                    if remainder < target:
                        # Every beta component is in [0,1], so this bounds the
                        # absolute truncation error in either tail of the mixture.
                        if abs(1 - lower - upper) > remainder + 100 * mp.eps:
                            raise ArithmeticError("Poisson mass bound failed")
                        return lower, upper, remainder, k + 1
                weight = next_weight
            raise ArithmeticError("Poisson mixture did not converge")

    cache = {}
    max_precision_delta = mp.mpf(0)
    max_omitted_mass = mp.mpf(0)
    terms_min, terms_max = 10000, 0
    for index, value in enumerate(values):
        first = mixture(value, args.dps)
        second = mixture(value, args.verify_dps)
        with mp.workdps(args.verify_dps):
            delta = max(abs(first[i] - second[i]) for i in (0, 1))
            if delta > mp.power(10, -(args.dps - 12)):
                raise ArithmeticError("Reference did not stabilize at x=" + repr(value))
            max_precision_delta = max(max_precision_delta, delta)
            max_omitted_mass = max(max_omitted_mass, second[2])
        terms_min, terms_max = min(terms_min, second[3]), max(terms_max, second[3])
        cache[value] = second[:2]
        if (index + 1) % 50 == 0:
            print("validated", index + 1, "of", len(values), "unique inputs", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    summaries = {}
    with mp.workdps(args.verify_dps):
        for row in rows:
            _, _, lower_flag, log_flag = row["key"].split(":")
            if lower_flag not in ("0", "1") or log_flag not in ("0", "1"):
                raise ValueError("Invalid tail or log flag")
            reference = cache[float(row["x"])][0 if lower_flag == "1" else 1]
            if log_flag == "1":
                reference = mp.log(reference)
            # Read printed R/Java results back as exactly their binary64 values.
            observed = {name: mp.mpf(float(row[name])) for name in ("R", "Java")}
            absolute = {name: abs(value - reference) for name, value in observed.items()}
            relative = {name: (abs(mp.expm1(value - reference)) if log_flag == "1"
                              else abs(value - reference) / abs(reference))
                        for name, value in observed.items()}
            winner = "Java" if absolute["Java"] < absolute["R"] else (
                "R" if absolute["R"] < absolute["Java"] else "tie")
            result = {"key": row["key"], "x": row["x"], "reference": mp.nstr(reference, 60),
                      "R": row["R"], "Java": row["Java"], "closer": winner}
            for name in ("R", "Java"):
                result[name + "_absolute_error"] = mp.nstr(absolute[name], 12)
                result[name + "_relative_probability_error"] = mp.nstr(relative[name], 12)
            results.append(result)
            summary = summaries.setdefault(row["key"], {"count": 0, "Java_closer": 0,
                "R_closer": 0, "ties": 0, "max_R_relative_probability_error": 0.0,
                "max_Java_relative_probability_error": 0.0, "max_R_absolute_error": 0.0,
                "max_Java_absolute_error": 0.0})
            summary["count"] += 1
            summary["ties" if winner == "tie" else winner + "_closer"] += 1
            for name in ("R", "Java"):
                key = "max_" + name + "_relative_probability_error"
                summary[key] = max(summary[key], float(relative[name]))
                key = "max_" + name + "_absolute_error"
                summary[key] = max(summary[key], float(absolute[name]))
    with (args.output / "reference-results.tsv").open("w", newline="", encoding="utf-8") as dest:
        writer = csv.DictWriter(dest, fieldnames=list(results[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(results)
    passed = (all(result["closer"] == "Java" for result in results)
              and all(summary["max_Java_relative_probability_error"] <= 1e-11
                      for summary in summaries.values()))
    report = {"passed": passed, "Java_relative_probability_tolerance": 1e-11,
              "input_sha256": hashlib.sha256(args.differences.read_bytes()).hexdigest(),
              "parameters": {"a": 2.5, "b": 7, "ncp": 11},
              "mpmath_version": mp.__version__, "precisions": [args.dps, args.verify_dps],
              "flagged_rows": len(rows), "unique_x": len(values),
              "mixture_terms_min": terms_min, "mixture_terms_max": terms_max,
              "max_omitted_probability_bound": mp.nstr(max_omitted_mass, 12),
              "max_precision_delta": mp.nstr(max_precision_delta, 12), "by_key": summaries}
    text = json.dumps(report, indent=2, sort_keys=True)
    (args.output / "summary.json").write_text(text + "\n", encoding="utf-8")
    print(text)
    if not passed:
        raise SystemExit("FAILED: Java must be closer on every row and have relative probability error <= 1e-11")


if __name__ == "__main__":
    main()
