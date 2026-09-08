"""Join the matched R/Java measurements without third-party dependencies.

python benchmarks/audit/summarize.py build/r-audit benchmarks/audit/results-2026-09-08.csv
"""
import csv
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])

def read(path):
    with path.open(newline="") as stream:
        return {row["key"]: row for row in csv.DictReader(stream, delimiter="\t")}

r = read(root / "r-timing.tsv")
before = read(root / "baseline" / "java-timing.tsv")
after = read(root / "candidate" / "java-timing.tsv")
accuracy = read(root / "candidate" / "java-accuracy.tsv")
assert r.keys() == before.keys() == after.keys() == accuracy.keys()
with output.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["workload", "r_ns", "before_ns", "after_ns", "r_over_java",
                     "before_over_after", "values", "r_discrepancies", "max_relative_difference"])
    for key, rr in r.items():
        rn, bn, an = (float(row["ns_per_value"]) for row in (rr, before[key], after[key]))
        writer.writerow([key, rn, bn, an, rn / an, bn / an, accuracy[key]["count"],
                         accuracy[key]["differences"], accuracy[key]["max_scaled_error"]])
        if key.endswith(":d:1:0"):
            print(f"| {key.split(':')[0]} | {rn:.1f} | {bn:.1f} | {an:.1f} | {rn/an:.2f}x |")
print("Values:", sum(int(row["count"]) for row in accuracy.values()))
print("R discrepancies:", sum(int(row["differences"]) for row in accuracy.values()))
print("Max central relative difference:", max(float(row["max_scaled_error"])
      for key, row in accuracy.items() if not key.startswith("nc")))
