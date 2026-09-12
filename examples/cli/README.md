# Running Stan/JDM scripts

Build from the JDistlib repository root with `./gradlew :jar :jdistlib-all:jar`
(Windows: `.\gradlew.bat :jar :jdistlib-all:jar`). The core executable JAR is in
`build/libs/`; the self-contained JAR, including runtime dependencies, is in
`distribution/all/build/libs/`. Keep the core JAR beside its generated `lib/`
directory when using `java -jar`; the all-in-one JAR needs no companion files.
In the commands below, replace `jdistlib-all.jar`
with your built JAR's path. This CLI is a source-checkout addition; older released
JARs do not have the new `--run` mode.

Merge several named JSON files:

```text
java -jar jdistlib-all.jar --run --input examples/cli/regression.stan --data examples/cli/dimensions.json --data examples/cli/predictors.json --data examples/cli/response.json --output output.txt --seed 42
```

Or explicitly bind separate CSV files to Stan variables:

```text
java -jar jdistlib-all.jar --run --input examples/cli/regression.stan --set N=6 --data x=examples/cli/predictors.csv --data-column y=examples/cli/response.csv:response --output output.txt --seed 42
```

Append `--validate` to check the source and data without sampling or writing output.
Use `--run --help` for all options. Every file path is relative to the current
working directory; quote the whole binding if the path contains spaces, e.g.
`--data "x=C:\My Data\predictors.csv"` or
`--data-column "y=C:\My Data\response.csv:response"`.

## Data rules

- `--data file.json` merges a JSON object whose keys match data-block variable
  names. Values must be finite numbers or rectangular numeric arrays. Multiple
  files can supply different variables; duplicate names fail, including repeats
  inside a single JSON file. Unknown names also fail.
- `--data X=matrix.csv` binds an entire headerless numeric CSV or TSV to `X`.
  A one-column file supplies a vector. Rows of a matrix are flattened in row-major
  order. `--data X=matrix.json` accepts a numeric JSON value/array without an
  enclosing named object.
- `--data-column y=table.csv:response` selects a header column from a CSV/TSV.
  The last colon separates the column name, so Windows drive letters work.
  Tables must be rectangular. Surrounding quotes on individual cells are accepted;
  embedded delimiters, embedded quotes, and multiline quoted CSV fields are not.
  Blank lines are skipped; missing/nonfinite selected values are errors.
- `--set N=6` supplies a scalar. `--set "y=[1,2,3]"` also accepts numeric JSON
  arrays. Supply sizes such as `N` and `K` explicitly; the runner does not guess
  which variables share a dimension or join/reorder rows from different files.
- All arrays cross the existing engine boundary as flat `double[]`. The script
  supplies the dimensions and verifies element counts, integer values and bounds.
  Nested JSON shape is checked for rectangularity, but is not matched against
  declared dimensions beyond total length. Keep JSON/CSV row and column order
  consistent with the script. For example, `matrix[2,3] X` accepts
  `{"X":[[1,2,3],[4,5,6]]}` in a merged JSON file. Integer inputs must be exactly
  representable as doubles. Null, boolean, string, R-dump, and missing-value inputs
  are not supported.

The named-object and row-major conventions follow the
[CmdStan JSON data format](https://mc-stan.org/docs/cmdstan-guide/json_apdx.html).
The finite-number restriction and flattening rules above describe this adapter.

## Sampling and output

Defaults: NUTS, 4 chains, 1 worker, 1,000 warmup iterations, 1,000 retained draws
per chain, thinning 1, seed 12345, target acceptance 0.8, maximum depth 10,
initial step size 0.25. Change them using `--chains`, `--threads`, `--warmup`,
`--samples`, `--thin`, `--seed`, `--target-accept`, `--max-depth`, and `--step-size`.
`--samples` counts retained draws: `--samples 100 --thin 2` runs 200 post-warmup
iterations. Chain seeds are independently derived; worker count does not alter
seeded results. All chains start from the model's default initial state.

Output is UTF-8 CSV text regardless of the file extension. Lines beginning with
`#` contain run settings and parameter/sampler diagnostics as JSON, including
R-hat, ESS, MCSE and divergence information. The table contains chain/draw IDs,
sampler statistics, constrained parameter values, and generated quantities for
every retained draw. Container names use one-based flat row-major indices.
Transformed parameters are evaluated internally; copy any desired transformed
value into a generated quantity to export it. `log_density__` remains the
unconstrained target, including the transformation Jacobian. This is JDistlib's
format, not a byte-compatible CmdStan CSV.

Fewer than four retained draws cannot produce a diagnostic report; a single chain
cannot produce R-hat. Warnings are written to stderr and must be assessed before
using posterior summaries. Successful completion does not certify convergence.
Draws are held in memory. Failed compilation/sampling/output generation leaves an
existing output intact; a successful run replaces it. Input files cannot be used
as output. Exit codes: 0 success/help, 2 invalid run arguments/data/model, 1 I/O or
execution failure. Legacy source-generation exceptions retain their old behavior.

The runner uses JDistlib's supported Stan/JDM language core, not CmdStan or a C++
toolchain. External Java function bindings and unsupported Stan features still
need an embedding application. Script NUTS currently uses CPU evaluation;
`--compute auto|cpu` and `--nuts-offload auto|off` are accepted. Explicit GPU/offload
requests fail rather than silently falling back.

## Existing Java source generation

Without `--run`, the three positional arguments keep their original meaning:

```text
java -jar jdistlib-all.jar examples/cli/regression.stan generated.Regression Regression.java
java -cp jdistlib-all.jar jdistlib.inference.lang.ModelScriptCli examples/cli/regression.stan generated.Regression Regression.java
```

Both commands generate the existing Java wrapper; neither starts inference.
