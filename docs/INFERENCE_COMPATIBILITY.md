# Inference compatibility and serialization

The public `jdistlib.inference` contracts are additive through the 0.8.x line.
Existing method behavior, constraint parameterizations, retained-sample array
orientation, explicit-stream ownership, and schema version meanings will not be
changed incompatibly within that line.

Machine-readable schemas are independently versioned:

| Schema | Producer |
| --- | --- |
| `jdistlib.chains/1` | `ChainExport.toJson` |
| `jdistlib.mcmc-diagnostics/1` | `McmcDiagnosticReport.toJson` |
| `jdistlib.chart/1` | `ChartSpec.toJson` |
| `jdistlib.model-graph/1` | `ModelGraphExport.toJson` |

New fields may be added to a schema version. A field will not change type or
meaning without a new schema version. CSV exporters use quoted names and
locale-independent numeric text. SVG and HTML are presentation artifacts, not
stable interchange formats.

Generated script wrappers record the SHA-256 hash of their exact source.
Compilation caches reject mismatched hashes. Script language semantics are
identified separately by `ModelScript.LANGUAGE_VERSION`.

Checkpoints retain the exact last state and a cloned `RandomEngine`; generic
checkpoints do not serialize sampler-specific NUTS/HMC metric adaptation. They
are not declared as a durable cross-version file format. Use chain JSON/CSV for
durable retained draws and rerun warmup when moving a model to a release with
changed sampler algorithms.
