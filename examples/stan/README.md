# Stan source-compatibility fixtures

These forty-one files use ordinary `.stan` source syntax and exercise the JDistlib
source-compatible scalar/container core: literals, modern arrays, matrices and
multidimensional slices/indexed assignment, container-valued and forward-declared
user functions, typed matrix algebra, structured constraints, vector broadcasting,
general bounds, and offset/multiplier declarations. Fixtures 31–41 add complex
and nested tuple values, tuple-valued functions, a Java-bound external function,
CSR kernels, additional reductions/SPD functions, `integrate_1d`, algebraic
sensitivities, stiff BDF integration, and an implicit DAE callback.

JDistlib executes the models through its Java runtime. Acceptance of the source
does not imply identical C++ code generation, floating-point values, RNG
streams, warmup trajectories, or posterior draw order. Run
`./gradlew validateModelScripts` to compile and gradient-check all forty-one fixtures
alongside the `.jdm` catalog.
