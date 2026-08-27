# Stan source-compatibility fixtures

These thirty files use ordinary `.stan` source syntax and exercise the JDistlib
source-compatible scalar/container core: literals, modern arrays, matrices and
multidimensional slices/indexed assignment, container-valued and forward-declared
user functions, typed matrix algebra, structured constraints, vector broadcasting,
general bounds, and offset/multiplier declarations.

JDistlib executes the models through its Java runtime. Acceptance of the source
does not imply identical C++ code generation, floating-point values, RNG
streams, warmup trajectories, or posterior draw order. Run
`./gradlew validateModelScripts` to compile and gradient-check all thirty fixtures
alongside the `.jdm` catalog.
