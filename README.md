# JDistlib

JDistlib is a Java library for probability distributions, statistical computing,
and numerical methods. Its core is a Java translation of R's `src/nmath`, extended
with additional distributions, Bayesian inference, and linear algebra.

See the [project website](https://robbyjo.github.io/JDistlib/) for documentation,
API references, examples, and downloads.

## Features

- Univariate, multivariate, and matrix distributions with density/mass,
  probability, quantile, and random-sampling APIs where applicable.
- Custom distributions, mixtures, transformations, copulas, and dependence modeling.
- Goodness-of-fit tests, multiple testing, and false discovery rate control.
- Bayesian modeling and MCMC, automatic differentiation, diagnostics, and
  JDistlib and supported Stan model scripts.
- Numerical integration, optimization, and algebraic, ODE, and DAE solvers.
- Dense and sparse linear algebra with optional native CPU, CUDA, OpenCL,
  and Vulkan acceleration.
- Financial distributions, tail risk, extreme-value modeling, and
  option-implied distributions.

## Requirements

- Java 8 or newer to run the library or the
  [all-in-one JAR](https://github.com/robbyjo/JDistlib/releases/latest/download/jdistlib-all.jar).
- JDK 17 or newer to build, plus internet access to download Gradle and dependencies
  on the first build. The included wrapper requires no separate Gradle installation.
- CPU-only use requires no accelerator software. Optional GPU backends require
  separately installed drivers and runtimes:
  - **CUDA:** [CUDA Toolkit 12.6](https://developer.nvidia.com/cuda-12-6-0-download-archive),
    including NVRTC, and a compatible NVIDIA driver.
  - **OpenCL:** a vendor OpenCL driver/runtime and a device supporting FP64.
  - **Vulkan:** a Vulkan loader, vendor driver, and device supporting `shaderFloat64`.

See the [acceleration guide](https://robbyjo.github.io/JDistlib/gpu-acceleration.html)
for backend setup and selection.

## Build

From the repository root, build the JARs and run the checks:

```sh
# Linux / macOS
./gradlew build
```

```powershell
# Windows
.\gradlew.bat build
```

The core JAR is written to `build/libs/`; the all-in-one JAR is written to
`distribution/all/build/libs/`. Generated class files target Java 8.
Use `./gradlew site` (or `.\gradlew.bat site` on Windows) to generate the website
and JavaDoc in `build/site/`.

## Run a Stan/JDM script

The executable JAR keeps Java source generation as its default mode. Add `--run`
to compile a supported script, load data, and sample with NUTS:

```text
java -jar jdistlib-all.jar --run --input script.stan --data dimensions.json --data observations.json --output output.txt
```

Repeat `--data` for multiple JSON files containing different named variables, or
bind files explicitly with `--data x=predictors.csv`,
`--data-column y=responses.csv:response`, and `--set N=100`.
`--run --validate` checks the script/data without sampling; `--run --help` lists
options. Output includes constrained draws, generated quantities and diagnostics.
The runner uses the supported Java-native Stan core on CPU.

See the [runnable examples and data rules](examples/cli/README.md). Build the JAR
from this checkout to use the new mode; earlier releases do not include it. The
core JAR also has an entry point and needs its generated `build/libs/lib/`
directory alongside it; the all-in-one JAR includes runtime dependencies.

Existing source-generation calls remain valid:

```text
java -jar jdistlib-all.jar script.stan generated.Model Model.java
```

## Vignettes

- [Using distributions](https://robbyjo.github.io/JDistlib/getting-started.html)
- [Response-time analysis](https://robbyjo.github.io/JDistlib/distribution-vignette.html)
- [Composing and transforming distributions](https://robbyjo.github.io/JDistlib/composition-tutorial.html)
- [Building custom distributions](https://robbyjo.github.io/JDistlib/custom-distributions.html#beginner-path)
- [Custom sensor-error distribution](https://robbyjo.github.io/JDistlib/custom-distribution-vignette.html)
- [Copulas and dependence](https://robbyjo.github.io/JDistlib/copula-tutorial.html)
- [Mixed insurance claims](https://robbyjo.github.io/JDistlib/copula-vignette.html)
- [Multiple testing and FDR](https://robbyjo.github.io/JDistlib/multiple-testing.html)
- [Bayesian modeling and MCMC](https://robbyjo.github.io/JDistlib/inference-tutorial.html)
- [Treatment-response posterior analysis](https://robbyjo.github.io/JDistlib/inference-vignette.html)
- [MCMC diagnostics](https://robbyjo.github.io/JDistlib/inference-diagnostics-vignette.html)
- [Data ingestion and model scripts](https://robbyjo.github.io/JDistlib/modeling-language-tutorial.html)
- [JDistlib for Stan users](https://robbyjo.github.io/JDistlib/stan-users.html)
- [Stan containers and matrices](https://robbyjo.github.io/JDistlib/stan-containers-tutorial.html)
- [Stan user functions](https://robbyjo.github.io/JDistlib/stan-functions-tutorial.html)
- [Algebraic, ODE, and DAE solvers](https://robbyjo.github.io/JDistlib/stan-solvers-tutorial.html)
- [Dense and sparse linear algebra](https://robbyjo.github.io/JDistlib/linear-algebra.html)
- [Probability-first finance](https://robbyjo.github.io/JDistlib/finance-tutorial.html)
- [Worked options analysis](https://robbyjo.github.io/JDistlib/options-trading-worked-example.html)

## License

JDistlib is distributed under the GNU General Public License, version 2 or later.
See [LICENSE](LICENSE).
