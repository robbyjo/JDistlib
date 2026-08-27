# Stan-inspired model script catalog

These `.jdm` files target JDistlib modeling-language version 0.8. They are
Stan-inspired examples, not claims of Stan source compatibility.

The catalog covers constrained scalar and vector parameters, simplexes,
ordered vectors, transformed data and parameters, vectorized and indexed
likelihoods, custom `target +=` terms, regression links, non-centered effects,
and every supported generated-quantity RNG. `./gradlew check` compiles every
script with representative data through `examples.ModelScriptCatalog`, so a
syntax or semantic regression fails the build.

Use a script from Java with:

```java
String source = new String(Files.readAllBytes(path), StandardCharsets.UTF_8);
CompiledModelScript compiled = ModelScript.compile(source, data);
ChainResult chain = new NoUTurnSampler().sample(
    compiled.model(), compiled.model().initialState(), options, random);
```

See `docs/model-script-examples.html` for a browsable description of every
model and `docs/MODELING_LANGUAGE.md` for the exact supported surface.
