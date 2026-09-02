# Publishing JDistlib 0.10.1

The normal release workflow builds and tests the native-free core, modular
CUDA/OpenCL/Vulkan providers, and unified distribution. It verifies manifests
and the merged provider descriptor, rejects SNAPSHOT filenames, creates
SHA-256 checksums, and attaches `jdistlib-all-0.10.1.jar`, the stable
`jdistlib-all.jar` latest-release alias, and the checksum file to the GitHub
tag. The Central bundle below contains the native-free core
and the four provider publications; it does not publish the fat JAR.

The build can also prepare a Maven Repository Layout bundle for the Maven
Central Publisher Portal without adding an unsupported third-party publishing
plugin. Build the released source from its immutable tag rather than from the
later development branch:

```text
git switch --detach v0.10.1
gradlew.bat clean centralBundle '-PreleaseVersion=0.10.1'
```

Supply the ASCII-armored private signing key and its password through Gradle's
`signingKey` and `signingPassword` properties, preferably as protected CI
environment variables (`ORG_GRADLE_PROJECT_signingKey` and
`ORG_GRADLE_PROJECT_signingPassword`). Never commit either value. The task
refuses SNAPSHOT versions and unsigned bundles. Its output is
`build/distributions/jdistlib-0.10.1-central.zip`, containing Maven-layout POM,
JARs, signatures, and MD5/SHA-1/SHA-256/SHA-512 checksums.

Before the first upload, the maintainer must verify ownership of the
`net.sourceforge.jdistlib` namespace in Central Portal and generate a portal
user token. Upload the bundle as a user-managed deployment, inspect Central's
validation results, and publish it only after the Git tag and GitHub release are
final. A released Central component cannot be replaced, so the upload is not an
automated side effect of the tag workflow.

The `jdistlib-nativecpu`, `jdistlib-cuda`, `jdistlib-opencl`, and
`jdistlib-vulkan` projects remain small dependency-managed artifacts.
Platform-specific JCuda/LWJGL native
classifiers are deliberately omitted from their portable POMs and must be
selected by the consuming build. `jdistlib-all` is the direct-download
convenience distribution and must not replace the core Maven coordinates.
