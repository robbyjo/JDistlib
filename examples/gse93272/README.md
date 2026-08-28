# GSE93272 sparse transcriptome RJMCMC input

This directory contains the reproducible preparation script, not a redistributed
copy of the study. From the repository root, install the listed Bioconductor
packages and run:

```sh
Rscript examples/gse93272/prepare-gse93272.R build/example-data/gse93272
```

The script downloads the public processed GPL570 Series Matrix from GEO, retains
complete rheumatoid-arthritis samples, maps probes to gene symbols, selects one
probe per gene without using the outcome, and writes the two files consumed by
`WorkedSparseTranscriptomeRjmcmcExample`.

See [`docs/sparse-transcriptome-rjmcmc-example.html`](../../docs/sparse-transcriptome-rjmcmc-example.html)
for the model, priors, restart protocol, GPU boundary, diagnostics, and caveats.
