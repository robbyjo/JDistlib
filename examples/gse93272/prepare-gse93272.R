#!/usr/bin/env Rscript
# Prepare the public GSE93272 Affymetrix series for the JDistlib sparse-RJMCMC example.
# Install once with:
# BiocManager::install(c("GEOquery", "Biobase", "AnnotationDbi", "hgu133plus2.db"))

args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1]] else file.path("build", "example-data", "gse93272")
dir.create(output, recursive = TRUE, showWarnings = FALSE)

required <- c("GEOquery", "Biobase", "AnnotationDbi", "hgu133plus2.db")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop("Missing Bioconductor packages: ", paste(missing, collapse = ", "))

message("Downloading the public processed GSE93272 Series Matrix...")
series <- GEOquery::getGEO("GSE93272", GSEMatrix = TRUE, AnnotGPL = FALSE,
                           destdir = output)
platform <- vapply(series, Biobase::annotation, character(1))
index <- which(platform == "GPL570")
if (length(index) != 1L) stop("Expected one GPL570 expression set; found: ", paste(platform, collapse = ", "))
eset <- series[[index]]
phenotype <- Biobase::pData(eset)

field <- function(label) {
  names_lower <- tolower(names(phenotype))
  hits <- which(names_lower == tolower(label) |
                  names_lower == paste0(tolower(label), ":ch1"))
  if (length(hits) != 1L) {
    stop("Could not uniquely locate phenotype field '", label,
         "'. Available fields: ", paste(names(phenotype), collapse = ", "))
  }
  as.character(phenotype[[hits]])
}
number <- function(value) suppressWarnings(as.numeric(gsub("[^0-9.+-]", "", value)))
zscore <- function(value) as.numeric((value - mean(value)) / stats::sd(value))

subject <- field("individual id")
disease <- field("disease state")
outcome <- number(field("crp.das28"))
age <- number(field("age"))
gender <- tolower(field("gender"))
batch <- field("batch")
rin <- number(field("rin"))

keep <- grepl("rheumatoid|^ra$", disease, ignore.case = TRUE) &
  complete.cases(subject, outcome, age, gender, batch, rin)
if (!any(keep)) stop("No complete rheumatoid-arthritis samples were found")
batch_factor <- droplevels(factor(batch[keep]))
if (nlevels(batch_factor) != 2L) stop("The worked model expects two batches, found ", nlevels(batch_factor))

sample_ids <- rownames(phenotype)[keep]
clinical <- data.frame(
  sample_id = sample_ids,
  subject_id = subject[keep],
  outcome = zscore(outcome[keep]),
  age_z = zscore(age[keep]),
  female = as.integer(grepl("female|^f$", gender[keep])),
  batch2 = as.integer(batch_factor == levels(batch_factor)[2]),
  rin_z = zscore(rin[keep]),
  check.names = FALSE
)

message("Mapping probes to gene symbols and applying an outcome-blind variance filter...")
probe_expression <- Biobase::exprs(eset)[, sample_ids, drop = FALSE]
symbols <- AnnotationDbi::mapIds(hgu133plus2.db::hgu133plus2.db,
  keys = rownames(probe_expression), column = "SYMBOL", keytype = "PROBEID",
  multiVals = "first")
valid <- !is.na(symbols) & nzchar(symbols)
probe_expression <- probe_expression[valid, , drop = FALSE]
symbols <- unname(symbols[valid])
probe_iqr <- apply(probe_expression, 1L, stats::IQR, na.rm = TRUE)

# One representative probe per symbol: the probe with largest IQR. Then retain at
# most 17,000 most-variable genes without looking at the clinical outcome.
ordering <- order(symbols, -probe_iqr, rownames(probe_expression))
first_probe <- !duplicated(symbols[ordering])
selected <- ordering[first_probe]
selected <- selected[is.finite(probe_iqr[selected]) & probe_iqr[selected] > 0]
selected <- selected[order(probe_iqr[selected], decreasing = TRUE)]
selected <- head(selected, 17000L)
selected <- selected[order(symbols[selected])]
expression <- probe_expression[selected, , drop = FALSE]
rownames(expression) <- make.unique(symbols[selected])
expression <- t(scale(t(expression)))
if (any(!is.finite(expression))) stop("Non-finite standardized expression values")

utils::write.table(clinical, file.path(output, "clinical.tsv"), sep = "\t",
                   row.names = FALSE, quote = FALSE)
connection <- gzfile(file.path(output, "expression.tsv.gz"), open = "wt")
utils::write.table(cbind(gene = rownames(expression), as.data.frame(expression,
                   check.names = FALSE)), connection, sep = "\t", row.names = FALSE,
                   col.names = c("gene", colnames(expression)), quote = FALSE)
close(connection)

manifest <- c(
  "accession\tGSE93272",
  "source\thttps://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE93272",
  paste0("prepared_utc\t", format(Sys.time(), tz = "UTC", usetz = TRUE)),
  paste0("samples\t", nrow(clinical)),
  paste0("subjects\t", length(unique(clinical$subject_id))),
  paste0("genes\t", nrow(expression)),
  "gene_filter\thighest-IQR probe per gene; top 17000 genes by outcome-blind IQR"
)
writeLines(manifest, file.path(output, "manifest.tsv"))
message("Prepared ", nrow(clinical), " samples and ", nrow(expression),
        " genes in ", normalizePath(output))
