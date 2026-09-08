x <- sin(seq_len(20000)*.37)+.3*cos(seq_len(20000)*.11)+seq_len(20000)/20000
measure <- function(name, f, calls=500) {
  for(i in 1:3) f()
  ms <- replicate(5,system.time(for(i in seq_len(calls)) value <- f())[["elapsed"]]*1000/calls)
  cat(name,median(ms),f(),sep="\t");cat("\n")
}
measure("SJ_DPI_20000",function() bw.SJ(x,method="dpi"))
measure("UCV_20000",function() suppressWarnings(bw.ucv(x)))
measure("Density_Gaussian_20000_512",function() density(x,bw=.4,n=512,from=-4,to=5)$y[257])
