# R 4.6.1, copula and mvtnorm. Run from the repository root.
.libPaths(c(normalizePath('build/audit2/Rlib'), .libPaths()))
library(copula)
library(mvtnorm)
dir.create('build/audit2/joint', recursive=TRUE, showWarnings=FALSE)
out <- file('build/audit2/joint/reference.tsv', 'wt')
writeLines('family\ttheta\tdelta\tu\tv\tcdf\tlogDensity\ttau', out)
points <- c(1e-12, 1e-6, .01, .15, .4, .7, .95, 1-1e-6, 1-1e-12)
for (family in c('clayton','gumbel','frank','joe','gaussian','student')) {
  parameters <- switch(family, clayton=c(0, .01, 1, 5, 50, 1000),
    gumbel=c(1, 1.01, 2, 10, 100), frank=c(-50,-5,-.01,.01,5,50,100),
    joe=c(1,1.01,2,2.00000001,10,100,1000), gaussian=c(-.9,0,.7), student=c(-.9,0,.7))
  for (theta in parameters) {
    obj <- switch(family, clayton=claytonCopula(theta), gumbel=gumbelCopula(theta),
      frank=frankCopula(theta), joe=joeCopula(theta), gaussian=normalCopula(theta),
      student=tCopula(theta, df=5))
    for (u in points) for (v in points) {
      uv <- matrix(c(u,v),nrow=1)
      vals <- suppressWarnings(c(pCopula(uv,obj), dCopula(uv,obj,log=TRUE), tau(obj)))
      writeLines(paste(c(family,theta,0,u,v,vals),collapse='\t'),out)
    }
  }
}
close(out)
# VineCopula provides independent BB1 density and conditional references.
library(VineCopula)
out <- file('build/audit2/joint/bb1.tsv','wt')
writeLines('theta\tdelta\tu\tv\tcdf\tlogDensity\tconditional',out)
for(theta in c(.01,.5,2,5)) for(delta in c(1.01,1.5,3))
  for(u in c(1e-6,.01,.2,.7,.99,1-1e-6)) for(v in c(1e-6,.01,.2,.7,.99,1-1e-6)) {
    vals <- c(BiCopCDF(u,v,7,theta,delta),log(BiCopPDF(u,v,7,theta,delta)),BiCopHfunc1(u,v,7,theta,delta))
    writeLines(paste(c(theta,delta,u,v,vals),collapse='\t'),out)
  }
close(out)
out <- file('build/audit2/joint/multivariate.tsv','wt')
writeLines('dimension\tdf\tx\tnormalLogDensity\ttLogDensity',out)
for (d in c(1,2,3,5,10)) for (df in c(.1,1,5,100,1e8,1e16)) for (x in c(-10,-1,0,.5,8)) {
  sigma <- matrix(.3,d,d); diag(sigma) <- 1
  xx <- rep(x,d)
  # For df=1e16, dmv t's direct lgamma subtraction is not a trusted oracle.
  writeLines(paste(c(d,df,x,dmvnorm(xx,sigma=sigma,log=TRUE),dmvt(xx,sigma=sigma,df=df,log=TRUE)),collapse='\t'),out)
}
close(out)
# Independent calculus references for BB1's density and conditional CDF are
# checked in the Java regression with Clayton/Gumbel limiting identities.
writeLines(capture.output(sessionInfo()),'build/audit2/joint/session-info.txt')
