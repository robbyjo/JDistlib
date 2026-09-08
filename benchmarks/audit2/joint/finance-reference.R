.libPaths(c(normalizePath('build/audit2/Rlib'), .libPaths()))
library(VGAM)
library(extraDistr)
library(actuar)
writeLines(c(grep('delap|polya',ls('package:VGAM'),value=TRUE),
             grep('delap|polya|wish|dirich',ls('package:extraDistr'),value=TRUE)),
           'build/audit2/joint/available-finance-functions.txt')
# Independent mixture sums using base R Poisson/NB (not the Java recurrence).
out <- file('build/audit2/joint/finance.tsv','wt')
writeLines('family\tlambda\tshape\tprob\tx\tlogDensity\tlogLower\tlogUpper',out)
logsum <- function(x) {a<-max(x);if(is.infinite(a)) a else a+log(sum(exp(x-a)))}
for(family in c('delaporte','polya')) for(lambda in c(0,.1,3,50))
for(shape in c(.2,2,20)) for(prob in c(.01,.4,.99,1)) for(x in c(0,1,5,50,150,1000)) {
  k <- 0:x
  if(family=='delaporte') {
    ld<-logsum(dpois(k,lambda,log=TRUE)+dnbinom(x-k,shape,prob,log=TRUE))
    lp<-logsum(dpois(k,lambda,log=TRUE)+pnbinom(x-k,shape,prob,log.p=TRUE))
    ls<-logsum(c(ppois(x,lambda,lower.tail=FALSE,log.p=TRUE),dpois(k,lambda,log=TRUE)+pnbinom(x-k,shape,prob,lower.tail=FALSE,log.p=TRUE)))
  } else {
    ld<-logsum(dpois(k,lambda,log=TRUE)+dnbinom(x-k,k,prob,log=TRUE))
    lp<-logsum(dpois(k,lambda,log=TRUE)+pnbinom(x-k,k,prob,log.p=TRUE))
    ls<-logsum(c(ppois(x,lambda,lower.tail=FALSE,log.p=TRUE),dpois(k,lambda,log=TRUE)+pnbinom(x-k,k,prob,lower.tail=FALSE,log.p=TRUE)))
  }
  writeLines(paste(c(family,lambda,shape,prob,x,ld,min(0,lp),min(0,ls)),collapse='\t'),out)
}
close(out)
# R package checks for the named multivariate densities, plus exact moments.
out <- file('build/audit2/joint/dirichlet.tsv','wt')
writeLines('a\tb\tc\tx\ty\tz\tlogDensity',out)
for(a in c(.1,1,3,100)) for(b in c(.2,2,40)) for(x in c(1e-8,.01,.2,.7)) {
  y<-(1-x)*.3;z<-(1-x)*.7
  writeLines(paste(c(a,b,2,x,y,z,ddirichlet(c(x,y,z),c(a,b,2),log=TRUE)),collapse='\t'),out)
}
close(out)
writeLines(capture.output(sessionInfo()),'build/audit2/joint/finance-session-info.txt')
# Freeze only comparisons whose R oracle is reliable; known strong-parameter
# copula and huge-df mvtnorm defects are tested by independent identities.
dir.create('src/test/resources/jdistlib',recursive=TRUE,showWarnings=FALSE)
out <- file('src/test/resources/jdistlib/audit2-joint.tsv','wt')
for(kind in c('copula','bb1','multivariate','finance','dirichlet')) {
  name<-if(kind=='copula') 'reference' else kind
  rows<-readLines(paste0('build/audit2/joint/',name,'.tsv'))[-1]
  if(kind=='copula')rows<-rows[vapply(strsplit(rows,'\t'),function(a)abs(as.numeric(a[2]))<=10,TRUE)]
  if(kind=='multivariate')rows<-rows[vapply(strsplit(rows,'\t'),function(a)as.numeric(a[2])<=100,TRUE)]
  writeLines(paste(kind,rows,sep='\t'),out)
}
close(out)
