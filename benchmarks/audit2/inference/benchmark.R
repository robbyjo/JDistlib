.libPaths(c(normalizePath("build/audit2/Rlib"),.libPaths()))
library(posterior)
args<-commandArgs(TRUE);n<-if(length(args))as.integer(args[1])else 16384
x<-sapply(0:3,function(c)sin((1:n)*(.003*4096/n)+c)+.05*cos((1:n)*.13+c))
work<-function() c(mean=mean(x),sd=sd(x),median=median(x),quantile(x,c(.025,.975)),rhat=rhat(x),bulk=ess_bulk(x),tail=ess_tail(x),mcse=mcse_mean(x))
for(i in 1:2) invisible(work())
ms<-replicate(5,system.time(work())[["elapsed"]]*1000)
cat(sprintf("diagnostics_4x%d\t%.6f\t%.17g\n",n,median(ms),ess_bulk(x)))
