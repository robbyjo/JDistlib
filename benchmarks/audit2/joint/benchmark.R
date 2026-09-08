.libPaths(c(normalizePath('build/audit2/Rlib'), .libPaths()))
library(copula)
library(VineCopula)
points<-cbind(.1+.8*(0:255)/256,.1+.8*((0:255*19)%%256)/256)
bench<-function(name,n,fun) {
  for(i in 1:2) invisible(fun())
  elapsed<-replicate(5,system.time(for(i in 1:10)invisible(fun()))[['elapsed']]*100)
  cat(sprintf('%s,%d,%.6f\n',name,n,median(elapsed)))
}
joe<-joeCopula(2.7)
uv<-points[(0:19999)%%256+1,]
bench('joe_density',20000,function() dCopula(uv,joe,log=TRUE))
bench('bb1_density',20000,function() log(BiCopPDF(uv[,1],uv[,2],7,1.3,2.1)))
bench('joe_tau',1000,function() replicate(1000,tau(joe)))
logsum<-function(x){m<-max(x);m+log(sum(exp(x-m)))}
bench('polya_cdf',100,function()sapply(0:99,function(i){n<-200+i%%50;k<-0:n;sum(dpois(k,30)*pnbinom(n-k,k,.4))}))
bench('delaporte_cdf',100,function()sapply(0:99,function(i){n<-200+i%%50;k<-0:n;sum(dpois(k,30)*pnbinom(n-k,10,.4))}))
bench('polya_density',1000,function()sapply(0:999,function(i){n<-200+i%%50;k<-0:n;logsum(dpois(k,30,log=TRUE)+dnbinom(n-k,k,.4,log=TRUE))}))
uv100<-points[1:100,]
bench('gaussian_cdf',100,function()pCopula(uv100,normalCopula(.7)))
bench('student_cdf',100,function()pCopula(uv100,tCopula(.7,df=5)))
bench('normal_maximum_random',1000,function()qnorm(runif(1000)^(1/1000)))
