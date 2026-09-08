.libPaths(c(normalizePath('build/audit2/Rlib'),.libPaths()))
n<-200000L
p<-(seq_len(n)%%997+.5)/997
x<-seq_len(n)%%80
tasks<-list(InvNormal.random=function()statmod::rinvgauss(n,mean=2,dispersion=.25),
            PositiveNormal.quantile=function()extraDistr::qtnorm(p,mean=-1,sd=2,a=0,b=Inf),
            BetaBinomial.cumulative=function()extraDistr::pbbinom(x,size=100,alpha=1.5,beta=3.5))
for(name in names(tasks)) {
  f<-tasks[[name]]; f(); f()
  repetitions<-50L
  times<-replicate(5,system.time(for(i in seq_len(repetitions)) f())[['elapsed']])
  cat(name, min(times)*1e9/(n*repetitions), 'ns/value (vectorized R)\n')
}
