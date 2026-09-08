# R 4.6.1: independent package fixtures. Run at repository root.
.libPaths(c(normalizePath('build/audit2/Rlib'), .libPaths()))
options(digits=17)
out <- list()
add <- function(java, params, rparams, pkg, family, x, tolerance=2e-9, quantiles=TRUE) {
  for(op in c('density','cumulative',if(quantiles)'quantile')) {
    fname<-paste0(switch(op,density='d',cumulative='p',quantile='q'),family)
    env<-if(pkg=='base-identities') globalenv() else asNamespace(pkg)
    if(!exists(fname,env,inherits=FALSE)) next
    f<-get(fname,env)
    for(logp in c(FALSE,TRUE)) for(lower in if(op=='density') TRUE else c(FALSE,TRUE)) {
      xx<-if(op=='quantile') c(1e-12,1e-8,.001,.1,.5,.9,1-1e-8,1-1e-12) else x
      if(op=='quantile' && logp) xx<-log(xx)
      args<-c(list(xx),rparams)
      if(op=='density') args$log<-logp else {args$lower.tail<-lower;args$log.p<-logp}
      nativeLog <- 'log.p' %in% names(formals(f))
      if(op!='density' && !nativeLog) {
        args$log.p<-NULL
        if(op=='quantile' && logp) args[[1]]<-exp(xx)
      }
      y<-vapply(seq_along(xx),function(i) { aa<-args;aa[[1]]<-args[[1]][i];
        tryCatch(suppressWarnings(do.call(f,aa)),error=function(e) NA_real_)
      },numeric(1))
      if(op=='cumulative' && logp && !nativeLog) y<-log(y)
      out[[length(out)+1]]<<-data.frame(java,op,x=sprintf('%.17g',xx),params=paste(sprintf('%.17g',params),collapse=';'),lower=tolower(lower),log=tolower(logp),expected=sprintf('%.17g',y),tolerance)
    }
  }
}
xpos<-c(.001,.03,.2,.7,1,2,8,30,100)
for(loc in c(-2,0,3)) for(scale in c(.2,1,5)) for(shape in c(-.8,-.1,0,.1,1,3)) {
  xx<-loc+scale*c(-4,-.3,0,.1,.5,1,3,10)
  add('evd.GEV',c(loc,scale,shape),list(loc=loc,scale=scale,shape=shape),'evd','gev',xx)
  add('evd.GeneralizedPareto',c(loc,scale,shape),list(loc=loc,scale=scale,shape=shape),'evd','gpd',xx)
}
for(loc in c(-2,0,3)) for(scale in c(.2,1,5)) for(shape in c(.3,1,3,8)) {
  add('evd.Fretchet',c(loc,scale,shape),list(loc=loc,scale=scale,shape=shape),'evd','frechet',loc+scale*xpos)
  add('evd.ReverseWeibull',c(loc,scale,shape),list(loc=loc,scale=scale,shape=shape),'evd','rweibull',loc-scale*xpos)
}
for(mu in c(.1,1,10)) for(sigma in c(.1,.5,2,10)) add('InvNormal',c(mu,sigma),list(mean=mu,dispersion=sigma^2),'statmod','invgauss',mu*xpos,tolerance=5e-8)
for(a in c(.2,1,3,10)) for(b in c(.2,1,3,10)) {
  add('BetaPrime',c(a,b),list(shape1=a,shape2=b),'extraDistr','betapr',xpos)
  add('Kumaraswamy',c(a,b),list(a=a,b=b),'extraDistr','kumar',c(.001,.03,.2,.5,.8,.97,.999))
  add('GeneralizedGamma',c(2,a,b),list(scale=2,d=a,k=b),'VGAM','gengamma.stacy',xpos)
  add('GeneralizedBetaSecondKind',c(2,.7,a,b),list(scale=2,shape1.a=.7,shape2.p=a,shape3.q=b),'VGAM','genbetaII',xpos)
}
for(scale in c(.1,1,10)) {
  add('evd.Rayleigh',scale,list(sigma=scale),'extraDistr','rayleigh',scale*xpos)
  add('Maxwell',scale,list(rate=scale),'VGAM','maxwell',xpos)
  add('Lindley',scale,list(theta=scale),'VGAM','lind',xpos)
  add('Laplace',c(2,scale),list(mu=2,sigma=scale),'extraDistr','laplace',c(-10,-2,0,2,5,10))
}
for(scale in c(.3,1,3)) {
  add('HalfNormal',scale,list(sigma=scale),'extraDistr','hnorm',xpos)
  add('HalfCauchy',scale,list(sigma=scale),'extraDistr','hcauchy',xpos)
  add('PositiveNormal',c(-1,scale),list(mean=-1,sd=scale,a=0,b=Inf),'extraDistr','tnorm',xpos)
  add('Slash',c(0,scale),list(mu=0,sigma=scale),'extraDistr','slash',c(-10,-1,-.01,0,.01,1,10),quantiles=FALSE)
  for(shape in c(.3,1,3)) {
    add('LogLogistic',c(shape,scale),list(shape=shape,scale=scale),'actuar','llogis',xpos)
    add('InvGamma',c(shape,scale),list(alpha=shape,beta=1/scale),'extraDistr','invgamma',xpos)
    add('HalfT',c(shape,scale),list(nu=shape,sigma=scale),'extraDistr','ht',xpos)
    add('BirnbaumSaunders',c(shape,scale,0),list(alpha=shape,beta=scale,mu=0),'extraDistr','fatigue',xpos)
    add('FellerPareto',c(0,shape,2,.7,scale),list(min=0,shape1=shape,shape2=2,shape3=.7,scale=scale),'actuar','fpareto',xpos)
    add('PoissonInverseGaussian',c(shape,scale),list(mean=shape,dispersion=scale),'actuar','poisinvgauss',c(0,1,2,5,10,20),quantiles=FALSE)
  }
}
for(prob in c(.1,.5,.9)) {
  add('Logarithmic',prob,list(prob=prob),'actuar','logarithmic',c(1,2,5,10,20))
  add('DiscreteLaplace',c(0,prob),list(location=0,scale=prob),'extraDistr','dlaplace',c(-10,-2,0,1,5,20))
  for(shape in c(.5,2)) {
    add('BetaBinomial',c(prob,shape,20),list(size=20,alpha=prob/shape,beta=(1-prob)/shape),'extraDistr','bbinom',c(0,1,2,5,10,19,20))
    add('DiscreteWeibull',c(prob,shape),list(shape1=prob,shape2=shape),'extraDistr','dweibull',c(0,1,2,5,10,20))
    add('BetaNegativeBinomial',c(2,prob/shape,(1-prob)/shape),list(size=2,alpha=prob/shape,beta=(1-prob)/shape),'extraDistr','bnbinom',c(0,1,2,5,10,20),quantiles=FALSE)
  }
  for(lambda in c(.2,2,20)) {
    add('ZeroInflatedPoisson',c(lambda,prob),list(lambda=lambda,pi=prob),'extraDistr','zip',c(0,1,2,5,10,20))
    add('ZeroTruncatedPoisson',lambda,list(lambda=lambda),'actuar','ztpois',c(0,1,2,5,10,20))
    add('ZeroInflatedNegativeBinomial',c(lambda,2,prob),list(size=2,prob=2/(lambda+2),pi=prob),'extraDistr','zinb',c(0,1,2,5,10,20))
    add('ZeroTruncatedNegativeBinomial',c(lambda,2),list(size=2,prob=2/(lambda+2)),'actuar','ztnbinom',c(0,1,2,5,10,20))
  }
}
for(mode in c(0,.3,1)) add('Triangular',c(0,1,mode),list(a=0,b=1,c=mode),'extraDistr','triang',c(.01,.1,.3,.5,.8,.99))
# These transformed laws have independent base-R reference identities.
dchiAudit<-function(x,df,log=FALSE) {v<-dchisq(x*x,df,log=TRUE)+log(2*x);if(log)v else exp(v)}
pchiAudit<-function(x,df,lower.tail=TRUE,log.p=FALSE)pchisq(x*x,df,lower.tail=lower.tail,log.p=log.p)
qchiAudit<-function(p,df,lower.tail=TRUE,log.p=FALSE)sqrt(qchisq(p,df,lower.tail=lower.tail,log.p=log.p))
darcsineAudit<-function(x,a,b,log=FALSE){v<-dbeta((x-a)/(b-a),.5,.5,log=TRUE)-log(b-a);if(log)v else exp(v)}
parcsineAudit<-function(x,a,b,lower.tail=TRUE,log.p=FALSE)pbeta((x-a)/(b-a),.5,.5,lower.tail=lower.tail,log.p=log.p)
qarcsineAudit<-function(p,a,b,lower.tail=TRUE,log.p=FALSE)a+(b-a)*qbeta(p,.5,.5,lower.tail=lower.tail,log.p=log.p)
dlevyAudit<-function(x,mu,sigma,log=FALSE){v<-dchisq(sigma/(x-mu),1,log=TRUE)+log(sigma)-2*log(x-mu);if(log)v else exp(v)}
plevyAudit<-function(x,mu,sigma,lower.tail=TRUE,log.p=FALSE)pchisq(sigma/(x-mu),1,lower.tail=!lower.tail,log.p=log.p)
qlevyAudit<-function(p,mu,sigma,lower.tail=TRUE,log.p=FALSE)mu+sigma/qchisq(p,1,lower.tail=!lower.tail,log.p=log.p)
for(df in c(.3,1,2,5,30))add('Chi',df,list(df=df),'base-identities','chiAudit',xpos)
for(scale in c(.3,1,3))add('Levy',c(0,scale),list(mu=0,sigma=scale),'base-identities','levyAudit',xpos)
for(a in c(-2,0))add('Arcsine',c(a,a+5),list(a=a,b=a+5),'base-identities','arcsineAudit',a+5*c(.001,.1,.3,.7,.9,.999))
dir.create('build/audit2',showWarnings=FALSE)
write.table(do.call(rbind,out),'build/audit2/scalar-reference.tsv',sep='\t',row.names=FALSE,quote=FALSE)
writeLines(capture.output(sessionInfo()),'build/audit2/scalar-R-session.txt')
