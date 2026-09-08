# Base-R and package numerical utilities and distribution-test references.
.libPaths(c(normalizePath("build/audit2/Rlib"), .libPaths()))
dir.create("src/test/resources/audit2", recursive=TRUE, showWarnings=FALSE)
rows <- list()
emit <- function(key, values) {
  for (i in seq_along(values)) rows[[length(rows)+1L]] <<- c(key, i-1L, sprintf("%.17g", values[i]))
}
x <- c(-2,-.9,-.3,0,.15,.4,.7,1.1,1.8,3)
w <- seq_along(x)/sum(seq_along(x))
for (kernel in c("gaussian","rectangular","triangular","epanechnikov","biweight","cosine","optcosine")) {
  for (mode in c("plain","weighted","infinite","weighted_infinite")) {
    xx <- if (mode == "infinite") c(x,Inf,-Inf) else x
    ww <- if (mode == "weighted") w else if (mode == "weighted_infinite") w*.75 else NULL
    # A finite subdensity is the independent reference for omitted infinite
    # weight. R density(x_with_Inf,weights) itself applies that mass twice.
    d <- density(xx,bw=.4,kernel=kernel,weights=ww,subdensity=TRUE,n=65,from=-4,to=5)
    emit(paste("density",kernel,mode,sep=":"),d$y)
  }
}
for (n in c(20,1000,10000)) {
  xx <- sin(seq_len(n)*.37)+cos(seq_len(n)*.11)*.3+seq_len(n)/n
  for (method in c("nrd0","nrd","ucv","bcv","SJ-ste","SJ-dpi")) {
    f <- switch(method,nrd0=bw.nrd0,nrd=bw.nrd,ucv=bw.ucv,bcv=bw.bcv,
                `SJ-ste`=function(x) bw.SJ(x,method="ste"),`SJ-dpi`=function(x) bw.SJ(x,method="dpi"))
    emit(paste("bandwidth",n,method,sep=":"),suppressWarnings(f(xx)))
  }
}
xx <- rep(c(0,1),50000)
emit("bandwidth:counts:bcv",suppressWarnings(bw.bcv(xx)))
xx <- 1e12+c(0,1,2,3,4)
emit("summary:offset",c(mean(xx),var(xx),sd(xx)))
emit("summary:ordinary",c(mean(x),var(x),sd(x),quantile(x,seq(0,1,.1))))
for (n in c(8,20,100,2000)) {
  xx <- sin(seq_len(n)*.37)+cos(seq_len(n)*.11)*.3+seq_len(n)/n
  sw <- shapiro.test(xx)
  emit(paste0("shapiro:",n),c(unname(sw$statistic),sw$p.value))
}
p <- c(.001,.003,.01,.02,.05,.15,.7,.99,NA,.4)
for (mode in c("ordinary", "offset", "outlier", "skew")) {
  xx <- if (mode == "offset") 1e12+seq_len(100) else if (mode == "outlier") c(rep(0,999),1) else if (mode == "skew") exp(seq(0,5,length.out=100)) else sin(seq_len(100)*.37)+seq_len(100)/100
  ad <- nortest::ad.test(xx); cvm <- nortest::cvm.test(xx); lillie <- nortest::lillie.test(xx)
  jb <- moments::jarque.test(xx)
  dp <- unname(moments::agostino.test(xx)$statistic[2]^2+moments::anscombe.test(xx)$statistic[2]^2)
  emit(paste0("normality:",mode),c(ad$statistic,ad$p.value,cvm$statistic,cvm$p.value,lillie$statistic,lillie$p.value,jb$statistic,jb$p.value,dp,pchisq(dp,2,lower.tail=FALSE)))
}
for (method in c("bonferroni","holm","hochberg","hommel","BH","BY","none"))
  emit(paste0("adjust:",method),p.adjust(p,method))
xx <- seq(0,1,length.out=30); yy <- sin(xx*6)+xx^2
# Independent exact roughness Gram matrix: B'' is piecewise linear, so
# Simpson's rule integrates each product exactly. R smooth.spline retains
# a historical 0.3330 approximation here; Java uses the exact 1/3.
knots <- c(rep(0,3),xx,rep(1,3))
B <- splines::splineDesign(knots,xx,ord=4)
S <- matrix(0,ncol(B),ncol(B))
for (i in seq_len(length(xx)-1)) {
  points <- c(xx[i],(xx[i]+xx[i+1])/2,xx[i+1])
  D <- splines::splineDesign(knots,points,ord=4,derivs=rep(2L,3))
  S <- S+(xx[i+1]-xx[i])/6*crossprod(D,D*c(1,4,1))
}
X <- crossprod(B); inner <- 3:(ncol(B)-3)
ratio <- sum(diag(X)[inner])/sum(diag(S)[inner])
for (spar in c(.3,.7)) {
  fit <- smooth.spline(xx,yy,spar=spar,all.knots=TRUE)
  emit(paste0("splineR:",spar),predict(fit,seq(-.1,1.1,length.out=25))$y)
  beta <- solve(X+ratio*16^(spar*6-2)*S,crossprod(B,yy))
  px <- seq(-.1,1.1,length.out=25); bound <- pmax(0,pmin(1,px))
  value <- as.vector(splines::splineDesign(knots,bound,ord=4)%*%beta)
  slope <- as.vector(splines::splineDesign(knots,bound,ord=4,derivs=rep(1L,length(px)))%*%beta)
  emit(paste0("spline:",spar),value+(px-bound)*slope)
}
emit("opt:root",uniroot(function(z) cos(z)-z,c(0,1),tol=1e-12)$root)
emit("opt:minimum",optimize(function(z) (z-.37)^2+exp(z)/10,c(-1,2),tol=1e-12)$minimum)
out <- do.call(rbind,rows)
write.table(out,"src/test/resources/audit2/math-reference.tsv",sep="\t",row.names=FALSE,col.names=c("key","index","value"),quote=FALSE)
cat(R.version.string,"\n",nrow(out),"reference values\n")
