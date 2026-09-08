# Run from repository root; records exact package versions with the references.
.libPaths(c(normalizePath("build/audit2/Rlib"), .libPaths()))
suppressPackageStartupMessages(library(posterior))
suppressPackageStartupMessages(library(loo))
suppressPackageStartupMessages(library(deSolve))
options(digits=17)
result <- data.frame(group=character(), case=character(), metric=character(), value=double())
add <- function(group, case, metric, value) {
 result[nrow(result)+1,] <<- list(group,case,metric,value)
}
chains <- function(kind,n) {
 x <- matrix(0,n,4)
 for(c in 0:3) for(i in 0:(n-1)) {
  v <- sin((i+1)*(c+2)*1.17)+cos((i+1)*.31+c)
  x[i+1,c+1] <- if(i==0) v else .7*x[i,c+1]+v
 }
 if(kind=="antithetic") x <- x * rep(c(1,-1),length.out=n)
 if(kind=="skewed") x <- exp(x*.4)
 if(kind=="tied") x <- round(x)
 x
}
for(kind in c("ordinary","antithetic","skewed","tied","odd")) {
 x <- chains(kind,if(kind=="odd")257 else 256)
 for(metric in c("rhat","ess_bulk","ess_tail","mcse_mean")) add("diagnostics",kind,metric,suppressWarnings(get(metric)(x)))
}
for(kind in c("smooth","heavy","bounded")) {
 i <- 0:199; p <- ((i*73)%%200+.5)/200
 lr <- switch(kind,smooth=.8*sin((i+1)*.7)+.4*cos((i+1)*1.3),heavy=-1.2*log1p(-p),bounded=.3*log1p(-p))
 fit <- suppressWarnings(psis(lr,r_eff=1))
 add("psis",kind,"k",pareto_k_values(fit))
 w <- weights(fit,normalize=TRUE,log=TRUE)
 for(j in c(1,32,200)) add("psis",kind,paste0("weight",j-1),w[j])
}
i <- 0:199
ll <- cbind(-2+.3*sin((i+1)*.4),-1+.2*cos((i+1)*.6),-3+.5*sin((i+1)*.2))
l <- suppressWarnings(loo(ll,r_eff=1));w <- suppressWarnings(waic(ll))
add("predictive","matrix","loo",l$estimates["elpd_loo","Estimate"])
add("predictive","matrix","waic",w$estimates["elpd_waic","Estimate"])
add("predictive","matrix","pwaic",w$estimates["p_waic","Estimate"])
add("predictive","matrix","waic_se",w$estimates["elpd_waic","SE"])
a <- matrix(c(1,2,3,4,5,6),nrow=3,byrow=TRUE)
for(j in 1:2) add("matrix","svd",paste0("s",j),svd(a)$d[j])
a <- matrix(c(4,1,1,3),2)
for(j in 1:2) add("matrix","solve",paste0("x",j),solve(a,c(1,2))[j])
rhs <- function(t,y,p) list(c(y[2],-y[1]))
z <- ode(c(1,0),c(0,.1,1,3),rhs,NULL,rtol=1e-11,atol=1e-13)
for(i in 2:4) for(j in 1:2) add("ode",as.character(c(.1,1,3)[i-1]),paste0("y",j),z[i,j+1])
f <- function(x) -.5*sum(x*x)+sin(x[1]*x[2])+log1p(exp(x[3]))
g <- numDeriv::grad(f,c(.7,-1.2,.3))
for(j in 1:3) add("gradient","smooth",paste0("g",j),g[j])
a <- Matrix::Matrix(matrix(c(4,1,1,3),2),sparse=TRUE)
r <- as.numeric(Matrix::solve(a,c(1,2)))
for(j in 1:2) add("matrix","sparse_solve",paste0("x",j),r[j])
result$value <- sprintf("%.17g",result$value)
write.csv(result,"src/test/resources/jdistlib/inference/audit2-reference.csv",row.names=FALSE)
writeLines(c(R.version.string,sapply(c("posterior","loo","deSolve","Matrix","numDeriv"),function(p) paste(p,packageVersion(p)))),"benchmarks/audit2/inference/reference-versions.txt")
