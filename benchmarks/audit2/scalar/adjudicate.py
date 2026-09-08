"""Adjudicate R discrepancies with 80-decimal independent formulas.
Run with PYTHONPATH=build/core-audit/python-deps after ScalarReferenceCheck.
"""
import csv, collections
import mpmath as m
m.mp.dps=80
def log1m(x): return m.log(-m.expm1(x))
def numeric(x):
    if x in ('NA','NaN'): return m.nan
    # Java and R were both given the nearest binary64 value, not the printed decimal.
    return m.mpf(float(x))
def calc(name,op,x,a,lower,logp):
    if op=='quantile':
        lp=x if logp else m.log(x); lq=log1m(x) if logp else m.log1p(-x)
        if not lower: lp,lq=lq,lp
        if name=='evd.GEV':
            loc,scale,shape=a; z=-m.log(-lp); return loc+scale*(m.expm1(shape*z)/shape if shape else z)
        if name=='evd.GeneralizedPareto':
            loc,scale,shape=a;return loc+scale*(m.expm1(-shape*lq)/shape if shape else -lq)
        if name=='evd.Fretchet':
            loc,scale,shape=a;return loc+scale*(-lp)**(-1/shape)
        if name=='evd.ReverseWeibull':
            loc,scale,shape=a;return loc-scale*(-lp)**(1/shape)
        if name=='evd.Rayleigh':return a[0]*m.sqrt(-2*lq)
        if name=='Kumaraswamy':return (-m.expm1(lq/a[1]))**(1/a[0])
        raise ValueError('invert')
    if name=='BirnbaumSaunders':
        aa,b,mu=a;u=m.sqrt((x-mu)/b);z=(u-1/u)/aa
        F=m.erfc(-z/m.sqrt(2))/2;S=m.erfc(z/m.sqrt(2))/2;f=(u+1/u)/(2*aa*(x-mu))*m.exp(-z*z/2)/m.sqrt(2*m.pi)
    elif name in ('HalfNormal','PositiveNormal'):
        mu,sd=(0,a[0]) if name=='HalfNormal' else a;z=(x-mu)/sd;Z=m.erfc(-mu/sd/m.sqrt(2))/2
        S=m.erfc(z/m.sqrt(2))/2/Z;F=1-S;f=m.exp(-z*z/2)/m.sqrt(2*m.pi)/sd/Z
    elif name=='HalfCauchy':
        z=x/a[0];F=2*m.atan(z)/m.pi;S=2*m.atan(1/z)/m.pi;f=2/(m.pi*a[0]*(1+z*z))
    elif name=='HalfT':
        df,sd=a;z=x/sd;S=m.betainc(df/2,m.mpf('.5'),0,df/(df+z*z),regularized=True);F=m.betainc(m.mpf('.5'),df/2,0,z*z/(df+z*z),regularized=True)
        f=2*m.gamma((df+1)/2)/(m.sqrt(df*m.pi)*m.gamma(df/2)*sd)*(1+z*z/df)**(-(df+1)/2)
    elif name=='InvGamma':
        shape,scale=a;z=1/(x*scale);F=m.gammainc(shape,z,m.inf,regularized=True);S=m.gammainc(shape,0,z,regularized=True);f=z**shape*m.exp(-z)/(m.gamma(shape)*x)
    elif name=='LogLogistic':
        shape,scale=a;z=(x/scale)**shape;F=z/(1+z);S=1/(1+z);f=shape*z/(x*(1+z)**2)
    elif name=='FellerPareto':
        loc,b,power,aa,scale=a;z=((x-loc)/scale)**power
        F=m.betainc(aa,b,0,z/(1+z),regularized=True);S=m.betainc(b,aa,0,1/(1+z),regularized=True)
        f=power*z**aa/(x-loc)/(1+z)**(aa+b)/m.beta(aa,b)
    elif name=='DiscreteLaplace':
        loc,p=a;k=m.floor(x-loc);f=(1-p)/(1+p)*p**abs(k)
        if k<0:F=p**(-k)/(1+p);S=1-F
        else:S=p**(k+1)/(1+p);F=1-S
    elif name=='DiscreteWeibull':
        q,b=a;k=m.floor(x);S=q**((k+1)**b);F=1-S;f=q**(k**b)-S
    elif name=='Logarithmic':
        p=a[0];k=int(x);f=-p**k/(k*m.log1p(-p));F=m.fsum(-p**i/(i*m.log1p(-p)) for i in range(1,k+1));S=1-F
    elif name=='PoissonInverseGaussian':
        mu,phi=a;k=int(x);t=2*phi*mu*mu;p0=m.exp(-2*mu/(1+m.sqrt(1+t)));p1=p0*mu/m.sqrt(1+t);mass=[p0,p1]
        for i in range(2,max(k+1,2)):mass.append(t/(1+t)*(1-m.mpf('1.5')/i)*mass[-1]+mu*mu/(1+t)*mass[-2]/(i*(i-1)))
        f=mass[k];F=m.fsum(mass[:k+1]);S=1-F
    elif name=='Slash':
        mu,sd=a;z=(x-mu)/sd;phi0=1/m.sqrt(2*m.pi);delta=phi0*(-m.expm1(-z*z/2))
        f=delta/(z*z*sd) if z else phi0/(2*sd);F=m.erfc(-z/m.sqrt(2))/2-delta/z if z else m.mpf('.5');S=1-F
    elif name=='Triangular':
        aa,b,c=a
        F=(x-aa)**2/((b-aa)*(c-aa)) if x<c else 1-(b-x)**2/((b-aa)*(b-c));S=1-F
        f=2*(x-aa)/((b-aa)*(c-aa)) if x<c else 2*(b-x)/((b-aa)*(b-c))
    elif name in ('ZeroInflatedPoisson','ZeroTruncatedPoisson','ZeroInflatedNegativeBinomial','ZeroTruncatedNegativeBinomial'):
        k=int(x);mu=a[0];nb='NegativeBinomial' in name
        if nb:
            size=a[1];p=size/(size+mu);pmf=lambda i:m.rf(size,i)/m.factorial(i)*p**size*(1-p)**i
        else:pmf=lambda i:m.exp(-mu)*mu**i/m.factorial(i)
        p0=pmf(0);F=m.fsum(pmf(i) for i in range(k+1));f=pmf(k);S=1-F
        if 'Inflated' in name:
            weight=a[-1];f=(1-weight)*f+(weight if k==0 else 0);F=weight+(1-weight)*F;S=(1-weight)*S
        else:
            f=0 if k==0 else f/(1-p0);F=(F-p0)/(1-p0);S=S/(1-p0)
    elif name in ('evd.GEV','evd.GeneralizedPareto'):
        loc,scale,shape=a;z=(x-loc)/scale
        if name=='evd.GEV':
            if shape and 1+shape*z<=0: f=0;F=0 if shape>0 else 1
            else:
                w=m.log1p(shape*z)/shape if shape else z; F=m.exp(-m.exp(-w));f=m.exp(-m.exp(-w)-(1+shape)*w)/scale
            S=1-F
        else:
            if z<0:f=0;F=0;S=1
            elif shape and 1+shape*z<=0:f=0;F=1;S=0
            else:
                w=m.log1p(shape*z)/shape if shape else z;S=m.exp(-w);F=-m.expm1(-w);f=m.exp(-(1+shape)*w)/scale
    elif name=='evd.Fretchet':
        loc,scale,shape=a;z=(x-loc)/scale
        w=z**(-shape); F=m.exp(-w);S=-m.expm1(-w);f=shape/scale*z**(-shape-1)*F
    elif name=='evd.ReverseWeibull':
        loc,scale,shape=a;z=(loc-x)/scale
        w=z**shape;F=m.exp(-w);S=-m.expm1(-w);f=shape/scale*z**(shape-1)*F
    elif name=='evd.Rayleigh':
        w=x*x/(2*a[0]**2);S=m.exp(-w);F=-m.expm1(-w);f=x/a[0]**2*S
    elif name=='Kumaraswamy':
        aa,b=a; ls=b*m.log1p(-x**aa);S=m.exp(ls);F=-m.expm1(ls);f=aa*b*x**(aa-1)*(1-x**aa)**(b-1)
    elif name=='BetaPrime':
        aa,b=a;f=x**(aa-1)*(1+x)**(-aa-b)/m.beta(aa,b)
        if x>1:S=m.betainc(b,aa,0,1/(1+x),regularized=True);F=1-S
        else:F=m.betainc(aa,b,0,x/(1+x),regularized=True);S=1-F
    elif name=='Maxwell':
        rate=a[0];w=rate*x*x/2;F=m.gammainc(m.mpf('1.5'),0,w,regularized=True);S=m.gammainc(m.mpf('1.5'),w,m.inf,regularized=True);f=m.sqrt(2/m.pi)*rate**m.mpf('1.5')*x*x*m.exp(-w)
    elif name=='Lindley':
        theta=a[0];ls=-theta*x+m.log1p(theta*x/(1+theta));F=-m.expm1(ls);S=m.exp(ls);f=theta**2/(1+theta)*(1+x)*m.exp(-theta*x)
    elif name=='InvNormal':
        mu,sigma=a;z=(x/mu-1)/(sigma*m.sqrt(x));zz=(x/mu+1)/(sigma*m.sqrt(x))
        term=m.exp(2/(mu*sigma*sigma))*m.erfc(zz/m.sqrt(2))/2
        F=m.erfc(-z/m.sqrt(2))/2+term;S=m.erfc(z/m.sqrt(2))/2-term
        f=m.exp(-z*z/2)/(sigma*m.sqrt(2*m.pi)*x**m.mpf('1.5'))
    else:raise ValueError(name)
    v=f if op=='density' else (F if lower else S)
    return m.log(v) if logp else v

def close(v,ref,tol=m.mpf('2e-8'), floor=m.mpf('1e-280')):
    if v==ref:return True
    if not m.isfinite(v) or not m.isfinite(ref):return False
    return abs(v-ref)<=tol*max(floor,abs(ref))
counts=collections.Counter();details=[]
for row in csv.reader(open('build/audit2/scalar-failures.tsv'),delimiter='\t'):
    name,op,x,params,low,logp,r,tol,j=row
    a=list(map(numeric,params.split(';')));lower=low=='true';logp=logp=='true';x=numeric(x);r=numeric(r);j=numeric(j)
    try:
        if op=='quantile' and name not in ('evd.GEV','evd.GeneralizedPareto','evd.Fretchet','evd.ReverseWeibull','evd.Rayleigh','Kumaraswamy'):
            # Inversion residual in the originally requested tail.
            ref=x if logp else m.log(x)
            jvalue=calc(name,'cumulative',j,a,lower,True) if m.isfinite(j) else m.nan
            rvalue=calc(name,'cumulative',r,a,lower,True) if m.isfinite(r) else m.nan
            if name.startswith('Zero') or name in ('DiscreteWeibull','Logarithmic','DiscreteLaplace'):
                def discrete_ok(v):
                    if not m.isfinite(v):return False
                    here=calc(name,'cumulative',v,a,lower,True);before=calc(name,'cumulative',v-1,a,lower,True) if v>0 or name=='DiscreteLaplace' else (-m.inf if lower else 0)
                    return (here>=ref and before<ref) if lower else (here<=ref and before>ref)
                jc=discrete_ok(j);rc=discrete_ok(r)
            else:jc=close(jvalue,ref,floor=1);rc=close(rvalue,ref,floor=1)
        else:
            ref=calc(name,op,x,a,lower,logp)
            floor=1 if logp and op!='quantile' else m.mpf('1e-280')
            # Location cancellation is limited by absolute binary64 resolution.
            if op=='quantile' and name.startswith('evd.') and abs(a[0])>0:floor=m.mpf('1e-6')
            jc=close(j,ref,floor=floor);rc=close(r,ref,floor=floor)
        label='both agree at 2e-8' if jc and rc else 'R discrepancy' if jc else 'Java discrepancy' if rc else 'both discrepancy'
    except Exception as exc:label='unresolved '+str(exc);ref=m.nan
    counts[(name,label)]+=1
    details.append('\t'.join(row+[label,str(ref)]))
open('build/audit2/scalar-adjudication.tsv','w').write('\n'.join(details)+'\n')
for (name,label),n in sorted(counts.items()):print(name,label,n,sep='\t')
