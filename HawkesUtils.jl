module HawkesUtils
using Distributions,LogExpFunctions,BSplineKit
export kernGam,KernGam,kernExp,KernExp,intCon,IntCon,intBsp,IntBsp
export tofreeGam,fromfreeGam,tofreeExp,fromfreeExp,tofreeBspExp,fromfreeBspExp
export hawkesllik #loglikelihood of Hawkes process
export exphawkesllik #loglikelihood of exponential Hawkes process

## define a default kernel function and its integral
function kernGam(x;pa) pa[1]*pdf(Gamma(pa[2],pa[3]),x) end
# kernGam.([0:20;]/5,pa=[1,4,0.5])
function KernGam(x;pa) pa[1]*cdf(Gamma(pa[2],pa[3]),x) end
# KernGam.([0:20;]/5,pa=[1,4,0.5])
function kernExp(x;pa) pa[1]*pdf(Exponential(pa[2]),x) end
# kernExp.([0:20;]/5,pa=[4,0.5])
function KernExp(x;pa) pa[1]*cdf(Exponential(pa[2]),x) end
# KernExp.([0:20;]/5,pa=[4,0.5])

function intCon(x;pa) return pa[1] end
# intCon.(1:10,pa=1.0)
function IntCon(x;pa) return pa[1]*x end
## parameter transformations
function intBsp(x;coef,iknts,ord) 
    return Spline(BSplineBasis(BSplineOrder(ord),iknts),coef)(x) 
end
# intCon.(1:10,pa=1.0)
function IntBsp(x;coef,iknts,ord) 
        return integral(Spline(BSplineBasis(BSplineOrder(ord),iknts),coef))(x) 
end
## parameter transformations

function tofreeGam(par) 
    [log(par[1]);logit(par[2]);log.(par[3:4])]
end
# tofreeGam([1,0.5,1,2])
function fromfreeGam(par) 
    [exp(par[1]);logistic(par[2]);exp.(par[3:4])]
end
## fromfreeGam(tofreeGam([1,0.5,1,2]))
function tofreeExp(par) 
    [log(par[1]);logit(par[2]);log(par[3])]
end

function fromfreeExp(par) 
    [exp(par[1]);logistic(par[2]);exp(par[3])]
end

function tofreeBspExp(par) 
    [log.(par[1:end-2]);logit(par[end-1]);log(par[end])]
end

function fromfreeBspExp(par) 
    [exp.(par[1:end-2]);logistic(par[end-1]);exp(par[end])]
end


function hawkesllik(pa,tms,cens;np=(b=1,e=3),basint=intCon,
                    basInt=IntCon,kern=kernGam,Kern=KernGam)
    if any(tms .> cens) 
        @warn("Event times beyond the censoring time truncated!")
        tms = tms[tms .<= cens]
    end
    if length(pa) != sum(np)
        @warn("length of pa does not match np.b+np.e")
    end
    n=length(tms)
    r=0.0
    for i in 1:n
        r += log(basint(tms[i],pa=pa[1:np.b])+ 
            (i<2 ? 0.0 : sum(kern.(tms[i] .- tms[1:i-1],pa=pa[np.b+1:end]) )) )
    end
    r -= basInt(cens,pa=pa[1:np.b])
    r -= (n==0 ? 0.0 : sum(Kern.(cens .- tms, pa=pa[np.b+1:end])) )
    r
end

function exphawkesllik(pa,tms,cens;npb=1,basint=intCon,basInt=IntCon)
    if any(tms .> cens) 
        @warn("Event times beyond the censoring time truncated!")
        tms = tms[tms .<= cens]
    end
    if length(pa) != npb+2
        @warn("length of pa does not match npb+2")
    end
    n=length(tms)
    r=0.0
    ex = 0.0
    for i in 1:n
        if i > 1
            ex += pa[npb+1]/pa[npb+2]
            ex *= exp(-(tms[i]-tms[i-1])/pa[npb+2])
            # ex = (ex + pa[npb+1]/pa[npb+2])*exp(-(tms[i]-tms[i-1])/pa[npb+2])
        end
        r += log(basint(tms[i],pa=pa[1:npb])+ ex) 
    end
    r -= basInt(cens,pa=pa[1:npb])
    r -= pa[npb+1]*(n==0 ? 0.0 : sum(1 .- exp.(-(cens .- tms)/pa[npb+2])))
    r
end

# The following is not needed in the package DisPaHawkes
# export lognormkde
# ## Mu: the vector of means of the normal distributions
# ## Sig: the vector of std's of the normal distributions
# ## lwts: logarithm of the weights of the normal distributions
# function lognormkde(x,lwts,Mu,Sig)
#     d=length(first(Mu))
#     if length(x) != d 
#         @warn("(dim(x)=$(length(x)))!= (dim(Mu)=$(length(first(Mu))))")
#     end
#     n=length(Mu)
#     lw = lwts .- logsumexp(lwts)
#     ld = logpdf.(MvNormal.(Mu,Sig),fill(x,n))
#     logsumexp(lw+ld)
# end

end
