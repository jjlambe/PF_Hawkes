module PaFiHawkes

using Distributions
using LogExpFunctions
using Random
using SpecialFunctions
using StatsBase
include("HawkesUtils.jl"); using .HawkesUtils
include("unif_ord_stats.jl")
# Test
export ntvppdatapreproc
export ntvxphpsmcll
export ntvhpsmcll
export hawkesllik #loglikelihood of Hawkes process
export exphawkesllik #loglikelihood of exponential Hawkes process
export kernGam,KernGam,kernExp,KernExp,intCon,IntCon,intBsp,IntBsp
export MHMCMCcontExpHawkes
export MHMCMCdiscHawkes
export MHMCMCdiscExpHawkes

## preprocess point process interval counts data
function ntvppdatapreproc(ot,ov)
    ots=[ot[i] for i in 1:length(ot) if i==1||i==length(ot)|| 
        ov[i]>ov[i-1] || (ov[i-1]==ov[i]<ov[i+1])]
    ovs=[ov[i] for i in 1:length(ot) if i==1||i==length(ot)|| 
        ov[i]>ov[i-1] || (ov[i-1]==ov[i]<ov[i+1])]
    ots,ovs
end
# ot = [0:10;]/2; ov=[0;cumsum(rand(Poisson(1),10))]; length(ot)
# ot,ov = ntvppdatapreproc(ot,ov); length(ot)

## SMC approximation to the loglikelihood of the exponential kernel
## Hawkes process
##  - ot: the observation times in ascending order
##  - ov: the total number of events at the observation times
##  - pa: the parameters of the hawkes model, consisting of nu the
##        background rate and parameters of the excitation kernel
##  - basint: the baseline intensity function
##  - basInt: the integral of the baseline intensity function
##  - J: the number of particles
##  - conf: a tuning parameter controlling the proposal Poisson process

function ntvxphpsmcll(ot,ov,pa;npb=1,basint=intCon,basInt=IntCon,J=100,conf=0.95)
    # print("Correct!\n")
    if npb+2 != length(pa) 
        @warn("length of pa = $(length(pa))!= $(npb) + 2") # Checking that the number of parameters is correct (3)
    end
    N = convert(Vector{Int64},ov) # values of the counting process at the obs times
    dN = diff(N) # num. of events each intervals
    dt = diff(convert(Vector{Int64},ot))
    mdN = maximum(dN) # Maximum number of events per window... currently unused.
    n = length(dN) # num. of intervals (=num. of obs times - 1)
    eta = pa[npb+1] # Branching ratio
    beta = pa[npb+2] # Excitation function parameter

    ptcls = fill(0.0,J) # Array with all J particle values, initialised at 0
    lwts = Vector{Float64}(undef,J) # log of weights of particles

    # This section is establishing the initial log-likelihood value
    # idx of 1st interval with 1+ events (idx<=2)
    idx = 1; while dN[idx]==0 idx += 1 end # Finding the first interval with an event in it
    cumBk0 = basInt(ot[1],pa=pa[1:npb]) # baseInt is the integrated baseline intensity function TntCon (intensity constant times x)
    cumBk1 = basInt(ot[idx],pa=pa[1:npb])
    ll = -cumBk1 + cumBk0 # Initialising the log-likelihood - probability of seeing no events between the first observation time and the first idx with an event.

    ## In the first interval with 1 or more events:
    # lmd = quantile(Gamma(dN[idx],1.0),conf)/(ot[idx+1]-ot[idx])

    # cumBk represents the cumulative value of the background intensity.
    cumBk0 = cumBk1
    cumBk1 = basInt(ot[idx+1],pa=pa[1:npb]) # Cumulative background at end of first interval with an observation.
    prop_vals = OrdUnifExpMultSamp(ot[idx:idx+1], dN[idx], J)
    Threads.@threads for j in 1:J
        lwts[j] = cumBk0 - cumBk1
        # etms = ot[idx] .+  
        #     cumsum(rand(Exponential(1/lmd),dN[idx]))  # Simulating the poisson process starting at t(i-1).
        # etms = OrdUnifExpSamp(ot[idx:idx+1], dN[idx]) # Sampling from ordered uniform.
        # Ordered Uniform sampling scheme
        # Y = rand(Exponential(1), dN[idx] + 1)
        # etms = ot[idx] .+ (cumsum(Y)[1:dN[idx]]*(ot[idx + 1] - ot[idx]))/sum(Y)
        etms = prop_vals[:, j]
        # This loop calculates the log of the numerator of the weights.
        ex = ptcls[j] # ex is a placeholder for the values of the particles.
        
        for k in 1:dN[idx]
            ex *= exp(-(k==1 ? etms[1]-ot[idx] :
                etms[k]-etms[k-1] )/beta) # Does this not always just return 0 for k = 1 because ex = 0?
            lwts[j] += log(basint(etms[k],pa=pa[1:npb])+ex)
            # print(log(basint(etms[k],pa=pa[1:npb]) + ex) == log(pa[1] + ex))
            ex += eta/beta
            lwts[j] -= eta*(1-exp(-(ot[idx+1]-etms[k])/beta))
        end

        # lwts[j] -= dN[idx]*log(lmd)-lmd*(etms[dN[idx]]-ot[idx]) # Here we are subtracting the denominator of the weights (log)
        lwts[j] -= sum(log.(2:dN[idx])) - dN[idx]*log(ot[idx+1] - ot[idx]) # Check whether indexing is correct on second term.
        ex *= exp( -(ot[idx+1]-etms[dN[idx]])/beta)
        ptcls[j] = ex
        if etms[dN[idx]] > ot[idx+1] 
            lwts[j] = -Inf
        end
    end
    lphat = logsumexp(lwts)-log(J)#sumexp(lwts); 
    ll += lphat

    ## For the idx+1st to the last interval

    for i in idx+1:n  # i=index of the interval = ot index - 1
        # print(i, "\n")
        smp = sample(1:J, weights(exp.(lwts)), J)
        ptcls = ptcls[smp] 
        cumBk0 = cumBk1
        cumBk1 = basInt(ot[i+1],pa=pa[1:npb])
        if dN[i]==0 
            lwts .= -cumBk1+cumBk0
            lwts -= ptcls * beta*
                (1-exp(-(ot[i+1] - ot[i])/beta))
            ptcls *= exp(-(ot[i+1] - ot[i])/beta)
        else
            # lmd = quantile(Gamma(dN[i],1.0),conf)/(ot[i+1]-ot[i])
            # prop_vals = OrdUnifExpMultSamp(ot[i:i+1], dN[i], J)
            Threads.@threads for j in 1:J
                # etms = ot[i] .+
                #     cumsum(rand(Exponential(1/lmd),dN[i])) 
                # etms = OrdUnifExpSamp(ot[i:i+1], dN[i]) # Sampling from ordered uniform.
                Y = rand(Exponential(1), dN[i] + 1)
                etms = ot[i] .+ (cumsum(Y)[1:dN[i]]*dt[i])/sum(Y) # Sampling from ordered uniform
                # etms = prop_vals[:, j]

                lwts[j] = -cumBk1 + cumBk0 # Accumulation of background intensity.
                ex = ptcls[j] # Establishing value of epsilon for this particle.
                lwts[j] -= ex*beta*cdf(Exponential(beta),
                                       ot[i+1]-ot[i]) # I think this contributes to the sum with epsilons.
        
                for k in 1:dN[i]
                    # This section recursively computes the log of the product of intensities evaluated at the event times.
                    ex *= exp(-(k==1 ? etms[1]-ot[i] :
                        etms[k]-etms[k-1])/beta) # Updating epsilon term to kth event time.
                    lwts[j] += log(basint(etms[k],pa=pa[1:npb])+ex) # Adding on the log of background intensity plus excitation effect.
                    ex += eta/beta # Updating epsilon to kth event time '+' for the next round
                    lwts[j] -= eta*(1 - exp(-(ot[i+1]-etms[k])/beta)) # I think this also contributes to the sum with epsilons.
                end
                # lwts[j] -= dN[i]*log(lmd)-lmd*(etms[dN[i]]-ot[i]) 
                lwts[j] -= sum(log.(2:dN[i])) - dN[i]*log(dt[i]) # Check whether indexing is correct on second term.
                ex *= exp(-(ot[i+1]-etms[dN[i]])/beta) # Updating ex to window between n_i th event and next time window.
                ptcls[j] = ex

                if etms[dN[i]] > ot[i+1] 
                    lwts[j] = -Inf 
                end
            end
        end
        lphat = logsumexp(lwts)-log(J);
        ll += lphat
    end
    return ll
end
# @time ntvxphpsmcll(ot,ov,[1,0.3,0.6])

## the same function, but without assuming the kernel is exponential
function ntvhpsmcll(ot,ov,pa;np=(b=1,e=3),basint=intCon,basInt=IntCon,
                        kern=kernGam,Kern=KernGam,J=100,conf=0.95) 
    if np.b+np.e != length(pa) 
        @warn("length of pa = $(length(pa))!= $(np.b) + $(np.e)") 
    end
    N=convert(Vector{Int64},ov) # values of the counting process at the obs times
    dN=diff(N) # num. of events each intervals
    n=length(dN) # num. of intervals (=num. of obs times - 1)
    NN=N[end] ## total number of events

    #idx of 1st interval with any events (idx<=2)
    idx = 1; while dN[idx]==0 idx += 1 end 
    cumBk = basInt(ot[idx],pa=pa[1:np.b])
    ll = -cumBk
    taus = Matrix{Float64}(undef,NN,J) # store the paths
    ptcls = fill(1,NN)*[1:J;]' # to store the ptcl idx
    lwts = log.(fill(1.0/J,J))

    cumEx = fill(0.0,J)
    smp = 1:J
    for i in idx:n  # i=index of the interval = ot index - 1
        lPr = fill(0.0,J)
        if i>idx 
            smp = sample(1:J, weights(exp.(lwts)),J)
            ptcls[1:N[i],:] = ptcls[1:N[i],smp] 
        end
        if dN[i] == 0
            lwts = log.(fill(1.0/J,J))
            lPr .=  cumBk 
            cumBk = basInt(ot[i+1],pa=pa[1:np.b])
            lPr .+= -cumBk 
            lPr .+= cumEx[smp]
            Threads.@threads for j in 1:J 
                cumEx[j] = sum(Kern.(ot[i+1] .- 
                    taus[CartesianIndex.(1:N[i+1],ptcls[1:N[i+1],j])],
                                     pa=pa[np.b+1:end]))
                lPr[j] += -cumEx[j]
            end
            lphat = logsumexp(lPr .+ lwts); 
            ll += lphat
            lwts .+= lPr
        else
            lmd = quantile(Gamma(dN[i],1.0),conf)/(ot[i+1]-ot[i])
            lwts .= cumBk .+ cumEx[smp]
            for j in 1:J
                # generate particles from the Poisson reference
                taus[N[i]+1:N[i+1],j] = ot[i] .+
                    cumsum(rand(Exponential(1/lmd),dN[i]))  
            end
            Threads.@threads for j in 1:J
                # calculate the log weights (density ratios)
                cumBk1 = basInt(taus[N[i+1],j],pa=pa[1:np.b])
                lwts[j] += -cumBk1-dN[i]*log(lmd)+ 
                    lmd*(taus[N[i+1],j]-ot[i])
                cumBk = basInt(ot[i+1],pa=pa[1:np.b])
                cumEx[j] = sum(Kern.(ot[i+1] .- 
                    taus[CartesianIndex.(1:N[i+1],
                                         ptcls[1:N[i+1],j])],
                                     pa=pa[np.b+1:end]))
                lPr[j] = -cumBk+cumBk1-cumEx[j]
                cumEx1 = sum(Kern.(taus[N[i+1],j] .- 
                    taus[CartesianIndex.(1:N[i+1],
                                         ptcls[1:N[i+1],j])],
                                   pa=pa[np.b+1:end]))
                lPr[j] += cumEx1
                lwts[j] += -cumEx1  #cumEx from previous time 
                for k in 1:dN[i]
                    lwts[j] += log(basint(taus[N[i]+k,j],pa=pa[1:np.b])+
                        sum(kern.(taus[N[i]+k,j] .-
                        taus[CartesianIndex.(1:N[i]+k-1,
                                             ptcls[1:N[i]+k-1,j])],
                                  pa=pa[np.b+1:end])) )
                end
                if taus[N[i+1],j]>ot[i+1] lPr[j]= -Inf end  
            end
             lphat = logsumexp(lPr .+ lwts) - log(J); 
            ll += lphat
            lwts += lPr
        end
    end
    return ll
end

# ntvhpsmcll(ot,ov,[1,0.3,0.6],np=(b=1,e=2),J=10,kern=kernExp,Kern=KernExp)
# Random.seed!(2023);@time ntvhpsmcll(ot,ov,[1,0.3,0.6],np=(b=1,e=2),J=1000000,kern=kernExp,Kern=KernExp)
# Random.seed!(2023);@time ntvxphpsmcll(ot,ov,[1,0.3,0.6],J=1000000)


function MHMCMCdiscHawkes(da,init;np=(b=1,e=3),basint=intCon,basInt=IntCon,
                         kern=kernGam,Kern=KernGam,tofree=tofreeGam,
                         fromfree=fromfreeGam,delta=0.1,N=10000,J=128,
                         verb=false)
    if np.b+np.e != length(init) 
        @warn("length of init = $(length(init))!= (np.b=$(np.b)) + (np.e=$(np.e))") 
    end
    ptcls = fill(0.0,np.b+np.e,N)
    llval = fill(0.0,N)
    ptcls[:,1] = init
    llval[1] = ntvhpsmcll(da.otms,da.ovls,init,np=np,basint=basint,
                          basInt=basInt,kern=kern,Kern=Kern,J=J)
    i = 2
    while i<=N 
        # y = [log(ptcls[1,i-1]);logit(ptcls[2,i-1]);
        #      log.(ptcls[3:4,i-1])] .+ randn(4)*delta
        # y = [exp(y[1]);logistic(y[2]);exp.(y[3:4])]
        y=fromfree(tofree(ptcls[:,i-1]) .+ randn(np.b+np.e)*delta)
        logacpt = ntvhpsmcll(da.otms,da.ovls,y,np=np,basint=basint,
                                 basInt=basInt,kern=kern,Kern=Kern,J=J)-
                                     llval[i-1];
        if log(rand())<logacpt 
            ptcls[:,i] = y
            llval[i] = llval[i-1]+logacpt
        else
            ptcls[:,i] = ptcls[:,i-1]
            llval[i] = llval[i-1];
        end
        if verb 
            println("i=$i, par=$(ptcls[:,i]), llval=$(llval[i])") 
        end
        i+=1
    end
    (ptcls=ptcls,llval=llval)
end
# initv=fromfreeExp(randn(3)); otms=[0:10;]/2; ovls=[0;cumsum(rand(Poisson(1),10))]
# @time res=MHMCMCdiscHawkes((otms=otms,ovls=ovls),initv,np=(b=1,e=2),
#                             kern=kernExp,Kern=KernExp,
#                             tofree=tofreeExp,
#                             fromfree=fromfreeExp,
#                             verb=true,N=10,
#                             J=128,delta=0.05)

function MHMCMCdiscExpHawkes(da,init;npb=1,basint=intCon,basInt=IntCon,
                             fromfree=fromfreeExp,tofree=tofreeExp,
                             delta=0.1,N=10000,J=128,verb=false)
    if npb+2 != length(init) 
        @warn("length of init = $(length(init))!= (npb=$npb) + 2") 
    end
    ptcls = fill(0.0,npb+2,N)
    llval = fill(0.0,N)
    ptcls[:,1] = init
    llval[1] = ntvxphpsmcll(da.otms,da.ovls,init,npb=npb,basint=basint,
                          basInt=basInt,J=J)
    i = 2
    while i<=N 
        y=fromfree(tofree(ptcls[:,i-1]) .+ randn(npb+2)*delta)
        logacpt = ntvxphpsmcll(da.otms,da.ovls,y,npb=npb,basint=basint,
                               basInt=basInt,J=J)-
                                   llval[i-1];
        if log(rand())<logacpt 
            ptcls[:,i] = y
            llval[i] = llval[i-1]+logacpt
        else
            ptcls[:,i] = ptcls[:,i-1]
            llval[i] = llval[i-1];
        end
        if verb 
            println("i=$i, par=$(ptcls[:,i]), llval=$(llval[i])") 
        end
        i+=1
    end
    (ptcls=ptcls,llval=llval)
end
# initv=fromfreeExp(randn(3)); otms=[0:10;]/2; ovls=[0;cumsum(rand(Poisson(1),10))]
# @time res=MHMCMCdiscExpHawkes((otms=otms,ovls=ovls),initv,
#                             verb=true,N=10,
#                             J=128,delta=0.05)


function MHMCMCcontExpHawkes(init,tms,cens;npb=1,basint=intCon,basInt=IntCon,
                             tofree=tofreeExp, fromfree=fromfreeExp,
                             delta=0.1, N=5500, verb=false)
    if npb+2 != length(init) 
        @warn("length of init = $(length(init))!= (npb=$(npb)) + 2") 
    end
    ptcls = fill(0.0,npb+2,N)
    llval = fill(0.0,N)
    ptcls[:,1] = init
    llval[1] = exphawkesllik(init,tms,cens,npb=npb,basint=basint,
                             basInt=basInt)
    i = 2
    while i<=N 
        y=fromfree(tofree(ptcls[:,i-1]) .+ randn(npb+2)*delta)
        logacpt = exphawkesllik(y,tms,cens,npb=npb,basint=basint,
                                basInt=basInt) -  llval[i-1];
        if log(rand())<logacpt 
            ptcls[:,i] = y
            llval[i] = llval[i-1]+logacpt
        else
            ptcls[:,i] = ptcls[:,i-1]
            llval[i] = llval[i-1];
        end
        if verb 
            println("i=$i, par=$(ptcls[:,i]), llval=$(llval[i])") 
        end
        i+=1
    end
    (ptcls=ptcls,llval=llval)
end

end
