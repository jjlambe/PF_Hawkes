using Random, Distributions, BenchmarkTools

function OrdUnifRecSamp(t_int, n)
    # Sampling n ordered uniform RVs in time window (t_int[1], t_int[2])
    # Uses an initial sample of uniforms and recursively constructs the ordered sample.
    U = Vector{Float64}(undef, n)
    V = rand(Uniform(0, 1), n)
    U[n] = V[n]^(1/n)
    for k in n-1:-1:1
        U[k] = U[k+1]*(V[k]^(1/k))
    end
    return (t_int[2] - t_int[1])*U .+ t_int[1]
end

# @btime OrdUnifRecSamp([10, 20], 1000)

function OrdUnifExpSamp(t_int, n)
    # Sampling n ordered uniform RVs in time window (t_int[1], t_int[2])
    # Uses an initial sample of exponentials and re-weights to generate the sample.
    Y = rand(Exponential(1), n + 1)
    # U = t_int[1] .+ (cumsum(Y)*(t_int[2] - t_int[1]))/sum(Y)
    return t_int[1] .+ (cumsum(Y)[1:end-1]*(t_int[2] - t_int[1]))/sum(Y)
end


function OrdUnifExpMultSamp(t_int, n, J)
    Y = rand(Exponential(1), n + 1, J)
    Y = cumsum(Y, dims = 1)
    Y = Y[1:end-1, :] ./ Y[end, :]'
    return (t_int[2] - t_int[1])*Y .+ t_int[1]
end

function SampType(n)
    # n is the vector of numbers
    indices = Int64.(ones(n[1]))
    for i in eachindex(n)[2:end]
        append!(indices, Int64.(i*ones(n[i])))
    end
    return(shuffle(indices))
end

SampType([10, 5])

# function OrdUnifProp(dt, dN, N, T)
#     K = length(dt)
#     U_vec = OrdUnifExpSamp([0, T], N)
#     ptcls = Vector{Vector{Float64}}(undef, K)
#     ptcls[1] = (dt[1]/U_vec[dN[1] + 1])*U_vec[1:dN[1]]
#     cumval = cumsum(dN)
#     cumtim = cumsum(dt)
#     for k in 2:K-1
#         ptcls[k] = (dt[k] / (U_vec[cumval[k] + 1] - U_vec[cumval[k-1]])) * U_vec[cumval[k-1] + 1: cumval[k]] .+ cumtim[k-1] .- U_vec[cumval[k-1]]
#     end
#     ptcls[K] = (dt[K] / (T - U_vec[cumval[K-1]])) * U_vec[cumval[K-1] + 1: end]
#     return(ptcls)
# end

