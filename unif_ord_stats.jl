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

# @btime vals = OrdUnifExpMultSamp([10, 20], 30, 128)

# x = [OrdUnifExpMultSamp([10, 20], 2, 2) for i in 1:2]
# print(x[1][:, 1])

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


# print(OrdUnifProp([1, 2, 3], [1, 4, 3], 8, 6))

# @btime OrdUnifExpSamp([10, 20], 1000)

# On 1000 samples, drawing from the exponential is just over 3 times faster.

# N = 100000
# joint_vec = Vector{Float64}(undef, N)
# @btime rand(JointOrderStatistics(Uniform(0, 100), N), 1)

# @btime OrdUnifExpSamp([0, 100], N) # The uniform with exp draws is 3 times faster than rand() from JointOrderStatistics()

# The OrdUnifExpSamp method is statistically identical to the inbuilt method.
# res1 = rand(JointOrderStatistics(Uniform(0, 1), 10, [6]), N)
# print(mean(res1), "\n")

# res2 = [OrdUnifExpSamp([0, 1], 10)[6] for i in 1:N]
# print(mean(res2), "\n")


# samp = [OrdUnifExpSamp([0, 1], 10) for i in 1:N]
# diff1 = [x[2] - x[1] for x in samp]
# diff2 = [x[6] - x[5] for x in samp]
# print(mean(diff1), "\n", mean(diff2))

# res3 = Vector{Float64}(undef, N)
# for i in 1:N
#     res3[i] = samp[i][6]/samp[i][11]
# end
# print(mean(res3))
# for i in 1:N
#     res2[i] = samp[i]
# end
