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

# @btime OrdUnifExpSamp([10, 20], 1000)

# On 1000 samples, drawing from the exponential is just over 3 times faster.

# N = 10000
# joint_vec = Vector{Float64}(undef, N)
# @btime rand(JointOrderStatistics(Uniform(0, 100), N), 1)

# @btime OrdUnifExpSamp([0, 100], N) # The uniform with exp draws is 3 times faster than rand() from JointOrderStatistics()

# The OrdUnifExpSamp method is statistically identical to the inbuilt method.
# res1 = rand(JointOrderStatistics(Uniform(0, 1), 10, [6]), N)
# print(mean(res1), "\n")

# res2 = [OrdUnifExpSamp([0, 1], 10)[6] for i in 1:N]
# print(mean(res2))
