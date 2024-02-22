using Random, Distributions
include("unif_ord_stats.jl")
Random.seed!(126789)
function SmoothCumDisUnif(x, w, J)
    # Generates a resample of x according to a smoothed CDF
    # print(J, "\n")
    sort_idx = sortperm(x)
    x_ord = x[sort_idx] # Sorting x
    w_ord = w[sort_idx] # Sorting w according to x
    U_vec =  OrdUnifExpSamp([0, 1], J) # Ordered sample of J uniform random variables

    pi_vec = Vector{Float64}(undef, J) # Initialising pi
    pi0 = w[1]/2; pi_vec[end] = w[J]/2 # pi0 kept separate to keep indexing consistent

    s = 1; r = pi0
    x_prop = Vector{Float64}(undef, J) # Initialising the vector of proposal values, i.e. the resampled particles
    # print(U_vec)
    for j in 1:J
        if U_vec[j] < pi0
            x_prop[j] = x_ord[1]
        elseif U_vec[j] > 1 - pi_vec[end]
            x_prop[j] = x_ord[J]
        else
            while U_vec[j] > r && s < J
                # print([s, U_vec[j], r], "\n")
                pi_vec[s] = (w_ord[s] + w_ord[s+1])/2 # This line is the issue
                r += pi_vec[s]
                s += 1
            end
            x_prop[j] = x_ord[s-1] + ((U_vec[j] - (r - pi_vec[s-1])) / pi_vec[s-1])*(x_ord[s] - x_ord[s-1])
        end
    end
    return x_prop
end


# x = [1, 3, 2, 5, 4, 7, 6, 9, 8]
# w = x/sum(x)

# SmoothCumDisUnif(x, w, length(x))