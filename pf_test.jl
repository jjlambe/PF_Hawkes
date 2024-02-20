using Random, Distributions, HawkesProcesses, Plots, DisPaHawkes, BenchmarkTools

Random.seed!(1234)
bg = 1
eta = 0.5
lambda = 2

kernel(x) = pdf.(Exponential(1/lambda), x)
T = 1000
simevents = HawkesProcesses.simulate(bg, eta, kernel, T)

# print(simevents)

n_windows = 100 
step = T/n_windows
windows = 0:step:T

function aggregate_data(data, windows)
    # aggregate_data takes event time data from any point process and
    # returns the total number of events in each specified time window.
    n_windows  = length(windows) - 1
    count_data = Vector{Int128}(undef, n_windows)
    for i in 1:n_windows
        count_data[i] = sum([if x >= windows[i] && x < windows[i+1] 1 else 0 end for x in data])
    end
    return count_data
end
obstms = Int.(collect(windows)[2:end])
obsvls = cumsum(aggregate_data(simevents, windows))

# print(DisPaHawkes.unif_ord_samp([0, 10], 10))


ots, ovs = ntvppdatapreproc(obstms, obsvls) # Cleaning the data



# The cleaning process makes no changes for seed 1234 as there are observations in every window.

# Likelihood estimate for the true parameters
param = [bg, eta, lambda]
true_loglik = @btime ntvxphpsmcll(ots, ovs, param, J = 100)
# print(true_loglik, "\n")

# Likelihood for an incorrect guess
# param_wrong = param .+ [-0.5, 0.1, 0.5]
# print(ntvxphpsmcll(ots, ovs, param_wrong, J = 100))
# The log-likelihood has dropped with an input of incorrect parameters

# res = MHMCMCdiscExpHawkes((otms=ots, ovls=ovs), param,
#                             verb = true, N = 1000,
#                             J = 128, delta = 0.05)


# bg_vals = 4.9:0.01:5.1
# loglik_vals = Vector{Float64}(undef, length(bg_vals))
# for i in eachindex(bg_vals)
#     loglik_vals[i] = ntvxphpsmcll(ots, ovs, [bg_vals[i], eta, lambda], J = 100)
# end

# plot(bg_vals, loglik_vals)
# ntvxphpsmcll(ots, ovs, param, J = 100)