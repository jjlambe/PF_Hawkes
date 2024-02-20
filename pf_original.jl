using Random, Distributions, HawkesProcesses, Plots
# include("HawkesUtils.jl")
include("DisPaHawkes_original.jl")
import .DisPaHawkes_original

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


ots, ovs = DisPaHawkes_original.ntvppdatapreproc(obstms, obsvls) # Cleaning the data



# The cleaning process makes no changes for seed 1234 as there are observations in every window.

# Likelihood estimate for the true parameters
param = [bg, eta, lambda]
true_loglik = DisPaHawkes_original.ntvxphpsmcll(ots, ovs, param, J = 100)
# print(true_loglik, "\n")

# Likelihood for an incorrect guess
pa_init = param .+ [0.5, -0.1, 0.5]
N = 5000

res = DisPaHawkes_original.MHMCMCdiscExpHawkes((otms=ots, ovls=ovs), pa_init,
                            verb = true, N = N,
                            J = 128, delta = 0.05)


# bg_est = [res[1][i][1] for i in 1:N]
# eta_est = [res[1][i][2] for i in 1:N]
# lambda_est = [res[1][i][3] for i in 1:N]

plot_orig = plot(1:N, [res[1][1:3:3*N-2] res[1][2:3:3*N-1] res[1][3:3:3*N]], label = ["Background" "Eta" "Lambda"])

savefig(plot_orig, raw"C:\\Users\\jlamb\\OneDrive\\PhD\\Code Projects\\PF_Hawkes\\Figures\\plot_orig.pdf")