using Random, Distributions, HawkesProcesses, Plots, DataFrames, CSV, BenchmarkTools
# include("HawkesUtils.jl")
include("..\\PFSmoothHawkes.jl")
import .PFSmoothHawkes
Random.seed!(2024)

data_name = raw"C:\\Users\\jlamb\\OneDrive\\PhD\\Code Projects\\PF_Hawkes\\Uniform_vs_Original\\Hawkes_sim_data.csv"

data = CSV.read(data_name, DataFrame, header = false)

ots, ovs = PFSmoothHawkes.ntvppdatapreproc(data[!, 1], data[!, 2]) # Cleaning the data. 
# The cleaning process will usually make no changes as the chance of a window with no observations is near 0.

param = [1, 0.5, 2] # True parameter
pa_init = param .+ [0.5, -0.1, 0.5] # Initial guess
N = 500 # Number of iterations

# @btime PFSmoothHawkes.ntvxphpsmcll(ots, ovs, param) # Takes 51 ms

@time begin
    res = PFSmoothHawkes.MHMCMCdiscExpHawkes((otms=ots, ovls=ovs), pa_init,
                            verb = true, N = N,
                            J = 128, delta = 0.1)
end

plot_smooth = plot(1:N, [res[1][1:3:3*N-2] res[1][2:3:3*N-1] res[1][3:3:3*N]], label = ["Background" "Eta" "Lambda"], title = "Ordered Uniform Proposal, Smoothed ECDF")
savefig(plot_smooth, raw"C:\\Users\\jlamb\\OneDrive\\PhD\\Code Projects\\PF_Hawkes\\Uniform_vs_Original\\Figures\\plot_smooth.pdf")
# 819.55 seconds

###### Manual Likelihood Estimate Comparison ##########
# n = 100
# ll_vec = Vector{Float64}(undef, n)

# beta_vec = 1:0.02:3-0.02

# for i in 1:n
#     print(i, "\n")
#     ll_vec[i] = PFSmoothHawkes.ntvxphpsmcll(ots, ovs, [1, 0.5, beta_vec[i]], J = 500)
# end

# unif_smooth = plot(beta_vec, ll_vec, label = "Log Likelihood Est.", xlabel = "Beta Value", title = "Uniform Proposal, J = 500 Particles")

# savefig(unif_smooth, raw"C:\\Users\\jlamb\\OneDrive\\PhD\\Code Projects\\PF_Hawkes\\Uniform_vs_Original\\Figures\\unif_smooth.pdf")