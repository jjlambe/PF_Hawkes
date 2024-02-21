using Random, Distributions, HawkesProcesses, Plots, DataFrames, CSV
# include("HawkesUtils.jl")
include("..\\PaFiHawkes.jl")
import .PaFiHawkes
Random.seed!(2024)

data_name = raw"C:\\Users\\jlamb\\OneDrive\\PhD\\Code Projects\\PF_Hawkes\\Uniform_vs_Original\\Hawkes_sim_data.csv"

data = CSV.read(data_name, DataFrame, header = false)

ots, ovs = PaFiHawkes.ntvppdatapreproc(data[!, 1], data[!, 2]) # Cleaning the data. 
# The cleaning process will usually make no changes as the chance of a window with no observations is near 0.

param = [1, 0.5, 2] # True parameter
pa_init = param .+ [0.5, -0.1, 0] # Initial guess
N = 5000 # Number of iterations

ll_est = PaFiHawkes.ntvxphpsmcll(ots, ovs, pa_init)

# @time begin
#     res = PaFiHawkes.MHMCMCdiscExpHawkes((otms=ots, ovls=ovs), pa_init,
#                             verb = true, N = N,
#                             J = 128, delta = 0.1)

# end




# plot_unif = plot(1:N, [res[1][1:3:3*N-2] res[1][2:3:3*N-1] res[1][3:3:3*N]], label = ["Background" "Eta" "Lambda"], xlabel = "Time = 397.54 seconds",
#                 title = "Ordered Uniform Proposal")
# savefig(plot_unif, raw"C:\\Users\\jlamb\\OneDrive\\PhD\\Code Projects\\PF_Hawkes\\Uniform_vs_Original\\Figures\\plot_unif.pdf")


####### Manual Likelihood Estimate Comparison ##########
# n = 100
# ll_vec = Vector{Float64}(undef, n)

# beta_vec = 1:0.02:3-0.02

# for i in 1:n
#     print(i, "\n")
#     ll_vec[i] = PaFiHawkes.ntvxphpsmcll(ots, ovs, [1, 0.5, beta_vec[i]], J = 500)
# end

# plot_unif_ll = plot(beta_vec, ll_vec, label = "Log Likelihood Est.", xlabel = "Beta Value", title = "Uniform Proposal, J = 500 Particles")

# savefig(plot_unif_ll, raw"C:\\Users\\jlamb\\OneDrive\\PhD\\Code Projects\\PF_Hawkes\\Uniform_vs_Original\\Figures\\plot_unif_ll.pdf")