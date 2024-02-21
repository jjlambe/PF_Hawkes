using Random, Distributions, HawkesProcesses, Plots, CSV
# include("HawkesUtils.jl")
include("..\\DisPaHawkes_original.jl")
import .DisPaHawkes_original

Random.seed!(2024)

data_name = raw"C:\\Users\\jlamb\\OneDrive\\PhD\\Code Projects\\PF_Hawkes\\Uniform_vs_Original\\Hawkes_sim_data.csv"

data = CSV.read(data_name, DataFrame, header = false)


ots, ovs = DisPaHawkes_original.ntvppdatapreproc(data[!, 1], data[!, 2]) # Cleaning the data. 
# The cleaning process will usually make no changes as the chance of a window with no observations is near 0.


param = [1, 0.5, 2] # True parameter
pa_init = param .+ [0.5, -0.1, 0.5] # Initial guess
N = 5000 # Number of iterations

# ll_est_orig = DisPaHawkes_original.ntvxphpsmcll(ots, ovs, pa_init)


@time begin
    res = DisPaHawkes_original.MHMCMCdiscExpHawkes((otms=ots, ovls=ovs), pa_init,
                            verb = true, N = N,
                            J = 128, delta = 0.1)

end

plot_orig = plot(1:N, [res[1][1:3:3*N-2] res[1][2:3:3*N-1] res[1][3:3:3*N]], label = ["Background" "Eta" "Lambda"], title = "Poisson Proposal", xlabel = "Time = 459.7 seconds")
savefig(plot_orig, raw"C:\\Users\\jlamb\\OneDrive\\PhD\\Code Projects\\PF_Hawkes\\Uniform_vs_Original\\Figures\\plot_orig.pdf")

# n = 100
# ll_vec = Vector{Float64}(undef, n)

# beta_vec = 1:0.02:3-0.02

# for i in 1:n
#     print(i, "\n")
#     ll_vec[i] = DisPaHawkes_original.ntvxphpsmcll(ots, ovs, [1, 0.5, beta_vec[i]], J = 500)
# end

# plot_poi_ll = plot(beta_vec, ll_vec, label = "Log Likelihood Est.", xlabel = "Beta Value", title = "Poisson Proposal, J = 500 Particles")

# savefig(plot_poi_ll, raw"C:\\Users\\jlamb\\OneDrive\\PhD\\Code Projects\\PF_Hawkes\\Uniform_vs_Original\\Figures\\plot_poi_ll.pdf")