using Random, Distributions, HawkesProcesses, Plots, DataFrames, CSV
include("HawkesUtils.jl")
import .HawkesUtils
include("PaFiHawkes.jl")
import .PaFiHawkes
Random.seed!(2024)

data_name = raw"C:\\Users\\jlamb\\OneDrive\\PhD\\Code Projects\\PF_Hawkes\\Uniform_vs_Original\\Hawkes_sim_data.csv"

data = CSV.read(data_name, DataFrame, header = false)

ots, ovs = PaFiHawkes.ntvppdatapreproc(data[!, 1], data[!, 2]) # Cleaning the data. 
# The cleaning process will usually make no changes as the chance of a window with no observations is near 0.

param = [1, 0.5, 2] # True parameter

# cumBk0 = HawkesUtils.IntCon(ots[1], pa = param)

# print(ots[1], "\n")
# print(ots[1]*param[1], "\n")
# print(cumBk0, "\n")

# dN = diff(ovs)

# global idx_test = 1; while dN[idx_test]==0 global idx_test += 1 end

# print(idx_test, "\n")

# cumBk1 = HawkesUtils.IntCon(ots[idx_test + 1],pa=param)
# ll = -cumBk1 + cumBk0
# print(ll)

# vec = Vector{Int128}(undef, 10)
# for k in 1:10
#     vec[k] = k==1 ? 100 : k
# end
# print(vec)
