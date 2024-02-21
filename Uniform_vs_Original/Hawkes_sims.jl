using Random, Distributions, CSV, Tables, HawkesProcesses

# Random.seed!(2024)
bg = 1
eta = 0.5
lambda = 4

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

CSV.write(raw"C:\\Users\\jlamb\\OneDrive\\PhD\\Code Projects\\PF_Hawkes\\Uniform_vs_Original\\Hawkes_sim_data.csv", Tables.table([obstms obsvls]), writeheader = false)
