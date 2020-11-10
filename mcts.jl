#Goldfrank 2020 - Liedtka wrote a lot of the MCTS code

include("observations.jl")
include("mcts_utils.jl")

# initialize Julia plotting environment
pyplot()
Plots.PyPlotBackend()

timenow = now()

#seed random number generator (rng)
rng = MersenneTwister(2)

arguments = parse_commandline()

# set hyperparameters for trials
N = parse(Int64, arguments["N"])
DEPTH = parse(Int64, arguments["depth"])
λ = parse(Float64, arguments["lambda"])
num_runs = parse(Int64, arguments["trials"])
iterations = parse(Int64, arguments["iterations"])
COLLISION_REWARD = parse(Float64, arguments["collision"])
LOSS_REWARD = parse(Float64, arguments["loss"])

plotting = false
testing = false
epochsize = 500

#output header file
header_string = string("MCTS Run: ", timenow)
header_string = string(header_string, "\n", "Depth: ", DEPTH)
header_string = string(header_string, "\n", "Lambda: ", λ)
header_string = string(header_string, "\n", "Iterations: ", iterations)
header_string = string(header_string, "\n", "Collision Reward: ", COLLISION_REWARD)
header_string = string(header_string, "\n", "Loss Reward: ", LOSS_REWARD)

#write output header
header_filename = string(timenow, "_header.txt")
open(header_filename, "w") do f
    write(f, header_string)
end



#cumulative collisions, losses, and number of trials
#total reward, and best average tracking
cum_coll = 0
cum_loss = 0
cum_trials = 0
total_reward = 0
best_average = 1


run_data = []

# trials
num_particles = N
mcts_loss = 0
mcts_coll = 0
for i = 1:num_runs
    global mcts_loss, mcts_coll, num_particles, DEPTH
    result = mcts_trial(DEPTH, 20, false, num_particles)
    mcts_coll += result[3]
    mcts_loss += result[4]
    push!(run_data,result[3:4])
    print(".")
    if i % 20 == 0
        println()
        println("==============================")
        println("Trials: ", i)
        println("NUM PARTICLES: ", num_particles)
        println("MCTS Depth ", DEPTH, " Results")
        println("Collision Rate: ", mcts_coll/i)
        println("Loss Rate: ", mcts_loss/i)
        println("==============================")
        namefile = string(timenow, "_data.csv")
        CSV.write(namefile, DataFrame(run_data))
    end
end

# print results





#
