#SARSA-lambda implementation of tracking problem
#Goldfrank 2020

include("observations.jl")
include("polar_interpolation.jl")
include("q_lambda_utils.jl")

# initialize Julia plotting environment
pyplot()
Plots.PyPlotBackend()

#time string for saving files
timenow = string(now())


#seed random number generator (rng)
rng = MersenneTwister(2)

arguments = parse_commandline()

# set hyperparameters for trials
N = parse(Int64, arguments["N"])
model = ParticleFilterModel{Vector{Float64}}(f2, g)
pfilter = SIRParticleFilter(model, N);
α = parse(Float64, arguments["alpha"])
γ = parse(Float64, arguments["gamma"])
ϵ = parse(Float64, arguments["epsilon"])
variance_enabled = parse(Bool, arguments["variance"])
COLLISION_REWARD = parse(Float64, arguments["collision"])
LOSS_REWARD = parse(Float64, arguments["loss"])
EPSILON = ϵ
λ = parse(Float64, arguments["lambda"])
num_runs = parse(Int64, arguments["trials"])
plotting = false
testing = false
epochsize = 500
burn_in_length = 15

# input file for weights (theta)
infile = arguments["theta"]

# initialize weights to all 0, naive rewards, or previous results, based on command line arguments
if infile == "zeros"
    θ = zeros(length(grid),6);
elseif infile == "rewards"
    θ = [r(Tuple(ind2x(grid, j)),3) for j in 1:length(grid), i in 1:6];
else
    θ = convert(Matrix, CSV.read(infile))
end

#output header file
header_string = string("Q-Lambda Run: ", timenow)
header_string = string(header_string, "\n", "Gamma: ", γ)
header_string = string(header_string, "\n", "Alpha: ", α)
header_string = string(header_string, "\n", "N: ", N)
header_string = string(header_string, "\n", "Collision Reward: ", COLLISION_REWARD)
header_string = string(header_string, "\n", "Loss Reward: ", LOSS_REWARD)
header_string = string(header_string, "\n", "Variance: ", variance_enabled)

#write output header
header_filename = string(timenow, "_header.txt")
open(header_filename, "w") do f
    write(f, header_string)
end

run_data = []

#cumulative collisions, losses, and number of trials
#total reward, and best average tracking
cum_coll = 0
cum_loss = 0
cum_trials = 0
total_reward = 0
best_average = 1
plotting_this_round = false

# q-lambda algorithm trials
for i in 1:num_runs
    global ϵ, EPSILON
    global plotting, plotting_this_round, variance_enabled
    global testing
    global cum_coll, cum_loss, cum_trials, run_data
    global total_reward,  best_average
    global θ
    if variance_enabled
        result = q_trial()
    else
        result = q_trial_no_variance()
    end


    cum_coll += result[1]
    cum_loss += result[2]
    reward = result[4]
    total_reward += reward
    cum_trials += 1

    θ = result[3]
    push!(run_data,result[1:2])

    if ((cum_coll+cum_loss)/cum_trials < best_average) && cum_trials > 40
        best_average = (cum_coll+cum_loss)/cum_trials
    end

    if cum_trials % 20 == 0 && false
        println("Current Score: ", (cum_coll+cum_loss)/cum_trials)
        println("θ Max: ", maximum(θ), " -- θ Min: ", minimum(θ))
    end

    if plotting_this_round
        println("PLOTTING")
        plots = result[5]
        if (result[1] == 1) || (result[2] == 1) || true
            name = string("plots/out_", i,".gif")
            anim = @animate for i in 1:length(plots)
                plot!(plots[i])
                fps=10
            end
            gif(anim, name, fps=10)
        end
    end

    if i % 5 == 0
        thetaframe = DataFrame(θ)
        CSV.write(outfile, thetaframe)
        namefile = string(timenow, "_data.csv")
        CSV.write(namefile, DataFrame(run_data))
    end

    if i % 20 == 0
        println("\n======================= MODEL STATUS ==========================")
        println("Round: ", i, " Best Average: ", round(best_average, sigdigits=4))
        println("Current Collison Rate: ", round((cum_coll)/cum_trials, sigdigits=4), " -- Col. Reward: ", COLLISION_REWARD)
        println("Current Loss Rate: ", round((cum_loss)/cum_trials, sigdigits=4), " -- Loss Reward: ", LOSS_REWARD)
        println("ϵ: ", ϵ, " α: ", α, " γ: ", γ, " λ: ", λ, " N: ", N)
        println("θ Max: ", maximum(θ), " -- θ Min: ", minimum(θ))
        println("===============================================================\n")

    end

    if cum_trials % 100 == 15
        plotting_this_round = plotting
    else
        plotting = false
    end

    #used for variable epsilon-greedy strategy only
    #ϵ = max(min(.8, (cum_coll+cum_loss)/i),.005)
end

thetaframe = DataFrame(θ)
CSV.write(outfile, thetaframe)

#
