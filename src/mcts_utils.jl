# Imports

using StaticArrays
using LinearAlgebra
using Random
using StatsBase
using SparseArrays
using Distributions
using GridInterpolations
using ParticleFilters
using DataStructures
using DataFrames
using CSV
using ArgParse
using Plots
using Dates


# parse command line arguments
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--lambda"
            help = ""
            default = "0.8"
        "--alpha"
            help = ""
            default = "0.7"
        "--gamma"
            help = ""
            default = "0.5"
        "--collision"
            help = ""
            default = "-2"
        "--loss"
            help = ""
            default = "-2"
        "--depth"
            help = ""
            default = "10"
        "--N"
            help = ""
            default = "500"
        "--trials"
            help = ""
            default = "100"
        "--iterations"
            help = ""
            default = "2000"
        "--plot_header"
            help = ""
            default = "out"
    end

    return parse_args(s)
end

######################################
### generative model
######################################

# generate next course given current course
function next_crs(crs,rng)
    if rand(rng) < .9
        return crs
    end
    crs = (crs + rand(rng,[-1,1])*30) % 360
    if crs < 0 crs += 360 end
    return crs
end

# alternate next course-- uses different probability to represent model error
function next_crs_gen(crs,rng)
    if rand(rng) < .75
        return crs
    end
    crs = (crs + rand(rng,[-1,1])*30) % 360
    if crs < 0 crs += 360 end
    return crs
end

# returns new state given last state and action (control)
function f(state, control, rng)
    TGT_SPD = 1
    r, θ, crs, spd = state
    spd = control[2]

    θ = θ % 360
    θ -= control[1]
    θ = θ % 360
    if θ < 0 θ += 360 end

    crs = crs % 360
    crs -= control[1]
    if crs < 0 crs += 360 end
    crs = crs % 360

    x = r*cos(π/180*θ)
    y = r*sin(π/180*θ)

    pos = [x + TGT_SPD*cos(π/180*crs) - spd, y +
        TGT_SPD*sin(π/180*crs)]
    crs = next_crs(crs,rng)

    r = sqrt(pos[1]^2 + pos[2]^2)
    θ = atan(pos[1],pos[2])*180/π
    if θ < 0 θ += 360 end
    return (r, θ, crs, spd)::NTuple{4, Real}
end

# deprecated  - idential to f(state, control, rng)
function f_gen(state, control, rng)
    TGT_SPD = 1
    r, θ, crs, spd = state
    spd = control[2]

    θ = θ % 360
    θ -= control[1]
    θ = θ % 360
    if θ < 0 θ += 360 end

    crs = crs % 360
    crs -= control[1]
    if crs < 0 crs += 360 end
    crs = crs % 360

    x = r*cos(π/180*θ)
    y = r*sin(π/180*θ)

    pos = [x + TGT_SPD*cos(π/180*crs) - spd, y +
        TGT_SPD*sin(π/180*crs)]
    crs = next_crs_gen(crs,rng)

    r = sqrt(pos[1]^2 + pos[2]^2)
    θ = atan(pos[2],pos[1])*180/π
    if θ < 0 θ += 360 end
    return (r, θ, crs, spd)::NTuple{4, Real}
end

ACTION_PENALTY = -.05

# returns reward as a function of range, action, and action penalty
function r(s, u, action_penalty=ACTION_PENALTY)
    if (2 < u < 5)
        action_penalty = 0
    end
    range = s[1]
    if range >= 150 return (-2 + action_penalty) end  # reward to not lose track of contact
    if range <= 10 return (-2 + action_penalty) end  # collision avoidance
    return (0.1 + action_penalty)  # being in "sweet spot" maximizes reward
end

# returns reward as a function of range only
function r(s)
    range = s[1]
    if range >= 150 return (-2) end  # reward to not lose track of contact
    if range <= 10 return (-200) end  # collision avoidance
    return (0.1)  # being in "sweet spot" maximizes reward
end

#same as f, except returns a list
function f2(x, u, rng)
    temp = [i for i in f(x, u, rng)]
    return temp
end

#same as f_gen, except returns a list
function f_gen_2(x, u, rng)
    temp = [i for i in f_gen(x, u, rng)]
    return temp
end

# Action space and function to convert from action to index and vice versa
action_space = ((-30,1), (-30, 2), (0, 1), (0, 2), (30, 1), (30, 2))

#returns action given an index
action_to_index(a) = trunc(Int, 2*(a[1]/30+1) + a[2])

#function version of action_space
function actions()
    return ((-30,1), (-30, 2), (0, 1), (0, 2), (30, 1), (30, 2))
end

#returns index of action given an action
function index_to_action(a)
    if a % 2 == 0
        return ( trunc(Int,(((a - 2) / 2) - 1) * 30), 2)
    else
        return ( trunc(Int,(((a - 1) / 2) - 1) * 30), 1)
    end
end


##################################################################
# MCTS Algorithm
##################################################################

function arg_max_action(Q, N, history, c=nothing, exploration_bonus=false)

    # only need to compute if exploration possibility
    if exploration_bonus
        N_h = 0
        for action in action_to_index.(action_space)
            new_index = copy(history)
            append!(new_index, action)
            N_h += N[new_index]
        end
    end

    values = Float64[]
    for action in action_to_index.(action_space)

        new_index = copy(history)
        append!(new_index, action)

        # best action with exploration possibility
        if exploration_bonus

            # ensure an action chosen zero times is always chosen
            if N[new_index] == 0
                return action
            end

            # compute exploration bonus, checking for zeroes (I don't think this will ever occur anyway...)
            if log(N_h) < 0
                numerator = 0
            else
                numerator = sqrt(log(N_h))
            end
            denominator = N[new_index]
            exp_bonus = c * numerator / denominator
            append!(values, Q[new_index] + exp_bonus)

        # strictly best action
        else
            append!(values, Q[new_index])
        end
    end

    return argmax(values)

end


##################################################################
# Rollout
##################################################################
function rollout_random(state, depth)

    if depth == 0 return 0 end

    # random action
    random_action_index = rand(rng,action_to_index.(action_space))
    action = index_to_action(random_action_index)

    # generate next state and reward with random action; observation doesn't matter
    state_prime = f2(state, action, rng)
    reward = r(Tuple(state_prime),action_to_index(action))

    return reward + lambda * rollout_random(state_prime, depth-1)

end

##################################################################
# Simulate
##################################################################
function simulate(Q, N, state, history, depth, c)

    if depth == 0 return (Q, N, 0) end


    # expansion
    test_index = copy(history)
    append!(test_index, 1)

    if !haskey(Q, test_index)

        for action in action_to_index.(action_space)
            # initialize Q and N to zeros
            new_index = copy(history)
            append!(new_index, action)
            Q[new_index] = 0
            N[new_index] = 0
        end

        # rollout
        return (Q, N, rollout_random(state, depth))

    end


    # search
    # find optimal action to explore
    search_action_index = arg_max_action(Q, N, history, c, true)
    action = index_to_action(search_action_index)

    # take action; get new state, observation, and reward
    state_prime = f2(state, action, rng)
    observation = h(state_prime, rng)
    reward = r(Tuple(state_prime), action_to_index(action))

    # recursive call after taking action and getting observation
    new_history = copy(history)
    append!(new_history, search_action_index)
    append!(new_history, observation)
    (Q, N, successor_reward) = simulate(Q, N, state_prime, new_history, depth-1, c)
    q = reward + lambda * successor_reward

    # update counts and values
    update_index = copy(history)
    append!(update_index, search_action_index)
    N[update_index] += 1
    Q[update_index] += ((q - Q[update_index]) / N[update_index])

    return (Q, N, q)

end

##################################################################
# Select Action
##################################################################

function select_action(Q, N, belief, depth, c, iterations)

    # empty history at top recursive call
    history = Int64[]

    # loop
    # timed loop, how long should intervals be?
    #counter = 0
    #start_time = time_ns()
    #while (time_ns() - start_time) / 1.0e9 < 1 # 1 second timer to start

    # number of iterations
    counter = 0
    while counter < iterations

        # draw state randomly based on belief state (pick a random particle)
        state = rand(rng,belief)

        # simulate
        simulate(Q, N, float(state), history, depth, c)

        counter+=1
    end
    #println(counter, " iterations")

    best_action_index = arg_max_action(Q, N, history)
    action = index_to_action(best_action_index)
    return (Q, N, action)

end


##################################################################
# Legacy - Unused
##################################################################

function modify_history_tree(Q, N, last_action, last_obs)

    newQ = Dict{Array{Int64,1},Float64}()
    newN = Dict{Array{Int64,1},Float64}()

    for key in keys(Q)
        # if key matches last action and observation, becomes root in new tree
        if length(key) > 2 && key[1] == action_to_index(last_action) && key[2] == last_obs
            newQ[key[3:length(key)]] = Q[key]
            newN[key[3:length(key)]] = N[key]
        else
            continue
        end
    end

    return (newQ, newN)

end

##################################################################
# Trial
##################################################################
lambda = 0.95
function mcts_trial(depth, c, plotting=false, num_particles=num_particles, iterations=iterations)



    # Initialize true state and belief state (particle filter); we assume perfect knowledge at start of simulation (could experiment otherwise with random beliefs)
    #num_particles = 500
    model = ParticleFilterModel{Vector{Float64}}(f2, g)
    pfilter = SIRParticleFilter(model, num_particles)

    # true state
    # state is [range, bearing, relative course, own speed]
    # assume a starting position within range of sensor and not too close
    true_state = [rand(rng, 25:100), rand(rng,0:359), rand(rng,0:11)*30, 1]

    # belief state
    # assume perfect knowledge at first time step
    belief = ParticleCollection([true_state for i in 1:num_particles])

        # Simulation prep/initialization; for now we start with no prior knowledge for Q values/N values, could incorporate this later

    # global Q and N dictionaries, indexed by history (and optionally action to follow all in same array; using ints)
    Q = Dict{Array{Int64,1},Float64}()
    N = Dict{Array{Int64,1},Float64}()

    # not manipulating these parameters for now, in global scope
    # lambda, discount factor
    lambda = 0.95

    # experimenting with different parameter values
    # experiment with different depth parameters
    depth = depth
    # exploration factor, experiment with different values
    c = c

    # don't need to modify history tree at first time step
    action = nothing
    observation = nothing



    # run simulation

    total_reward = 0
    total_col = 0
    total_loss = 0

    # 500 time steps with an action to be selected at each
    num_iters = 500
    plots = []
    for time_step = 1:num_iters

        #if time_step % 100 == 0
        #    @show time_step
        #end

        # NOTE: we found restarting history tree at each time step yielded better results
        # if action taken, modify history tree
        if action != nothing
            #(Q,N) = modify_history_tree(Q, N, action, observation)
            Q = Dict{Array{Int64,1},Float64}()
            N = Dict{Array{Int64,1},Float64}()
        end


        # select an action
        (Q, N, action) = select_action(Q, N, belief, depth, c, iterations)

        # take action; get next true state, obs, and reward
        next_state = f2(true_state, action, rng)
        observation = h(next_state, rng)
        reward = r(Tuple(next_state), action_to_index(action))
        true_state = next_state

        # update belief state (particle filter)
        belief = update(pfilter, belief, action, observation)

        # accumulate reward
        total_reward += reward
        if true_state[1] < 10
            total_col = 1
        end


        if plotting
            push!(plots, build_plot(true_state, belief))
        end

        # TODO: flags for collision, lost track, end of simulation lost track

    end

    if true_state[1] > 150
        total_loss = 1
    end
    return (total_reward, plots, total_col, total_loss)

end

##################################################################
# Plotting
##################################################################

#build plot, including particles and grid
# returns a plot given state, particle filter, grid interpolants, and grid
function build_plot(xp, b, ξ, grid=grid)
    grid_r, grid_theta = [],[]
    plot_r = [row[1] for row in particles(b)]
    plot_theta = [row[2] for row in particles(b)]*π/180
    plot_x_theta = xp[2]*π/180
    plot_x_r = xp[1]

    for i in ξ.nzind
        coords = ind2x(grid,i)
        push!(grid_r, coords[1])
        push!(grid_theta, coords[2]*π/180)
    end

    plt = plot(proj=:polar, lims=(0,200), size=(1000,1000))
    scatter!([grid_theta], [grid_r], label="grid", color=:blue)
    scatter!(plot_theta, plot_r, label="particles", markershape=:+, seriescolor=:black,markeralpha=.3,markersize=3)
    scatter!([plot_x_theta], [plot_x_r],label="location", seriescolor=:red, markershape=:diamond, markersize=5)

    return plt
end

#build plot without grid - returns a plot given state and particle filter
function build_plot(xp, b)
    grid_r, grid_theta = [],[]
    plot_r = [row[1] for row in particles(b)]
    plot_theta = [row[2] for row in particles(b)]*π/180
    plot_x_theta = xp[2]*π/180
    plot_x_r = xp[1]

    plt = plot(proj=:polar, lims=(0,200), size=(1000,1000))
    scatter!([grid_theta], [grid_r], label="grid", color=:blue)
    scatter!(plot_theta, plot_r, label="particles", markershape=:+, seriescolor=:black,markeralpha=.3,markersize=3)
    scatter!([plot_x_theta], [plot_x_r],label="location", seriescolor=:red, markershape=:diamond, markersize=5)

    return plt
end
