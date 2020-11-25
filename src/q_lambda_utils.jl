#SARSA-lambda implementation of tracking problem
#Goldfrank 2020

######################################
### imports
######################################

#using ParticleFilters
#using Distributions
using StaticArrays
using LinearAlgebra
using Random
using StatsBase
using SparseArrays
using DataStructures
using DataFrames
using CSV
using ArgParse
using Plots
using Dates


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--theta"
            help = ""
            default = "zeros"
        "--lambda"
            help = ""
            default = "0.9"
        "--alpha"
            help = ""
            default = "0.7"
        "--gamma"
            help = ""
            default = "0.9"
        "--epsilon"
            help = ""
            default = "0.1"
        "--collision"
            help = ""
            default = "-2"
        "--loss"
            help = ""
            default = "-2"
        "--N"
            help = ""
            default = "1000"
        "--trials"
            help = ""
            default = "1000"
        "--variance"
            help = ""
            default = "true"
        "--outfile"
            help = ""
            default = "sarsa_lambda.csv"
        "--plot_header"
            help = ""
            default = "out"
        "--determ"
            help = ""
            default = "false"
    end

    return parse_args(s)
end

######################################
### generative model
######################################

#input is course in degrees and rng
#returns next course in degrees
function next_crs(crs,rng)
    if rand(rng) < .9
        return crs
    end
    crs = (crs + rand(rng,[-1,1])*30) % 360
    if crs < 0 crs += 360 end
    return crs
end

function next_crs_gen(crs,rng)
    if rand(rng) < .75
        return crs
    end
    crs = (crs + rand(rng,[-1,1])*30) % 360
    if crs < 0 crs += 360 end
    return crs
end

# state as tuple (x, y, crs, spd) of target (spd of o/s)
# returns next state as a function of current state and control(action)
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
    θ = atan(pos[2],pos[1])*180/π
    if θ < 0 θ += 360 end
    return (r, θ, crs, spd)::NTuple{4, Real}
end
#
ACTION_PENALTY = -.05

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
# returns reward as a function of range
function r(s,u, action_penalty=ACTION_PENALTY)
    global COLLISION_REWARD, LOSS_REWARD
    if (2 < u < 5)
        action_penalty = 0
    end
    range = s[1]
    if range >= 150 return (COLLISION_REWARD + action_penalty) end  # reward to not lose track of contact
    if range <= 10 return (LOSS_REWARD + action_penalty) end  # collision avoidance
    return (0.1 + action_penalty)  # being in "sweet spot" maximizes reward
end

## defines action space and creates indexing function for actions
function actions()
    return ((-30,1), (-30, 2), (0, 1), (0, 2), (30, 1), (30, 2))
end

function action_index(action)
    return trunc(Int, 2*(action[1]/30+1) + action[2]) #clever, perhaps too clever
end

# returns vector rather than Tuple, for particle filter
function f2(x, u, rng)
    temp = [i for i in f(x, u, rng)]
    return temp
end

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



###########################################
## Implements Q-Lambda Learning Algorithm
###########################################
totals = [0.0]

function q_trial(θ=θ,trial_length=epochsize, λ=λ, α=α, γ=γ, ϵ=ϵ, N=N,
    burn_in_length=burn_in_length, plotting=plotting)
    e = sparse(zeros(length(grid),6))
    x = [rand(rng, 30:135), rand(rng,0:359), rand(rng,0:11)*30, 1, 1];
    xp = x
    y = h(xp, rng)
    b = ParticleCollection([x[1:4] for i in 1:N])
    ξ = sparse(weighted_grid_2(b)/N)
    particle_collection = []
    starting_x = x

    cur = 0
    last = 0
    uu = next_action([transpose(θ[:,j])*ξ for j in 1:size(θ)[2]], ϵ, rng)
    u = uu[1]
    collision_list = []
    loss_list = []
    zoof = 0
    cum_rew = 0
    collisions = 0
    loss = 0

    burn_in = true
    plots = []

    i = 0
    counter = false
    while i <= (trial_length + burn_in_length)
        #observe reward and new state
        rew = r(Tuple(xp),u)
        #b = update(pfilter, b, actions()[u], y)
        xp = f2(x, actions()[u], rng)

        y = h(xp, rng)
        b = update(pfilter, b, actions()[u], y)

        #update N(s,a) with state (t) weights
        e[:,u] += ξ
        old_u = u

        # choose next action
        uu = next_action([transpose(θ[:,j])*ξ for j in 1:size(θ)[2]], ϵ, rng)
        u = trunc(Int64,uu[1])
        #b = update(pfilter, b, actions()[u], y)

        #reset eligibility trace if action is random
        if uu[2] == 1
            e = sparse(zeros(length(grid),6))
            e[:,old_u] += ξ
        end

        #ξ is belief state at (t+1)
        ξ = sparse(weighted_grid_2(b)/N)

        #intermediate math
        cur = transpose(θ[:,uu[3]])*ξ #NOT the argmax!!
        δ = rew + γ * cur - last
        #last = transpose(θ[:,uu[1]])*ξ

        cum_rew += rew
        θ += α * δ * e
        e *= λ*γ

        #should this be here??
        last = transpose(θ[:,uu[1]])*ξ

        x = xp

        if length(particles(b)) != N
            println("PARTICLE FILTER SIZE ERROR: ", length(particles(b)))
        end

        if xp[1] < 10
            zoof = 1
        end

        #PLOTTING
        if plotting
            push!(plots, build_plot(xp, b, ξ))
        end
        i += 1
        if (i == (trial_length + burn_in_length)) && xp[1] > 160 && counter == false
            i -= 10
            counter = true
        end
    end
    collisions += zoof
    if xp[1] > 160
        loss = 1
    end

    return (collisions, loss, θ, cum_rew, plots)
end

function q_trial_no_variance(θ=θ,trial_length=epochsize, λ=λ, α=α, γ=γ, ϵ=ϵ, N=N,
    burn_in_length=burn_in_length, plotting=plotting)
    e = sparse(zeros(length(grid),6))
    x = [rand(rng, 30:135), rand(rng,0:359), rand(rng,0:11)*30, 1];
    xp = x
    y = h(xp, rng)
    b = ParticleCollection([x[1:4] for i in 1:N])
    ξ = sparse(weighted_grid_simple(b)/N)
    particle_collection = []
    starting_x = x

    cur = 0
    last = 0
    uu = next_action([transpose(θ[:,j])*ξ for j in 1:size(θ)[2]], ϵ, rng)
    u = uu[1]
    collision_list = []
    loss_list = []
    zoof = 0
    cum_rew = 0
    collisions = 0
    loss = 0

    burn_in = true
    plots = []

    i = 0
    counter = false
    while i <= (trial_length + burn_in_length)
        #observe reward and new state
        rew = r(Tuple(xp),u)
        #b = update(pfilter, b, actions()[u], y)
        xp = f2(x, actions()[u], rng)

        y = h(xp, rng)
        b = update(pfilter, b, actions()[u], y)

        #update N(s,a) with state (t) weights
        e[:,u] += ξ
        old_u = u

        # choose next action
        uu = next_action([transpose(θ[:,j])*ξ for j in 1:size(θ)[2]], ϵ, rng)
        u = trunc(Int64,uu[1])
        #b = update(pfilter, b, actions()[u], y)

        #reset eligibility trace if action is random
        if uu[2] == 1
            e = sparse(zeros(length(grid),6))
            e[:,old_u] += ξ
        end

        #ξ is belief state at (t+1)
        ξ = sparse(weighted_grid_simple(b)/N)

        #intermediate math
        cur = transpose(θ[:,uu[3]])*ξ #NOT the argmax!!
        δ = rew + γ * cur - last
        #last = transpose(θ[:,uu[1]])*ξ

        cum_rew += rew
        θ += α * δ * e
        e *= λ*γ

        #should this be here??
        last = transpose(θ[:,uu[1]])*ξ

        x = xp

        if length(particles(b)) != N
            println("PARTICLE FILTER SIZE ERROR: ", length(particles(b)))
        end

        if xp[1] < 10
            zoof = 1
        end

        #PLOTTING
        if plotting
            push!(plots, build_plot(xp, b, ξ))
        end
        i += 1
        if (i == (trial_length + burn_in_length)) && xp[1] > 160 && counter == false
            i -= 10
            counter = true
        end
    end
    collisions += zoof
    if xp[1] > 160
        loss = 1
    end

    return (collisions, loss, θ, cum_rew, plots)
end

# trial with random actions - for calibration only
function simple_trial(θ=θ,trial_length=epochsize, λ=λ, α=α, γ=γ, ϵ=ϵ, N=N,
    burn_in_length=burn_in_length, plotting=plotting, quant = QUANTILE, action=DETERM_ACTION)
    e = sparse(zeros(4,6))
    x = [rand(rng, 25:135), rand(rng,0:359), rand(rng,0:11)*30, 1];
    xp = x
    y = h(xp, rng)
    #b = ParticleCollection([x[1:4] for i in 1:N])
    #ξ = sparse(weighted_grid_2(b)/N)
    ξ = zeros(4)
    ξ[y+1] = 1
    #particle_collection = []
    #starting_x = x

    cur = 0
    last = 0
    u = rand(rng, 1:6)
    uu = [u, 1, u]
    collision_list = []
    loss_list = []
    zoof = 0
    cum_rew = 0
    collisions = 0
    loss = 0

    burn_in = true
    plots = []

    for i in 1:(trial_length + burn_in_length)
        #observe reward and new state
        rew = r(Tuple(xp),u)
        #b = update(pfilter, b, actions()[u], y)
        xp = f2(x, actions()[u], rng)

        y = h(xp, rng)


        #b = update(pfilter, b, actions()[u], y)

        #update N(s,a) with state (t) weights
        e[:,u] += ξ
        old_u = u

        # choose next action
        #uu = next_action([transpose(θ[:,j])*ξ for j in 1:size(θ)[2]], ϵ, rng)
        u = rand(rng, 1:6)
        uu = [u, 1, u]

        #u = trunc(Int64,uu[1])

        #b = update(pfilter, b, actions()[u], y)

        #reset eligibility trace if action is random
        if uu[2] == 1
            e = sparse(zeros(4,6))
            e[:,old_u] += ξ
        end

        #ξ is belief state at (t+1)
        #ξ = sparse(weighted_grid_2(b)/N)
        ξ = zeros(4)
        ξ[y+1] = 1

        #intermediate math
        cur = transpose(θ[:,uu[3]])*ξ #NOT the argmax!!
        δ = rew + γ * cur - last
        #last = transpose(θ[:,uu[1]])*ξ

        cum_rew += rew
        θ += α * δ * e
        e *= λ*γ

        #should this be here??
        last = transpose(θ[:,uu[1]])*ξ
        #last = cur

        x = xp

        #if length(particles(b)) != N
        #    println("PARTICLE FILTER SIZE ERROR: ", length(particles(b)))
        #end

        if xp[1] < 10
            zoof = 1
        end

        #PLOTTING
        #if plotting
        #    push!(plots, build_plot(xp, b, ξ))
        #end
    end
    collisions += zoof
    if xp[1] > 160
        loss = 1
    end

    return (collisions, loss, θ, cum_rew, plots)
end

#
