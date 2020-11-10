using Distributions
using GridInterpolations
using ParticleFilters
using Distributed
using SharedArrays

states_r = [9,11,50,140,160,500]
states_θ = [0, 60, 90, 120, 240, 270, 300, 360]
states_crs = [90.0*c for c in 0:4]
states_theta = [90.0*c for c in 0:4]
states_qual = [q for q in [0,1000]]
states_spd = [1,2];

# rectangular grid
grid = RectangleGrid(states_r, states_θ, states_crs, states_spd, states_qual)

##########################################################################################
# wrapmap is maps the "wrap around" of grid points due to the polar system, such that
#    angle 360 = angle 0
wrapmap = Dict()

#wraps around theta = 360 to theta = 0
for r in states_r
    for crs in states_crs
        for spd in states_spd
            for qual in states_qual
                key = interpolants(grid, [r, 360, crs, spd, qual])[1][1]
                val = interpolants(grid, [r, 0, crs, spd, qual])[1][1]
                wrapmap[key] = val
            end
        end
    end
end

#wraps around course = 360 to course = 0
for r in states_r
    for theta in states_theta
        for spd in states_spd
            for qual in states_qual
                key = interpolants(grid, [r, theta, 360, spd, qual])[1][1]
                val = interpolants(grid, [r, theta, 0, spd, qual])[1][1]
                wrapmap[key] = val
            end
        end
    end
end

#wraps around theta, course = 360,260 to theta, course = 0, 0
for r in states_r
    for spd in states_spd
        for qual in states_qual
            key = interpolants(grid, [r, 360, 360, spd, qual])[1][1]
            val = interpolants(grid, [r, 0, 0, spd, qual])[1][1]
            wrapmap[key] = val
        end
    end
end
##########################################################################################


#returns grid interpolants given a vector and grid
function polar_grid(vec, grid=grid)
    polants = interpolants(grid, vec)
    for (i, p) in enumerate(polants[1])
        if haskey(wrapmap, p)
            polants[1][i] = wrapmap[p]
        end
    end
    return polants
end

#returns a weighted interpolant grid given a particle collection
#includes the variance term
function weighted_grid_2(b::ParticleCollection)
    N = length(particles(b))
    beta = zeros(length(grid));
    for row in particles(b)
        for (i, x) in enumerate(polar_grid(row)[1])
            beta[x] += polar_grid(row)[2][i]
        end
    end
    vv = var(beta)/(N/2000)^2
    beta = zeros(length(grid));
    for row in particles(b)
        row = [float(r) for r in row]
        push!(row, vv)
        for (i, x) in enumerate(polar_grid(row)[1])
            beta[x] += polar_grid(row)[2][i]
        end
    end
    return beta
end

#argmax function, but selects randomly between arguments of multiple equal maxima
#uses user-specified random number generator(rng)
function argmax2(possible_actions, rng)
    options = []
    max = maximum([i for i in possible_actions if !isnan(i)])
    for (i, x) in enumerate(possible_actions)
        if x == max
            push!(options, i)
        end
    end
    return rand(rng, options)
end

#max function, but selects randomly between multiple equal maxima
#uses user-specified random number generator(rng)
function max2(possible_actions, rng)
    options = []
    max = maximum([i for i in possible_actions if !isnan(i)])
    for (i, x) in enumerate(possible_actions)
        if x == max
            push!(options, i)
        end
    end
    return pose_actions[rand(rng, options)]
end

#epsilon-greedy exploration strategy
#returns an action from list of possible actions, given epsilon
#uses user-specified random number generator(rng)
function next_action(possible_actions, epsilon, rng)
    choice = argmax2(possible_actions, rng)
    if rand(rng) > epsilon
        return (trunc(Int, choice),0, trunc(Int, choice))
    end
    return (trunc(Int, rand(rng, 1:length(possible_actions))),1, trunc(Int, choice))
end

#softmax function for choosing next action
#uses user-specified random number generator(rng)
function softmax_action(possible_actions, k, rng)
    possible_actions = exp.(k*possible_actions)/sum(exp.(k*possible_actions))
    next = trunc(Int, sample(rng, 1:length(possible_actions), StatsBase.weights(possible_actions)))
    best = trunc(Int, argmax2(possible_actions, rng))
    return(next, (next == best)*1, best)
end


#returns a weighted interpolant grid given a particle collection
#for comparison - does not include the variance term
function weighted_grid_simple(b::ParticleCollection)
    beta = zeros(length(grid));
    for row in particles(b)
        for (i, x) in enumerate(polar_grid(row)[1])
            beta[x] += polar_grid(row)[2][i]
        end
    end
    return beta
end


##################################################
# Quantile Choose Action - Not Used
##################################################

#returns a distribution of particle weights given a collection of particles
function particle_distribution(b::ParticleCollection, θ)
    beta = sparse(zeros(length(grid)))
    weights = zeros(6,N)
    for (j, particle) in enumerate(particles(b))
        for (i, x) in enumerate(polar_grid(particle)[1])
            beta[x] += polar_grid(particle)[2][i]
        end
        weight = transpose(θ)*beta
        beta = sparse(zeros(length(grid)))
        for (i, w) in enumerate(weight)
            weights[i,j] += w
        end
    end
    return weights
end

#chooses action based on distribution of particles, algorithm weights,
# desired quantile, randomness factor epsilon, and specified random number generator
function choose_action(b, θ, quant, epsilon, rng)
    o = particle_distribution(b, θ)
    possible_actions = zeros(size(o)[1])
    argmax_actions = zeros(size(o)[1])

    for i in 1:size(o)[1]
       possible_actions[i] = quantile(o[i,:],quant)
       argmax_actions[i] = mean(o[i,:])
    end

    choice = argmax2(possible_actions, rng)
    best = argmax2(argmax_actions, rng)

    if rand(rng) > epsilon
        return (trunc(Int, choice),0, trunc(Int, best))
    end

    return (trunc(Int, rand(rng, 1:length(possible_actions))),1, trunc(Int, best))
end
