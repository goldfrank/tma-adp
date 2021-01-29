SENSOR_RANGE = 150

function obs1(state)
    rel_brg = state[2]
    range = state[1]
    if rel_brg < 0
        rel_brg += 360
    end
    if ((60 < rel_brg < 90) || (270 < rel_brg < 300)) && (range < SENSOR_RANGE/2)
        return 1::Int64
    elseif ((60 < rel_brg < 90) || (270 < rel_brg < 300)) && (range < SENSOR_RANGE)
        return 2-2*range/SENSOR_RANGE
    end
    return 0::Int64
end

function obs2(state)
    rel_brg = state[2]
    range = state[1]
    if rel_brg < 0 rel_brg += 360 end
    if ((90 <= rel_brg < 120) || (240 < rel_brg <= 270)) && (range < SENSOR_RANGE/2)
        return 1::Int64
    elseif ((90 <= rel_brg < 120) || (240 < rel_brg <= 270)) && (range < SENSOR_RANGE)
        return 2-2*range/SENSOR_RANGE
    end
    return 0::Int64
end

function obs3(state)
    rel_brg = state[2]
    range = state[1]
    if rel_brg < 0 rel_brg += 360 end
    if (120 <= rel_brg <= 240) && (range < SENSOR_RANGE/2)
        return 1::Int64
    elseif (120 <= rel_brg <= 240) && (range < SENSOR_RANGE)
        return 2-2*range/SENSOR_RANGE
    end
    return 0::Int64
end

#null observation, also referred to as "observation 4" in paper
function obs0(state)
    rel_brg = state[2]
    range = state[1]
    if rel_brg < 0 rel_brg += 360 end
    if (rel_brg <= 60) || (rel_brg >= 300) || (range >= SENSOR_RANGE)
        return 1::Int64
    end
    if (!(obs1(state) > 0) && !(obs2(state) > 0) && !(obs3(state) > 0))
        return 1::Int64
    elseif (120 <= rel_brg <= 240) && (SENSOR_RANGE/2 < range < SENSOR_RANGE)
        return 2*range/SENSOR_RANGE - 1
    elseif ((90 <= rel_brg < 120) || (240 < rel_brg <= 270)) && (SENSOR_RANGE/2 < range < SENSOR_RANGE)
        return 2*range/SENSOR_RANGE - 1
    elseif ((60 <= rel_brg < 90) || (270 < rel_brg <= 300)) && (SENSOR_RANGE/2 < range < SENSOR_RANGE)
        return 2*range/SENSOR_RANGE - 1
    end
    return 0::Int64
end

# returns observation given state
function g(x, a, xp, o)
    if o == 0 return obs0(xp) end
    if o == 1 return obs1(xp) end
    if o == 2 return obs2(xp) end
    if o == 3 return obs3(xp) end
end

# samples state given observation
function h(x, rng)
    weights = [obs0(x), obs1(x), obs2(x), obs3(x)]
    obsers::Array{Int64,1} = [0, 1, 2, 3]
    return sample(obsers, Weights(weights))::Int64
end
