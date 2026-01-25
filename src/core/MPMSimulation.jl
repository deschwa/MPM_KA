"""
MPMSimulation Type. Contains all necessary information about an MPM simulation.
"""
mutable struct MPMSimulation{T, G<:Grid, MPG<:Tuple, SF}
    mp_groups::MPG
    grid::G
    dt::T
    t::T
    total_time::T
    shape_function::SF
end

function MPMSimulation(mp_groups::MPG, grid::G, total_time::T, dt::T, shape::SF) where {MPG<:Tuple, G<:Grid, T<:Real, SF<:AbstractShapeFunction}
    return MPMSimulation{T, G, MPG, SF}(mp_groups, grid, dt, zero(T), total_time, shape)
end