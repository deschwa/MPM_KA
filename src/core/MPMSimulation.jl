"""
MPMSimulation Type. Contains all necessary information about an MPM simulation.
"""
mutable struct MPMSimulation{T, G<:Grid, MPG<:Tuple, SF<:AbstractShapeFunction, BC<:AbstractBoundaryCondition}
    mp_groups::MPG
    grid::G
    dt::T
    t::T
    total_time::T
    shape_function::SF
    boundary_condition::BC
end


function MPMSimulation(mp_groups::MPG, grid::G, total_time::T, dt::T, shape::SF, boundarycondition::BC) where {MPG<:Tuple, G<:Grid, T<:Real, SF<:AbstractShapeFunction, BC<:AbstractBoundaryCondition}
    return MPMSimulation{T, G, MPG, SF, BC}(mp_groups, grid, dt, zero(T), total_time, shape, boundarycondition)
end


# Legacy constructor without boundary condition
function MPMSimulation(mp_groups::MPG, grid::G, total_time::T, dt::T, shape::SF) where {MPG<:Tuple, G<:Grid, T<:Real, SF<:AbstractShapeFunction}
    return MPMSimulation{T, G, MPG, SF, NoBoundaryCondition}(mp_groups, grid, dt, zero(T), total_time, shape, NoBoundaryCondition())
end