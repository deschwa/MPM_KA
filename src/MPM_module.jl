module MPM

using StaticArrays
using LinearAlgebra
using Base.Threads
using StructArrays
using KernelAbstractions
using Adapt
using Atomix
using CUDA
using Plots


# Small Helpers used across the module
@inline I_3(T) = SMatrix{3,3,T}(I)

@inline epsilon(::Type{Float64}) = 1e-15
@inline epsilon(::Type{Float32}) = 1f-7





include("core/abstract_types.jl")
export LinearElastic, LinearElastic_E_ν, LinearElastic_λ_ν
export NeoHookean, NeoHookean_E_ν, NeoHookean_λ_ν

export AbstractMaterial, AbstractMaterialCache
export AbstractShapeFunction, LinearHat, QuadraticBSpline


include("core/particles.jl")
export MaterialPoint, MaterialPointGroup

include("core/grid.jl")
export Grid, GridNode, reset_grid!

include("core/MPMSimulation.jl")
export MPMSimulation

include("interpolation/shapefunctions.jl")
export shape_function, get_grid_position, get_support_base, get_support_offsets

include("material_models/solids.jl")
export stress_update_kernel!, soundspeed

include("boundary_conditions/boundaries.jl")
export fix_boundaries!

include("solver/timestep.jl")
export timestep!, timestep_fixed_dt!, p2g!, p2g_kernel!, g2p!, g2p_kernel!, grid_update!, grid_update_kernel!, stress_update!
export p2g_barrier!


include("benchmark_utils/generate_test_setups.jl")
export random_particle_sim, get_quantities

include("output/plotting_utils.jl")
export plot_material_points


end # Nodule MPM