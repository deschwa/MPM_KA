include("../src/MPM_module.jl")
using .MPM
using BenchmarkTools
using StructArrays
using StaticArrays
using KernelAbstractions
using Adapt
using Atomix
using Profile
using InteractiveUtils

T = Float64

N_particles = 100

positions = [SVector{3, T}(rand(T, 3)) for _ in 1:N_particles]

material, cache = NeoHookean_E_ν(T(1e6), T(0.3), T(1000))


cpu_points = [MaterialPoint(
    x,
    SVector{3, T}(zeros(T, 3)),
    SVector{3, T}(zeros(T, 3)),
    one(T),
    one(T),
    cache
) for x in positions]


mp_group = MaterialPointGroup(Array, cpu_points, material, "NeoHookean")



min_coords = T(1.5) * SVector{3, T}(zeros(T, 3))
max_coords = SVector{3, T}(ones(T, 3))
Ns = SVector{3, Int}(62, 62, 62)
grid = Grid(Array, Ns, 2, min_coords, max_coords)


sim = MPMSimulation((mp_group,), grid, T(1.0), T(1e-3), LinearHat())

linhat = LinearHat() 

p2g!(mp_group, grid, linhat)



# p2g_kernel!(grid.state, mp_group.material_points, grid.inv_spacings, grid.min_coords, grid.ghost_width, LinearHat())

# # display(@benchmark p2g_kernel!($grid.state, $mp_group.material_points, $grid.inv_spacings, $grid.min_coords, $grid.ghost_width, LinearHat()))
# # display(@profile p2g_kernel!(grid.state, mp_group.material_points, grid.inv_spacings, grid.min_coords, grid.ghost_width, LinearHat()))
# mp = mp_group.material_points[1]


# reset_grid!(grid)
# display(@allocated p2g_barrier!(grid.state, grid.inv_spacings, grid.min_coords, grid.ghost_width, linhat, 
# mp.m, mp.x, mp.v, mp.a_ext, mp.L, mp.F, mp.volume_0, mp.σ))
#precompile
