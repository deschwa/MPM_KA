function random_particle_sim(N_particles::Int, N_grid_cells::Int, T::Type{<:AbstractFloat}, material::mat, cache::matcache) where {mat<:AbstractMaterial, matcache<:AbstractMaterialCache}
    # Initialize Particles
    positions = [SVector{3, T}(rand(T, 3)) for _ in 1:N_particles]
    cpu_points = [MaterialPoint(
        x,
        SVector{3, T}(zeros(T, 3)),
        SVector{3, T}(zeros(T, 3)),
        one(T),
        one(T),
        cache
    ) for x in positions]

    # Create Group and Grid
    mp_group = MaterialPointGroup(Array, cpu_points, material, "benchmark_particles")
    min_coords = SVector{3, T}(ones(T, 3))
    max_coords = SVector{3, T}(ones(T, 3))
    Ns = SVector{3, Int}(N_grid_cells, N_grid_cells, N_grid_cells)
    grid = Grid(Array, Ns, 2, min_coords, max_coords)

    # Create Simulation container
    sim = MPMSimulation((mp_group,), grid, T(1.0), T(1e-3), LinearHat())

    return sim 
end

"""
Helper function to extract commonly used quantities from simulation#
returns mp_groups, grid, dt, shapefunction
"""
function get_quantities(sim::MPMSimulation)
    mp_groups = sim.mp_groups
    grid = sim.grid
    dt = sim.dt
    shapefunction = sim.shape_function
    return mp_groups, grid, dt, shapefunction
end