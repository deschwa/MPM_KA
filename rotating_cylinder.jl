include("src/MPM_module.jl")
using .MPM
using StaticArrays
using LinearAlgebra

using Plots
gr()

function generate_rotating_cylinder(radius, height, spacing, angular_velocity, material, cache)
    particles = MaterialPoint{Float64, typeof(cache)}[] # Prep Output

    # Ceiling handles float errors 
    N_r = ceil(Int, radius/spacing)
    N_h = ceil(Int, (height/2)/spacing)
    volume = spacing^3
    mass = material.ρ * volume

    omega_vec = SVector(0, 0, angular_velocity)

    for i_x in -N_r:N_r
        for i_y in -N_r:N_r
            for i_z in -N_h:N_h

                pos = spacing .* SVector(i_x, i_y, i_z)

                abs_rho = sqrt(pos[1]^2 + pos[2]^2)
                if abs_rho >= radius
                    continue #Reject Point
                end

                #check z coordinate because of the earlier ceil op
                if abs(pos[3]) > height / 2
                    continue
                end

                vel = cross(omega_vec, pos)

                particle = MaterialPoint(pos, vel, zero(SVector{3,Float64}), mass, volume, cache)

                push!(particles, particle)
            end
        end
    end
    
    return MaterialPointGroup(Array, particles, material, "cylinder")
    
end



println("Setting up Simulation...")

rho = 1000.0
material, cache = NeoHookean_E_ν(1e6, 0.3, rho)

R = 0.5  # Cylinder Radius (m)
h = 1.0  # Cylinder Height (m)
T = 5.0  # Rotation Period (s)
angular_velocity = 2π / T

# Define Simulation Domain
min_coords = SVector(-1.0, -1.0, -1.0)
max_coords = SVector(1.0, 1.0, 1.0)
padding = 3


# 1. Set a FIXED Grid Resolution (Physics Quality)
# A 0.5m radius needs at least ~10-20 cells to look like a circle.
grid_dx = R/10  # 5cm grid cells (Radius = 10 cells)

# 2. Set Particle Density (Sampling Quality)
# We want 8 particles per cell (2x2x2) for stability
particles_per_cell_axis = 2 
particle_spacing = grid_dx / particles_per_cell_axis # 0.025

# 3. Generate Particles
mp_group = generate_rotating_cylinder(R, h, particle_spacing, angular_velocity, material, cache)
println("generated $(mp_group.N) particles.")
mp_group_tuple = (mp_group,)

# 4. Create Grid
# Calculate N based on the FIXED grid_dx, not the particle spacing
N_1D = ceil.(Int, (max_coords .- min_coords) ./ grid_dx)
N_vec = SVector(N_1D...)
grid = Grid(Array, N_vec, padding, min_coords, max_coords)
println("Grid Size: $(N_vec) nodes.")


sim = MPMSimulation(mp_group_tuple, grid, 10.0, 1e-3, LinearHat())

savefig(plot_material_points(sim, "Initial State"), "initial_state.png")


# Animation Setup
frames = []
fps = 30
next_frame_time = 0


println("Starting Simulation")
timestep!(sim, 1.0, 0.1)
while sim.t < sim.total_time
    
    if sim.t >= next_frame_time
        p = plot_material_points(sim, "t=$(sim.t)")
        push!(frames, p)
        display(p)
        global next_frame_time += 1/fps
        print("Calculating time $(sim.t)       \r")
    end

    timestep!(sim, 1.0, 0.4)
end

println("\nCalculated Simulation. Starting Animation...")


# Create the Animation
anim = Animation()
anim = @animate for frame in frames 
    plot(frame)
end

gif(anim, "simulation.gif", fps=fps)
println("Saved Animation to simulation.gif")

