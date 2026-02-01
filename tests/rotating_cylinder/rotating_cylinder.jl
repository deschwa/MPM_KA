include("../../src/MPM_module.jl")
using .MPM
using StaticArrays
using LinearAlgebra
using JLD2

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

function calculate_angular_momentum(mp_group)
    total_L = zero(SVector{3,Float64})
    total_mass = 0.0

    # go into constant CM frame
    com = zero(SVector{3,Float64})
    com_v = zero(SVector{3,Float64})
    for mp in mp_group.material_points
        com += mp.x * mp.m
        com_v += mp.v * mp.m
        total_mass += mp.m
    end
    com /= total_mass
    com_v /= total_mass

    for mp in mp_group.material_points
        r = mp.x - com
        total_L += cross(r, mp.m * (mp.v - com_v))
    end

    return norm(total_L)
end



println("Setting up Simulation...")

rho = 1000.0
material, cache = NeoHookean_E_ν(1e6, 0.3, rho)

R = 0.5  # Cylinder Radius (m)
h = 2.0  # Cylinder Height (m)
T = 3.0  # Rotation Period (s)
angular_velocity = 2π / T

# Define Simulation Domain
min_coords = SVector(-0.75, -0.75, -1.25)
max_coords = SVector(0.75, 0.75, 1.25)
padding = 3


# 1. Set a FIXED Grid Resolution (Physics Quality)
# A 0.5m radius needs at least ~10-20 cells to look like a circle.
grid_dx = R/10  # 10cm grid cells (Radius = 5 cells)

# 2. Set Particle Density (Sampling Quality)
# We want 27 particles per cell (3x3x3) for stability
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


sim = MPMSimulation(mp_group_tuple, grid, 10.0, 1e-3, QuadraticBSpline())

savefig(plot_material_points(sim, title="Initial State", scheme=:velocity, color_limits=(-0.7, 0.7)), "initial_state.png")


# Animation Setup
fps = 15
next_frame_time = 0
anim = Animation()


# particle save configuration
particle_filename(t) = "long_cylinder/mp_group_t=$(lpad((round(t, digits=4)), 4, '0')).jld2"

# Angular momentum preparation
angular_momenta = Float64[]
times = Float64[]



println("Starting Simulation")
timestep!(sim, 1.0, 0.1)
while sim.t < sim.total_time
    
    print("Calculating time $(sim.t)       \r")
    if sim.t >= next_frame_time
        p = plot_material_points(sim, title="t=$(sim.t)", scheme=:velocity, color_limits=(-0.7, 0.7))
        frame(anim, p)
        display(p)

        jldsave(particle_filename(sim.t), particles=sim.mp_groups[1], time=sim.t)
        global next_frame_time += 1/fps
    end

    timestep!(sim, 1.0, 0.5)

    push!(angular_momenta, calculate_angular_momentum(mp_group))
    push!(times, sim.t)
end

println("\nCalculated Simulation. Starting Animation...")



gif(anim, "simulation.gif", fps=fps)
println("Saved Animation to simulation.gif")


println("\nPlotting Angular Momentum Conservation...")

logerror(x) = log10(abs(x - angular_momenta[1]) )

p2 = plot(times, angular_momenta, xlabel="Time (s)", ylabel="Angular Momentum (kg·m²/s)", title="Angular Momentum Conservation", legend=false)
plot!(p2, xlims=(0, sim.total_time), ylims=(minimum(angular_momenta)*0.95, maximum(angular_momenta)*1.05))

logerrors = logerror.(angular_momenta)
p3 = plot(times[2:end], logerrors[2:end], xlabel="Time (s)", ylabel="Log10 Error in Angular Momentum", title="Angular Momentum Error (Log Scale)", legend=false)
plot!(p3, xlims=(0, sim.total_time), ylims=(minimum(logerrors)*2, 0))

savefig(p2, "angular_momentum.png")
savefig(p3, "angular_momentum_error.png")
println("Saved Angular Momentum plot to angular_momentum.png")
println("Saved Angular Momentum Error plot to angular_momentum_error.png")