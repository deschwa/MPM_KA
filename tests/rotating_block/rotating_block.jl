include("../../src/MPM_module.jl")
using .MPM
using StaticArrays
using LinearAlgebra
using JLD2
using LaTeXStrings
using Statistics
using DataFrames
using CSV

using Plots
gr()
theme(:vibrant) # oder :dao, :bright
default(
    tickfontfamily="Computer Modern", 
    guidefontfamily="Computer Modern", 
    framestyle=:box, 
    gridalpha=0.3
)




function generate_rotating_block(width, height, Nx, omega, material, cache)
    dx = width / (Nx - 1)
    Nz = ceil(Int, height / dx) + 1
    Ns = SVector{3}(Nx, Nx, Nz)

    V = dx^3
    m = V * material.ρ

    MPs = Vector{MaterialPoint{Float64, typeof(cache)}}()
    
    for ix in 1:Nx, iy in 1:Nx, iz in 1:Nz
        x = (ix - 1) * dx - width / 2
        y = (iy - 1) * dx - width / 2
        z = (iz - 1) * dx - height / 2
        r = sqrt(x^2 + y^2)
        v_theta = omega * r
        theta = atan(y, x)
        vx = -v_theta * sin(theta)
        vy = v_theta * cos(theta)
        vz = 0.0
        pos = SVector(x, y, z)
        vel = SVector(vx, vy, vz)
        # println("Position: ($x, $y, $z), Velocity: ($vx, $vy, $vz)")
        push!(MPs, MaterialPoint(pos, vel, zero(SVector{3, Float64}), m, V, cache))
    end

    mp_group = MaterialPointGroup(Array, MPs, material, "rotating_block")

    return mp_group, dx
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


function set_plot_attributes(plotobject)
    plot!(plotobject,
        lw=2,
        grid=:both,
        gridalpha=0.3,
        minorgrid=true,
        framestyle=:box,
        tickfontfamily = "Computer Modern", # LaTeX-Look
        guidefontfamily = "Computer Modern",
        titlefontfamily = "Computer Modern",
        legendfontfamily = "Computer Modern",
        titlefontsize=14,
        guidefontsize=12,
        tickfontsize=10,
        legendfontsize=10,

        dpi=300,

        left_margin = (5,:mm),
        bottom_margin = (5,:mm),
        right_margin = (2,:mm),
        top_margin = (2,:mm)
    )
end


function export_as_csv(filename, x, y)
    df = DataFrame(t = x, L = y)
    CSV.write(filename, df, precision=16)
end



"""
Setup
"""
# Block
b = 1.0 # block width
h = 1.0 # block height
T = 2.0 # Rotation period
ω = 2π / T # angular velocity
Nx_block = 15 # number of MPs along one edge

# Material
E = 1.0e6
ν = 0.4
ρ = 1000.0
material, cache = NeoHookean_E_ν(E, ν, ρ)

# Material Point Group
mp_group, dx = generate_rotating_block(b, h, Nx_block, ω, material, cache)
mp_groups = (mp_group,)
println("Generated $(length(mp_group.material_points)) material points with dx = $(round(dx, sigdigits=4)) m")


# Grid
domain_width = b * 2.0
domain_height = h * 1.5
max_coords = SVector(domain_width/2, domain_width/2, domain_height/2)
min_coords = SVector(-domain_width/2, -domain_width/2, -domain_height/2)
dx_grid = dx * 2.0
Ns = SVector{3}(ceil(Int, domain_width/dx_grid) + 1,
                 ceil(Int, domain_width/dx_grid) + 1,
                 ceil(Int, domain_height/dx_grid) + 1)
grid = Grid(Array, Ns, 3, min_coords, max_coords)
println("Created grid with spacings = $(1 ./ grid.inv_spacings) m and Ns = $(size(grid.state))")

shape_function = QuadraticBSpline()

# Simulation
dt = 0.05
total_time = 600.0 # 10 minutes
sim = MPMSimulation(mp_groups, grid, total_time, dt, shape_function)

# Prepare for logging
L_list = Float64[]
time_list = Float64[]
t_interval = 1 # seconds
next_log_time = t_interval

timestep_runtime = []

println("Starting Simulation...\n")

while sim.t < total_time
    time = @elapsed timestep!(sim)
    push!(timestep_runtime, time)

    if sim.t >= next_log_time
        L = calculate_angular_momentum(sim.mp_groups[1])
        push!(L_list, L)
        push!(time_list, sim.t)
        print("Time: $(round(sim.t, digits=4))s, Angular Momentum: $(round(L, sigdigits=6)) kg·m²/s       \r")
        global next_log_time += t_interval
    end
end


# # Plot Angular Momentum over Time
# p_zoomed = plot(
#     time_list, 
#     L_list, 
#     xlabel="Time [s]", 
#     ylabel="Angular Momentum [kg·m²/s]",
#     title="Angular Momentum Conservation in Rotating Block (zoomed)",
#     legend=false)
# # set_plot_attributes(p_zoomed)

# savefig(p_zoomed, "angular_momentum_conservation_$(total_time)s.png")
# println("Saved zoomed angular momentum plot to angular_momentum_conservation_$(total_time)s.png")



# p_not_zoomed = plot(time_list, 
#     L_list, 
#     xlabel="Time [s]", 
#     ylabel="Angular Momentum [kg·m²/s]",
#     title="Angular Momentum Conservation in Rotating Block (Full View)", 
#     ylims=(0, maximum(L_list)*1.2),
#     legend=false)
# # set_plot_attributes(p_not_zoomed)

# savefig(p_not_zoomed, "angular_momentum_conservation_full_$(total_time)s.png")
# println("Saved full angular momentum plot to angular_momentum_conservation_full_$(total_time)s.png")

# mean_L = mean(L_list)
# p_relative_error = plot(time_list, log10.(abs.(L_list .- mean_L) ./ mean_L),
#     xlabel="Time [s]", ylabel=L"log_{10} \left( \frac{ | L(t) - \bar{L} | }{ \bar{L} } \right)",
#     title="Relative Error in Angular Momentum Conservation",
#     legend=false)
# # set_plot_attributes(p_relative_error)

# savefig(p_relative_error, "angular_momentum_relative_error_$(total_time)s.png")
# println("Saved relative error plot to angular_momentum_relative_error_$(total_time)s.png")


# Export data as CSV
export_as_csv("angular_momentum_data_$(total_time)s.csv", time_list, L_list)
println("Exported angular momentum data to angular_momentum_data_$(total_time)s.csv")


max_derivation = (maximum(L_list) - minimum(L_list)) / mean(L_list)
println("Maximum Relative Deviation in Angular Momentum: $(round(max_derivation * 100, sigdigits=4)) %")


mean_timestep_time = mean(timestep_runtime)
println("Average Timestep Runtime: $(round(mean_timestep_time * 1000, sigdigits=10)) ms")



