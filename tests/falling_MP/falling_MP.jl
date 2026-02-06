include("../../src/MPM.jl")
using .MPM
using StaticArrays
using LinearAlgebra
using LaTeXStrings


using Plots
gr()


"""
Setup
"""
# Material Point
h_0 = 2.0
g = -9.81
E = 1.0e3
ν = 0.3
ρ = 1000.0
m = 1.0

x_0 = SVector(0.0, 0.0, h_0)
v_0 = SVector(0.0, 0.0, 0.0)
a_ext = SVector(0.0, 0.0, g)
V = m / ρ
material, cache = NeoHookean_E_ν(E, ν, ρ)

MP = MaterialPoint(x_0, v_0, a_ext, m, V, cache)
mp_group = MaterialPointGroup(Array, [MP], material, "falling_MP")
mp_groups = (mp_group,)


# Grid
domain_width = 1.0
domain_height = h_0 * 1.2
max_coords = SVector(domain_width/2, domain_width/2, domain_height)
min_coords = SVector(-domain_width/2, -domain_width/2, -domain_height)


Nx = 20
Ny = Nx
dx = domain_width / (Nx - 1)
Nz = ceil(Int, domain_height / dx)
Ns = SVector{3}(Nx, Ny, Nz)
dt_list = [0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01]


"""
Prepare analytical solution
"""
h_analytical(t) = h_0 + 0.5 * g * t^2
function log_error(analytical_function, x, y)
    return log10(abs(analytical_function(x) - y))
end
t_arr_dict = Dict{Float64, Vector{Float64}}()
h_arr_dict = Dict{Float64, Vector{Float64}}()
logerror_arr_dict = Dict{Float64, Vector{Float64}}()


for dt in dt_list
    dt_sim = deepcopy(dt)

    println("Starting simulation with dt = $(dt_sim)s")

    t_arr = Float64[]
    h_arr = Float64[]
    logerror_arr = Float64[]

    # Create grid
    grid = Grid(Array, Ns, 3, min_coords, max_coords)
    println("Grid created with spacings = $(1 ./ grid.inv_spacings) m and Ns = $(size(grid.state))")

    # Simulation
    sim = MPMSimulation(deepcopy(mp_groups), grid, 100.0, dt_sim, QuadraticBSpline())

    while sim.mp_groups[1].material_points[1].x[3] > 1.0
        timestep!(sim)
        push!(t_arr, sim.t)
        h_num = sim.mp_groups[1].material_points[1].x[3]
        push!(h_arr, h_num)
        push!(logerror_arr, log_error(h_analytical, sim.t, h_num))
    end
    
    t_arr_dict[dt] = t_arr
    h_arr_dict[dt] = h_arr
    logerror_arr_dict[dt] = logerror_arr
end



#Plotting
p1 = plot(title="Falling Material Point - Height over Time", xlabel="Time [s]", ylabel="Height [m]", legend=:topright)
t_analytical = 0:0.01:sqrt(2*h_0/abs(g))
h_analytical_arr = [h_analytical(t) for t in t_analytical]
plot!(p1, t_analytical, h_analytical_arr, label="Analytical Solution", lw=2, ls=:dash, color=:black)
for dt in dt_list
    plot!(p1, t_arr_dict[dt], h_arr_dict[dt], label="dt = $(dt)s", lw=2)
end
savefig(p1, "falling_MP_height_over_time.png")


p2 = plot(title="Logarithmic Error over Time - dt Dependence", xlabel="Time [s]", ylabel="Logarithmic Error", legend=:topright)
for dt in dt_list
    plot!(p2, t_arr_dict[dt], logerror_arr_dict[dt], label="dt = $(dt)s", lw=2)
end
savefig(p2, "falling_MP_log_error_over_time_dt.png")   



# COnvergence rate
final_logerrors = [logerror_arr_dict[dt][end] for dt in dt_list]
p3 = scatter(dt_list, final_logerrors, xscale=:log10, xticks = (dt_list, string.(dt_list)), title="Convergence Rate of the Euler Integrator", xlabel="Time Step Size [s]", ylabel="Final Logarithmic Error", legend=false, marker=:o)
savefig(p3, "falling_MP_convergence_rate.png")