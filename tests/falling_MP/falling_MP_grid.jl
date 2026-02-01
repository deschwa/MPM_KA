include("../../src/MPM_module.jl")
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


Nx_list = [2, 4, 8, 16, 32]


dt = 0.0005

"""
Prepare analytical solution
"""
h_analytical(t) = h_0 + 0.5 * g * t^2
function log_error(analytical_function, x, y)
    return log10(abs(analytical_function(x) - y))
end
t_arr_dict = Dict{Int, Vector{Float64}}()
h_arr_dict = Dict{Int, Vector{Float64}}()
logerror_arr_dict = Dict{Int, Vector{Float64}}()


for Nx in Nx_list
    dx = domain_width / (Nx - 1)
    Nz = ceil(Int, domain_height / dx)
    Ns = SVector{3}(Nx, Nx, Nz)
    grid = Grid(Array, Ns, 2, min_coords, max_coords)
    println("Created grid with spacings = $(1 ./ grid.inv_spacings) m and Ns = $(size(grid.state))")
    shape_function = QuadraticBSpline()
    sim = MPMSimulation(deepcopy(mp_groups), grid, 2.0 * sqrt(2*h_0/abs(g)), deepcopy(dt), shape_function)

    t_arr = Float64[]
    h_arr = Float64[]
    logerror_arr = Float64[]

    while sim.mp_groups[1].material_points[1].x[3] > 1.0
        timestep_fixed_dt!(sim)

        t = sim.t
        h = sim.mp_groups[1].material_points[1].x[3]
        push!(t_arr, t)
        push!(h_arr, h)
        push!(logerror_arr, log_error(h_analytical, t, h))
    end

    t_arr_dict[Nx] = t_arr
    h_arr_dict[Nx] = h_arr
    logerror_arr_dict[Nx] = logerror_arr
end



#Plotting
# p1 = plot(title="Falling Material Point - Height over Time", xlabel="Time [s]", ylabel="Height [m]", legend=:topright)
# t_analytical = 0:0.01:sqrt(2*h_0/abs(g))
# h_analytical_arr = [h_analytical(t) for t in t_analytical]
# plot!(p1, t_analytical, h_analytical_arr, label="Analytical Solution", lw=2, ls=:dash, color=:black)
# for Nx in Nx_list
#     plot!(p1, t_arr_dict[Nx], h_arr_dict[Nx], label="Nx = $(Nx)", lw=2)
# end
# savefig(p1, "falling_MP_height_over_time_dx.png")


p2 = plot(title="Logarithmic Error over Time - dx Dependence", xlabel="Time [s]", ylabel="Logarithmic Error", legend=:topright)
for Nx in Nx_list
    plot!(p2, t_arr_dict[Nx], logerror_arr_dict[Nx], label="Nx = $(Nx)", lw=2)
end
savefig(p2, "falling_MP_log_error_over_time_dx.png")   