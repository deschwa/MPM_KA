include("../../src/MPM.jl")
using .MPM

using Base.Threads
using StaticArrays
using Plots 
using LinearAlgebra


function generate_sphere(R::T, N_radius::Int, rho_0::T, material, material_cache) where {T}
    MPList = MaterialPoint{T, typeof(material_cache)}[]
    
    dx = R / (N_radius - 1)
    V = dx^3

    for i in -N_radius:N_radius
        for j in -N_radius:N_radius
            for k in -N_radius:N_radius
                x = SVector{3,T}(i,j,k) * dx
                if norm(x) <= R
                    v = SVector{3,T}(0,0,0)
                    a_ext = SVector{3,T}(0,0,0)
                    m = rho_0 * V
                    push!(MPList, MaterialPoint(x, v, a_ext, m, V, deepcopy(material_cache)))
                end
            end
        end
    end

    mp_group = MaterialPointGroup(Array, MPList, material, "Sphere")
    return mp_group
end


function set_forces(mp_group, lambda, nu)
    @inbounds @threads for i in 1:mp_group.N
        pos = mp_group.material_points.x[i]
        vel = mp_group.material_points.v[i]

        # Gravity force
        a_grav = - pos * lambda
        # Damping force
        a_damp = - vel * nu

        mp_group.material_points.a_ext[i] = a_grav + a_damp
    end
end


# ---------------------
# Simulation parameters
# ---------------------
n = 1.0                 # polytropic index
γ = 1.0 + 1/n             # polytropic exponent
K = 0.1                 # polytropic constant
ν = 0.1                 # viscosity coefficient
λ = 2.01203286081606    # linear gravity
t_end = 20.0            # end time
Δt = 0.001               # time step
R = 1.0                 # initial radius of sphere
println("Simulation parameters set: n=$n, γ=$γ, K=$K, ν=$ν, R=$R, t_end=$t_end, Δt=$Δt")



# Generate Grid
# Grid should have 20 cells per radius
dx_grid = R / 10
min_coords = SVector{3,Float64}(-R, -R, -R) * 2
max_coords = SVector{3,Float64}(R, R, R) * 2
Ns = ceil.(Int, (max_coords - min_coords) / dx_grid)
grid = Grid(Array, SVector{3,Int}(Ns...), 0, min_coords, max_coords)


# Generate Material
material = IsentropicGas(γ, K)
mat_cache = IsentropicGasCache(0.0)


# Generate Material Points
N_radius = 2 * ceil(Int, R / dx_grid) + 1
M_tot = 1.0

mp_group = generate_sphere(R, N_radius, M_tot, material, mat_cache)
println("Generated $(mp_group.N) material points with total mass $M_tot in a sphere of radius $R.")


# Boundary conditions
bc = FreeSlipBoundary()

# Shape Function
sf = QuadraticBSpline()


# create Simulation
sim = MPMSimulation((mp_group,), grid, t_end, Δt, sf, bc)


# Animation Setup
anim_density = Animation()
function get_densities(mp_group)
    densities = zeros(mp_group.N)
    rs = zeros(mp_group.N)
    for i in 1:mp_group.N
        rho_p = mp_group.material_points.m[i] / (mp_group.material_points.volume_0[i] * det(mp_group.material_points.F[i]))
        densities[i] = rho_p
        rs[i] = norm(mp_group.material_points.x[i])
    end
    return rs, densities
end
calculated_steps = 0


# Run Simulation
println("Starting simulation...")
while sim.t < sim.total_time
    set_forces(sim.mp_groups[1], λ, ν)
    timestep!(sim)
    # if sim.t % 0.1 == 0.0
    #     println("Time: $(round(sim.t, digits=2)) / $(sim.total_time)")
    #     rs, densities = get_densities(sim.mp_groups[1])
    #     s = scatter(rs, densities, xlabel="Radius", ylabel="Density", title="Density vs Radius at t=$(round(sim.t, digits=2))", label="")
    #     frame(anim_density, s)
    # end
    if calculated_steps % 100 == 0
        print("Time: $(round(sim.t, digits=2)) / $(sim.total_time)        \r")
    end
end

# gif(anim_density, "density_vs_radius.gif", fps=10)


r_final, densities_final = get_densities(sim.mp_groups[1])
s_final = scatter(r_final, densities_final, xlabel="Radius", ylabel="Density", title="Final Density vs Radius at t=$(round(sim.t, digits=2))", label="")  
savefig(s_final, "final_density_vs_radius.png")


