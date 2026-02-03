include("src/MPM.jl")
using .MPM
using BenchmarkTools
using StaticArrays
using StructArrays
using LinearAlgebra
using KernelAbstractions
using InteractiveUtils

using Profile


mat, cache = NeoHookean_E_ν(1e6, 0.3, 1000.0)

sim = random_particle_sim(1000, 20, Float64, mat, cache)

mp_groups, grid, dt, shapefunction = get_quantities(sim)
mpg = mp_groups[1]

# 1. Define helper FIRST so it can be correctly inlined
@inline function barrier(F::SMatrix{3,3,T,9}, λ::T, μ::T)::SMatrix{3,3,T,9} where {T}
    J = det(F)
    J = max(J, T(1e-8))

    I_mat = one(F)
    
    # Force the math to stay in SMatrix world
    # (F * F') is usually safe, but let's be 100% explicit
    FFt = F * F' 

    σ_new = (μ / J) * (FFt - I_mat) + (λ * log(J) / J) * I_mat 
    
    return σ_new
end

# 2. Define kernel/function SECOND
function stress_update_neo!(material::NeoHookean{T}, mps::StructVector{MP}, dt::T) where {T, MP<:MaterialPoint{T}}

    p_idx = 1
    
    # Direct column access is safe and fast
    val_F = mps.F[p_idx]
    
    σ_new = barrier(val_F, material.λ, material.μ)

    mps.σ[p_idx] = σ_new
    return nothing
end

# 3. Benchmark Harness
function verify_allocations(mat, mps, dt)
    stress_update_neo!(mat, mps, dt)
    return nothing
end

# 4. Run Test
println("--- Final Benchmark ---")
# Use local variables to simulate real usage
local_mat = mat
local_mps = mpg.material_points
local_dt = dt

# Warmup
verify_allocations(local_mat, local_mps, local_dt)

# Measure
allocs = @allocated verify_allocations(local_mat, local_mps, local_dt)

if allocs == 0
    println("SUCCESS: 0 Bytes allocated!")
else
    println("Still allocating: $allocs bytes (Should be 0!)")
end