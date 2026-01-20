# -----------
# Rigid Solid
# -----------
@kernel function stress_update_kernel!(material::RigidSolid, mps, dt::T) where {T}
    p_idx = @index(Global, Linear)

    mps.σ[p_idx] = zero(SMatrix{3,3,T,9})
end

function soundspeed(material::RigidSolid)
    return 0
end


# --------------------
# Neo-Hookean Material
# --------------------
@kernel function stress_update_kernel!(material::NeoHookean{T}, mps::StructVector{MP}, dt::T) where {T, MP<:MaterialPoint{T}}
    p_idx = @index(Global, Linear)
    
    # Direct column access is safe and fast
    F = mps.F[p_idx]
    
    σ_new = barrier_neohookean(F, material.λ, material.μ)::SMatrix{3,3,T,9}

    mps.σ[p_idx] = σ_new
end

@inline function barrier_neohookean(F::SMatrix{3,3,T,9}, λ::T, μ::T)::SMatrix{3,3,T,9} where {T}
    J = det(F)
    J = max(J, epsilon(T))
    
    # Force the math to stay in SMatrix world
    # (F * F') is usually safe, but let's be 100% explicit
    FFt = F * F' 

    σ_new = (μ / J) * (FFt - I_3(T)) + (λ * log(J) / J) * I_3(T)
    
    return σ_new
end

function soundspeed(material::NeoHookean{T}) where {T}
    return sqrt(material.E / material.ρ)
end

# ---------------------------------
# Linear Elastic Isotropic Material
# ---------------------------------
@kernel function stress_update_kernel!(material::LinearElastic{T}, mps, dt::T) where {T}
    p_idx = @index(Global, Linear)

    L_p = mps.L[p_idx]
    σ_p = mps.σ[p_idx]

    ε_new = 0.5 * dt * (L_p + transpose(L_p))
    tr_ε = tr(ε_new)

    λ = material.λ
    μ = material.μ

    σ_new = σ_p + λ*tr_ε*I_3 + 2*μ*ε_new

    mps.σ[p_idx] = σ_new
end

function soundspeed(material::LinearElastic{T}) where {T}
    return sqrt(material.E / material.ρ)
end


