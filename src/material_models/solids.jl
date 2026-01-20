# -----------
# Rigid Solid
# -----------
@kernel function stress_update_kernel!(material::RigidSolid, mps, dt::T) where {T}
    p_idx = @index(Global, Linear)

    mps.σ[p_idx] = zero(SMatrix{3,3,T})
end

function soundspeed(material::RigidSolid)
    return 0
end


# --------------------
# Neo-Hookean Material
# --------------------
@kernel function stress_update_kernel!(material::NeoHookean{T}, mps, dt::T) where {T}
    p_idx = @index(Global, Linear)

    F_p = mps.F[p_idx]
    J = det(F_p)
    J = max(J, epsilon(T))

    λ = material.λ
    μ = material.μ

    σ_new = (μ / J) * (F_p * transpose(F_p) - I_3(T)) + (λ * log(J) / J) * I_3(T)    

    mps.σ[p_idx] = σ_new
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


