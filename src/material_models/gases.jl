"""
Gamma Law Ideal Gas
"""
struct GammaLawGas{T} <: AbstractMaterial
    γ::T
    c_0::T  # quadratic artificial viscosity coefficient
    c_1::T  # linear artificial viscosity coefficient
end

struct GammaLawGasCache{T} <: AbstractMaterialCache
    e::T
    p::T
end

function max_speed(material::GammaLawGas, mps, p_idx)
    rho_p = mps.m[p_idx] / (mps.volume_0[p_idx] * det(mps.F[p_idx]))
    a = sqrt(material.γ  * max(0.0, mps.mat_cache.p[p_idx]) / rho_p)
    return norm(mps.v[p_idx]) + a
end


@kernel function stress_update_kernel!(material::GammaLawGas{T}, mps, dt::T) where {T}
    p_idx = @index(Global, Linear)

    γ = material.γ
    rho_p = mps.m[p_idx] / (mps.volume_0[p_idx] * det(mps.F[p_idx]))
    e_p = mps.mat_cache.e[p_idx]
    L_p = mps.L[p_idx]

    p_p = - (γ - 1) * rho_p * e_p

    σ_p = p_p * one(SMatrix{3,3,T,9})

    D_p = T(0.5) * (L_p + L_p')
    trD_p = tr(D_p)

    rho_new = rho_p / (1 + dt*trD_p)


    e_new = e_p + dt * (dot(σ_p, D_p)/rho_new)

    mps.σ[p_idx] = σ_p
    mps.mat_cache.e[p_idx] = e_new
    mps.mat_cache.p[p_idx] = p_p
end



"""
Isentropic Gas
"""
struct IsentropicGas{T} <: AbstractMaterial
    γ::T
    K::T
end

struct IsentropicGasCache{T} <: AbstractMaterialCache
    p::T
end

function max_speed(material::IsentropicGas, mps, p_idx)
    rho_p = mps.m[p_idx] / (mps.volume_0[p_idx] * det(mps.F[p_idx]))
    a = sqrt(material.γ  * max(0.0, mps.mat_cache.p[p_idx]) / rho_p)
    return norm(mps.v[p_idx]) + a
end


@kernel function stress_update_kernel!(material::IsentropicGas{T}, mps, dt::T) where {T}
    p_idx = @index(Global, Linear)
    
    γ = material.γ
    rho_p = mps.m[p_idx] / (mps.volume_0[p_idx] * det(mps.F[p_idx]))
    K_p = mps.mat_cache.K[p_idx]

    p_p = K_p * rho_p ^ γ

    σ_p = p_p * one(SMatrix{3,3,T,9})

    mps.σ[p_idx] = σ_p
    mps.mat_cache.p[p_idx] = p_p
end
