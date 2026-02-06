"""
Linear Elastic Isotropic Material
"""
struct LinearElastic{T}<:AbstractMaterial
    E::T        # Young's Modulus
    ν::T        # Poisson's Ratio

    λ::T        # Lame's First Parameter
    μ::T        # Lame's Second Parameter

    ρ::T        # Density

    c::T        # soundspeed
end


function LinearElastic_E_ν(E::T, ν::T, ρ::T) where {T}
    λ = (E * ν) / ((1 + ν) * (1 - 2 * ν))
    μ = E / (2 * (1 + ν))
    c = sqrt(E/ρ)
    return LinearElastic{T}(E, ν, λ, μ, ρ, c), NoMaterialCache()
end
function LinearElastic_λ_ν(λ::T, ν::T, ρ::T) where {T}
    E = λ * (1 + ν) * (1 - 2 * ν) / ν
    μ = E / (2 * (1 + ν))
    c = sqrt(E/ρ)
    return LinearElastic{T}(E, ν, λ, μ, ρ, c), NoMaterialCache()
end

function max_speed(material::LinearElastic{T}, mps, p_idx) where {T}
    return norm(mps.v[p_idx]) + material.c
end

@kernel function stress_update_kernel!(material::NeoHookean{T}, mps::StructVector{MP}, dt::T) where {T, MP<:MaterialPoint{T}}
    p_idx = @index(Global, Linear)
    
    # Direct column access is safe and fast
    F = mps.F[p_idx]
    
    σ_new = barrier_neohookean(F, material.λ, material.μ)::SMatrix{3,3,T,9}

    mps.σ[p_idx] = σ_new
end

@inline function barrier_neohookean(F::SMatrix{3,3,T,9}, λ::T, μ::T)::SMatrix{3,3,T,9} where {T}
    J = det(F)::Float64
    J = max(J, eps(T))

    σ_new = (μ / J) * (F * F'  - I_3(T)) + (λ * log(J) / J) * I_3(T)
    
    return σ_new
end




"""
NeoHookean Material
"""
struct NeoHookean{T}<:AbstractMaterial
    E::T        # Young's Modulus
    ν::T        # Poisson's Ratio

    μ::T        # Lame's First Parameter
    λ::T        # Lame's Second Parameter

    ρ::T        # Density

    c::T        # Soundspeed
end

function NeoHookean(E::T, ν::T, ρ::T) where {T}
    λ = (E * ν) / ((1 + ν) * (1 - 2 * ν))
    μ = E / (2 * (1 + ν))
    c = sqrt(E/ρ)
    return NeoHookean{T}(E, ν, μ, λ, ρ, c), NoMaterialCache()
end

function max_speed(material::NeoHookean{T}, mps, p_idx) where {T}
    return norm(mps.v[p_idx]) + material.c
end


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




