"""
Materials
"""
abstract type AbstractMaterial end
abstract type AbstractMaterialCache end

"""
Linear Elastic Isotropic Material
"""
struct LinearElastic{T}<:AbstractMaterial
    E::T        # Young's Modulus
    ν::T        # Poisson's Ratio

    λ::T        # Lame's First Parameter
    μ::T        # Lame's Second Parameter

    ρ::T        # Density
end

struct NoMaterialCache <: AbstractMaterialCache end

function LinearElastic_E_ν(E::T, ν::T, ρ::T) where {T}
    λ = (E * ν) / ((1 + ν) * (1 - 2 * ν))
    μ = E / (2 * (1 + ν))
    return LinearElastic{T}(E, ν, λ, μ, ρ), NoMaterialCache()
end
function LinearElastic_λ_ν(λ::T, ν::T, ρ::T) where {T}
    E = λ * (1 + ν) * (1 - 2 * ν) / ν
    μ = E / (2 * (1 + ν))
    return LinearElastic{T}(E, ν, λ, μ, ρ), NoMaterialCache()
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
end

function NeoHookean_E_ν(E::T, ν::T, ρ::T) where {T}
    λ = (E * ν) / ((1 + ν) * (1 - 2 * ν))
    μ = E / (2 * (1 + ν))
    return NeoHookean{T}(E, ν, μ, λ, ρ), NoMaterialCache()
end
function NeoHookean_λ_ν(λ::T, ν::T, ρ::T) where {T}
    E = λ * (1 + ν) * (1 - 2 * ν) / ν
    μ = E / (2 * (1 + ν))
    return NeoHookean{T}(E, ν, μ, λ, ρ), NoMaterialCache()
end


"""
Rigid Solid
"""
struct RigidSolid<:AbstractMaterial end


"""
Shape Function Types
"""
abstract type AbstractShapeFunction end
struct LinearHat <: AbstractShapeFunction end