"""
Material Point
"""
struct MaterialPoint{T, MaterialCache}
    x::SVector{3, T}          # Position
    v::SVector{3, T}          # Velocity
    a_ext::SVector{3, T}      # External Acceleration

    m::T                      # Mass
    volume_0::T               # Volume
    F::SMatrix{3,3,T}         # Deformation Gradient
    σ::SMatrix{3,3,T}         # Cauchy Stress
    L::SMatrix{3,3,T}         # Velocity Gradient/Affine Matrix

    mat_cache::MaterialCache      # Cache for additional data (eg damage, ...)
end

function MaterialPoint(x::SVector{3, T}, v::SVector{3, T}, a_ext::SVector{3, T}, 
                        m::T, volume::T, material_cache::matcache) where {T, matcache<:AbstractMaterialCache}
    F = SMatrix{3,3,T}(I)
    σ = zero(SMatrix{3,3,T})
    L = zero(SMatrix{3,3,T})
    return MaterialPoint(x, v, a_ext, m, volume, F, σ, L, material_cache)
end


"""
MateiralPointGroup
"""
struct MaterialPointGroup{MPs<:StructArray, MaterialType<:AbstractMaterial}
    N::Int                          # Nr of Particles in Group
    
    material_points::MPs            # Tuple of MaterialPoints

    material::MaterialType          # Material Model
    label::String                    # Material Type
    
end


function MaterialPointGroup(::Type{ArrayType}, MPList::Vector{<:MaterialPoint}, material::mat, label::String) where {ArrayType<:AbstractArray, mat<:AbstractMaterial}
    N = length(MPList)
    MP_structarray = StructArray(MPList)

    MP_structarray = StructArrays.replace_storage(ArrayType, MP_structarray)
    return MaterialPointGroup(N, MP_structarray, material, label)
end