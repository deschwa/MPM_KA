using KernelAbstractions
using StaticArrays
using StructArrays
using Adapt

struct GridNode{T}
    m::T                    # Mass
    p::SVector{3, T}        # Momentum
    p_new::SVector{3, T}    # New Momentum
    f::SVector{3, T}        # Force
end

struct Grid{T, S <: AbstractArray}
    state::S            # StructArray of GridNodes

    ghost_width::Int    # Padding width

    min_coords::SVector{3, T}       # Minimum coordinates of grid
    inv_spacings::SVector{3, T}     # Inverse of grid spacings
end

function Grid(::Type{ArrayType}, N_physical::SVector{3, Int}, ghost_width::Int, 
              min_coords::SVector{3, T}, max_coords::SVector{3, T}) where {ArrayType, T}
    N_total = N_physical .+ 2 * ghost_width
    dims = Tuple(N_total)

    spacings = (max_coords .- min_coords) ./ (N_physical .- 1)
    inv_spacings = one(T) ./ spacings

    cpu_state = StructArray{GridNode{T}}(
        undef, dims; 
        unwrap = t -> t <: SVector
    )
    
    fill!(cpu_state, zero(GridNode{T}))

    device_state = StructArrays.replace_storage(ArrayType, cpu_state)

    return Grid(device_state, ghost_width, min_coords, inv_spacings)
end

Grid(args...) = Grid(Array, args...)



@kernel function reset_grid_kernel!(grid_state)
    I = @index(Global, Cartesian)
    grid_state[I] = zero(eltype(grid_state))
end

function reset_grid!(grid::Grid)
    backend = KernelAbstractions.get_backend(grid.state)
    kernel = reset_grid_kernel!(backend)
    kernel(grid.state, ndrange=size(grid.state))
    KernelAbstractions.synchronize(backend)
end

function Base.zero(::Type{GridNode{T}}) where {T}
    return GridNode(
        zero(T),
        zero(SVector{3, T}),
        zero(SVector{3, T}),
        zero(SVector{3, T})
    )
end