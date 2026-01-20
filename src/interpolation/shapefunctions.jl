@inline function get_grid_position(pos_p::SVector{3, T}, inv_spacings::SVector{3, T}, min_coords::SVector{3, T}, ghost_width::Int) where {T}
    return (pos_p - min_coords) .* inv_spacings .+ (ghost_width + one(T))  # +1 for 1-based indexing
end



@inline function get_support_base(::LinearHat, grid_coords::SVector{3, T}) where {T}
    i = floor(Int, grid_coords[1])
    j = floor(Int, grid_coords[2])
    k = floor(Int, grid_coords[3])

    return i,j,k
end


@inline get_support_offsets(::LinearHat) = (0:1, 0:1, 0:1)
    


@inline function shape_function(natural_coords::SVector{3, T}, inv_spacings::SVector{3, T}) where {T}
    Nx = max(one(T) - abs(natural_coords[1]), zero(T))
    Ny = max(one(T) - abs(natural_coords[2]), zero(T))
    Nz = max(one(T) - abs(natural_coords[3]), zero(T))
    N_I = Nx * Ny * Nz

    dN_xdx = Nx>0 ? - sign(natural_coords[1]) * inv_spacings[1] : zero(T)
    dN_ydy = Ny>0 ? - sign(natural_coords[2]) * inv_spacings[2] : zero(T)
    dN_zdz = Nz>0 ? - sign(natural_coords[3]) * inv_spacings[3] : zero(T)
    ∇N_Ix = dN_xdx * Ny * Nz
    ∇N_Iy = Nx * dN_ydy * Nz
    ∇N_Iz = Nx * Ny * dN_zdz

    return N_I, ∇N_Ix, ∇N_Iy, ∇N_Iz
end
