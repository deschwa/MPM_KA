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
    


@inline function shape_function(natural_coords::SVector{3, T}, inv_spacings::SVector{3, T})::NTuple{4, T} where {T}
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



@inline function get_support_base(::QuadraticBSpline, grid_coords::SVector{3, T}) where {T}
    i = floor(Int, grid_coords[1] - T(0.5))
    j = floor(Int, grid_coords[2] - T(0.5))
    k = floor(Int, grid_coords[3] - T(0.5))

    return i, j, k
end

@inline get_support_offsets(::QuadraticBSpline) = (0:2, 0:2, 0:2)

@inline function shape_function(natural_coords::SVector{3, T}, inv_spacings::SVector{3, T})::NTuple{4, T} where {T}
    # 1D B-Spline and deriv
    wx, dwx = bspline_quad_1d(natural_coords[1], inv_spacings[1])
    wy, dwy = bspline_quad_1d(natural_coords[2], inv_spacings[2])
    wz, dwz = bspline_quad_1d(natural_coords[3], inv_spacings[3])

    N_I = wx * wy * wz

    ∇N_Ix = dwx * wy * wz
    ∇N_Iy = wx * dwy * wz
    ∇N_Iz = wx * wy * dwz

    return N_I, ∇N_Ix, ∇N_Iy, ∇N_Iz
end

@inline function bspline_quad_1d(dist::T, inv_h::T) where {T}
    abs_d = abs(dist)
    
    w = zero(T)
    dw = zero(T)

    if abs_d < 0.5
        # Fall 1: |x| < 0.5
        # N(x) = 0.75 - x^2
        w = T(0.75) - abs_d^2
        dw = -2 * dist
    elseif abs_d < 1.5
        # Fall 2: 0.5 <= |x| < 1.5
        # N(x) = 0.5 * (1.5 - |x|)^2
        val = T(1.5) - abs_d
        w = T(0.5) * val^2
        dw = (dist > 0) ? -val : val
    else
        # Fall 3: Außerhalb des Supports
        w = zero(T)
        dw = zero(T)
    end

    # Gradient muss mit 1/dx skaliert werden (Kettenregel)
    return w, dw * inv_h
end