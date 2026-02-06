function fix_boundaries!(::NoSlipBoundary, sim::MPMSimulation)
    grid = sim.grid
    v = grid.state.v  # Zugriff beschleunigen
    Nx, Ny, Nz = size(v)
    p = grid.ghost_width
    T = eltype(v)

    # Ganze Padding-Blöcke auf Null setzen
    @views v[1:p, :, :]          .= zero(T)     # x=0
    @views v[(Nx-p+1):Nx, :, :]  .= zero(T)     # x=Nx
    
    @views v[:, 1:p, :]          .= zero(T)     # y=0
    @views v[:, (Ny-p+1):Ny, :]  .= zero(T)     # y=Ny
    
    @views v[:, :, 1:p]          .= zero(T)     # z=0
    @views v[:, :, (Nz-p+1):Nz]    .= zero(T)     # z=Nz
end


function fix_boundaries!(::FreeSlipBoundary, sim::MPMSimulation)
    grid = sim.grid
    v = grid.state.v  # Zugriff beschleunigen
    Nx, Ny, Nz = size(v)
    p = grid.ghost_width
    vec_T = eltype(v)
    T = eltype(vec_T)
    zero_T = zero(T)


    @views v.x[1:p, :, :] .= max(zero_T, v.x[1:p, :, :])

    @views v.x[(Nx-p+1):Nx, :, :] .= min(zero_T, v.x[(Nx-p+1):Nx, :, :])   

    @views v.y[:, 1:p, :] .= max(zero_T,v.y[:, 1:p, :])

    @views v.y[:, (Ny-p+1):Ny, :] .= min(zero_T, v.y[:, (Ny-p+1):Ny, :])

    @views v.z[:, :, 1:p] .= max(zero_T, v.z[:, :, 1:p])

    @views v.z[:, :, (Nz-p+1):Nz] .= min(zero_T, v.z[:, :, (Nz-p+1):Nz])
end