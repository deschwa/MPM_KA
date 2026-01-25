using StaticArrays
using LinearAlgebra
using Base.Threads
using Atomix


"""
Compute the Courant condition based timestep
Note: This runs on the host (CPU)
"""
function courant_cond(sim::MPMSimulation, courant_factor::T=0.4) where {T}
    mp_groups = sim.mp_groups

    spacing = one(T) / maximum(sim.grid.inv_spacings)

    max_soundspeed = zero(T)
    max_v = zero(T)
    for mp_group in mp_groups
        # v_vec = mp_group.material_points.v
        v_temp = maximum(norm, Array(mp_group.material_points.v))

        soundspeed_temp = soundspeed(mp_group.material)
        if soundspeed_temp > max_soundspeed
            max_soundspeed = soundspeed_temp
        end

        if v_temp > max_v
            max_v = v_temp
        end
    end

    return min(courant_factor * spacing / (max_soundspeed + max_v), sim.dt)
end


"""
Perform a single timestep of the MPM simulation
"""
function timestep!(sim::MPMSimulation, alpha::T=1.0, courant_factor::T=0.3) where {T}
    mp_groups = sim.mp_groups
    grid = sim.grid
    shapefunction = sim.shape_function
    dt = courant_cond(sim, courant_factor)
    
    # ============
    # Reset Kernel
    # ============
    reset_grid!(grid)

    # =================
    # Particles to Grid
    # =================
    for mp_group in mp_groups
        p2g!(mp_group, grid, shapefunction)
    end

    # ===========
    # Grid Update
    # ===========
    grid_update!(grid, dt)

    # ================
    # Grid to Particle
    # ================
    for mp_group in mp_groups
        g2p!(mp_group, grid, alpha, dt, shapefunction)
    end

    # ==============
    # Stress Update
    # ==============
    for mp_group in mp_groups
        stress_update!(mp_group, dt)
    end

    sim.t += dt
end


"""
Helper function for atomic updates of vector fields
"""
@inline function atomic_add_states!(grid_state, m::T, p::SVector{3,T}, f::SVector{3,T}, i, j, k) where {T}
    Atomix.@atomic grid_state.m[i, j, k] += m

    Atomix.@atomic grid_state.p.x[i, j, k] += p[1]
    Atomix.@atomic grid_state.p.y[i, j, k] += p[2]
    Atomix.@atomic grid_state.p.z[i, j, k] += p[3]

    Atomix.@atomic grid_state.f.x[i, j, k] += f[1]
    Atomix.@atomic grid_state.f.y[i, j, k] += f[2]
    Atomix.@atomic grid_state.f.z[i, j, k] += f[3]
end



"""
Particles to Grid host function
"""
function p2g!(mp_group::MaterialPointGroup, grid::Grid, shapefunction=LinearHat())
    mps = mp_group.material_points
    grid_state = grid.state

    backend = KernelAbstractions.get_backend(grid_state)

    kernel = p2g_kernel!(backend)

    kernel(grid_state, mps, grid.inv_spacings, grid.min_coords, grid.ghost_width, shapefunction; ndrange=mp_group.N)

    KernelAbstractions.synchronize(backend)
end # p2g! function

# Particles to Grid kernel function
@kernel function p2g_kernel!(grid_state, mps::StructVector{MP}, 
                            inv_spacings::SVector{3,T}, min_coords::SVector{3,T}, ghost_width::Int,
                            shapefunction::SF) where {T, MP<:MaterialPoint{T}, SF<:AbstractShapeFunction}

    # Thread index
    p_idx = @index(Global, Linear)
    # p_idx = 

    # extract mp
    # mp = mps[p_idx]
    m_p = mps.m[p_idx]
    x_p = mps.x[p_idx]::SVector{3,T}
    v_p = mps.v[p_idx]::SVector{3,T}
    a_ext_p = mps.a_ext[p_idx]::SVector{3,T}
    L_p = mps.L[p_idx]::SMatrix{3,3,T,9}
    F_p = mps.F[p_idx]::SMatrix{3,3,T,9}
    V0_p = mps.volume_0[p_idx]::T
    σ_p = mps.σ[p_idx]::SMatrix{3,3,T,9}


    p2g_barrier!(grid_state, inv_spacings, min_coords, ghost_width,
                shapefunction,
                m_p, x_p, v_p, a_ext_p, L_p, F_p,
                V0_p, σ_p)
end

@inline function p2g_barrier!(grid_state, inv_spacings::SVector{3,T}, min_coords::SVector{3,T}, ghost_width::Int,
                            shapefunction::SF,
                            m_p::T, x_p::SVector{3,T}, v_p::SVector{3,T}, a_ext_p::SVector{3,T}, L_p::SMatrix{3,3,T,9}, F_p::SMatrix{3,3,T,9},
                            V0_p::T, σ_p::SMatrix{3,3,T,9}) where {T, SF<:AbstractShapeFunction}
    J = LinearAlgebra.det(F_p)

    grid_position = get_grid_position(x_p, inv_spacings, min_coords, ghost_width)
    i_base, j_base, k_base = get_support_base(shapefunction, grid_position)
    i_offsets, j_offsets, k_offsets = get_support_offsets(shapefunction)


    @inbounds for di in i_offsets, dj in j_offsets, dk in k_offsets
        i = i_base + di
        j = j_base + dj
        k = k_base + dk

        if !checkbounds(Bool, grid_state.m, i,j,k)
            continue
        end

        natural_coords = SVector(
                            grid_position[1] - i,
                            grid_position[2] - j,
                            grid_position[3] - k
                        )

        r_rel = - natural_coords ./ inv_spacings

        N_Ip, ∇Nx, ∇Ny, ∇Nz = shape_function(natural_coords, inv_spacings)
        ∇N_Ip = SVector{3, T}(∇Nx, ∇Ny, ∇Nz)

        m_update = N_Ip * m_p

        σ∇N = σ_p * ∇N_Ip
        Lp_rrel = L_p * r_rel
        vol_p = V0_p * J

        f_update = N_Ip * a_ext_p * m_p - vol_p * σ∇N
        p_update = m_p * (v_p + Lp_rrel) * N_Ip

        atomic_add_states!(grid_state, m_update, p_update, f_update, i, j, k)

    end # loop over support nodes
end




"""
Grid Update host function
"""
function grid_update!(grid::Grid, dt::T) where {T}
    backend = KernelAbstractions.get_backend(grid.state)
    
    kernel = grid_update_kernel!(backend)

    kernel(grid.state, dt; ndrange=size(grid.state))
    KernelAbstractions.synchronize(backend) 
end

# Grid Update kernel function
@kernel function grid_update_kernel!(grid_state, dt::T) where {T}
    I = @index(Global, Cartesian)

    local_m = grid_state.m[I]
    if local_m > epsilon(T)
        grid_state.p_new.x[I] = grid_state.p.x[I] + dt * grid_state.f.x[I]
        grid_state.p_new.y[I] = grid_state.p.y[I] + dt * grid_state.f.y[I]
        grid_state.p_new.z[I] = grid_state.p.z[I] + dt * grid_state.f.z[I]
    else
        grid_state.p_new.x[I] = zero(T)
        grid_state.p_new.y[I] = zero(T)
        grid_state.p_new.z[I] = zero(T)
    end
end





"""
Grid to Particles host function
"""
function g2p!(mp_group::MaterialPointGroup, grid::Grid, α::T, dt::T, shapefunction::AbstractShapeFunction) where {T}
    mps = mp_group.material_points
    grid_state = grid.state

    backend = KernelAbstractions.get_backend(grid_state)

    kernel = g2p_kernel!(backend)

    kernel(grid_state, mps, grid.inv_spacings, grid.min_coords, grid.ghost_width,
           α, dt, shapefunction; ndrange=mp_group.N)

    KernelAbstractions.synchronize(backend)
end

# Grid to Particles kernel function
@kernel function g2p_kernel!(grid_state, mps::StructVector{MP},
                            inv_spacings::SVector{3,T}, min_coords::SVector{3,T}, ghost_width::Int,
                            α::T, dt::T,
                            shapefunction::AbstractShapeFunction) where {T, MP<:MaterialPoint{T}}

    # Thread index
    p_idx = @index(Global, Linear)

    # extract mp
    mp = mps[p_idx]

    grid_position = get_grid_position(mp.x, inv_spacings, min_coords, ghost_width)
    i_base, j_base, k_base = get_support_base(shapefunction, grid_position)
    i_offsets, j_offsets, k_offsets = get_support_offsets(shapefunction)

    v_p_apic = zero(SVector{3, T})
    v_p_flip = mp.v
    B_p_new = zero(SMatrix{3,3,T,9})

    for di in i_offsets, dj in j_offsets, dk in k_offsets
        i = i_base + di
        j = j_base + dj
        k = k_base + dk
        if !checkbounds(Bool, grid_state.m, i,j,k)
            continue
        end

        natural_coords = SVector(
                            grid_position[1] - i,
                            grid_position[2] - j,
                            grid_position[3] - k
                        )

        r_rel = - natural_coords ./ inv_spacings

        N = shape_function(natural_coords, inv_spacings)
        N_Ip = N[1]
        ∇N_Ip = SVector{3, T}(N[2], N[3], N[4])

        grid_mass = grid_state.m[i, j, k]
        if grid_mass > 1e-14
            v_i = grid_state.p[i, j, k] / grid_state.m[i, j, k]
            v_i_new = grid_state.p_new[i, j, k] / grid_state.m[i, j, k]

            v_p_apic = v_p_apic + N_Ip * v_i_new
            v_p_flip = v_p_flip + N_Ip * (v_i_new - v_i)

            B_p_new = B_p_new + update_B_matrix(N_Ip, v_i_new, r_rel)
        end
    end
    mps.v[p_idx] = (1-α) * v_p_flip + α * v_p_apic
    mps.x[p_idx] = mp.x + mps.v[p_idx] * dt
    mps.L[p_idx] = finalize_apic(shapefunction, B_p_new, inv_spacings)
    mps.F[p_idx] = (I_3(T) + mps.L[p_idx]*dt) * mp.F
end

@inline function update_B_matrix(N_Ip::T, v_i_new::SVector{3,T}, r_rel::SVector{3,T}) where {T}
    v_i_new * r_rel' * N_Ip
end

@inline function finalize_apic(::LinearHat, B_sum::SMatrix{3,3,T,9}, inv_spacings::SVector{3,T}) where {T}
    D_inv = SMatrix{3,3,T,9}(
        inv_spacings[1], zero(T), zero(T),
        zero(T), inv_spacings[2], zero(T),
        zero(T), zero(T), inv_spacings[3]
    )
    return B_sum * 4 * D_inv.^2
end

"""
Stress Update host function
"""
function stress_update!(mp_group::MaterialPointGroup{<:Any, MT}, dt::T) where {MT, T}
    mps = mp_group.material_points
    material = mp_group.material

    backend = KernelAbstractions.get_backend(mps)
    kernel = stress_update_kernel!(backend)
    kernel(material, mps, dt; ndrange=mp_group.N)

    KernelAbstractions.synchronize(backend)
end

# Stress Update kernel functions in ../material_models/solids.jl