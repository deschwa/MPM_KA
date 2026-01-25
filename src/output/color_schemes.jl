function get_color_scheme(mp_group, scheme::Symbol)
    mps = mp_group.material_points
    
    if scheme == :velocity
        return norm.(mps.v)
        
    elseif scheme == :acceleration
        return norm.(mps.a_ext)
        
    elseif scheme == :compression
        # Plot Jacobian (J = det(F)). J < 1 is compression, J > 1 is expansion.
        # F is stored in mps.F 
        return det.(mps.F)
        
    elseif scheme == :pressure
        # Pressure is -1/3 trace(σ)
        # σ is stored in mps.σ 
        return [ -tr(mp.σ)/3 for mp in mps ]
        
    elseif scheme == :von_mises
        # Von Mises Stress (useful for yield criteria)
        return [von_mises(mp.σ) for mp in mps]
        
    else
        error("Unknown coloring scheme: $scheme")
    end
end


function von_mises(σ::SMatrix{3,3,T}) where {T}
    s = σ .- (trace(σ)/3) * I # Deviatoric stress
    return sqrt(1.5 * sum(s .* s)) # Von Mises stress
end