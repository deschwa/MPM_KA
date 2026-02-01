include("color_schemes.jl")



# Helper to extract coordinates efficiently
function extract_coords(mp_group)
    # mps.x is a StructArray of SVectors. 
    # We broadcast getindex to separate components.
    pos = mp_group.material_points.x
    return (
        getindex.(pos, 1), # x coordinates
        getindex.(pos, 2), # y coordinates
        getindex.(pos, 3)  # z coordinates
    )
end


function plot_material_points(sim::MPMSimulation; title="", scheme=:velocity, color_limits=nothing)
    p = plot(layout=(1,1), size=(600,600), camera=(45, 30),
             xlabel="X", ylabel="Y", zlabel="Z", title=title)
    
    grid_size = SVector(size(sim.grid.state)) .- sim.grid.ghost_width .* 2 .- 1
    min_c = sim.grid.min_coords
    max_c = min_c .+ grid_size .* (1.0 ./ sim.grid.inv_spacings)

    plot!(xlims=(min_c[1], max_c[1]), 
          ylims=(min_c[2], max_c[2]), 
          zlims=(min_c[3], max_c[3]))

    # 3. Plot Particles
    for group in sim.mp_groups
        x, y, z = extract_coords(group)
        
        # Calculate velocity magnitude for coloring
        color_values = get_color_scheme(group, scheme)
        
        scatter!(p, x, y, z, 
            markersize = 2, 
            markerstrokewidth = 0,
            marker_z = color_values,    # Color by velocity
            color = :plasma,     # Colormap
            clims = color_limits,
            label = group.label  # Legend entry
        )
    end
    
    return p
end