using JLD2
using StaticArrays
using LinearAlgebra
using Plots
using LaTeXStrings
gr()

include("../../src/MPM.jl")
using .MPM


function calculate_angular_momentum(mp_group)
    total_L = zero(SVector{3,Float64})
    total_mass = 0.0

    # go into constant CM frame
    com = zero(SVector{3,Float64})
    com_v = zero(SVector{3,Float64})
    for mp in mp_group.material_points
        com += mp.x * mp.m
        com_v += mp.v * mp.m
        total_mass += mp.m
    end
    com /= total_mass
    com_v /= total_mass

    for mp in mp_group.material_points
        r = mp.x - com
        total_L += cross(r, mp.m * (mp.v - com_v))
    end

    return norm(total_L)
end

function get_sigma_rr(mp_group)
    rs = Float64[]
    σ_rrs = Float64[]

    for mp in mp_group.material_points
        r = sqrt(mp.x[1]^2 + mp.x[2]^2)
        if r > 0 && abs(mp.x[3])<0.05
            e_r = SVector{3,Float64}(mp.x[1]/r, mp.x[2]/r, 0.0)
            σ = mp.σ
            σ_rr = dot(e_r, σ * e_r)
            push!(rs, r)
            push!(σ_rrs, σ_rr)
        end
    end
    return rs, σ_rrs
end

function get_sigma_phiphi(mp_group)
    σ_φφs = Float64[]
    rs = Float64[]

    for mp in mp_group.material_points
        r = sqrt(mp.x[1]^2 + mp.x[2]^2)
        if r > 0 && abs(mp.x[3])<0.05
            e_φ = SVector{3,Float64}(-mp.x[2]/r, mp.x[1]/r, 0.0)
            σ = mp.σ
            σ_φφ = dot(e_φ, σ * e_φ)
            push!(σ_φφs, σ_φφ)
            push!(rs, r)
        end
    end

    return rs, σ_φφs
end

function analyze_timestep(file_path::String)
    @load file_path particles time

    L = calculate_angular_momentum(particles)
    σ_rr = get_sigma_rr(particles)
    σ_φφ = get_sigma_phiphi(particles)

    return (time=time, angular_momentum=L, sigma_rr=σ_rr, sigma_phiphi=σ_φφ)
end


# Material Parameters
E = 1.0e6       # Young's Modulus in Pa
ν = 0.3         # Poisson's Ratio
ρ = 1000.0      # Density in kg/m^3
ω = 2pi/3         # Angular Velocity in rad/s
R = 0.5        # Radius of the cylinder in meters



"""
Stress Analysis of Rotating Cylinder
"""
example_file = "long_cylinder_N=100845/mp_group_t=6.2005.jld2"

result = analyze_timestep(example_file)

analytical_radial_stress(r) = 1/8 * ρ * ω^2 * ((3-2ν)/(1-ν)) * (R^2 - r^2)
r_linspace = collect(0.0:0.01:R)
analytical_values = [analytical_radial_stress(r) for r in r_linspace]

rr_plot = plot(result.sigma_rr[1], result.sigma_rr[2], seriestype=:scatter, xlabel=L"Radius $r$", ylabel=L"\sigma_{rr}", title="Radial Stress Distribution", label="MPM Result")
plot!(rr_plot, r_linspace, analytical_values, label="Analytical Solution", lw=2, lc=:red, legend=:topright)
savefig(rr_plot, "sigma_rr_distribution.png")


analytical_tangential_stress(r) = 1/8 * ρ * ω^2 * ( ((3-2ν)/(1-ν))*R^2 - ((1+2ν)/(1-ν))*r^2)

φφ_plot = plot(result.sigma_phiphi[1], result.sigma_phiphi[2], seriestype=:scatter, xlabel=L"Radius $r$", ylabel=L"\sigma_{\phi\phi}", title="Tangential Stress Distribution", label="MPM Result")
plot!(φφ_plot, r_linspace, [analytical_tangential_stress(r) for r in r_linspace], label="Analytical Solution", lw=2, lc=:red, legend=:topright)
savefig(φφ_plot, "sigma_phiphi_distribution.png")



