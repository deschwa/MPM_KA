include("src/MPM_module.jl")
using .MPM
using BenchmarkTools
using StaticArrays
using StructArrays
using KernelAbstractions

# ==========================================
# 1. SETUP (Identical to check_allocations.jl)
# ==========================================
T = Float64
N_particles = 100 

# Define Material
material, cache = NeoHookean_E_ν(T(1e6), T(0.3), T(1000))

# Initialize Particles
positions = [SVector{3, T}(rand(T, 3)) for _ in 1:N_particles]
cpu_points = [MaterialPoint(
    x,
    SVector{3, T}(zeros(T, 3)),
    SVector{3, T}(zeros(T, 3)),
    one(T),
    one(T),
    cache
) for x in positions]

# Create Group and Grid
mp_group = MaterialPointGroup(Array, cpu_points, material, "NeoHookean")
min_coords = T(1.5) * SVector{3, T}(zeros(T, 3))
max_coords = SVector{3, T}(ones(T, 3))
Ns = SVector{3, Int}(62, 62, 62) # ~238k nodes
grid = Grid(Array, Ns, 2, min_coords, max_coords)

# Create Simulation container
sim = MPMSimulation((mp_group,), grid, T(1.0), T(1e-3), LinearHat())

# Extract references for easier benchmarking
mpg = sim.mp_groups[1]
grd = sim.grid
dt = sim.dt
sf = sim.shape_function

# ==========================================
# 2. WARMUP
# ==========================================
println("🔥 Warming up kernels...")
timestep!(sim) # Compiles all kernels once
println("✅ Warmup complete.\n")

# ==========================================
# 3. KERNEL BENCHMARKS
# ==========================================

function check_kernel(name, func)
    println("---------------------------------------------------")
    println("Testing: $name")
    
    # Check Allocations
    allocs = @allocated func()
    if allocs == 0
        println("   💾 Allocations: 0 bytes (✅ PASS)")
    else
        println("   💾 Allocations: $(Base.format_bytes(allocs)) (⚠️ FAIL)")
    end

    # Check Speed
    # We use $ interpolation to prevent global variable lookup overhead
    bench = @benchmark $func()
    
    min_time = minimum(bench).time / 1e6 # convert ns to ms
    med_time = median(bench).time / 1e6
    println("   ⏱️  Time (min):  $(round(min_time, digits=4)) ms")
    println("   ⏱️  Time (med):  $(round(med_time, digits=4)) ms")
end

function check_scaling()
    # 1. Setup small and large particle counts
    mat, cache = NeoHookean_E_ν(1e6, 0.3, 1000.0)
    grid = Grid(Array, SVector(62,62,62), 2, SVector(0.0,0.0,0.0), SVector(1.0,1.0,1.0))
    sf = LinearHat()

    # Case A: 10 Particles
    pts_small = [MaterialPoint(SVector(rand(3)...), SVector(0.0,0.0,0.0), SVector(0.0,0.0,0.0), 1.0, 1.0, cache) for _ in 1:10]
    mpg_small = MaterialPointGroup(Array, pts_small, mat, "Small")
    
    # Case B: 10,000 Particles
    pts_large = [MaterialPoint(SVector(rand(3)...), SVector(0.0,0.0,0.0), SVector(0.0,0.0,0.0), 1.0, 1.0, cache) for _ in 1:10000]
    mpg_large = MaterialPointGroup(Array, pts_large, mat, "Large")

    # 2. Warmup
    stress_update!(mpg_small, 0.001)

    # 3. Measure
    println("--- Scaling Test: Stress Update ---")
    alloc_small = @allocated stress_update!(mpg_small, 0.001)
    alloc_large = @allocated stress_update!(mpg_large, 0.001)

    println("N=10    Allocations: $(Base.format_bytes(alloc_small))")
    println("N=10000 Allocations: $(Base.format_bytes(alloc_large))")
    
    if alloc_large > alloc_small * 10
        println("👉 VERDICT: Allocations scale with N. The leak is INSIDE the kernel loop.")
    else
        println("👉 VERDICT: Constant allocations. Likely fixed KernelAbstractions overhead.")
    end
end

# check_scaling()

# --- 1. Reset Grid ---
# Iterates over all Grid Nodes (High Memory Bandwidth)
check_kernel("reset_grid!", () -> reset_grid!(grd))

# --- 2. P2G (Particles to Grid) ---
# Iterates over Particles (Atomic Adds to Grid)
check_kernel("p2g!", () -> p2g!(mpg, grd, sf))

# --- 3. Grid Update ---
# Iterates over all Grid Nodes (Update positions/velocities)
check_kernel("grid_update!", () -> grid_update!(grd, dt))

# --- 4. G2P (Grid to Particles) ---
# Iterates over Particles (Read from Grid)
check_kernel("g2p!", () -> g2p!(mpg, grd, 1.0, dt, sf))

# --- 5. Stress Update ---
# Iterates over Particles (Constitutive Model)
check_kernel("stress_update!", () -> stress_update!(mpg, dt))

println("\n---------------------------------------------------")
println("Total Simulation Loop")
check_kernel("timestep!", () -> timestep!(sim))