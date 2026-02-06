"""
Materials
"""
abstract type AbstractMaterial end
abstract type AbstractMaterialCache end

struct NoMaterialCache <: AbstractMaterialCache end



"""
Shape Function Types
"""
abstract type AbstractShapeFunction end
struct LinearHat <: AbstractShapeFunction end
struct QuadraticBSpline <: AbstractShapeFunction end



"""
Boundary Condition Types
"""
abstract type AbstractBoundaryCondition end
struct NoSlipBoundary <: AbstractBoundaryCondition end
struct FreeSlipBoundary <: AbstractBoundaryCondition end
struct NoBoundaryCondition <: AbstractBoundaryCondition end # Should never be used in practice!